# nexus-processor

Convert NeXus HDF5 files to Apache Parquet format for efficient analysis in modern data lakehouse architectures.

## Overview

NeXus files (`.nxs.h5`) are the standard data format for neutron scattering facilities. While HDF5 is excellent for data acquisition, it's not optimal for large-scale analytics. This tool converts NeXus files to Parquet format, enabling:

- **Efficient querying** with column-oriented storage
- **Scalable analytics** via Apache Spark, Trino, DuckDB, etc.
- **Data lakehouse integration** with Apache Iceberg table format
- **Time-based event extraction** using pulse correlation

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Basic conversion (excludes events and users by default)
nexus-to-parquet input.nxs.h5 -o ./output

# Include neutron events
nexus-to-parquet input.nxs.h5 -o ./output --include-events

# Include events with limit (for testing)
nexus-to-parquet input.nxs.h5 -o ./output --include-events --max-events 100000

# Include user information
nexus-to-parquet input.nxs.h5 -o ./output --include-users

# Chunk large event files for Iceberg compatibility
nexus-to-parquet input.nxs.h5 -o ./output --include-events --max-events-per-file 10000000
```

## Output Files

Each NeXus file produces multiple Parquet files:

| File | Description |
|------|-------------|
| `*_metadata.parquet` | Run metadata (title, times, proton charge, etc.) |
| `*_sample.parquet` | Sample information (name, formula, mass) |
| `*_instrument.parquet` | Instrument configuration and XML definition |
| `*_daslogs.parquet` | Time-series DAS log data |
| `*_software.parquet` | Software component versions |
| `*_users.parquet` | Experiment participants (opt-in) |
| `*_<bank>_events.parquet` | Neutron events per detector bank (opt-in) |
| `*_event_summary.parquet` | Event count summaries per bank |

## Schema Design

### Partition Strategy

All tables use a composite partition key for efficient querying:

- **`instrument_id`**: Instrument identifier (e.g., `REF_L`, `VULCAN`)
- **`run_number`**: Integer run number

This enables efficient queries like:
```sql
SELECT * FROM events WHERE instrument_id = 'REF_L' AND run_number = 218386
```

### Run Identifier

Since `run_number` alone is not unique across a facility (different instruments may have the same run number), we include:

- **`run_id`**: Composite string identifier in the format `instrument_id:run_number` (e.g., `REF_L:218386`)

The colon separator was chosen because:
- It's visually clear and readable
- It's valid in most file systems
- It's commonly used in namespacing (e.g., Docker images, URIs)

### Type Choices

| Design Choice | Rationale |
|---------------|-----------|
| `large_string` over `string` | Handles variable-length text without size limits |
| `map<string, string>` for attributes | Flexible key-value storage for HDF5 attributes |
| Nullable fields | Gracefully handles missing data in NeXus files |
| Explicit schemas | Ensures type consistency across all files |
| Nested structs | Iceberg-compatible alternative to string serialization |

### Event Files

Neutron event data can be extremely large (billions of events). Design decisions:

1. **Separate files per detector bank**: Enables parallel processing
2. **Chunking support**: `--max-events-per-file` splits large banks (e.g., 10M events ≈ 200MB)
3. **Pulse correlation**: Each event includes `pulse_index` for time reconstruction
4. **Time extraction**: Events can be correlated to wall-clock time via `proton_charge` DAS log

### Iceberg Compatibility

Schemas are designed for Apache Iceberg table format:

- Partition columns (`instrument_id`, `run_number`) as first columns
- ZSTD compression for optimal size/speed
- Nested types (struct, list, map) instead of JSON strings
- Aggregated `experiment_runs` table with denormalized sample/instrument data

## Event Time Extraction

Neutron events are recorded relative to accelerator pulses. To extract events by wall-clock time:

```bash
# Extract events in 60-second intervals
python scripts/extract_events_by_time.py ./output --run-id REF_L:218386 --interval 60

# Extract events between 30 and 90 seconds
python scripts/extract_events_by_time.py ./output --run-id REF_L:218386 --start 30 --end 90
```

The correlation works via:
1. `proton_charge` DAS log records wall-clock time of each neutron pulse
2. Each event has a `pulse_index` indicating which pulse it belongs to
3. Absolute time = pulse_time + time_offset (microseconds)

## Iceberg Table Setup

Generate Iceberg table definitions:

```bash
# Generate SQL for Trino/Athena/etc.
python scripts/init_iceberg_tables.py --sql-only --output create_tables.sql

# Create tables with PySpark
spark-submit scripts/init_iceberg_tables.py --catalog my_catalog --database neutron_data
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_parquet.py::TestExtractEvents -v
```

## License

See [LICENSE](LICENSE) for details.