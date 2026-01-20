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
4. **Embedded pulse time**: Events include `pulse_time` (seconds from run start) for efficient time-based queries without joining DAS logs
5. **Event weights**: Events include `event_weight` (default 1.0) for weighted histogramming

Event columns:
- `pulse_index`: Index of the accelerator pulse this event belongs to
- `pulse_time`: Time of the pulse in seconds from run start (enables direct time filtering)
- `event_id`: Detector pixel identifier  
- `time_offset`: Time offset within the pulse (microseconds)
- `event_weight`: Weight for this event (default 1.0)

### Iceberg Compatibility

Schemas are designed for Apache Iceberg table format:

- Partition columns (`instrument_id`, `run_number`) as first columns
- ZSTD compression for optimal size/speed
- Nested types (struct, list, map) instead of JSON strings
- Aggregated `experiment_runs` table with denormalized sample/instrument data

## Event Time Extraction

Neutron events are recorded relative to accelerator pulses. Each event now includes `pulse_time` directly, enabling efficient time-based filtering without joining with DAS logs.

```bash
# Extract events in 60-second intervals
python scripts/extract_events_by_time.py ./output --run-id REF_L:218386 --interval 60

# Extract events between 30 and 90 seconds
python scripts/extract_events_by_time.py ./output --run-id REF_L:218386 --start 30 --end 90

# Include error/unmapped events (excluded by default)
python scripts/extract_events_by_time.py ./output --run-id REF_L:218386 --interval 60 --include-error-events
```

The time fields work as follows:
- `pulse_time`: Time of the pulse in seconds from run start (stored directly in events)
- `time_offset`: Time within the pulse (microseconds)
- Absolute time = `pulse_time` + `time_offset` / 1,000,000

**Note**: Error events (`bank_error_events`) and unmapped events (`bank_unmapped_events`) are excluded by default. Use `--include-error-events` to include them.

## Event Replay (Streaming)

Stream historical events from Parquet files to downstream consumers for testing, development, or backfilling:

```bash
# Replay to stdout (pipe to other processes)
python scripts/replay_events.py \
    --run-id REF_L:218386 \
    --source parquet \
    --path ./output/REF_L_218386_events.parquet \
    --output stdout \
    --rate 100000

# Start Arrow Flight server for high-performance streaming
python scripts/replay_events.py \
    --run-id REF_L:218386 \
    --source parquet \
    --path ./output/ \
    --output flight \
    --port 8815

# Pipe to a reduction script
python scripts/replay_events.py ... --output stdout | python my_reduction.py
```

See [docs/event-replay.md](docs/event-replay.md) for full documentation.

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