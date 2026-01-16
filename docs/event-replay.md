# Event Replay Service

Stream historical neutron event data to downstream consumers for on-the-fly processing, testing, or simulation.

## Overview

There are two replay tools depending on your data source:

| Tool | Source | Output | When to Use |
|------|--------|--------|-------------|
| `replay_events.py` | Local Parquet files | stdout, Arrow Flight | Local development, testing |
| `spark_replay_events.py` | Iceberg lakehouse | Kafka | Production, distributed systems |

## Use Cases

- **Algorithm Testing**: Test reduction algorithms with real historical data
- **ML Training**: Feed past experiment data to machine learning pipelines
- **Development**: Simulate instrument output without access to beamline
- **System Integration**: Backfill or synchronize other data systems
- **Live Dashboards**: Replay historical data for demonstration or debugging

---

## Local Replay (Parquet Files)

The `replay_events.py` script streams events from local Parquet files. It's lightweight and doesn't require Spark.

### Supported Output Formats

| Format | Throughput | Use Case |
|--------|-----------|----------|
| **stdout** | ~100K events/sec | Unix pipes, simple scripts |
| **flight** | ~10M events/sec | Python/C++ consumers, ML pipelines |

### Installation

```bash
# Arrow Flight support (optional)
pip install "pyarrow[flight]"
```

### Usage

```bash
# Replay to stdout (JSON lines)
python scripts/replay_events.py \
    --run-id REF_L:218386 \
    --path ./data/parquet_output/REF_L_218386_events.parquet \
    --output stdout

# Rate-limited replay (100K events/second)
python scripts/replay_events.py \
    --run-id REF_L:218386 \
    --path ./data/parquet_output/ \
    --output stdout \
    --rate 100000

# Start Arrow Flight server
python scripts/replay_events.py \
    --run-id REF_L:218386 \
    --path ./data/parquet_output/ \
    --output flight \
    --port 8815
```

### Piping to Other Processes

```bash
# Pipe to a reduction script
python scripts/replay_events.py \
    --run-id REF_L:218386 \
    --path ./data/events.parquet \
    --output stdout \
    | python my_reduction_algorithm.py

# Filter with jq before processing
python scripts/replay_events.py \
    --run-id REF_L:218386 \
    --path ./data/events.parquet \
    --output stdout \
    | jq 'select(.bank == "bank1")' \
    | python process_bank1_only.py

# Count events per bank
python scripts/replay_events.py \
    --run-id REF_L:218386 \
    --path ./data/events.parquet \
    --output stdout \
    | jq -r '.bank' \
    | sort | uniq -c
```

### Command Reference (Local)

```
usage: replay_events.py [-h] --run-id RUN_ID --path PATH
                        --output {stdout,flight} [--format {jsonl,csv}]
                        [--rate RATE] [--batch-size BATCH_SIZE]
                        [--port PORT] [--host HOST]

Required arguments:
  --run-id RUN_ID       Run ID to replay (e.g., 'REF_L:218386')
  --path PATH           Path to parquet file or directory
  --output {stdout,flight}
                        Output destination

Output configuration:
  --format {jsonl,csv}  Output format for stdout (default: jsonl)
  --rate RATE           Events per second rate limit (default: unlimited)
  --batch-size BATCH_SIZE
                        Batch size for streaming (default: 1000)

Flight configuration:
  --port PORT           Arrow Flight server port (default: 8815)
  --host HOST           Arrow Flight server host (default: 0.0.0.0)
```

---

## Distributed Replay (Iceberg → Kafka)

The `spark_replay_events.py` script is a distributed Spark job that replays events from Iceberg tables to Kafka. Use this for production workloads and integration with downstream systems.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Spark Cluster                            │
│  ┌─────────┐    ┌──────────────┐    ┌─────────┐            │
│  │ Iceberg │───►│ Spark Workers│───►│  Kafka  │            │
│  │ Tables  │    │ (distributed)│    │         │            │
│  └─────────┘    └──────────────┘    └─────────┘            │
└─────────────────────────────────────────────────────────────┘
                                            │
        ┌───────────────────────────────────┼────────────────┐
        │                                   │                │
        ▼                                   ▼                ▼
  ┌───────────┐                      ┌───────────┐    ┌───────────┐
  │ Python    │                      │ Reduction │    │ Dashboard │
  │ Consumer  │                      │ Service   │    │           │
  └───────────┘                      └───────────┘    └───────────┘
```

### Modes

| Mode | Description |
|------|-------------|
| **batch** | Replay all events for a run at once (default) |
| **streaming** | Continuously stream new events as they're ingested |

### Usage

```bash
# Replay a single run to Kafka
lakehouse submit scripts/spark_replay_events.py -- \
    --run-id REF_L:218386 \
    --topic event-replay \
    --bootstrap-servers kafka:9092

# Replay with rate limiting
lakehouse submit scripts/spark_replay_events.py -- \
    --run-id REF_L:218386 \
    --topic event-replay \
    --rate-limit 100000

# Replay multiple runs
lakehouse submit scripts/spark_replay_events.py -- \
    --instrument REF_L \
    --run-start 218000 \
    --run-end 218500 \
    --topic event-replay

# Replay specific detector bank
lakehouse submit scripts/spark_replay_events.py -- \
    --run-id REF_L:218386 \
    --bank bank1 \
    --topic bank1-events

# Continuous streaming mode
lakehouse submit scripts/spark_replay_events.py -- \
    --mode streaming \
    --topic event-stream \
    --checkpoint /tmp/replay-checkpoint

# Dry run (show what would be replayed)
lakehouse submit scripts/spark_replay_events.py -- \
    --run-id REF_L:218386 \
    --dry-run
```

### Kafka Message Format

Events are written to Kafka as JSON:

- **Key**: `run_id` (string) - for partitioning
- **Value**: JSON object with all event fields

Example message:
```json
{
    "instrument_id": "REF_L",
    "run_number": 218386,
    "run_id": "REF_L:218386",
    "bank": "bank1",
    "event_idx": 0,
    "pulse_index": 42,
    "event_id": 12345,
    "time_offset": 1234.567
}
```

### Command Reference (Spark)

```
usage: spark_replay_events.py [-h] [--mode {batch,streaming}]
                              [--run-id RUN_ID] [--instrument INSTRUMENT]
                              [--run-start RUN_START] [--run-end RUN_END]
                              [--bank BANK] [--topic TOPIC]
                              [--bootstrap-servers BOOTSTRAP_SERVERS]
                              [--rate-limit RATE_LIMIT]
                              [--checkpoint CHECKPOINT]
                              [--trigger-interval TRIGGER_INTERVAL]
                              [--catalog CATALOG] [--database DATABASE]
                              [--dry-run] [--console]

Mode selection:
  --mode {batch,streaming}
                        Replay mode (default: batch)

Event selection (batch mode):
  --run-id RUN_ID       Specific run ID to replay
  --instrument INSTRUMENT
                        Filter by instrument ID
  --run-start RUN_START Minimum run number
  --run-end RUN_END     Maximum run number
  --bank BANK           Filter by detector bank

Kafka configuration:
  --topic TOPIC         Kafka topic name (default: event-replay)
  --bootstrap-servers BOOTSTRAP_SERVERS
                        Kafka bootstrap servers (default: kafka:9092)
  --rate-limit RATE_LIMIT
                        Maximum events per second (batch mode)

Streaming configuration:
  --checkpoint CHECKPOINT
                        Checkpoint location (default: /tmp/spark-replay-checkpoint)
  --trigger-interval TRIGGER_INTERVAL
                        Trigger interval (default: '10 seconds')
  --start-timestamp START_TIMESTAMP
                        ISO timestamp to start streaming from

Debug options:
  --dry-run             Show what would be replayed
  --console             Write to console instead of Kafka
```

---

## Consuming Replayed Events

### From Kafka (Python)

```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'event-replay',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for message in consumer:
    event = message.value
    print(f"Event: bank={event['bank']}, pulse={event['pulse_index']}")
```

### From Arrow Flight (Python)

```python
import pyarrow.flight as flight

# Connect to server
client = flight.connect("grpc://localhost:8815")

# List available data
for info in client.list_flights():
    print(f"Available: {info.total_records} events")

# Request data stream (ticket = batch size)
ticket = flight.Ticket(b"10000")
reader = client.do_get(ticket)

# Process batches
for batch in reader:
    df = batch.data.to_pandas()
    process_events(df)
```

### From stdout (Shell)

```bash
# Process with Python
python scripts/replay_events.py ... --output stdout | python process.py

# Filter with jq
python scripts/replay_events.py ... --output stdout | jq '.bank'

# Count lines
python scripts/replay_events.py ... --output stdout | wc -l
```

---

## Integration Examples

### Real-Time Histogram

```python
#!/usr/bin/env python
"""Build histogram from streaming events."""
import sys
import json
import numpy as np

bins = np.linspace(0, 20000, 201)
histogram = np.zeros(len(bins) - 1)

for line in sys.stdin:
    event = json.loads(line)
    idx = np.searchsorted(bins, event['time_offset']) - 1
    if 0 <= idx < len(histogram):
        histogram[idx] += 1

# Output histogram
for i in range(len(histogram)):
    print(f"{bins[i]},{bins[i+1]},{int(histogram[i])}")
```

### Mantid Integration

```python
#!/usr/bin/env python
"""Process replayed events with Mantid."""
import sys
import json
from mantid.simpleapi import CreateWorkspace

events = []
for line in sys.stdin:
    event = json.loads(line)
    events.append(event)
    
    if len(events) >= 10000:
        ws = CreateWorkspace(
            DataX=[e['time_offset'] for e in events],
            DataY=[1] * len(events),
            NSpec=1
        )
        # Process workspace...
        events = []
```

---

## Troubleshooting

### No events found

- Check run_id format matches (e.g., `REF_L:218386` not `REF_L_218386`)
- Verify parquet file contains the run
- List run_ids: `python -c "import pyarrow.parquet as pq; print(pq.read_table('file.parquet').column('run_id').unique())"`

### Broken pipe (stdout)

Normal when consumer exits early (e.g., `head -n 100`). Handled gracefully.

### Flight server connection refused

- Check port not in use: `lsof -i :8815`
- Try localhost: `--host 127.0.0.1`

### Kafka connection issues

- Verify Kafka is running: `docker ps | grep kafka`
- Check topic exists or auto-creation enabled
- Use correct internal hostname in Docker: `kafka:9092`

### Spark job fails

- Check Iceberg tables exist: `lakehouse spark-sql` then `SHOW TABLES`
- Verify run exists: `SELECT COUNT(*) FROM nessie.neutron_data.events WHERE run_id = 'REF_L:218386'`
