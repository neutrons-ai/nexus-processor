#!/usr/bin/env python3
"""
Event Replay Service (Local Parquet Files)

A lightweight tool that streams events from local Parquet files to various
consumers for on-the-fly processing, testing, or simulation purposes.

This script is designed for:
- Local development and testing with Parquet files
- Piping events to other Python scripts via stdout
- High-performance streaming via Arrow Flight server

For replaying events from the Iceberg lakehouse at scale, use the Spark job:
    lakehouse submit scripts/spark_replay_events.py -- --help

Supported output formats:
- stdout: JSON lines for piping to other processes
- flight: High-performance Arrow Flight server for Python/C++ clients

Usage:
    # Replay to stdout (for piping to other processes)
    python replay_events.py --run-id REF_L:218386 \
        --path ./data/events.parquet --output stdout --rate 100000

    # Replay to stdout as fast as possible
    python replay_events.py --run-id REF_L:218386 \
        --path ./data/events.parquet --output stdout

    # Start Arrow Flight server for high-performance streaming
    python replay_events.py --run-id REF_L:218386 \
        --path ./data/events.parquet --output flight --port 8815

Examples:
    # Pipe events to a reduction script
    python replay_events.py --run-id REF_L:218386 \
        --path ./data/events.parquet --output stdout --rate 50000 \
        | python my_reduction.py

    # Filter with jq and process
    python replay_events.py --run-id REF_L:218386 \
        --path ./data/events.parquet --output stdout \
        | jq 'select(.bank == "bank1")' \
        | python process_bank1.py

    # Connect to Flight server from another process
    python -c "
    import pyarrow.flight as flight
    client = flight.connect('grpc://localhost:8815')
    for batch in client.do_get(flight.Ticket(b'10000')):
        print(f'Received {len(batch)} events')
    "

See also:
    spark_replay_events.py - Distributed Spark job for Iceberg → Kafka streaming
"""

import argparse
import json
import sys
import time
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq


def read_events_from_parquet(path: str, run_id: str) -> pa.Table:
    """
    Read events from Parquet file with predicate pushdown.

    
    Args:
        path: Path to parquet file or directory
        run_id: Run ID to filter events
        
    Returns:
        PyArrow Table containing filtered events
    """
    print(f"Reading events from {path}...", file=sys.stderr)
    
    try:
        table = pq.read_table(
            path,
            filters=[("run_id", "=", run_id)]
        )
    except Exception as e:
        # Try without filter if schema doesn't support it
        print(f"Warning: Predicate pushdown failed, reading all data: {e}", file=sys.stderr)
        table = pq.read_table(path)
        if "run_id" in table.column_names:
            mask = pa.compute.equal(table.column("run_id"), run_id)
            table = table.filter(mask)
    
    return table


def replay_to_stdout(
    table: pa.Table, 
    rate: Optional[int] = None, 
    batch_size: int = 1000,
    format: str = "jsonl"
):
    """
    Stream events to stdout as JSON lines.
    
    Args:
        table: PyArrow Table of events
        rate: Events per second (None for max speed)
        batch_size: Number of events per batch
        format: Output format ('jsonl' or 'csv')
    """
    delay = batch_size / rate if rate else 0
    total_events = 0
    start_time = time.time()
    
    print(f"Streaming {table.num_rows:,} events to stdout...", file=sys.stderr)
    if rate:
        print(f"Rate limited to {rate:,} events/second", file=sys.stderr)
    
    try:
        for batch in table.to_batches(max_chunksize=batch_size):
            if format == "jsonl":
                for row in batch.to_pylist():
                    print(json.dumps(row), flush=True)
            elif format == "csv":
                # CSV format for simpler consumers
                for row in batch.to_pylist():
                    print(",".join(str(v) for v in row.values()), flush=True)
            
            total_events += len(batch)
            
            if delay:
                time.sleep(delay)
                
    except BrokenPipeError:
        # Handle pipe closed by consumer
        pass
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - start_time
        actual_rate = total_events / elapsed if elapsed > 0 else 0
        print(f"\nStreamed {total_events:,} events in {elapsed:.1f}s ({actual_rate:,.0f} events/sec)", 
              file=sys.stderr)


def start_flight_server(table: pa.Table, port: int, host: str = "0.0.0.0"):
    """
    Start Arrow Flight server for high-performance streaming.
    
    Clients can connect and request batches of events. The ticket
    specifies the batch size (default 10000).
    
    Args:
        table: PyArrow Table of events to serve
        port: Port to listen on
        host: Host to bind to
    """
    try:
        import pyarrow.flight as flight
    except ImportError:
        print("ERROR: pyarrow[flight] not installed. Install with:", file=sys.stderr)
        print("  pip install pyarrow[flight]", file=sys.stderr)
        sys.exit(1)
    
    class EventReplayServer(flight.FlightServerBase):
        """Arrow Flight server that streams events to clients."""
        
        def __init__(self, location: str, data: pa.Table):
            super().__init__(location)
            self.data = data
            self._bytes_sent = 0
            self._batches_sent = 0
        
        def do_get(self, context, ticket):
            """
            Handle client request for event stream.
            
            Ticket format: batch_size as string (e.g., b"10000")
            Default batch size: 10000 events
            """
            try:
                batch_size = int(ticket.ticket.decode()) if ticket.ticket else 10000
            except ValueError:
                batch_size = 10000
            
            print(f"Client requested stream with batch_size={batch_size}", file=sys.stderr)
            
            batches = list(self.data.to_batches(max_chunksize=batch_size))
            result_table = pa.Table.from_batches(batches, schema=self.data.schema)
            
            return flight.RecordBatchStream(result_table)
        
        def list_flights(self, context, criteria):
            """List available data streams."""
            descriptor = flight.FlightDescriptor.for_path("events")
            endpoints = [flight.FlightEndpoint(b"events", [])]
            
            yield flight.FlightInfo(
                self.data.schema,
                descriptor,
                endpoints,
                self.data.num_rows,
                self.data.nbytes
            )
        
        def get_flight_info(self, context, descriptor):
            """Get info about the event stream."""
            endpoints = [flight.FlightEndpoint(b"events", [])]
            
            return flight.FlightInfo(
                self.data.schema,
                descriptor,
                endpoints,
                self.data.num_rows,
                self.data.nbytes
            )
    
    location = f"grpc://{host}:{port}"
    server = EventReplayServer(location, table)
    
    print(f"=" * 60, file=sys.stderr)
    print(f"Arrow Flight Server", file=sys.stderr)
    print(f"=" * 60, file=sys.stderr)
    print(f"Listening on: {location}", file=sys.stderr)
    print(f"Events ready: {table.num_rows:,}", file=sys.stderr)
    print(f"Data size:    {table.nbytes / 1e6:.1f} MB", file=sys.stderr)
    print(f"Schema:       {table.schema}", file=sys.stderr)
    print(f"=" * 60, file=sys.stderr)
    print(f"Connect with:", file=sys.stderr)
    print(f"  client = pyarrow.flight.connect('{location}')", file=sys.stderr)
    print(f"  reader = client.do_get(flight.Ticket(b'10000'))", file=sys.stderr)
    print(f"  for batch in reader: process(batch)", file=sys.stderr)
    print(f"=" * 60, file=sys.stderr)
    print(f"Press Ctrl+C to stop", file=sys.stderr)
    
    try:
        server.serve()
    except KeyboardInterrupt:
        print("\nShutting down...", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Replay neutron events from storage to various consumers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID to replay (e.g., 'REF_L:218386')"
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to parquet file or directory"
    )
    
    # Output configuration
    parser.add_argument(
        "--output",
        required=True,
        choices=["stdout", "flight"],
        help="Output destination (stdout for piping, flight for Arrow server)"
    )
    parser.add_argument(
        "--format",
        default="jsonl",
        choices=["jsonl", "csv"],
        help="Output format for stdout (default: jsonl)"
    )
    
    # Rate control
    parser.add_argument(
        "--rate",
        type=int,
        help="Events per second rate limit (stdout only, default: unlimited)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for streaming (default: 1000)"
    )
    
    # Flight configuration
    parser.add_argument(
        "--port",
        type=int,
        default=8815,
        help="Arrow Flight server port (default: 8815)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Arrow Flight server host (default: 0.0.0.0)"
    )
    
    args = parser.parse_args()
    
    # Load events from Parquet
    table = read_events_from_parquet(args.path, args.run_id)
    
    # Check if we got any data
    if table.num_rows == 0:
        print(f"ERROR: No events found for run_id '{args.run_id}'", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {table.num_rows:,} events for {args.run_id}", file=sys.stderr)
    
    # Output to selected destination
    if args.output == "stdout":
        replay_to_stdout(table, args.rate, args.batch_size, args.format)
    elif args.output == "flight":
        start_flight_server(table, args.port, args.host)


if __name__ == "__main__":
    main()
