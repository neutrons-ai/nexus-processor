#!/usr/bin/env python3
"""
Extract Events by Time Window

This script extracts neutron detector events from parquet files based on
time intervals. Events now include pulse_time directly, so no join with
daslogs is required.

Since the time within a pulse is very small (microseconds), filtering is
done on pulse_time only for efficiency.

By default, error events (bank_error_events, bank_unmapped_events) are
excluded. Use --include-error-events to include them.

Usage:
    python extract_events_by_time.py ./parquet_output --run-id REF_L:218386 --interval 60
    python extract_events_by_time.py ./parquet_output --run-id REF_L:218386 --start 0 --end 120

Example:
    # Extract events in 60-second intervals
    python extract_events_by_time.py ./output --run-id REF_L:218386 --interval 60
    
    # Extract events between 30 and 90 seconds
    python extract_events_by_time.py ./output --run-id REF_L:218386 --start 30 --end 90
    
    # Include error events
    python extract_events_by_time.py ./output --run-id REF_L:218386 --interval 60 --include-error-events
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np


def find_parquet_files(directory: str, pattern: str) -> List[Path]:
    """Find parquet files matching a pattern."""
    dir_path = Path(directory)
    files = sorted(dir_path.glob(pattern))
    return files


def load_events(
    directory: str, 
    run_id: str, 
    bank: str = None,
    include_error_events: bool = False
) -> pd.DataFrame:
    """
    Load event parquet files for a specific run.
    
    Args:
        directory: Directory containing parquet files
        run_id: Run identifier to filter by
        bank: Specific bank to load (optional)
        include_error_events: If False, exclude error and unmapped events
        
    Returns:
        DataFrame with events
    """
    dir_path = Path(directory)
    
    # Find all event files (may be chunked: bank1_events_part001.parquet)
    if bank:
        patterns = [f"*_{bank}.parquet", f"*_{bank}_part*.parquet"]
    else:
        patterns = ["*_events.parquet", "*_events_part*.parquet", 
                    "*bank*_events.parquet", "*bank*_events_part*.parquet"]
    
    event_files = []
    for pattern in patterns:
        event_files.extend(dir_path.glob(pattern))
    
    # Remove duplicates and sort
    event_files = sorted(set(event_files))
    
    # Filter out summary files
    event_files = [f for f in event_files if 'summary' not in f.name]
    
    # Filter out error/unmapped events unless requested
    if not include_error_events:
        event_files = [
            f for f in event_files 
            if 'error' not in f.name.lower() and 'unmapped' not in f.name.lower()
        ]
    
    if not event_files:
        raise FileNotFoundError(f"No event parquet files found in {directory}")
    
    print(f"  Found {len(event_files)} event file(s)")
    
    # Load and filter by run_id
    dfs = []
    total_events = 0
    for f in event_files:
        df = pd.read_parquet(f)
        if 'run_id' in df.columns:
            df = df[df['run_id'] == run_id]
        total_events += len(df)
        dfs.append(df)
        print(f"    {f.name}: {len(df):,} events")
    
    result = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print(f"  Total events: {len(result):,}")
    return result


def extract_by_interval(
    events: pd.DataFrame,
    interval_seconds: float,
) -> pd.DataFrame:
    """
    Add an interval label to each event based on its pulse time.
    
    Args:
        events: Events DataFrame with 'pulse_time' column
        interval_seconds: Size of each time interval in seconds
        
    Returns:
        Events DataFrame with 'interval' column added
    """
    events = events.copy()
    
    # Calculate interval number (0, 1, 2, ...) based on pulse_time
    events['interval'] = (events['pulse_time'] // interval_seconds).astype('Int64')
    
    # Print interval summary
    interval_counts = events.groupby('interval').size()
    print(f"\nEvents by {interval_seconds}s intervals:")
    for interval, count in interval_counts.items():
        start = interval * interval_seconds
        end = start + interval_seconds
        print(f"  [{start:6.1f}s - {end:6.1f}s): {count:,} events")
    
    return events


def extract_by_time_range(
    events: pd.DataFrame,
    start_time: float,
    end_time: float,
) -> pd.DataFrame:
    """
    Extract events within a specific time range based on pulse time.
    
    Args:
        events: Events DataFrame with 'pulse_time' column
        start_time: Start of time range in seconds
        end_time: End of time range in seconds
        
    Returns:
        Filtered events DataFrame
    """
    mask = (events['pulse_time'] >= start_time) & (events['pulse_time'] < end_time)
    filtered = events[mask].copy()
    
    print(f"\nExtracted {len(filtered):,} events in range [{start_time:.1f}s, {end_time:.1f}s)")
    
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Extract events by time window from split parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "directory",
        help="Directory containing parquet files from nexus-processor"
    )
    
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run identifier (e.g., 'REF_L:218386')"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output parquet file path (default: derived from run-id)"
    )
    
    parser.add_argument(
        "--interval",
        type=float,
        help="Split events into intervals of this many seconds"
    )
    
    parser.add_argument(
        "--start",
        type=float,
        help="Start time in seconds (requires --end)"
    )
    
    parser.add_argument(
        "--end",
        type=float,
        help="End time in seconds (requires --start)"
    )
    
    parser.add_argument(
        "--bank",
        help="Filter to specific detector bank (e.g., 'bank1_events')"
    )
    
    parser.add_argument(
        "--include-error-events",
        action="store_true",
        help="Include error and unmapped events (excluded by default)"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics only, don't write output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start is not None and args.end is None:
        parser.error("--start requires --end")
    if args.end is not None and args.start is None:
        parser.error("--end requires --start")
    if args.interval is None and args.start is None:
        parser.error("Must specify either --interval or --start/--end")
    
    print(f"Loading data for run: {args.run_id}")
    if not args.include_error_events:
        print("  (excluding error/unmapped events, use --include-error-events to include)")
    
    # Load events
    print("\nLoading events...")
    events = load_events(
        args.directory, 
        args.run_id, 
        args.bank,
        include_error_events=args.include_error_events
    )
    
    if len(events) == 0:
        print(f"ERROR: No events found for run {args.run_id}")
        print("  Did you run nexus-processor with --include-events?")
        sys.exit(1)
    
    # Check for pulse_time column
    if 'pulse_time' not in events.columns or events['pulse_time'].isna().all():
        print("ERROR: Events don't have pulse_time information.")
        print("  This file may have been created with an older version.")
        print("  Re-run nexus-processor with --include-events to regenerate.")
        sys.exit(1)
    
    # Print time range info
    valid_times = events['pulse_time'].dropna()
    print(f"\nPulse time range: {valid_times.min():.2f}s to {valid_times.max():.2f}s")
    print(f"Events with valid pulse_time: {len(valid_times):,} / {len(events):,}")
    
    # Apply time selection
    if args.interval:
        events = extract_by_interval(events, args.interval)
    else:
        events = extract_by_time_range(events, args.start, args.end)
    
    # Output
    if args.summary:
        print("\n=== Summary ===")
        print(f"Total events: {len(events):,}")
        if 'interval' in events.columns:
            print(f"Intervals: {events['interval'].nunique()}")
        print(f"Detector banks: {events['bank'].unique().tolist()}")
        print(f"Pulse time range: {events['pulse_time'].min():.3f}s - {events['pulse_time'].max():.3f}s")
    else:
        # Determine output path
        if args.output:
            output_path = args.output
            print(f"\nWriting: {output_path}")
            events.to_parquet(output_path, index=False)
            print(f"  Wrote {len(events):,} events")


if __name__ == "__main__":
    main()
