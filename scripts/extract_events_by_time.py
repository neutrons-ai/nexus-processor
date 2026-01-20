#!/usr/bin/env python3
"""
Extract Events by Time Window

This script extracts neutron detector events from parquet files based on
time intervals. It correlates events to wall-clock time using the
proton_charge daslog, which records the time of each neutron pulse.

The correlation works as follows:
1. Each neutron pulse has a wall-clock time recorded in the proton_charge log
2. Each event has a pulse_index indicating which pulse it belongs to
3. The event's absolute time = pulse_time + time_offset (microseconds)

Works with split parquet files from nexus-processor (events and daslogs
are in separate files).

Usage:
    python extract_events_by_time.py ./parquet_output --run-id REF_L:218386 --interval 60
    python extract_events_by_time.py ./parquet_output --run-id REF_L:218386 --start 0 --end 120

Example:
    # Extract events in 60-second intervals
    python extract_events_by_time.py ./output --run-id REF_L:218386 --interval 60
    
    # Extract events between 30 and 90 seconds
    python extract_events_by_time.py ./output --run-id REF_L:218386 --start 30 --end 90
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


def load_daslogs(directory: str, run_id: str) -> pd.DataFrame:
    """Load daslogs parquet file for a specific run."""
    dir_path = Path(directory)
    
    # Try to find daslogs file
    daslog_files = list(dir_path.glob("*_daslogs.parquet"))
    
    if not daslog_files:
        raise FileNotFoundError(f"No daslogs parquet file found in {directory}")
    
    # Load and filter by run_id if multiple runs present
    dfs = []
    for f in daslog_files:
        df = pd.read_parquet(f)
        if 'run_id' in df.columns:
            df = df[df['run_id'] == run_id]
        dfs.append(df)
    
    result = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print(f"  Loaded {len(result):,} daslog records")
    return result


def load_events(directory: str, run_id: str, bank: str = None) -> pd.DataFrame:
    """Load event parquet files for a specific run."""
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


def get_pulse_times(daslogs: pd.DataFrame, time_log: str = "proton_charge") -> pd.Series:
    """
    Extract pulse times from the proton_charge (or similar) DAS log.
    
    The proton_charge log has one entry per pulse, with the 'time' column
    giving the wall-clock time offset from run start in seconds.
    
    Args:
        daslogs: DAS logs DataFrame
        time_log: Name of the DAS log to use for pulse timing
        
    Returns:
        Series with pulse index as index and time (seconds) as values
    """
    # Find the proton_charge log
    pulse_log = daslogs[daslogs['log_name'] == time_log].copy()
    
    if len(pulse_log) == 0:
        # Try alternative names
        alternatives = ['proton_charge', 'SampleProtonCharge', 'pcharge', 'ProtonCharge']
        for alt in alternatives:
            pulse_log = daslogs[daslogs['log_name'] == alt].copy()
            if len(pulse_log) > 0:
                print(f"  Using '{alt}' as pulse timing source")
                break
        else:
            available = daslogs['log_name'].unique()[:20]
            raise ValueError(
                f"Could not find pulse timing log '{time_log}'. "
                f"Available logs (first 20): {list(available)}"
            )
    else:
        print(f"  Using '{time_log}' as pulse timing source")
    
    # Sort by time and create pulse index
    pulse_log = pulse_log.sort_values('time').reset_index(drop=True)
    pulse_times = pulse_log['time'].values
    
    print(f"  Found {len(pulse_times):,} pulses")
    print(f"  Time range: {pulse_times.min():.2f}s to {pulse_times.max():.2f}s")
    
    return pd.Series(pulse_times, name='pulse_time')


def add_absolute_time_to_events(
    events: pd.DataFrame, 
    pulse_times: pd.Series
) -> pd.DataFrame:
    """
    Add absolute time (from run start) to each event.
    
    Absolute time = pulse_time + time_offset (converted to seconds)
    
    Args:
        events: DataFrame of events with pulse_index and time_offset columns
        pulse_times: Series of pulse times indexed by pulse number
        
    Returns:
        Events DataFrame with 'absolute_time' column added (in seconds)
    """
    events = events.copy()
    
    # Get pulse time for each event's pulse_index
    events['pulse_time'] = events['pulse_index'].map(
        lambda idx: pulse_times.iloc[int(idx)] if pd.notna(idx) and int(idx) < len(pulse_times) else np.nan
    )
    
    # Calculate absolute time (time_offset is in microseconds)
    #events['absolute_time'] = events['pulse_time'] + (events['time_offset'] / 1e6)
    
    # Count events with valid times
    valid = events['absolute_time'].notna()
    print(f"  Events with valid absolute time: {valid.sum():,} / {len(events):,}")
    
    return events


def extract_by_interval(
    events: pd.DataFrame,
    interval_seconds: float,
) -> pd.DataFrame:
    """
    Add an interval label to each event based on its absolute time.
    
    Args:
        events: Events DataFrame with 'absolute_time' column
        interval_seconds: Size of each time interval in seconds
        
    Returns:
        Events DataFrame with 'interval' column added
    """
    events = events.copy()
    
    # Calculate interval number (0, 1, 2, ...)
    events['interval'] = (events['absolute_time'] // interval_seconds).astype('Int64')
    
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
    Extract events within a specific time range.
    
    Args:
        events: Events DataFrame with 'absolute_time' column
        start_time: Start of time range in seconds
        end_time: End of time range in seconds
        
    Returns:
        Filtered events DataFrame
    """
    mask = (events['absolute_time'] >= start_time) & (events['absolute_time'] < end_time)
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
        "--time-log",
        default="proton_charge",
        help="DAS log name for pulse timing (default: proton_charge)"
    )
    
    parser.add_argument(
        "--bank",
        help="Filter to specific detector bank (e.g., 'bank1_events')"
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
    
    # Load daslogs
    print("\nLoading DAS logs...")
    daslogs = load_daslogs(args.directory, args.run_id)
    
    if len(daslogs) == 0:
        print(f"ERROR: No daslogs found for run {args.run_id}")
        sys.exit(1)
    
    # Load events
    print("\nLoading events...")
    events = load_events(args.directory, args.run_id, args.bank)
    
    if len(events) == 0:
        print(f"ERROR: No events found for run {args.run_id}")
        print("  Did you run nexus-processor with --include-events?")
        sys.exit(1)
    
    # Check for pulse_index
    if 'pulse_index' not in events.columns or events['pulse_index'].isna().all():
        print("ERROR: Events don't have pulse_index information.")
        print("  This file may have been created with an older version.")
        print("  Re-run nexus-processor with --include-events to regenerate.")
        sys.exit(1)
    
    # Get pulse times
    print("\nExtracting pulse timing...")
    pulse_times = get_pulse_times(daslogs, args.time_log)
    
    # Add absolute time to events
    print("\nCalculating absolute event times...")
    events = add_absolute_time_to_events(events, pulse_times)
    
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
        print(f"Time range: {events['absolute_time'].min():.3f}s - {events['absolute_time'].max():.3f}s")
    else:
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            run_id_safe = args.run_id.replace(':', '_')
            if args.interval:
                suffix = f"_interval_{int(args.interval)}s"
            else:
                suffix = f"_time_{int(args.start)}-{int(args.end)}s"
            output_path = Path(args.directory) / f"{run_id_safe}_events{suffix}.parquet"
        
        print(f"\nWriting: {output_path}")
        events.to_parquet(output_path, index=False)
        print(f"  Wrote {len(events):,} events")


if __name__ == "__main__":
    main()
