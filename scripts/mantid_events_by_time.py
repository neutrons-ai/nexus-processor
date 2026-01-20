#!/usr/bin/env python3
"""
Mantid Script: Count Events by Time Slice

This Mantid script filters neutron detector events by time intervals and returns
the count of events in each time slice. It reads directly from NeXus HDF5 files.

The correlation works as follows:
1. Each neutron pulse has a wall-clock time recorded in the proton_charge log
2. Each event has a pulse time indicating which pulse it belongs to
3. The event's absolute time = pulse_time + time_offset

This is the Mantid equivalent of spark_events_by_time_parquet.py and provides
the same functionality using Mantid's built-in event filtering capabilities.

Usage:
    python mantid_events_by_time.py \
        --file data/REF_L_218386.nxs.h5 \
        --interval 60

    python mantid_events_by_time.py \
        --file data/REF_L_218386.nxs.h5 \
        --start 30 --end 120

Requirements:
    - Mantid framework installed (conda install -c mantid mantid)
    - NeXus file with event data

Arguments:
    --file          Path to NeXus HDF5 file
    --interval      Time interval in seconds for grouping events
    --start         Start time in seconds (requires --end)
    --end           End time in seconds (requires --start)
    --bank          Filter to specific detector bank
    --by-bank       Show counts per detector bank
    --output        Output CSV file for results
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

try:
    from mantid.simpleapi import *
    from mantid.api import WorkspaceGroup
    import mantid.kernel as mk
except ImportError:
    print("ERROR: Mantid framework not found.")
    print("Install with: conda install -c mantid mantid")
    sys.exit(1)


def load_nexus_file(file_path: str) -> str:
    """
    Load NeXus file into Mantid workspace.
    
    Args:
        file_path: Path to NeXus file
        
    Returns:
        Name of the loaded workspace
    """
    ws_name = Path(file_path).stem
    
    print(f"  Loading: {file_path}")
    Load(Filename=file_path, OutputWorkspace=ws_name)
    
    ws = mtd[ws_name]
    
    # Get basic info
    if isinstance(ws, WorkspaceGroup):
        # For grouped workspaces, get info from first workspace
        ws = ws[0]
    
    run = ws.getRun()
    
    print(f"  Instrument: {ws.getInstrument().getName()}")
    print(f"  Run number: {run.getProperty('run_number').value}")
    print(f"  Total events: {ws.getNumberEvents():,}")
    
    # Get time range
    if run.hasProperty('proton_charge'):
        pc_log = run.getProperty('proton_charge')
        times = pc_log.times
        print(f"  Run duration: {(times[-1] - times[0]).total_seconds():.2f}s")
        print(f"  Number of pulses: {len(times):,}")
    
    return ws_name


def get_workspace_names(ws_name: str, bank: Optional[str] = None) -> List[str]:
    """
    Get list of workspace names to analyze.
    
    Args:
        ws_name: Base workspace name
        bank: Optional bank filter
        
    Returns:
        List of workspace names
    """
    ws = mtd[ws_name]
    
    if isinstance(ws, WorkspaceGroup):
        # Grouped workspace (multiple banks)
        ws_names = [ws[i].name() for i in range(ws.size())]
        
        if bank:
            # Filter by bank name
            ws_names = [name for name in ws_names if bank in name]
            if not ws_names:
                available = [ws[i].name() for i in range(ws.size())]
                print(f"ERROR: Bank '{bank}' not found.")
                print(f"Available banks: {available}")
                sys.exit(1)
    else:
        # Single workspace
        ws_names = [ws_name]
    
    return ws_names


def count_by_interval(
    ws_names: List[str],
    interval_seconds: float,
    by_bank: bool = False
) -> pd.DataFrame:
    """
    Count events in each time interval.
    
    Args:
        ws_names: List of workspace names
        interval_seconds: Size of each time interval in seconds
        by_bank: If True, return per-bank counts
        
    Returns:
        DataFrame with interval statistics
    """
    results = []
    
    for ws_name in ws_names:
        ws = mtd[ws_name]
        run = ws.getRun()
        
        # Get the proton charge log for pulse times
        if not run.hasProperty('proton_charge'):
            print(f"WARNING: No proton_charge log in {ws_name}")
            continue
        
        pc_log = run.getProperty('proton_charge')
        pulse_times_abs = pc_log.times  # Absolute times
        
        # Convert to seconds from run start
        start_time = pulse_times_abs[0]
        pulse_times = np.array([(t - start_time).total_seconds() for t in pulse_times_abs])
        
        # Get all events
        n_spectra = ws.getNumberHistograms()
        
        for spec_idx in range(n_spectra):
            spectrum = ws.getSpectrum(spec_idx)
            event_list = spectrum.getEvents()
            
            if len(event_list) == 0:
                continue
            
            # Get event times (pulse times + offsets)
            pulse_indices = event_list.getPulseIndices()
            time_offsets = event_list.getTofs()  # Time of flight, but also used for offsets
            
            # Calculate absolute times for each event
            # Note: in event mode, getTofs() returns time offsets in microseconds
            event_times = pulse_times[pulse_indices] + time_offsets / 1e6
            
            # Assign to intervals
            intervals = np.floor(event_times / interval_seconds).astype(int)
            
            # Count events per interval
            unique_intervals, counts = np.unique(intervals, return_counts=True)
            
            for interval_num, count in zip(unique_intervals, counts):
                interval_start = interval_num * interval_seconds
                interval_end = (interval_num + 1) * interval_seconds
                
                # Get actual min/max times in this interval
                mask = intervals == interval_num
                interval_event_times = event_times[mask]
                
                result = {
                    'interval': interval_num,
                    'interval_start': interval_start,
                    'interval_end': interval_end,
                    'event_count': count,
                    'min_time': interval_event_times.min(),
                    'max_time': interval_event_times.max(),
                }
                
                if by_bank:
                    result['bank'] = ws_name
                
                results.append(result)
    
    # Combine results
    df = pd.DataFrame(results)
    
    if by_bank:
        # Group by interval and bank
        df = df.groupby(['interval', 'bank'], as_index=False).agg({
            'interval_start': 'first',
            'interval_end': 'first',
            'event_count': 'sum',
            'min_time': 'min',
            'max_time': 'max',
        }).sort_values(['interval', 'bank'])
    else:
        # Group by interval only
        df = df.groupby('interval', as_index=False).agg({
            'interval_start': 'first',
            'interval_end': 'first',
            'event_count': 'sum',
            'min_time': 'min',
            'max_time': 'max',
        }).sort_values('interval')
    
    return df


def count_in_time_range(
    ws_names: List[str],
    start_time: float,
    end_time: float
) -> pd.DataFrame:
    """
    Count events within a specific time range.
    
    Args:
        ws_names: List of workspace names
        start_time: Start of time range in seconds
        end_time: End of time range in seconds
        
    Returns:
        DataFrame with event count and statistics
    """
    total_events = 0
    min_time = float('inf')
    max_time = float('-inf')
    n_banks = 0
    
    for ws_name in ws_names:
        ws = mtd[ws_name]
        run = ws.getRun()
        
        # Get the proton charge log for pulse times
        if not run.hasProperty('proton_charge'):
            print(f"WARNING: No proton_charge log in {ws_name}")
            continue
        
        pc_log = run.getProperty('proton_charge')
        pulse_times_abs = pc_log.times
        
        # Convert to seconds from run start
        start_time_abs = pulse_times_abs[0]
        pulse_times = np.array([(t - start_time_abs).total_seconds() for t in pulse_times_abs])
        
        # Get all events
        n_spectra = ws.getNumberHistograms()
        bank_events = 0
        
        for spec_idx in range(n_spectra):
            spectrum = ws.getSpectrum(spec_idx)
            event_list = spectrum.getEvents()
            
            if len(event_list) == 0:
                continue
            
            # Get event times
            pulse_indices = event_list.getPulseIndices()
            time_offsets = event_list.getTofs()
            
            # Calculate absolute times
            event_times = pulse_times[pulse_indices] + time_offsets / 1e6
            
            # Filter by time range
            mask = (event_times >= start_time) & (event_times < end_time)
            filtered_times = event_times[mask]
            
            bank_events += len(filtered_times)
            
            if len(filtered_times) > 0:
                min_time = min(min_time, filtered_times.min())
                max_time = max(max_time, filtered_times.max())
        
        if bank_events > 0:
            total_events += bank_events
            n_banks += 1
    
    result = pd.DataFrame([{
        'start_time': start_time,
        'end_time': end_time,
        'event_count': total_events,
        'min_time': min_time if min_time != float('inf') else None,
        'max_time': max_time if max_time != float('-inf') else None,
        'n_banks': n_banks,
    }])
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Count events by time slice using Mantid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--file",
        required=True,
        help="Path to NeXus HDF5 file"
    )
    
    parser.add_argument(
        "--interval",
        type=float,
        help="Time interval in seconds for grouping events"
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
        help="Filter to specific detector bank"
    )
    
    parser.add_argument(
        "--by-bank",
        action="store_true",
        help="Show counts per detector bank"
    )
    
    parser.add_argument(
        "--output",
        help="Output CSV file for results"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start is not None and args.end is None:
        parser.error("--start requires --end")
    if args.end is not None and args.start is None:
        parser.error("--end requires --start")
    if args.interval is None and args.start is None:
        parser.error("Must specify either --interval or --start/--end")
    
    if not Path(args.file).exists():
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)
    
    print(f"=" * 60)
    print(f"Mantid Events by Time Analysis")
    print(f"=" * 60)
    
    # Load NeXus file
    print("\n[1/3] Loading NeXus file...")
    ws_name = load_nexus_file(args.file)
    
    # Get workspaces to analyze
    print("\n[2/3] Preparing workspaces...")
    ws_names = get_workspace_names(ws_name, args.bank)
    print(f"  Analyzing {len(ws_names)} workspace(s)")
    for name in ws_names[:5]:  # Show first 5
        print(f"    - {name}")
    if len(ws_names) > 5:
        print(f"    ... and {len(ws_names) - 5} more")
    
    # Count by time
    print("\n[3/3] Counting events by time slice...")
    
    if args.interval:
        result = count_by_interval(ws_names, args.interval, args.by_bank)
    else:
        result = count_in_time_range(ws_names, args.start, args.end)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if args.interval:
        print(f"\nEvents by {args.interval}s intervals:")
        print(result.to_string(index=False))
        
        # Summary statistics
        total_events = result['event_count'].sum()
        n_intervals = len(result)
        print(f"\nSummary:")
        print(f"  Total events: {total_events:,}")
        print(f"  Number of intervals: {n_intervals}")
    else:
        print(f"\nEvents in range [{args.start:.1f}s, {args.end:.1f}s):")
        print(result.to_string(index=False))
    
    # Write output if requested
    if args.output:
        print(f"\nWriting results to: {args.output}")
        result.to_csv(args.output, index=False)
        print("  Done!")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
