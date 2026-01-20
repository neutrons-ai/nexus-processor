#!/usr/bin/env python3
"""
Spark Job: Count Events by Time Slice (Parquet Version)

This PySpark job filters neutron detector events by time intervals and returns
the count of events in each time slice. It reads directly from parquet files
created by the nexus-processor tool.

The correlation works as follows:
1. Each neutron pulse has a wall-clock time recorded in the proton_charge daslog
2. Each event has a pulse_index indicating which pulse it belongs to
3. The event's absolute time = pulse_time + time_offset (microseconds)

This is the parquet-based equivalent of spark_events_by_time.py but queries data
directly from parquet files instead of Iceberg tables.

Usage:
    spark-submit spark_events_by_time_parquet.py \
        --parquet-dir data/parquet_output \
        --run-id REF_L_218386 \
        --interval 60

    spark-submit spark_events_by_time_parquet.py \
        --parquet-dir data/parquet_output \
        --run-id REF_L_218386 \
        --start 30 --end 120

Arguments:
    --parquet-dir   Directory containing parquet files for the run
    --run-id        Run identifier (e.g., 'REF_L_218386')
    --interval      Time interval in seconds for grouping events
    --start         Start time in seconds (requires --end)
    --end           End time in seconds (requires --start)
    --time-log      DAS log name for pulse timing (default: proton_charge)
    --bank          Filter to specific detector bank
    --output        Output path for results (optional, writes Parquet)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def get_spark_session() -> SparkSession:
    """Get or create SparkSession."""
    return SparkSession.builder \
        .appName("NeutronEventsTimeAnalysis") \
        .getOrCreate()


def discover_parquet_files(parquet_dir: str, run_id: str) -> dict:
    """
    Discover parquet files for a given run.
    
    Args:
        parquet_dir: Directory containing parquet files
        run_id: Run identifier (e.g., 'REF_L_218386')
        
    Returns:
        Dictionary mapping file types to paths
    """
    parquet_path = Path(parquet_dir)
    
    # Build expected filename patterns
    # Handle both 'REF_L_218386' and 'REF_L:218386' formats
    run_id_safe = run_id.replace(':', '_')
    
    file_types = {
        'metadata': f'{run_id_safe}_metadata.parquet',
        'daslogs': f'{run_id_safe}_daslogs.parquet',
        'sample': f'{run_id_safe}_sample.parquet',
        'instrument': f'{run_id_safe}_instrument.parquet',
        'software': f'{run_id_safe}_software.parquet',
    }
    
    found_files = {}
    
    for file_type, filename in file_types.items():
        file_path = parquet_path / filename
        if file_path.exists():
            found_files[file_type] = str(file_path)
    
    # Look for event files (can have multiple banks)
    event_files = list(parquet_path.glob(f'{run_id_safe}_*_events.parquet'))
    if event_files:
        found_files['events'] = [str(f) for f in event_files]
    
    return found_files


def load_pulse_times(
    spark: SparkSession,
    daslogs_path: str,
    time_log: str = "proton_charge",
) -> DataFrame:
    """
    Load pulse times from the daslogs parquet file.
    
    The proton_charge log has one entry per neutron pulse, with the 'time' column
    giving the wall-clock time offset from run start in seconds.
    
    Returns DataFrame with:
        - pulse_index: Index of the pulse (0-based)
        - pulse_time: Time in seconds from run start
    """
    # Read daslogs parquet
    daslogs_df = spark.read.parquet(daslogs_path)
    
    # Filter for the pulse timing log and create pulse index
    pulse_df = daslogs_df.filter(F.col("log_name") == time_log) \
        .select("time") \
        .withColumnRenamed("time", "pulse_time") \
        .orderBy("pulse_time")
    
    # Add pulse_index as row number
    window = Window.orderBy("pulse_time")
    pulse_df = pulse_df.withColumn(
        "pulse_index",
        F.row_number().over(window) - 1
    )
    
    pulse_count = pulse_df.count()
    
    if pulse_count == 0:
        # Try alternative log names
        alternatives = ['proton_charge', 'SampleProtonCharge', 'pcharge', 'ProtonCharge']
        
        available_logs = daslogs_df.select("log_name").distinct().limit(20).collect()
        
        for alt in alternatives:
            if alt != time_log:
                pulse_df = daslogs_df.filter(F.col("log_name") == alt) \
                    .select("time") \
                    .withColumnRenamed("time", "pulse_time") \
                    .orderBy("pulse_time")
                
                pulse_df = pulse_df.withColumn(
                    "pulse_index",
                    F.row_number().over(window) - 1
                )
                
                if pulse_df.count() > 0:
                    print(f"  Using '{alt}' as pulse timing source")
                    break
        else:
            log_names = [r.log_name for r in available_logs]
            raise ValueError(
                f"Could not find pulse timing log '{time_log}'. "
                f"Available logs (first 20): {log_names}"
            )
    else:
        print(f"  Using '{time_log}' as pulse timing source")
    
    # Cache for reuse during join
    pulse_df = pulse_df.cache()
    
    pulse_count = pulse_df.count()
    time_stats = pulse_df.agg(
        F.min("pulse_time").alias("min_time"),
        F.max("pulse_time").alias("max_time")
    ).collect()[0]
    
    print(f"  Found {pulse_count:,} pulses")
    print(f"  Time range: {time_stats.min_time:.2f}s to {time_stats.max_time:.2f}s")
    
    return pulse_df


def load_events(
    spark: SparkSession,
    event_paths: list,
    bank: Optional[str] = None,
) -> DataFrame:
    """
    Load events from parquet files.
    
    Args:
        spark: SparkSession
        event_paths: List of paths to event parquet files
        bank: Optional bank name to filter by
        
    Returns:
        DataFrame with all event columns.
    """
    # Read all event files
    events_df = spark.read.parquet(*event_paths)
    
    if bank:
        events_df = events_df.filter(F.col("bank") == bank)
    
    event_count = events_df.count()
    print(f"  Found {event_count:,} events")
    
    if event_count == 0 and bank:
        # Show available banks
        all_events = spark.read.parquet(*event_paths)
        available_banks = [r.bank for r in all_events.select("bank").distinct().collect()]
        print(f"  Available banks: {available_banks}")
    
    return events_df


def add_absolute_time(
    events_df: DataFrame,
    pulse_times_df: DataFrame,
) -> DataFrame:
    """
    Add absolute time (from run start) to each event.
    
    Absolute time = pulse_time + time_offset (converted to seconds)
    
    Args:
        events_df: DataFrame of events with pulse_index and time_offset columns
        pulse_times_df: DataFrame with pulse_index and pulse_time columns
        
    Returns:
        Events DataFrame with 'absolute_time' column added (in seconds)
    """
    # Join events with pulse times on pulse_index
    events_with_time = events_df.join(
        pulse_times_df,
        on="pulse_index",
        how="left"
    )
    
    # Calculate absolute time (time_offset is in microseconds)
    events_with_time = events_with_time.withColumn(
        "absolute_time",
        F.col("pulse_time") + (F.col("time_offset") / 1e6)
    )
    
    # Count valid times
    valid_count = events_with_time.filter(F.col("absolute_time").isNotNull()).count()
    total_count = events_with_time.count()
    
    print(f"  Events with valid absolute time: {valid_count:,} / {total_count:,}")
    
    return events_with_time


def count_by_interval(
    events_df: DataFrame,
    interval_seconds: float,
) -> DataFrame:
    """
    Count events in each time interval.
    
    Args:
        events_df: Events DataFrame with 'absolute_time' column
        interval_seconds: Size of each time interval in seconds
        
    Returns:
        DataFrame with interval statistics
    """
    # Calculate interval number (0, 1, 2, ...)
    events_with_interval = events_df.withColumn(
        "interval",
        F.floor(F.col("absolute_time") / interval_seconds).cast("long")
    )
    
    # Group by interval and count
    interval_counts = events_with_interval.groupBy("interval").agg(
        F.count("*").alias("event_count"),
        F.min("absolute_time").alias("min_time"),
        F.max("absolute_time").alias("max_time"),
        F.countDistinct("bank").alias("n_banks"),
        F.countDistinct("pulse_index").alias("n_pulses")
    ).orderBy("interval")
    
    # Add interval time bounds
    result = interval_counts.withColumn(
        "interval_start",
        F.col("interval") * interval_seconds
    ).withColumn(
        "interval_end",
        (F.col("interval") + 1) * interval_seconds
    ).select(
        "interval",
        "interval_start",
        "interval_end",
        "event_count",
        "n_banks",
        "n_pulses",
        "min_time",
        "max_time"
    )
    
    return result


def count_in_time_range(
    events_df: DataFrame,
    start_time: float,
    end_time: float,
) -> DataFrame:
    """
    Count events within a specific time range.
    
    Args:
        events_df: Events DataFrame with 'absolute_time' column
        start_time: Start of time range in seconds
        end_time: End of time range in seconds
        
    Returns:
        DataFrame with event count and statistics
    """
    filtered = events_df.filter(
        (F.col("absolute_time") >= start_time) & 
        (F.col("absolute_time") < end_time)
    )
    
    result = filtered.agg(
        F.lit(start_time).alias("start_time"),
        F.lit(end_time).alias("end_time"),
        F.count("*").alias("event_count"),
        F.min("absolute_time").alias("min_time"),
        F.max("absolute_time").alias("max_time"),
        F.countDistinct("bank").alias("n_banks"),
        F.countDistinct("pulse_index").alias("n_pulses")
    )
    
    return result


def count_by_bank_and_interval(
    events_df: DataFrame,
    interval_seconds: float,
) -> DataFrame:
    """
    Count events per detector bank in each time interval.
    
    Args:
        events_df: Events DataFrame with 'absolute_time' column
        interval_seconds: Size of each time interval in seconds
        
    Returns:
        DataFrame with per-bank interval statistics
    """
    events_with_interval = events_df.withColumn(
        "interval",
        F.floor(F.col("absolute_time") / interval_seconds).cast("long")
    )
    
    result = events_with_interval.groupBy("interval", "bank").agg(
        F.count("*").alias("event_count"),
        F.min("absolute_time").alias("min_time"),
        F.max("absolute_time").alias("max_time"),
        F.countDistinct("pulse_index").alias("n_pulses")
    ).orderBy("interval", "bank")
    
    # Add interval time bounds
    result = result.withColumn(
        "interval_start",
        F.col("interval") * interval_seconds
    ).withColumn(
        "interval_end",
        (F.col("interval") + 1) * interval_seconds
    )
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Count events by time slice using Spark on parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--parquet-dir",
        required=True,
        help="Directory containing parquet files for the run"
    )
    
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run identifier (e.g., 'REF_L_218386' or 'REF_L:218386')"
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
        "--time-log",
        default="proton_charge",
        help="DAS log name for pulse timing (default: proton_charge)"
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
        help="Output path for results (Parquet format)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start is not None and args.end is None:
        parser.error("--start requires --end")
    if args.end is not None and args.start is None:
        parser.error("--end requires --start")
    if args.interval is None and args.start is None:
        parser.error("Must specify either --interval or --start/--end")
    
    # Initialize Spark
    spark = get_spark_session()
    
    print(f"=" * 60)
    print(f"Spark Events by Time Analysis (Parquet)")
    print(f"=" * 60)
    print(f"Run ID: {args.run_id}")
    print(f"Parquet directory: {args.parquet_dir}")
    
    # Discover parquet files
    print("\n[1/5] Discovering parquet files...")
    parquet_files = discover_parquet_files(args.parquet_dir, args.run_id)
    
    if 'daslogs' not in parquet_files:
        print(f"ERROR: Could not find daslogs parquet file for run {args.run_id}")
        print(f"Expected: {args.parquet_dir}/{args.run_id.replace(':', '_')}_daslogs.parquet")
        spark.stop()
        sys.exit(1)
    
    if 'events' not in parquet_files:
        print(f"ERROR: Could not find event parquet files for run {args.run_id}")
        print(f"Expected: {args.parquet_dir}/{args.run_id.replace(':', '_')}_*_events.parquet")
        print("\nTo extract events from a NeXus file, use:")
        print("  nexus-processor <nexus_file> --output-dir <dir> --include-events")
        spark.stop()
        sys.exit(1)
    
    print(f"  Found files:")
    for file_type, path in parquet_files.items():
        if isinstance(path, list):
            print(f"    {file_type}: {len(path)} files")
            for p in path[:3]:  # Show first 3
                print(f"      - {Path(p).name}")
            if len(path) > 3:
                print(f"      ... and {len(path) - 3} more")
        else:
            print(f"    {file_type}: {Path(path).name}")
    
    # Load pulse times
    print(f"\n[2/5] Loading pulse timing from daslogs...")
    pulse_times = load_pulse_times(spark, parquet_files['daslogs'], args.time_log)
    
    # Load events
    print(f"\n[3/5] Loading events...")
    events = load_events(spark, parquet_files['events'], args.bank)
    
    if events.count() == 0:
        print(f"ERROR: No events found")
        if args.bank:
            print(f"  Bank filter: {args.bank}")
        spark.stop()
        sys.exit(1)
    
    # Add absolute time to events
    print(f"\n[4/5] Calculating absolute event times...")
    events_with_time = add_absolute_time(events, pulse_times)
    
    # Cache for multiple aggregations
    events_with_time = events_with_time.cache()
    
    # Count by time
    print(f"\n[5/5] Counting events by time slice...")
    
    if args.interval:
        if args.by_bank:
            result = count_by_bank_and_interval(events_with_time, args.interval)
        else:
            result = count_by_interval(events_with_time, args.interval)
    else:
        result = count_in_time_range(events_with_time, args.start, args.end)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if args.interval:
        print(f"\nEvents by {args.interval}s intervals:")
        result.show(100, truncate=False)
        
        # Summary statistics
        total_events = result.agg(F.sum("event_count")).collect()[0][0]
        n_intervals = result.count()
        print(f"\nSummary:")
        print(f"  Total events: {total_events:,}")
        print(f"  Number of intervals: {n_intervals}")
    else:
        print(f"\nEvents in range [{args.start:.1f}s, {args.end:.1f}s):")
        result.show(truncate=False)
    
    # Write output if requested
    if args.output:
        print(f"\nWriting results to: {args.output}")
        result.write.mode("overwrite").parquet(args.output)
        print("  Done!")
    
    # Clean up
    events_with_time.unpersist()
    pulse_times.unpersist()
    spark.stop()
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
