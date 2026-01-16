#!/usr/bin/env python3
"""
Spark Job: Count Events by Time Slice

This PySpark job filters neutron detector events by time intervals and returns
the count of events in each time slice. It's designed to run in the 
neutron-lakehouse environment.

The correlation works as follows:
1. Each neutron pulse has a wall-clock time recorded in the proton_charge daslog
2. Each event has a pulse_index indicating which pulse it belongs to
3. The event's absolute time = pulse_time + time_offset (microseconds)

This is the Spark equivalent of extract_events_by_time.py but queries data
directly from the Iceberg tables in the lakehouse.

Usage (via neutron-lakehouse CLI):
    lakehouse submit scripts/spark_events_by_time.py -- \
        --run-id REF_L:218386 --interval 60

    lakehouse submit scripts/spark_events_by_time.py -- \
        --run-id REF_L:218386 --start 30 --end 120

Arguments:
    --run-id        Run identifier (e.g., 'REF_L:218386')
    --interval      Time interval in seconds for grouping events
    --start         Start time in seconds (requires --end)
    --end           End time in seconds (requires --start)
    --time-log      DAS log name for pulse timing (default: proton_charge)
    --bank          Filter to specific detector bank
    --output        Output path for results (optional, writes Parquet)
    --catalog       Catalog name (default: nessie)
    --database      Database name (default: neutron_data)
"""

import argparse
import sys
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def get_spark_session() -> SparkSession:
    """Get or create SparkSession (assumes Iceberg config is already set)."""
    return SparkSession.builder.getOrCreate()


def load_pulse_times(
    spark: SparkSession,
    run_id: str,
    time_log: str = "proton_charge",
    catalog: str = "nessie",
    database: str = "neutron_data",
) -> DataFrame:
    """
    Load pulse times from the daslogs table.
    
    The proton_charge log has one entry per neutron pulse, with the 'time' column
    giving the wall-clock time offset from run start in seconds.
    
    Returns DataFrame with:
        - pulse_index: Index of the pulse (0-based)
        - pulse_time: Time in seconds from run start
    """
    daslogs_table = f"{catalog}.{database}.daslogs"
    
    # Query the daslogs table for the timing log
    pulse_df = spark.sql(f"""
        SELECT 
            time as pulse_time,
            ROW_NUMBER() OVER (ORDER BY time) - 1 as pulse_index
        FROM {daslogs_table}
        WHERE run_id = '{run_id}'
          AND log_name = '{time_log}'
        ORDER BY time
    """)
    
    pulse_count = pulse_df.count()
    
    if pulse_count == 0:
        # Try alternative log names
        alternatives = ['proton_charge', 'SampleProtonCharge', 'pcharge', 'ProtonCharge']
        
        available_logs = spark.sql(f"""
            SELECT DISTINCT log_name 
            FROM {daslogs_table}
            WHERE run_id = '{run_id}'
            LIMIT 20
        """).collect()
        
        for alt in alternatives:
            if alt != time_log:
                pulse_df = spark.sql(f"""
                    SELECT 
                        time as pulse_time,
                        ROW_NUMBER() OVER (ORDER BY time) - 1 as pulse_index
                    FROM {daslogs_table}
                    WHERE run_id = '{run_id}'
                      AND log_name = '{alt}'
                    ORDER BY time
                """)
                
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
    run_id: str,
    bank: Optional[str] = None,
    catalog: str = "nessie",
    database: str = "neutron_data",
) -> DataFrame:
    """
    Load events from the Iceberg events table.
    
    Returns DataFrame with all event columns.
    """
    events_table = f"{catalog}.{database}.events"
    
    query = f"""
        SELECT *
        FROM {events_table}
        WHERE run_id = '{run_id}'
    """
    
    if bank:
        query += f" AND bank = '{bank}'"
    
    events_df = spark.sql(query)
    
    event_count = events_df.count()
    print(f"  Found {event_count:,} events")
    
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
        description="Count events by time slice using Spark on Iceberg tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run identifier (e.g., 'REF_L:218386')"
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
    
    parser.add_argument(
        "--catalog",
        default="nessie",
        help="Catalog name (default: nessie)"
    )
    
    parser.add_argument(
        "--database",
        default="neutron_data",
        help="Database name (default: neutron_data)"
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
    print(f"Spark Events by Time Analysis")
    print(f"=" * 60)
    print(f"Run ID: {args.run_id}")
    print(f"Catalog: {args.catalog}.{args.database}")
    
    # Load pulse times
    print("\n[1/4] Loading pulse timing from daslogs...")
    pulse_times = load_pulse_times(
        spark, args.run_id, args.time_log, args.catalog, args.database
    )
    
    # Load events
    print("\n[2/4] Loading events...")
    events = load_events(
        spark, args.run_id, args.bank, args.catalog, args.database
    )
    
    if events.count() == 0:
        print(f"ERROR: No events found for run {args.run_id}")
        print("  Ensure events have been ingested with:")
        print("    nexus-to-parquet <file> --include-events")
        print("    lakehouse ingest <path>")
        spark.stop()
        sys.exit(1)
    
    # Add absolute time to events
    print("\n[3/4] Calculating absolute event times...")
    events_with_time = add_absolute_time(events, pulse_times)
    
    # Cache for multiple aggregations
    events_with_time = events_with_time.cache()
    
    # Count by time
    print("\n[4/4] Counting events by time slice...")
    
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
