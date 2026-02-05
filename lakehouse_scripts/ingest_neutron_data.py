#!/usr/bin/env python3
"""
Ingest Parquet files into Neutron Data Iceberg tables.

This script loads Parquet files into the neutron_data lakehouse tables.
It's designed to be run inside the Spark container via spark-submit.

Routing Strategy:
1. Read 'iceberg_table' metadata from each parquet file's schema metadata
2. Route to the table specified in that metadata
3. Fall back to filename patterns if metadata is missing

File naming convention:
- *experiment_runs*.parquet -> experiment_runs table
- *daslogs*.parquet -> daslogs table
- *events*.parquet -> events table
- *event_summary*.parquet -> event_summary table

Usage:
    # Via spark-submit
    spark-submit ingest_neutron_data.py --input-dir /path/to/data --catalog nessie --namespace neutron_data

    # Via lakehouse CLI (once integrated)
    lakehouse ingest /path/to/data --namespace neutron_data --script ingest_neutron_data.py
"""

import argparse
import glob
import os
import sys
from typing import Optional

# Mapping from file patterns to table names (fallback when metadata is missing)
FILE_TABLE_MAPPING = {
    "experiment_runs": "experiment_runs",
    "daslogs": "daslogs",
    "events": "events",
    "event_summary": "event_summary",
}


def get_target_table_from_metadata(file_path: str) -> Optional[str]:
    """
    Read the target Iceberg table name from a parquet file's schema metadata.

    Args:
        file_path: Path to the parquet file (local or S3)

    Returns:
        Table name if found in metadata, None otherwise
    """
    try:
        # For S3 paths, we need to use Spark to read - handled separately
        if file_path.startswith("s3"):
            return None

        # Only import pyarrow if we need it (for local files)
        try:
            import pyarrow.parquet as pq
        except ImportError:
            print(
                f"  Warning: pyarrow not available, cannot read metadata from {file_path}"
            )
            return None

        pf = pq.ParquetFile(file_path)
        metadata = pf.schema_arrow.metadata
        if metadata and b"iceberg_table" in metadata:
            return metadata[b"iceberg_table"].decode("utf-8")
    except Exception as e:
        print(f"  Warning: Could not read metadata from {file_path}: {e}")
    return None


def get_target_table_from_filename(file_path: str) -> Optional[str]:
    """
    Determine target table from filename pattern (fallback method).

    Args:
        file_path: Path to the parquet file

    Returns:
        Table name if pattern matches, None otherwise
    """
    filename = os.path.basename(file_path).lower()

    # Check for exact matches first
    for pattern, table in FILE_TABLE_MAPPING.items():
        if pattern in filename and filename.endswith(".parquet"):
            return table

    return None


def find_parquet_files(input_dir: str) -> dict:
    """
    Find Parquet files and map them to tables using embedded metadata.

    Strategy:
    1. First, try to read 'iceberg_table' from each file's parquet metadata
    2. Fall back to filename pattern matching if metadata is missing

    Args:
        input_dir: Directory containing Parquet files

    Returns:
        Dictionary mapping table names to list of file paths
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    # Dynamic result dict - tables discovered from metadata
    result = {}
    files_without_metadata = []

    # Handle S3 paths
    if input_dir.startswith("s3"):
        # Use Spark's Hadoop filesystem to list files
        hadoop_conf = spark._jsc.hadoopConfiguration()
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jvm.java.net.URI(input_dir), hadoop_conf
        )
        path = spark._jvm.org.apache.hadoop.fs.Path(input_dir)

        if fs.exists(path):
            file_statuses = fs.listStatus(path)
            for file_status in file_statuses:
                file_path = str(file_status.getPath())
                if file_path.endswith(".parquet"):
                    # For S3, fall back to filename patterns
                    # (reading individual file metadata from S3 is expensive)
                    table = get_target_table_from_filename(file_path)
                    if table:
                        if table not in result:
                            result[table] = []
                        result[table].append(file_path)
                    else:
                        files_without_metadata.append(file_path)
    else:
        # Local filesystem - can read metadata efficiently
        search_patterns = [
            os.path.join(input_dir, "*.parquet"),
            os.path.join(input_dir, "**", "*.parquet"),
        ]

        all_files = set()
        for pattern in search_patterns:
            all_files.update(glob.glob(pattern, recursive=True))

        for file_path in all_files:
            # Try metadata first
            table = get_target_table_from_metadata(file_path)

            # Fall back to filename pattern
            if not table:
                table = get_target_table_from_filename(file_path)

            if table:
                if table not in result:
                    result[table] = []
                result[table].append(file_path)
            else:
                files_without_metadata.append(file_path)

    if files_without_metadata:
        print(
            f"\n  Warning: {len(files_without_metadata)} file(s) could not be routed:"
        )
        for f in files_without_metadata[:5]:  # Show first 5
            print(f"    - {os.path.basename(f)}")
        if len(files_without_metadata) > 5:
            print(f"    ... and {len(files_without_metadata) - 5} more")

    return result


def ingest_files(
    spark, files: list, table_name: str, catalog: str, namespace: str, mode: str
) -> int:
    """
    Ingest a list of Parquet files into an Iceberg table.

    Args:
        spark: SparkSession
        files: List of file paths to ingest
        table_name: Target table name
        catalog: Catalog name
        namespace: Namespace name
        mode: Write mode ('append' or 'overwrite')

    Returns:
        Number of rows ingested
    """
    full_table = f"{catalog}.{namespace}.{table_name}"

    print(f"\nIngesting into {full_table}...")
    print(f"  Files: {len(files)}")
    print(f"  Mode: {mode}")

    # Read all files
    df = spark.read.parquet(*files)
    row_count = df.count()

    print(f"  Rows: {row_count:,}")

    # Write to Iceberg table
    if mode == "overwrite":
        df.writeTo(full_table).using("iceberg").createOrReplace()
    else:
        df.writeTo(full_table).using("iceberg").append()

    print(f"  ✓ Ingested {row_count:,} rows")
    return row_count


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Parquet files into neutron data Iceberg tables"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Input directory with Parquet files"
    )
    parser.add_argument(
        "--catalog", default="nessie", help="Catalog name (default: nessie)"
    )
    parser.add_argument(
        "--namespace",
        default="neutron_data",
        help="Namespace name (default: neutron_data)",
    )
    # Keep --database as alias for backwards compatibility
    parser.add_argument(
        "--database", dest="namespace", help="[DEPRECATED] Use --namespace instead"
    )
    parser.add_argument(
        "--mode",
        choices=["append", "overwrite"],
        default="append",
        help="Write mode (default: append)",
    )
    parser.add_argument(
        "--table", help="Specific table to ingest (skip auto-detection)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show routing without ingesting"
    )
    args = parser.parse_args()

    try:
        from pyspark.sql import SparkSession
    except ImportError:
        print("ERROR: PySpark not available. Run this with spark-submit.")
        sys.exit(1)

    # Get existing Spark session
    spark = SparkSession.builder.getOrCreate()

    print("Neutron Data Ingestion")
    print(f"{'=' * 50}")
    print(f"Scanning for Parquet files in: {args.input_dir}")
    print(f"Target namespace: {args.catalog}.{args.namespace}")

    if args.table:
        # Ingest specific table
        if args.input_dir.endswith(".parquet"):
            files = [args.input_dir]
        else:
            files = glob.glob(os.path.join(args.input_dir, "*.parquet"))

        if not files:
            print("No Parquet files found")
            sys.exit(1)

        if args.dry_run:
            print(f"\n[DRY RUN] Would ingest {len(files)} file(s) into {args.table}")
            for f in files:
                print(f"  - {os.path.basename(f)}")
            sys.exit(0)

        total_rows = ingest_files(
            spark, files, args.table, args.catalog, args.namespace, args.mode
        )
    else:
        # Auto-detect and ingest all tables
        file_mapping = find_parquet_files(args.input_dir)

        if not file_mapping:
            print("\nNo routable Parquet files found.")
            print("Files must either:")
            print("  1. Have 'iceberg_table' in parquet schema metadata, or")
            print("  2. Match a known filename pattern:")
            for pattern, table in FILE_TABLE_MAPPING.items():
                print(f"     *{pattern}*.parquet -> {table}")
            sys.exit(1)

        # Show routing summary
        print("\nFile routing summary:")
        for table_name, files in sorted(file_mapping.items()):
            print(f"  {table_name}: {len(files)} file(s)")

        if args.dry_run:
            print("\n[DRY RUN] Would ingest the following:")
            for table_name, files in sorted(file_mapping.items()):
                print(f"\n  Table: {args.catalog}.{args.namespace}.{table_name}")
                for f in files:
                    print(f"    - {os.path.basename(f)}")
            sys.exit(0)

        total_rows = 0
        tables_ingested = 0
        failed_tables = []

        for table_name, files in sorted(file_mapping.items()):
            if files:
                try:
                    rows = ingest_files(
                        spark,
                        files,
                        table_name,
                        args.catalog,
                        args.namespace,
                        args.mode,
                    )
                    total_rows += rows
                    tables_ingested += 1
                except Exception as e:
                    failed_tables.append((table_name, str(e)))
                    print(f"  ✗ Failed to ingest {table_name}: {e}")

        if tables_ingested == 0 and failed_tables:
            print(f"\n✗ All {len(failed_tables)} table(s) failed to ingest.")
            for table_name, error in failed_tables:
                print(f"  {table_name}: {error}")
            sys.exit(1)

    print(f"\n{'=' * 50}")
    print("✓ Ingestion complete!")
    print(f"Total rows ingested: {total_rows:,}")
    if failed_tables:
        print(f"Failed tables: {len(failed_tables)}")
        for table_name, _ in failed_tables:
            print(f"  - {table_name}")
    print(f"{'=' * 50}")

    spark.stop()


if __name__ == "__main__":
    main()
