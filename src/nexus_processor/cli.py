"""
Command-line interface for nexus-processor.

This module provides a Click-based CLI for converting NeXus HDF5 files
to Parquet format.
"""

import os
import sys

import click

from nexus_processor.parquet import process_nexus_file


@click.command()
@click.argument(
    'input_file',
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(file_okay=False, resolve_path=True),
    default=None,
    help='Output directory for Parquet files (default: parquet_output next to input file)',
)
@click.option(
    '--include-events/--no-events',
    default=False,
    help='Include event data extraction (default: skip events)',
)
@click.option(
    '--include-users/--no-users',
    default=False,
    help='Include user information (default: skip users)',
)
@click.option(
    '--max-events', '-m',
    type=int,
    default=None,
    help='Maximum number of events to extract per bank (default: all)',
)
def main(input_file: str, output_dir: str, include_events: bool, 
         include_users: bool, max_events: int) -> None:
    """
    Convert NeXus HDF5 files to Parquet format.

    INPUT_FILE is the path to the NeXus HDF5 file to convert.

    \b
    Examples:
      nexus-processor ~/data/REF_L_218389.nxs.h5
      nexus-processor ~/data/REF_L_218389.nxs.h5 --output-dir ./output
      nexus-processor ~/data/REF_L_218389.nxs.h5 --include-events
      nexus-processor ~/data/REF_L_218389.nxs.h5 --include-events --max-events 100000
      nexus-processor ~/data/REF_L_218389.nxs.h5 --include-users
    """
    # Determine output directory
    if output_dir is None:
        input_dir = os.path.dirname(input_file)
        if not input_dir:
            input_dir = '.'
        output_dir = os.path.join(input_dir, 'parquet_output')

    # Process the file
    try:
        output_files = process_nexus_file(
            input_file,
            output_dir,
            max_events=max_events,
            include_events=include_events,
            include_users=include_users,
        )
    except Exception as e:
        click.echo(f"Error processing file: {e}", err=True)
        sys.exit(1)

    click.echo("\nProcessing complete!")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Files created: {len(output_files)}")

    # Print summary
    click.echo("\nOutput files:")
    for data_type, path in output_files.items():
        file_size = os.path.getsize(path) / 1024  # KB
        if file_size > 1024:
            size_str = f"{file_size/1024:.1f} MB"
        else:
            size_str = f"{file_size:.1f} KB"
        click.echo(f"  {data_type}: {os.path.basename(path)} ({size_str})")


if __name__ == '__main__':
    main()
