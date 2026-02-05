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
@click.option(
    '--max-events-per-file',
    type=int,
    default=None,
    help='Maximum events per output file for chunking (default: no limit). '
         'Recommended: 10000000 (~200MB files)',
)
@click.option(
    '--format', '-f',
    type=click.Choice(['auto', 'standard', 'mantid'], case_sensitive=False),
    default='auto',
    help='Force file format (auto: detect automatically, standard: /entry/ structure, '
         'mantid: /mantid_workspace_*/ structure)',
)
def main(input_file: str, output_dir: str, include_events: bool, 
         include_users: bool, max_events: int, max_events_per_file: int,
         format: str) -> None:
    """
    Convert NeXus HDF5 files to Parquet format.

    INPUT_FILE is the path to the NeXus HDF5 file to convert.
    
    Supports two NeXus formats:
    - Standard NeXus: /entry/ structure with bank*_events (e.g., REF_L, VULCAN)
    - Mantid processed: /mantid_workspace_*/ structure (e.g., SNAP .lite files)

    \b
    Examples:
      nexus-processor ~/data/REF_L_218389.nxs.h5
      nexus-processor ~/data/REF_L_218389.nxs.h5 --output-dir ./output
      nexus-processor ~/data/REF_L_218389.nxs.h5 --include-events
      nexus-processor ~/data/REF_L_218389.nxs.h5 --include-events --max-events 100000
      nexus-processor ~/data/REF_L_218389.nxs.h5 --include-events --max-events-per-file 10000000
      nexus-processor ~/data/REF_L_218389.nxs.h5 --include-users
      
      # SNAP files (Mantid format, auto-detected)
      nexus-processor ~/data/SNAP/SNAP_64413.lite.nxs.h5 --include-events
      nexus-processor ~/data/SNAP/SNAP_64413.lite.nxs.h5 --include-events --max-events-per-file 10000000
    """
    # Determine output directory
    if output_dir is None:
        input_dir = os.path.dirname(input_file)
        if not input_dir:
            input_dir = '.'
        output_dir = os.path.join(input_dir, 'parquet_output')

    # Determine format override
    force_format = None if format == 'auto' else format

    # Process the file
    try:
        output_files = process_nexus_file(
            input_file,
            output_dir,
            max_events=max_events,
            max_events_per_file=max_events_per_file,
            include_events=include_events,
            include_users=include_users,
            force_format=force_format,
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
