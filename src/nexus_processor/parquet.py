"""
NeXus HDF5 to Parquet Conversion Module

This module provides functions to read NeXus format HDF5 files (commonly used
in neutron scattering experiments) and convert data into Parquet files organized
by data category.

Output modes:
  Split files (default): Creates separate parquet files for each category:
    - metadata.parquet: Run-level metadata (title, times, experiment info)
    - daslogs.parquet: Data Acquisition System time series logs
    - sample.parquet: Sample information
    - instrument.parquet: Instrument configuration
    - software.parquet: Software/provenance information
    - users.parquet: User/experimenter information (optional)
    - *_events.parquet: Neutron detector event data per bank (optional)

  Single file (--single-file): Creates one combined parquet file:
    - combined.parquet: All data (daslogs, events, users) with metadata
      denormalized as additional columns (prefixed with meta_, sample_, etc.)
    - A 'record_type' column distinguishes between 'daslog' and 'event' rows

Splitting advantages:
  - Consistent schema per file (same columns for all rows)
  - Query only what you need without loading everything
  - Better for very large event datasets
  - Easier to append or update individual components
  - Works well with data processing pipelines

Single file advantages:
  - Single file contains everything for a run
  - Run metadata travels with every record
  - Simpler for sharing and archiving
  - Easy to filter by record_type for analysis

For command-line usage, use the `nexus-processor` CLI command.
See `nexus-processor --help` for details.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nexus_processor.schemas import (
    METADATA_SCHEMA,
    SAMPLE_SCHEMA,
    INSTRUMENT_SCHEMA,
    SOFTWARE_SCHEMA,
    USERS_SCHEMA,
    DASLOGS_SCHEMA,
    EVENTS_SCHEMA,
    EVENT_SUMMARY_SCHEMA,
    try_parse_numeric,
    normalize_to_string,
    build_attribute_map,
    extract_known_fields,
)

from nexus_processor.mantid import (
    detect_nexus_format,
    get_mantid_workspace_name,
    get_mantid_instrument_id,
    extract_mantid_metadata,
    extract_mantid_instrument_info,
    extract_mantid_sample_info,
    extract_mantid_logs,
    extract_mantid_events,
)



def _write_table_with_metadata(table: pa.Table, output_path: str, iceberg_table: str) -> None:
    """
    Write a PyArrow table with Iceberg routing metadata.
    
    Embeds 'iceberg_table' in the parquet file's schema metadata so that
    downstream ingestion tools can automatically route files to the correct
    Iceberg table without hardcoded filename patterns.
    
    Args:
        table: PyArrow table to write
        output_path: Path to write the parquet file
        iceberg_table: Target Iceberg table name (e.g., 'daslogs', 'events')
    """
    existing_metadata = table.schema.metadata or {}
    new_metadata = {
        **existing_metadata,
        b'iceberg_table': iceberg_table.encode('utf-8'),
    }
    table = table.replace_schema_metadata(new_metadata)
    pq.write_table(table, output_path)


class _ChunkedParquetWriter:
    """
    Context manager for writing parquet files in chunks.
    
    Uses pq.ParquetWriter to append row groups instead of overwriting.
    Adds Iceberg routing metadata to the schema.
    """
    def __init__(self, output_path: str, schema: pa.Schema, iceberg_table: str):
        self.output_path = output_path
        # Add Iceberg metadata to schema
        existing_metadata = schema.metadata or {}
        new_metadata = {
            **existing_metadata,
            b'iceberg_table': iceberg_table.encode('utf-8'),
        }
        self.schema = schema.with_metadata(new_metadata)
        self.writer = None
    
    def __enter__(self):
        self.writer = pq.ParquetWriter(self.output_path, self.schema)
        return self
    
    def write_chunk(self, table: pa.Table) -> None:
        """Write a chunk (row group) to the parquet file."""
        if self.writer is None:
            raise RuntimeError("Writer not initialized. Use as context manager.")
        # Ensure table matches the schema with metadata
        table = table.cast(self.schema)
        self.writer.write_table(table)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            self.writer.close()


def safe_decode(value: Any) -> Any:
    """
    Safely decode bytes to string and handle numpy arrays.
    
    Args:
        value: Any value that might need decoding
        
    Returns:
        Decoded/converted value suitable for a DataFrame
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        # Flatten the array
        flat = value.flatten()
        if flat.dtype.kind in ('S', 'U', 'O'):  # String types
            decoded = [safe_decode(v) for v in flat]
            if len(decoded) == 1:
                return decoded[0]
            return decoded
        else:
            if flat.size == 1:
                return flat[0].item() if hasattr(flat[0], 'item') else flat[0]
            return flat.tolist()
    elif isinstance(value, (np.integer, np.floating)):
        return value.item()
    elif isinstance(value, (list, tuple)):
        decoded = [safe_decode(v) for v in value]
        return decoded
    return value


def make_run_id(instrument_id: str, run_number: int) -> str:
    """
    Create a unique run identifier by combining instrument_id and run_number.
    
    Args:
        instrument_id: Instrument identifier (e.g., 'REF_L')
        run_number: Run number (e.g., 218386)
        
    Returns:
        Composite run ID in format 'instrument_id:run_number' (e.g., 'REF_L:218386')
    """
    return f"{instrument_id}:{run_number}"


def read_dataset_value(dataset: h5py.Dataset) -> Any:
    """
    Read a dataset value and decode it appropriately.
    
    Args:
        dataset: HDF5 dataset to read
        
    Returns:
        The decoded value
    """
    try:
        value = dataset[()]
        return safe_decode(value)
    except Exception as e:
        print(f"Warning: Could not read dataset {dataset.name}: {e}")
        return None


def extract_entry_metadata(h5file: h5py.File, entry_name: str = 'entry') -> Dict[str, Any]:
    """
    Extract run-level metadata from the NeXus entry group.
    
    Args:
        h5file: Open HDF5 file handle
        entry_name: Name of the entry group (default 'entry')
        
    Returns:
        Dictionary of metadata fields
    """
    entry = h5file.get(entry_name)
    if entry is None:
        return {}
    
    metadata = {}
    
    # Direct scalar datasets in entry
    scalar_fields = [
        'definition', 'duration', 'end_time', 'entry_identifier',
        'experiment_identifier', 'experiment_title', 'notes',
        'proton_charge', 'raw_frames', 'run_number', 'start_time',
        'title', 'total_counts', 'total_other_counts', 'total_uncounted_counts'
    ]
    
    for field in scalar_fields:
        if field in entry:
            metadata[field] = read_dataset_value(entry[field])
    
    # Add file-level attributes
    for attr_name, attr_value in h5file.attrs.items():
        metadata[f'file_attr_{attr_name}'] = safe_decode(attr_value)
    
    # Add entry-level attributes
    for attr_name, attr_value in entry.attrs.items():
        metadata[f'entry_attr_{attr_name}'] = safe_decode(attr_value)
    
    return metadata


def extract_sample_info(h5file: h5py.File, entry_name: str = 'entry') -> Dict[str, Any]:
    """
    Extract sample information from the NeXus file.
    
    Args:
        h5file: Open HDF5 file handle
        entry_name: Name of the entry group
        
    Returns:
        Dictionary of sample fields
    """
    sample_path = f'{entry_name}/sample'
    sample = h5file.get(sample_path)
    if sample is None:
        return {}
    
    sample_info = {}
    
    # Iterate through all datasets in sample group
    for key in sample.keys():
        if isinstance(sample[key], h5py.Dataset):
            sample_info[key] = read_dataset_value(sample[key])
    
    return sample_info


def extract_instrument_info(h5file: h5py.File, entry_name: str = 'entry') -> Dict[str, Any]:
    """
    Extract instrument configuration from the NeXus file.
    
    Args:
        h5file: Open HDF5 file handle
        entry_name: Name of the entry group
        
    Returns:
        Dictionary of instrument fields
    """
    instrument_path = f'{entry_name}/instrument'
    instrument = h5file.get(instrument_path)
    if instrument is None:
        return {}
    
    instrument_info = {}
    
    # Direct datasets
    for key in instrument.keys():
        item = instrument[key]
        if isinstance(item, h5py.Dataset):
            instrument_info[key] = read_dataset_value(item)
        elif isinstance(item, h5py.Group):
            # Handle nested groups (like instrument_xml)
            if key == 'instrument_xml':
                for subkey in item.keys():
                    if isinstance(item[subkey], h5py.Dataset):
                        instrument_info[f'{key}_{subkey}'] = read_dataset_value(item[subkey])
    
    return instrument_info


def extract_users(h5file: h5py.File, entry_name: str = 'entry') -> List[Dict[str, Any]]:
    """
    Extract user information from the NeXus file.
    
    Args:
        h5file: Open HDF5 file handle
        entry_name: Name of the entry group
        
    Returns:
        List of user dictionaries
    """
    entry = h5file.get(entry_name)
    if entry is None:
        return []
    
    users = []
    
    # Find all userN groups
    for key in sorted(entry.keys()):
        if key.startswith('user'):
            user_group = entry[key]
            if isinstance(user_group, h5py.Group):
                user_info = {'user_id': key}
                for field in user_group.keys():
                    if isinstance(user_group[field], h5py.Dataset):
                        user_info[field] = read_dataset_value(user_group[field])
                users.append(user_info)
    
    return users


def extract_daslogs(h5file: h5py.File, entry_name: str = 'entry') -> List[Dict[str, Any]]:
    """
    Extract Data Acquisition System logs from the NeXus file.
    
    Each DASlog entry typically contains time-series data with time/value pairs,
    plus aggregate statistics (average, min, max, etc.).
    
    Args:
        h5file: Open HDF5 file handle
        entry_name: Name of the entry group
        
    Returns:
        List of DASlog records (one per time point per log)
    """
    daslogs_path = f'{entry_name}/DASlogs'
    daslogs = h5file.get(daslogs_path)
    if daslogs is None:
        return []
    
    records = []
    
    for log_name in daslogs.keys():
        log_group = daslogs[log_name]
        if not isinstance(log_group, h5py.Group):
            continue
        
        # Get device metadata
        device_name = None
        device_id = None
        average_value = None
        min_value = None
        max_value = None
        
        if 'device_name' in log_group:
            device_name = read_dataset_value(log_group['device_name'])
        if 'device_id' in log_group:
            device_id = read_dataset_value(log_group['device_id'])
        if 'average_value' in log_group:
            average_value = read_dataset_value(log_group['average_value'])
        if 'minimum_value' in log_group:
            min_value = read_dataset_value(log_group['minimum_value'])
        if 'maximum_value' in log_group:
            max_value = read_dataset_value(log_group['maximum_value'])
        
        # Get time series data
        times = None
        values = None
        
        if 'time' in log_group:
            time_ds = log_group['time']
            if isinstance(time_ds, h5py.Dataset):
                times = time_ds[()]
        
        if 'value' in log_group:
            value_ds = log_group['value']
            if isinstance(value_ds, h5py.Dataset):
                values = value_ds[()]
        
        # Handle special cases (like Veto_pulse with veto_pulse_time)
        if times is None and 'veto_pulse_time' in log_group:
            times = log_group['veto_pulse_time'][()]
            values = np.ones_like(times)  # Pulse indicator
        
        # Create records
        if times is not None and values is not None:
            times = np.atleast_1d(times)
            values = np.atleast_1d(values)
            
            # Handle multi-dimensional values (flatten if needed)
            if values.ndim > 1:
                # For string arrays, join them
                if values.dtype.kind in ('S', 'U', 'O'):
                    values = np.array([safe_decode(v) for v in values.flatten()])
                else:
                    values = values.flatten()
            
            # Ensure same length
            n_points = min(len(times), len(values))
            
            for i in range(n_points):
                val = values[i]
                if isinstance(val, bytes):
                    val = val.decode('utf-8', errors='replace')
                elif isinstance(val, (np.integer, np.floating)):
                    val = val.item()
                
                record = {
                    'log_name': log_name,
                    'device_name': device_name,
                    'device_id': device_id,
                    'time': float(times[i]),
                    'value': val,
                    'average_value': average_value,
                    'min_value': min_value,
                    'max_value': max_value,
                }
                records.append(record)
        else:
            # No time series, just metadata
            record = {
                'log_name': log_name,
                'device_name': device_name,
                'device_id': device_id,
                'time': None,
                'value': None,
                'average_value': average_value,
                'min_value': min_value,
                'max_value': max_value,
            }
            records.append(record)
    
    return records


def extract_events(h5file: h5py.File, entry_name: str = 'entry', 
                   max_events: Optional[int] = None,
                   pulse_times: Optional[np.ndarray] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract neutron detector event data from the NeXus file.
    
    Event data includes:
    - event_id: Detector pixel ID
    - event_time_offset: Time offset within pulse (microseconds typically)
    - event_index: Index into event arrays per pulse
    - pulse_time: Pulse time in seconds from run start (if pulse_times provided)
    - event_weight: Event weight (default 1.0)
    
    Args:
        h5file: Open HDF5 file handle
        entry_name: Name of the entry group
        max_events: Maximum number of events to read (None for all)
        pulse_times: Array of pulse times (seconds from run start), indexed by pulse_index
        
    Returns:
        Dictionary with bank names as keys and lists of event records as values
    """
    entry = h5file.get(entry_name)
    if entry is None:
        return {}
    
    events_by_bank = {}
    
    # Find all event banks (bank*_events groups)
    for key in entry.keys():
        if '_events' in key or key.startswith('monitor'):
            group = entry[key]
            if not isinstance(group, h5py.Group):
                continue
            
            # Check for event data
            has_events = 'event_id' in group or 'event_time_offset' in group
            
            if has_events:
                bank_records = []
                
                event_ids = None
                event_offsets = None
                event_index = None
                total_counts = None
                
                if 'event_id' in group:
                    event_ids = group['event_id'][()]
                if 'event_time_offset' in group:
                    event_offsets = group['event_time_offset'][()]
                if 'event_index' in group:
                    event_index = group['event_index'][()]
                if 'total_counts' in group:
                    total_counts = read_dataset_value(group['total_counts'])
                
                # Combine event data
                if event_ids is not None and event_offsets is not None:
                    n_events = len(event_ids)
                    if max_events and n_events > max_events:
                        n_events = max_events
                    
                    # Build pulse_index mapping from event_index
                    # event_index[i] is the starting index of events for pulse i
                    pulse_indices = None
                    if event_index is not None and len(event_index) > 0:
                        pulse_indices = np.zeros(len(event_ids), dtype=np.int64)
                        for pulse_idx in range(len(event_index)):
                            start_idx = event_index[pulse_idx]
                            end_idx = event_index[pulse_idx + 1] if pulse_idx + 1 < len(event_index) else len(event_ids)
                            pulse_indices[start_idx:end_idx] = pulse_idx
                    
                    for i in range(n_events):
                        pulse_idx = int(pulse_indices[i]) if pulse_indices is not None else None
                        # Look up pulse_time if pulse_times array is provided
                        pulse_time = None
                        if pulse_times is not None and pulse_idx is not None:
                            if 0 <= pulse_idx < len(pulse_times):
                                pulse_time = float(pulse_times[pulse_idx])
                        
                        record = {
                            'bank': key,
                            'event_idx': i,
                            'pulse_index': pulse_idx,
                            'pulse_time': pulse_time,
                            'event_id': int(event_ids[i]),
                            'time_offset': float(event_offsets[i]),
                            'event_weight': 1.0,
                        }
                        bank_records.append(record)
                
                # Store summary even if no individual events
                events_by_bank[key] = {
                    'records': bank_records,
                    'total_counts': total_counts,
                    'n_pulses': len(event_index) if event_index is not None else 0,
                }
    
    return events_by_bank


def extract_software_info(h5file: h5py.File, entry_name: str = 'entry') -> List[Dict[str, Any]]:
    """
    Extract software/provenance information from the NeXus file.
    
    Args:
        h5file: Open HDF5 file handle
        entry_name: Name of the entry group
        
    Returns:
        List of software component dictionaries
    """
    software_path = f'{entry_name}/Software'
    software = h5file.get(software_path)
    if software is None:
        return []
    
    software_list = []
    
    for component_name in software.keys():
        component = software[component_name]
        if isinstance(component, h5py.Group):
            info = {'component': component_name}
            for field in component.keys():
                if isinstance(component[field], h5py.Dataset):
                    info[field] = read_dataset_value(component[field])
            software_list.append(info)
    
    return software_list


def _save_split_parquets(
    output_dir: str,
    base_name: str,
    instrument_id: str,
    run_number: int,
    daslogs: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    sample_info: Dict[str, Any],
    instrument_info: Dict[str, Any],
    software: List[Dict[str, Any]],
    users: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Save each data category as a separate parquet file with explicit Iceberg-compatible schemas.
    
    Args:
        output_dir: Directory to write parquet files
        base_name: Base name for output files
        instrument_id: Instrument identifier (e.g., 'REF_L')
        run_number: Run number for partition key
        daslogs: List of DAS log records
        metadata: Run metadata dictionary
        sample_info: Sample information dictionary
        instrument_info: Instrument information dictionary
        software: List of software component dictionaries
        users: List of user dictionaries
        
    Returns:
        Dictionary mapping data type to output file path
    """
    output_files = {}
    
    # Create run_id for all records
    run_id = make_run_id(instrument_id, run_number)
    
    # Known fields for each schema (others go to additional_fields map)
    METADATA_KNOWN = ['run_number', 'title', 'start_time', 'end_time', 'duration',
                      'proton_charge', 'total_counts', 'experiment_identifier', 
                      'definition', 'source_file', 'source_path', 'ingestion_time']
    SAMPLE_KNOWN = ['name', 'nature', 'chemical_formula', 'mass', 'temperature']
    INSTRUMENT_KNOWN = ['name', 'beamline', 'instrument_xml_data']
    SOFTWARE_KNOWN = ['component', 'name', 'version']
    USERS_KNOWN = ['user_id', 'name', 'facility_user_id', 'role']
    
    if metadata:
        record = {
            'instrument_id': instrument_id,
            'run_number': run_number,
            'run_id': run_id,
            'title': normalize_to_string(metadata.get('title')),
            'start_time': normalize_to_string(metadata.get('start_time')),
            'end_time': normalize_to_string(metadata.get('end_time')),
            'duration': metadata.get('duration'),
            'proton_charge': metadata.get('proton_charge'),
            'total_counts': metadata.get('total_counts'),
            'experiment_identifier': normalize_to_string(metadata.get('experiment_identifier')),
            'definition': normalize_to_string(metadata.get('definition')),
            'source_file': normalize_to_string(metadata.get('source_file')),
            'source_path': normalize_to_string(metadata.get('source_path')),
            'ingestion_time': normalize_to_string(metadata.get('ingestion_time')),
            'file_attributes': build_attribute_map(metadata, 'file_attr_'),
            'entry_attributes': build_attribute_map(metadata, 'entry_attr_'),
        }
        table = pa.Table.from_pylist([record], schema=METADATA_SCHEMA)
        output_path = os.path.join(output_dir, f'{base_name}_metadata.parquet')
        _write_table_with_metadata(table, output_path, 'experiment_runs')
        output_files['metadata'] = output_path
        print(f"    Saved: {output_path}")
    
    if sample_info:
        record = {
            'instrument_id': instrument_id,
            'run_number': run_number,
            'run_id': run_id,
            'name': normalize_to_string(sample_info.get('name')),
            'nature': normalize_to_string(sample_info.get('nature')),
            'chemical_formula': normalize_to_string(sample_info.get('chemical_formula')),
            'mass': try_parse_numeric(sample_info.get('mass')),
            'temperature': try_parse_numeric(sample_info.get('temperature')),
            'additional_fields': extract_known_fields(sample_info, SAMPLE_KNOWN),
        }
        table = pa.Table.from_pylist([record], schema=SAMPLE_SCHEMA)
        output_path = os.path.join(output_dir, f'{base_name}_sample.parquet')
        _write_table_with_metadata(table, output_path, 'sample')
        output_files['sample'] = output_path
        print(f"    Saved: {output_path}")
    
    if instrument_info:
        record = {
            'instrument_id': instrument_id,
            'run_number': run_number,
            'run_id': run_id,
            'name': normalize_to_string(instrument_info.get('name')),
            'beamline': normalize_to_string(instrument_info.get('beamline')),
            'instrument_xml_data': normalize_to_string(instrument_info.get('instrument_xml_data')),
            'additional_fields': extract_known_fields(instrument_info, INSTRUMENT_KNOWN),
        }
        table = pa.Table.from_pylist([record], schema=INSTRUMENT_SCHEMA)
        output_path = os.path.join(output_dir, f'{base_name}_instrument.parquet')
        _write_table_with_metadata(table, output_path, 'instrument')
        output_files['instrument'] = output_path
        print(f"    Saved: {output_path}")
    
    if users:
        records = []
        for user in users:
            records.append({
                'instrument_id': instrument_id,
                'run_number': run_number,
                'run_id': run_id,
                'user_id': normalize_to_string(user.get('user_id')),
                'name': normalize_to_string(user.get('name')),
                'facility_user_id': normalize_to_string(user.get('facility_user_id')),
                'role': normalize_to_string(user.get('role')),
                'additional_fields': extract_known_fields(user, USERS_KNOWN),
            })
        table = pa.Table.from_pylist(records, schema=USERS_SCHEMA)
        output_path = os.path.join(output_dir, f'{base_name}_users.parquet')
        _write_table_with_metadata(table, output_path, 'users')
        output_files['users'] = output_path
        print(f"    Saved: {output_path}")
    
    if software:
        records = []
        for sw in software:
            records.append({
                'instrument_id': instrument_id,
                'run_number': run_number,
                'run_id': run_id,
                'component': normalize_to_string(sw.get('component')),
                'name': normalize_to_string(sw.get('name')),
                'version': normalize_to_string(sw.get('version')),
                'additional_fields': extract_known_fields(sw, SOFTWARE_KNOWN),
            })
        table = pa.Table.from_pylist(records, schema=SOFTWARE_SCHEMA)
        output_path = os.path.join(output_dir, f'{base_name}_software.parquet')
        _write_table_with_metadata(table, output_path, 'software')
        output_files['software'] = output_path
        print(f"    Saved: {output_path}")
    
    if daslogs:
        import time
        start_time = time.time()
        n_records = len(daslogs)
        
        # For large log datasets, use vectorized approach
        if n_records > 100_000:
            print(f"    Building {n_records:,} DASlog records (vectorized)...")
            
            # Pre-allocate lists
            log_names = []
            device_names = []
            device_ids = []
            times = []
            values = []
            value_numerics = []
            average_values = []
            min_values = []
            max_values = []
            
            for log in daslogs:
                log_names.append(normalize_to_string(log.get('log_name')))
                device_names.append(normalize_to_string(log.get('device_name')))
                device_ids.append(normalize_to_string(log.get('device_id')))
                times.append(log.get('time'))
                val = log.get('value')
                values.append(normalize_to_string(val))
                value_numerics.append(try_parse_numeric(val))
                average_values.append(log.get('average_value'))
                min_values.append(log.get('min_value'))
                max_values.append(log.get('max_value'))
            
            # Build table directly from arrays
            table = pa.table({
                'instrument_id': pa.array([instrument_id] * n_records, type=pa.large_string()),
                'run_number': pa.array([run_number] * n_records, type=pa.int64()),
                'run_id': pa.array([run_id] * n_records, type=pa.large_string()),
                'log_name': pa.array(log_names, type=pa.large_string()),
                'device_name': pa.array(device_names, type=pa.large_string()),
                'device_id': pa.array(device_ids, type=pa.large_string()),
                'time': pa.array(times, type=pa.float64()),
                'value': pa.array(values, type=pa.large_string()),
                'value_numeric': pa.array(value_numerics, type=pa.float64()),
                'average_value': pa.array(average_values, type=pa.float64()),
                'min_value': pa.array(min_values, type=pa.float64()),
                'max_value': pa.array(max_values, type=pa.float64()),
            })
            elapsed = time.time() - start_time
            print(f"    DASlog table built in {elapsed:.1f}s ({n_records/elapsed:,.0f} records/sec)")
        else:
            records = []
            for log in daslogs:
                records.append({
                    'instrument_id': instrument_id,
                    'run_number': run_number,
                    'run_id': run_id,
                    'log_name': normalize_to_string(log.get('log_name')),
                    'device_name': normalize_to_string(log.get('device_name')),
                    'device_id': normalize_to_string(log.get('device_id')),
                    'time': log.get('time'),
                    'value': normalize_to_string(log.get('value')),
                    'value_numeric': try_parse_numeric(log.get('value')),
                    'average_value': log.get('average_value'),
                    'min_value': log.get('min_value'),
                    'max_value': log.get('max_value'),
                })
            table = pa.Table.from_pylist(records, schema=DASLOGS_SCHEMA)
        
        output_path = os.path.join(output_dir, f'{base_name}_daslogs.parquet')
        _write_table_with_metadata(table, output_path, 'daslogs')
        output_files['daslogs'] = output_path
        print(f"    Saved: {output_path} ({len(daslogs):,} records)")
    
    return output_files


def _save_events(
    output_dir: str,
    base_name: str,
    instrument_id: str,
    run_number: int,
    events_data: Dict[str, Any],
    max_events_per_file: Optional[int] = None,
) -> Dict[str, str]:
    """
    Save event data to parquet files with explicit Iceberg-compatible schema.
    
    Large event banks can be split into multiple files for better Iceberg
    performance. Files are named with part numbers: bank1_events_part001.parquet
    
    Args:
        output_dir: Directory to write parquet files
        base_name: Base name for output files
        instrument_id: Instrument identifier (e.g., 'REF_L')
        run_number: Run number for partition key
        events_data: Dictionary with bank names as keys and event data as values
        max_events_per_file: Maximum events per file (None for no limit)
        
    Returns:
        Dictionary mapping bank names to output file paths
    """
    output_files = {}
    run_id = make_run_id(instrument_id, run_number)
    
    for bank_name, bank_data in events_data.items():
        records = bank_data['records']
        if records:
            # Add instrument_id, run_number, run_id to each record
            records_with_ids = [
                {**record, 'instrument_id': instrument_id, 'run_number': run_number, 'run_id': run_id}
                for record in records
            ]
            
            # Chunk if needed
            if max_events_per_file and len(records_with_ids) > max_events_per_file:
                num_chunks = (len(records_with_ids) + max_events_per_file - 1) // max_events_per_file
                print(f"    Splitting {bank_name} into {num_chunks} files ({len(records_with_ids):,} events)")
                
                for i in range(num_chunks):
                    start_idx = i * max_events_per_file
                    end_idx = min((i + 1) * max_events_per_file, len(records_with_ids))
                    chunk = records_with_ids[start_idx:end_idx]
                    
                    table = pa.Table.from_pylist(chunk, schema=EVENTS_SCHEMA)
                    part_name = f"{bank_name}_part{i+1:03d}"
                    output_path = os.path.join(output_dir, f'{base_name}_{part_name}.parquet')
                    _write_table_with_metadata(table, output_path, 'events')
                    output_files[part_name] = output_path
                    print(f"      Saved: {output_path} ({len(chunk):,} events)")
            else:
                table = pa.Table.from_pylist(records_with_ids, schema=EVENTS_SCHEMA)
                output_path = os.path.join(output_dir, f'{base_name}_{bank_name}.parquet')
                _write_table_with_metadata(table, output_path, 'events')
                output_files[bank_name] = output_path
                print(f"    Saved: {output_path} ({len(records):,} events)")
        else:
            print(f"    Skipped {bank_name}: no events (total_counts={bank_data['total_counts']})")
    
    # Save summary of all banks with schema
    bank_summary = [
        {
            'instrument_id': instrument_id,
            'run_number': run_number,
            'run_id': run_id,
            'bank': bank_name,
            'total_counts': bank_data['total_counts'],
            'n_pulses': bank_data['n_pulses'],
            'events_extracted': len(bank_data['records']),
        }
        for bank_name, bank_data in events_data.items()
    ]
    
    if bank_summary:
        table = pa.Table.from_pylist(bank_summary, schema=EVENT_SUMMARY_SCHEMA)
        output_path = os.path.join(output_dir, f'{base_name}_event_summary.parquet')
        _write_table_with_metadata(table, output_path, 'event_summary')
        output_files['event_summary'] = output_path
        print(f"    Saved: {output_path}")
    
    return output_files


def _save_mantid_events_chunked(
    output_dir: str,
    base_name: str,
    instrument_id: str,
    run_number: int,
    h5file: h5py.File,
    workspace_name: str,
    max_events: Optional[int] = None,
    max_events_per_file: Optional[int] = None,
) -> Dict[str, str]:
    """
    Save Mantid event data to parquet files with chunked processing.
    
    This function uses VECTORIZED operations to process events efficiently.
    Events are processed in chunks to handle large files (100M+ events)
    without loading everything into memory.
    
    Args:
        output_dir: Directory to write parquet files
        base_name: Base name for output files
        instrument_id: Instrument identifier
        run_number: Run number for partition key
        h5file: Open HDF5 file handle
        workspace_name: Name of the mantid workspace group
        max_events: Maximum number of events to read (None for all)
        max_events_per_file: Maximum events per output file
        
    Returns:
        Dictionary mapping output names to file paths
    """
    import time
    output_files = {}
    run_id = make_run_id(instrument_id, run_number)
    
    event_path = f'{workspace_name}/event_workspace'
    if event_path not in h5file:
        print("    No event_workspace found")
        return output_files
    
    event_ws = h5file[event_path]
    
    if 'tof' not in event_ws or 'indices' not in event_ws:
        print("    Missing tof or indices datasets")
        return output_files
    
    tof_ds = event_ws['tof']
    indices_ds = event_ws['indices']
    
    n_events_total = tof_ds.shape[0]
    n_spectra = indices_ds.shape[0] - 1
    
    print(f"    Found {n_events_total:,} events across {n_spectra:,} spectra")
    
    # Limit events if requested
    n_events = n_events_total
    if max_events and n_events > max_events:
        n_events = max_events
        print(f"    Limiting to {n_events:,} events")
    
    # Read indices array (needed for event-to-spectrum mapping)
    indices = indices_ds[:]
    
    # Determine chunk size for writing
    if max_events_per_file:
        write_chunk_size = max_events_per_file
    else:
        write_chunk_size = n_events  # Single file
    
    # Check for weights
    has_weights = 'weight' in event_ws
    
    # Pre-compute constant arrays for the schema
    # These will be broadcast/repeated for each chunk
    
    file_number = 0
    total_written = 0
    start_time = time.time()
    
    # Process in chunks - use larger chunks for better throughput
    read_chunk_size = min(write_chunk_size, 10_000_000)  # 10M events per read
    
    # Determine output path once before the loop
    if max_events_per_file and n_events > max_events_per_file:
        # Will create multiple files
        output_path = None  # Set per chunk in loop
    else:
        # Single output file for all chunks
        output_path = os.path.join(output_dir, f'{base_name}_event_workspace.parquet')
        output_files['event_workspace'] = output_path
    
    # Open writer for single-file mode (will append chunks)
    writer = None
    if output_path is not None:
        writer = _ChunkedParquetWriter(output_path, EVENTS_SCHEMA, 'events')
        writer.__enter__()
    
    try:
        for start_idx in range(0, n_events, read_chunk_size):
            end_idx = min(start_idx + read_chunk_size, n_events)
            chunk_size = end_idx - start_idx
            
            # Read TOF chunk directly as numpy array
            tof_chunk = tof_ds[start_idx:end_idx]
            
            # Read weights if available
            if has_weights:
                weight_chunk = event_ws['weight'][start_idx:end_idx]
            else:
                weight_chunk = np.ones(chunk_size, dtype=np.float32)
            
            # VECTORIZED: Map events to spectrum IDs using searchsorted
            event_indices = np.arange(start_idx, end_idx, dtype=np.int64)
            event_ids = np.searchsorted(indices, event_indices, side='right') - 1
            
            # VECTORIZED: Build PyArrow arrays directly (no Python loop!)
            table = pa.table({
                'instrument_id': pa.array([instrument_id] * chunk_size, type=pa.large_string()),
                'run_number': pa.array(np.full(chunk_size, run_number, dtype=np.int64)),
                'run_id': pa.array([run_id] * chunk_size, type=pa.large_string()),
                'bank': pa.array(['event_workspace'] * chunk_size, type=pa.large_string()),
                'event_idx': pa.array(event_indices),
                'pulse_index': pa.array([None] * chunk_size, type=pa.int64()),
                'pulse_time': pa.array([None] * chunk_size, type=pa.float64()),
                'event_id': pa.array(event_ids),
                'time_offset': pa.array(tof_chunk.astype(np.float64)),
                'event_weight': pa.array(weight_chunk.astype(np.float64)),
            })
            
            # Write chunk to file
            if max_events_per_file and n_events > max_events_per_file:
                # Multi-file mode: create new file for each chunk
                file_number += 1
                part_name = f"event_workspace_part{file_number:03d}"
                chunk_output_path = os.path.join(output_dir, f'{base_name}_{part_name}.parquet')
                output_files[part_name] = chunk_output_path
                _write_table_with_metadata(table, chunk_output_path, 'events')
            else:
                # Single-file mode: append chunk to writer
                writer.write_chunk(table)
            
            total_written += chunk_size
            
            # Progress reporting with throughput
            elapsed = time.time() - start_time
            rate = total_written / elapsed / 1_000_000 if elapsed > 0 else 0
            print(f"      Written {total_written:,} / {n_events:,} events "
                  f"({100*total_written/n_events:.1f}%) - {rate:.1f}M events/sec")
    finally:
        # Close writer if opened
        if writer is not None:
            writer.__exit__(None, None, None)
    
    # Save event summary
    summary = [{
        'instrument_id': instrument_id,
        'run_number': run_number,
        'run_id': run_id,
        'bank': 'event_workspace',
        'total_counts': n_events_total,
        'n_pulses': 0,  # No pulse info in .lite files
        'events_extracted': n_events,
    }]
    
    table = pa.Table.from_pylist(summary, schema=EVENT_SUMMARY_SCHEMA)
    output_path = os.path.join(output_dir, f'{base_name}_event_summary.parquet')
    _write_table_with_metadata(table, output_path, 'event_summary')
    output_files['event_summary'] = output_path
    print(f"    Saved: {output_path}")
    
    total_time = time.time() - start_time
    print(f"    Total event processing time: {total_time:.1f}s "
          f"({n_events/total_time/1_000_000:.1f}M events/sec)")
    
    return output_files
    output_files['event_summary'] = output_path
    print(f"    Saved: {output_path}")
    
    return output_files


def process_mantid_file(filepath: str, output_dir: str,
                        max_events: Optional[int] = None,
                        max_events_per_file: Optional[int] = None,
                        include_events: bool = True) -> Dict[str, str]:
    """
    Process a Mantid-format NeXus file and write data to Parquet files.
    
    Mantid files have a different structure than standard NeXus:
    - Root: /mantid_workspace_1/ instead of /entry/
    - Events: /event_workspace/ with tof, indices, weight
    - Logs: /logs/ instead of /DASlogs/
    - No pulse timing in .lite files
    
    Args:
        filepath: Path to the Mantid NeXus file
        output_dir: Directory to write Parquet files
        max_events: Maximum number of events to read (None for all)
        max_events_per_file: Maximum events per output file for chunking
        include_events: Whether to include event data
        
    Returns:
        Dictionary mapping data type to output file path
    """
    output_files = {}
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(filepath).stem.replace('.nxs', '').replace('.h5', '').replace('.lite', '')
    
    print(f"Processing (Mantid format): {filepath}")
    
    with h5py.File(filepath, 'r') as h5file:
        # Find the workspace
        workspace_name = get_mantid_workspace_name(h5file)
        if not workspace_name:
            raise ValueError(f"No mantid_workspace found in {filepath}")
        
        print(f"  Found workspace: {workspace_name}")
        
        # Get instrument ID
        instrument_id = get_mantid_instrument_id(h5file, workspace_name)
        print(f"  Instrument: {instrument_id}")
        
        # Extract metadata
        print("  Extracting metadata...")
        metadata = extract_mantid_metadata(h5file, workspace_name)
        metadata['source_file'] = os.path.basename(filepath)
        metadata['source_path'] = os.path.abspath(filepath)
        metadata['ingestion_time'] = datetime.now().isoformat()
        
        # Get run_number
        run_number = metadata.get('run_number', 0)
        if run_number is None:
            run_number = 0
        run_number = int(run_number)
        
        run_id = make_run_id(instrument_id, run_number)
        print(f"  Run identifier: {run_id}")
        
        # Extract sample info
        print("  Extracting sample info...")
        sample_info = extract_mantid_sample_info(h5file, workspace_name)
        
        # Extract instrument info
        print("  Extracting instrument info...")
        instrument_info = extract_mantid_instrument_info(h5file, workspace_name)
        
        # Extract logs (DASlog equivalent)
        print("  Extracting logs...")
        daslogs = extract_mantid_logs(h5file, workspace_name)
        print(f"    Found {len(daslogs):,} log records")
        
        # Save metadata, sample, instrument, and daslogs
        output_files.update(_save_split_parquets(
            output_dir, base_name, instrument_id, run_number, daslogs, metadata,
            sample_info, instrument_info, software=[], users=[]
        ))
        
        # Extract and save events
        if include_events:
            print("  Extracting events (chunked processing for large files)...")
            output_files.update(_save_mantid_events_chunked(
                output_dir, base_name, instrument_id, run_number,
                h5file, workspace_name,
                max_events=max_events,
                max_events_per_file=max_events_per_file,
            ))
    
    return output_files


def process_nexus_file(filepath: str, output_dir: str, 
                       max_events: Optional[int] = None,
                       max_events_per_file: Optional[int] = None,
                       include_events: bool = True,
                       include_users: bool = True,
                       force_format: Optional[str] = None) -> Dict[str, str]:
    """
    Process a NeXus HDF5 file and write data to Iceberg-compatible Parquet files.
    
    Automatically detects the file format:
    - Standard NeXus: /entry/ structure with bank*_events
    - Mantid: /mantid_workspace_*/ structure with event_workspace
    
    All output files use explicit PyArrow schemas with:
    - Consistent column types across files
    - instrument_id and run_number as composite partition key
    - run_id as unique identifier (instrument_id:run_number)
    - value_numeric for queryable numeric DAS log values
    - Map types for flexible additional fields
    
    Args:
        filepath: Path to the NeXus HDF5 file
        output_dir: Directory to write Parquet files
        max_events: Maximum number of events per bank (None for all)
        max_events_per_file: Maximum events per output file for chunking (None for no limit)
        include_events: Whether to include event data (can be large)
        include_users: Whether to include user information
        force_format: Force format detection ('standard', 'mantid', or None for auto)
        
    Returns:
        Dictionary mapping data type to output file path
    """
    # Detect file format
    with h5py.File(filepath, 'r') as h5file:
        if force_format:
            file_format = force_format
        else:
            file_format = detect_nexus_format(h5file)
    
    print(f"Detected format: {file_format}")
    
    # Route to appropriate processor
    if file_format == 'mantid':
        return process_mantid_file(
            filepath, output_dir,
            max_events=max_events,
            max_events_per_file=max_events_per_file,
            include_events=include_events,
        )
    
    # Standard NeXus processing (existing code)
    output_files = {}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base name for output files
    base_name = Path(filepath).stem.replace('.nxs', '').replace('.h5', '')
    
    print(f"Processing: {filepath}")
    
    with h5py.File(filepath, 'r') as h5file:
        # 1. Extract metadata
        print("  Extracting metadata...")
        metadata = extract_entry_metadata(h5file)
        metadata['source_file'] = os.path.basename(filepath)
        metadata['source_path'] = os.path.abspath(filepath)
        metadata['ingestion_time'] = datetime.now().isoformat()
        
        # Extract run_number for partition key (default to 0 if not found)
        run_number = metadata.get('run_number', 0)
        if run_number is None:
            run_number = 0
        run_number = int(run_number)
        
        # 2. Extract sample info
        print("  Extracting sample info...")
        sample_info = extract_sample_info(h5file)
        
        # 3. Extract instrument info and get instrument_id
        print("  Extracting instrument info...")
        instrument_info = extract_instrument_info(h5file)
        
        # Get instrument_id from instrument name (e.g., 'REF_L')
        instrument_id = normalize_to_string(instrument_info.get('name', 'UNKNOWN'))
        if instrument_id is None:
            instrument_id = 'UNKNOWN'
        
        run_id = make_run_id(instrument_id, run_number)
        print(f"  Run identifier: {run_id}")
        
        # 4. Extract users (optional)
        users = []
        if include_users:
            print("  Extracting user info...")
            users = extract_users(h5file)
        
        # 5. Extract software info
        print("  Extracting software info...")
        software = extract_software_info(h5file)
        
        # 6. Extract DAS logs
        print("  Extracting DAS logs (this may take a moment)...")
        daslogs = extract_daslogs(h5file)
        
        # Build pulse_times array from proton_charge log for event time correlation
        pulse_times = None
        if include_events:
            # Find proton_charge times from daslogs
            proton_charge_times = [
                record['time'] for record in daslogs 
                if record.get('log_name') == 'proton_charge' and record.get('time') is not None
            ]
            if proton_charge_times:
                pulse_times = np.array(sorted(proton_charge_times), dtype=np.float64)
                print(f"  Found {len(pulse_times):,} pulse times from proton_charge log")
            else:
                print("  Warning: No proton_charge log found, pulse_time will be null in events")
        
        # 7. Extract events (optional, can be very large)
        events_data = None
        if include_events:
            print("  Extracting event data (this may take a while for large files)...")
            events_data = extract_events(h5file, max_events=max_events, pulse_times=pulse_times)
        
        # Save data as separate parquet files
        output_files.update(_save_split_parquets(
            output_dir, base_name, instrument_id, run_number, daslogs, metadata,
            sample_info, instrument_info, software, users
        ))
        
        # Save events separately (with optional chunking)
        if events_data:
            output_files.update(_save_events(
                output_dir, base_name, instrument_id, run_number, events_data,
                max_events_per_file=max_events_per_file
            ))
    
    return output_files
