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
    COMBINED_SCHEMA,
    try_parse_numeric,
    normalize_to_string,
    build_attribute_map,
    extract_known_fields,
)


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
                   max_events: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract neutron detector event data from the NeXus file.
    
    Event data includes:
    - event_id: Detector pixel ID
    - event_time_offset: Time offset within pulse (microseconds typically)
    - event_index: Index into event arrays per pulse
    
    Args:
        h5file: Open HDF5 file handle
        entry_name: Name of the entry group
        max_events: Maximum number of events to read (None for all)
        
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
                    
                    for i in range(n_events):
                        record = {
                            'bank': key,
                            'event_idx': i,
                            'event_id': int(event_ids[i]),
                            'time_offset': float(event_offsets[i]),
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


def _add_metadata_columns(
    df: pd.DataFrame,
    run_number: int,
    metadata: Dict[str, Any],
    sample_info: Dict[str, Any],
    instrument_info: Dict[str, Any],
    software: List[Dict[str, Any]],
    users: List[Dict[str, Any]],
) -> None:
    """
    Add metadata as nested struct columns for Iceberg compatibility.
    
    Uses PyArrow-compatible nested structures instead of flat prefixed columns.
    
    Args:
        df: DataFrame to modify in place
        run_number: Run number for partition key
        metadata: Run metadata dictionary
        sample_info: Sample information dictionary
        instrument_info: Instrument information dictionary
        software: List of software component dictionaries
        users: List of user dictionaries
    """
    # Add run_number as partition key
    df['run_number'] = run_number
    
    # Build metadata struct
    df['metadata'] = [{
        'title': normalize_to_string(metadata.get('title')),
        'start_time': normalize_to_string(metadata.get('start_time')),
        'end_time': normalize_to_string(metadata.get('end_time')),
        'duration': metadata.get('duration'),
        'proton_charge': metadata.get('proton_charge'),
        'total_counts': metadata.get('total_counts'),
        'experiment_identifier': normalize_to_string(metadata.get('experiment_identifier')),
        'definition': normalize_to_string(metadata.get('definition')),
        'source_file': normalize_to_string(metadata.get('source_file')),
    }] * len(df)
    
    # Build sample struct
    df['sample'] = [{
        'name': normalize_to_string(sample_info.get('name')),
        'nature': normalize_to_string(sample_info.get('nature')),
        'chemical_formula': normalize_to_string(sample_info.get('chemical_formula')),
        'mass': sample_info.get('mass'),
        'temperature': sample_info.get('temperature'),
    }] * len(df)
    
    # Build instrument struct
    df['instrument'] = [{
        'name': normalize_to_string(instrument_info.get('name')),
        'beamline': normalize_to_string(instrument_info.get('beamline')),
    }] * len(df)
    
    # Build users list of structs
    user_structs = [
        {
            'name': normalize_to_string(u.get('name', u.get('user_id', ''))),
            'role': normalize_to_string(u.get('role')),
        }
        for u in users
    ] if users else None
    df['users'] = [user_structs] * len(df)


def _save_combined_parquet(
    output_dir: str,
    base_name: str,
    run_number: int,
    daslogs: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    sample_info: Dict[str, Any],
    instrument_info: Dict[str, Any],
    software: List[Dict[str, Any]],
    users: List[Dict[str, Any]],
    events_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Save all data as a single combined parquet file with explicit Iceberg-compatible schema.
    
    DAS logs and events form the rows, with metadata as nested struct columns.
    A 'record_type' column distinguishes between 'daslog' and 'event'.
    
    Args:
        output_dir: Directory to write the parquet file
        base_name: Base name for the output file
        run_number: Run number for partition key
        daslogs: List of DAS log records
        metadata: Run metadata dictionary
        sample_info: Sample information dictionary
        instrument_info: Instrument information dictionary
        software: List of software component dictionaries
        users: List of user dictionaries
        events_data: Optional dictionary with bank names as keys and event data
        
    Returns:
        Dictionary with 'combined' key mapping to output file path
    """
    dataframes = []
    
    # String columns that need normalization
    STRING_COLS = ['log_name', 'device_name', 'device_id', 'value', 'bank', 'record_type']
    
    # Add daslogs with value_numeric column
    if daslogs:
        df_daslogs = pd.DataFrame(daslogs)
        # Add value_numeric for Iceberg compatibility
        df_daslogs['value_numeric'] = df_daslogs['value'].apply(try_parse_numeric)
        df_daslogs['value'] = df_daslogs['value'].apply(normalize_to_string)
        df_daslogs['record_type'] = 'daslog'
        # Normalize all string columns
        for col in STRING_COLS:
            if col in df_daslogs.columns:
                df_daslogs[col] = df_daslogs[col].apply(normalize_to_string)
        # Ensure event columns exist as NULL for daslogs
        for col in ['bank', 'event_idx', 'event_id', 'time_offset']:
            if col not in df_daslogs.columns:
                df_daslogs[col] = None
        dataframes.append(df_daslogs)
    
    # Add events
    if events_data:
        all_events = []
        for bank_name, bank_data in events_data.items():
            all_events.extend(bank_data['records'])
        
        if all_events:
            df_events = pd.DataFrame(all_events)
            df_events['record_type'] = 'event'
            # Normalize string columns
            for col in STRING_COLS:
                if col in df_events.columns:
                    df_events[col] = df_events[col].apply(normalize_to_string)
            # Ensure daslog columns exist as NULL for events
            for col in ['log_name', 'device_name', 'device_id', 'time', 'value', 
                        'value_numeric', 'average_value', 'min_value', 'max_value']:
                if col not in df_events.columns:
                    df_events[col] = None
            dataframes.append(df_events)
            print(f"    Including {len(all_events):,} events in combined file")
    
    # Combine all dataframes or create empty row for metadata-only
    if dataframes:
        df = pd.concat(dataframes, ignore_index=True)
    else:
        # Create a metadata-only row with all required columns
        df = pd.DataFrame([{
            'record_type': 'metadata',
            'log_name': None, 'device_name': None, 'device_id': None,
            'time': None, 'value': None, 'value_numeric': None,
            'average_value': None, 'min_value': None, 'max_value': None,
            'bank': None, 'event_idx': None, 'event_id': None, 'time_offset': None,
        }])
    
    # Add metadata columns as nested structs
    _add_metadata_columns(df, run_number, metadata, sample_info, instrument_info, software, users)
    
    # Write with explicit schema using PyArrow
    output_path = os.path.join(output_dir, f'{base_name}_combined.parquet')
    table = pa.Table.from_pandas(df, schema=COMBINED_SCHEMA, preserve_index=False)
    pq.write_table(table, output_path)
    
    print(f"    Saved: {output_path} ({len(df):,} records)")
    
    return {'combined': output_path}


def _save_split_parquets(
    output_dir: str,
    base_name: str,
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
            'run_number': run_number,
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
        pq.write_table(table, output_path)
        output_files['metadata'] = output_path
        print(f"    Saved: {output_path}")
    
    if sample_info:
        record = {
            'run_number': run_number,
            'name': normalize_to_string(sample_info.get('name')),
            'nature': normalize_to_string(sample_info.get('nature')),
            'chemical_formula': normalize_to_string(sample_info.get('chemical_formula')),
            'mass': sample_info.get('mass'),
            'temperature': sample_info.get('temperature'),
            'additional_fields': extract_known_fields(sample_info, SAMPLE_KNOWN),
        }
        table = pa.Table.from_pylist([record], schema=SAMPLE_SCHEMA)
        output_path = os.path.join(output_dir, f'{base_name}_sample.parquet')
        pq.write_table(table, output_path)
        output_files['sample'] = output_path
        print(f"    Saved: {output_path}")
    
    if instrument_info:
        record = {
            'run_number': run_number,
            'name': normalize_to_string(instrument_info.get('name')),
            'beamline': normalize_to_string(instrument_info.get('beamline')),
            'instrument_xml_data': normalize_to_string(instrument_info.get('instrument_xml_data')),
            'additional_fields': extract_known_fields(instrument_info, INSTRUMENT_KNOWN),
        }
        table = pa.Table.from_pylist([record], schema=INSTRUMENT_SCHEMA)
        output_path = os.path.join(output_dir, f'{base_name}_instrument.parquet')
        pq.write_table(table, output_path)
        output_files['instrument'] = output_path
        print(f"    Saved: {output_path}")
    
    if users:
        records = []
        for user in users:
            records.append({
                'run_number': run_number,
                'user_id': normalize_to_string(user.get('user_id')),
                'name': normalize_to_string(user.get('name')),
                'facility_user_id': normalize_to_string(user.get('facility_user_id')),
                'role': normalize_to_string(user.get('role')),
                'additional_fields': extract_known_fields(user, USERS_KNOWN),
            })
        table = pa.Table.from_pylist(records, schema=USERS_SCHEMA)
        output_path = os.path.join(output_dir, f'{base_name}_users.parquet')
        pq.write_table(table, output_path)
        output_files['users'] = output_path
        print(f"    Saved: {output_path}")
    
    if software:
        records = []
        for sw in software:
            records.append({
                'run_number': run_number,
                'component': normalize_to_string(sw.get('component')),
                'name': normalize_to_string(sw.get('name')),
                'version': normalize_to_string(sw.get('version')),
                'additional_fields': extract_known_fields(sw, SOFTWARE_KNOWN),
            })
        table = pa.Table.from_pylist(records, schema=SOFTWARE_SCHEMA)
        output_path = os.path.join(output_dir, f'{base_name}_software.parquet')
        pq.write_table(table, output_path)
        output_files['software'] = output_path
        print(f"    Saved: {output_path}")
    
    if daslogs:
        records = []
        for log in daslogs:
            records.append({
                'run_number': run_number,
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
        pq.write_table(table, output_path)
        output_files['daslogs'] = output_path
        print(f"    Saved: {output_path} ({len(daslogs):,} records)")
    
    return output_files


def _save_events(
    output_dir: str,
    base_name: str,
    run_number: int,
    events_data: Dict[str, Any],
) -> Dict[str, str]:
    """
    Save event data to parquet files with explicit Iceberg-compatible schema.
    
    Args:
        output_dir: Directory to write parquet files
        base_name: Base name for output files
        run_number: Run number for partition key
        events_data: Dictionary with bank names as keys and event data as values
        
    Returns:
        Dictionary mapping bank names to output file paths
    """
    output_files = {}
    
    for bank_name, bank_data in events_data.items():
        records = bank_data['records']
        if records:
            # Add run_number to each record
            records_with_run = [
                {**record, 'run_number': run_number}
                for record in records
            ]
            table = pa.Table.from_pylist(records_with_run, schema=EVENTS_SCHEMA)
            output_path = os.path.join(output_dir, f'{base_name}_{bank_name}.parquet')
            pq.write_table(table, output_path)
            output_files[bank_name] = output_path
            print(f"    Saved: {output_path} ({len(records):,} events)")
        else:
            print(f"    Skipped {bank_name}: no events (total_counts={bank_data['total_counts']})")
    
    # Save summary of all banks with schema
    bank_summary = [
        {
            'run_number': run_number,
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
        pq.write_table(table, output_path)
        output_files['event_summary'] = output_path
        print(f"    Saved: {output_path}")
    
    return output_files


def process_nexus_file(filepath: str, output_dir: str, 
                       max_events: Optional[int] = None,
                       include_events: bool = True,
                       include_users: bool = True,
                       single_file: bool = False) -> Dict[str, str]:
    """
    Process a NeXus HDF5 file and write data to Iceberg-compatible Parquet files.
    
    All output files use explicit PyArrow schemas with:
    - Consistent column types across files
    - run_number as partition key on every file
    - Nested structs for metadata in combined mode
    - value_numeric for queryable numeric DAS log values
    - Map types for flexible additional fields
    
    Args:
        filepath: Path to the NeXus HDF5 file
        output_dir: Directory to write Parquet files
        max_events: Maximum number of events per bank (None for all)
        include_events: Whether to include event data (can be large)
        include_users: Whether to include user information
        single_file: If True, combine all data (including events) into a single
            parquet file with metadata as nested structs. A 'record_type' column
            distinguishes between 'daslog' and 'event' rows. If False (default),
            create separate files for each data category.
        
    Returns:
        Dictionary mapping data type to output file path
    """
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
        
        # 3. Extract instrument info
        print("  Extracting instrument info...")
        instrument_info = extract_instrument_info(h5file)
        
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
        
        # 7. Extract events (optional, can be very large)
        events_data = None
        if include_events:
            print("  Extracting event data (this may take a while for large files)...")
            events_data = extract_events(h5file, max_events=max_events)
        
        # Save data based on single_file option
        if single_file:
            print("  Combining data into single file...")
            output_files.update(_save_combined_parquet(
                output_dir, base_name, run_number, daslogs, metadata,
                sample_info, instrument_info, software, users,
                events_data=events_data
            ))
        else:
            output_files.update(_save_split_parquets(
                output_dir, base_name, run_number, daslogs, metadata,
                sample_info, instrument_info, software, users
            ))
            # Save events separately in split mode
            if events_data:
                output_files.update(_save_events(output_dir, base_name, run_number, events_data))
    
    return output_files
