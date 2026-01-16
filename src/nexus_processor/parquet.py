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
                        record = {
                            'bank': key,
                            'event_idx': i,
                            'pulse_index': int(pulse_indices[i]) if pulse_indices is not None else None,
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
        pq.write_table(table, output_path)
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
        pq.write_table(table, output_path)
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
        pq.write_table(table, output_path)
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
        pq.write_table(table, output_path)
        output_files['software'] = output_path
        print(f"    Saved: {output_path}")
    
    if daslogs:
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
        pq.write_table(table, output_path)
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
                    pq.write_table(table, output_path)
                    output_files[part_name] = output_path
                    print(f"      Saved: {output_path} ({len(chunk):,} events)")
            else:
                table = pa.Table.from_pylist(records_with_ids, schema=EVENTS_SCHEMA)
                output_path = os.path.join(output_dir, f'{base_name}_{bank_name}.parquet')
                pq.write_table(table, output_path)
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
        pq.write_table(table, output_path)
        output_files['event_summary'] = output_path
        print(f"    Saved: {output_path}")
    
    return output_files


def process_nexus_file(filepath: str, output_dir: str, 
                       max_events: Optional[int] = None,
                       max_events_per_file: Optional[int] = None,
                       include_events: bool = True,
                       include_users: bool = True) -> Dict[str, str]:
    """
    Process a NeXus HDF5 file and write data to Iceberg-compatible Parquet files.
    
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
        
        # 7. Extract events (optional, can be very large)
        events_data = None
        if include_events:
            print("  Extracting event data (this may take a while for large files)...")
            events_data = extract_events(h5file, max_events=max_events)
        
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
