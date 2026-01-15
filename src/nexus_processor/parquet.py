"""
NeXus HDF5 to Parquet Conversion Module

This module provides functions to read NeXus format HDF5 files (commonly used
in neutron scattering experiments) and convert data into Parquet files organized
by data category.

Output Parquet files:
  - metadata.parquet: Run-level metadata (title, times, experiment info)
  - daslogs.parquet: Data Acquisition System time series logs (flattened)
  - events.parquet: Neutron detector event data (optional)
  - sample.parquet: Sample information
  - instrument.parquet: Instrument configuration
  - software.parquet: Software/provenance information
  - users.parquet: User/experimenter information (optional)

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


def process_nexus_file(filepath: str, output_dir: str, 
                       max_events: Optional[int] = None,
                       include_events: bool = True,
                       include_users: bool = True) -> Dict[str, str]:
    """
    Process a NeXus HDF5 file and write data to Parquet files.
    
    Args:
        filepath: Path to the NeXus HDF5 file
        output_dir: Directory to write Parquet files
        max_events: Maximum number of events per bank (None for all)
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
        # 1. Extract and save metadata
        print("  Extracting metadata...")
        metadata = extract_entry_metadata(h5file)
        metadata['source_file'] = os.path.basename(filepath)
        metadata['source_path'] = os.path.abspath(filepath)
        metadata['ingestion_time'] = datetime.now().isoformat()
        
        if metadata:
            df_metadata = pd.DataFrame([metadata])
            output_path = os.path.join(output_dir, f'{base_name}_metadata.parquet')
            df_metadata.to_parquet(output_path, index=False)
            output_files['metadata'] = output_path
            print(f"    Saved: {output_path}")
        
        # 2. Extract and save sample info
        print("  Extracting sample info...")
        sample_info = extract_sample_info(h5file)
        if sample_info:
            df_sample = pd.DataFrame([sample_info])
            output_path = os.path.join(output_dir, f'{base_name}_sample.parquet')
            df_sample.to_parquet(output_path, index=False)
            output_files['sample'] = output_path
            print(f"    Saved: {output_path}")
        
        # 3. Extract and save instrument info
        print("  Extracting instrument info...")
        instrument_info = extract_instrument_info(h5file)
        if instrument_info:
            df_instrument = pd.DataFrame([instrument_info])
            output_path = os.path.join(output_dir, f'{base_name}_instrument.parquet')
            df_instrument.to_parquet(output_path, index=False)
            output_files['instrument'] = output_path
            print(f"    Saved: {output_path}")
        
        # 4. Extract and save users (optional)
        if include_users:
            print("  Extracting user info...")
            users = extract_users(h5file)
            if users:
                df_users = pd.DataFrame(users)
                output_path = os.path.join(output_dir, f'{base_name}_users.parquet')
                df_users.to_parquet(output_path, index=False)
                output_files['users'] = output_path
                print(f"    Saved: {output_path}")
        
        # 5. Extract and save software info
        print("  Extracting software info...")
        software = extract_software_info(h5file)
        if software:
            df_software = pd.DataFrame(software)
            output_path = os.path.join(output_dir, f'{base_name}_software.parquet')
            df_software.to_parquet(output_path, index=False)
            output_files['software'] = output_path
            print(f"    Saved: {output_path}")
        
        # 6. Extract and save DAS logs
        print("  Extracting DAS logs (this may take a moment)...")
        daslogs = extract_daslogs(h5file)
        if daslogs:
            df_daslogs = pd.DataFrame(daslogs)
            # Convert 'value' column to string to handle mixed types
            df_daslogs['value'] = df_daslogs['value'].astype(str)
            output_path = os.path.join(output_dir, f'{base_name}_daslogs.parquet')
            df_daslogs.to_parquet(output_path, index=False)
            output_files['daslogs'] = output_path
            print(f"    Saved: {output_path} ({len(daslogs):,} records)")
        
        # 7. Extract and save events (optional, can be very large)
        if include_events:
            print("  Extracting event data (this may take a while for large files)...")
            events_data = extract_events(h5file, max_events=max_events)
            
            for bank_name, bank_data in events_data.items():
                records = bank_data['records']
                if records:
                    df_events = pd.DataFrame(records)
                    output_path = os.path.join(output_dir, f'{base_name}_{bank_name}.parquet')
                    df_events.to_parquet(output_path, index=False)
                    output_files[bank_name] = output_path
                    print(f"    Saved: {output_path} ({len(records):,} events)")
                else:
                    print(f"    Skipped {bank_name}: no events (total_counts={bank_data['total_counts']})")
            
            # Also save a summary of all banks
            bank_summary = []
            for bank_name, bank_data in events_data.items():
                bank_summary.append({
                    'bank': bank_name,
                    'total_counts': bank_data['total_counts'],
                    'n_pulses': bank_data['n_pulses'],
                    'events_extracted': len(bank_data['records']),
                })
            if bank_summary:
                df_bank_summary = pd.DataFrame(bank_summary)
                output_path = os.path.join(output_dir, f'{base_name}_event_summary.parquet')
                df_bank_summary.to_parquet(output_path, index=False)
                output_files['event_summary'] = output_path
                print(f"    Saved: {output_path}")
    
    return output_files
