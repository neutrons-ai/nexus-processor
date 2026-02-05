"""
Mantid-processed NeXus file extraction module.

This module provides functions to extract data from Mantid-processed NeXus files
(*.lite.nxs.h5 and similar formats) which have a different structure than
standard NeXus event files.

Mantid workspace structure:
  /mantid_workspace_1/
    ├── definition          # NeXus definition
    ├── title               # Run title
    ├── workspace_name      # Mantid workspace name
    ├── event_workspace/    # Event data
    │   ├── tof             # Time-of-flight values (microseconds)
    │   ├── indices         # Cumulative event count per spectrum
    │   ├── weight          # Event weights
    │   └── error_squared   # Error values squared
    ├── instrument/         # Instrument definition
    │   ├── name
    │   ├── detector/
    │   └── instrument_xml/
    ├── logs/               # Time-series logs (similar to DASlogs)
    ├── sample/             # Sample information
    └── process/            # Processing history

Compared to standard NeXus:
  - Events stored by spectrum index, not by detector bank
  - No pulse timing information in .lite files
  - Logs stored under /logs/ instead of /entry/DASlogs/
  - Metadata stored in logs rather than entry-level datasets
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from nexus_processor.schemas import (
    MANTID_METADATA_FIELDS,
    MANTID_PRIORITY_LOGS,
    MANTID_INSTRUMENT_LOGS,
    MANTID_SAMPLE_LOG_PREFIXES,
    normalize_to_string,
    try_parse_numeric,
)


def detect_nexus_format(h5file: h5py.File) -> str:
    """
    Detect the NeXus file format type.
    
    Args:
        h5file: Open HDF5 file handle
        
    Returns:
        Format string: 'standard' for /entry structure, 'mantid' for /mantid_workspace_*
    """
    # Check for Mantid workspace structure
    for key in h5file.keys():
        if key.startswith('mantid_workspace'):
            return 'mantid'
    
    # Check for standard NeXus structure
    if 'entry' in h5file:
        return 'standard'
    
    # Unknown format - try to process as standard
    return 'unknown'


def get_mantid_workspace_name(h5file: h5py.File) -> Optional[str]:
    """
    Find the first mantid_workspace_* group in the file.
    
    Args:
        h5file: Open HDF5 file handle
        
    Returns:
        Name of the first mantid workspace group, or None if not found
    """
    for key in sorted(h5file.keys()):
        if key.startswith('mantid_workspace'):
            return key
    return None


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
        flat = value.flatten()
        if flat.dtype.kind in ('S', 'U', 'O'):
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
        return [safe_decode(v) for v in value]
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


def extract_mantid_metadata(h5file: h5py.File, workspace_name: str) -> Dict[str, Any]:
    """
    Extract run-level metadata from a Mantid workspace.
    
    Metadata is collected from:
    - Direct datasets (title, workspace_name, definition)
    - Priority logs (run_number, run_start, duration, proton_charge, etc.)
    
    Args:
        h5file: Open HDF5 file handle
        workspace_name: Name of the mantid workspace group
        
    Returns:
        Dictionary of metadata fields
    """
    workspace = h5file[workspace_name]
    metadata = {}
    
    # Read direct datasets
    for field in MANTID_METADATA_FIELDS:
        if field in workspace:
            metadata[field] = read_dataset_value(workspace[field])
    
    # Read priority logs for metadata
    logs_path = f'{workspace_name}/logs'
    if logs_path in h5file:
        logs = h5file[logs_path]
        for log_name in MANTID_PRIORITY_LOGS:
            if log_name in logs:
                log_group = logs[log_name]
                if isinstance(log_group, h5py.Group) and 'value' in log_group:
                    value = read_dataset_value(log_group['value'])
                    # Handle array values - take first element
                    if isinstance(value, (list, np.ndarray)):
                        value = value[0] if len(value) > 0 else None
                    metadata[log_name] = value
    
    # Map log names to standard metadata fields
    field_mapping = {
        'run_number': 'run_number',
        'run_start': 'start_time',
        'start_time': 'start_time',
        'end_time': 'end_time',
        'run_title': 'title',
        'duration': 'duration',
        'proton_charge': 'proton_charge',
        'gd_prtn_chrg': 'proton_charge',
        'experiment_identifier': 'experiment_identifier',
        'IPTS': 'experiment_identifier',
    }
    
    for log_key, meta_key in field_mapping.items():
        if log_key in metadata and meta_key not in metadata:
            metadata[meta_key] = metadata[log_key]
    
    # Try to extract run_number from workspace_name if not in logs
    # e.g., "SNAP_64413" -> 64413
    if 'run_number' not in metadata or metadata['run_number'] is None:
        ws_name = metadata.get('workspace_name', '')
        if ws_name:
            # Try to extract number from end of workspace name
            import re
            match = re.search(r'_(\d+)$', str(ws_name))
            if match:
                metadata['run_number'] = int(match.group(1))
    
    return metadata


def extract_mantid_instrument_info(h5file: h5py.File, workspace_name: str) -> Dict[str, Any]:
    """
    Extract instrument configuration from a Mantid workspace.
    
    Args:
        h5file: Open HDF5 file handle
        workspace_name: Name of the mantid workspace group
        
    Returns:
        Dictionary of instrument fields
    """
    instrument_path = f'{workspace_name}/instrument'
    if instrument_path not in h5file:
        return {}
    
    instrument = h5file[instrument_path]
    instrument_info = {}
    
    # Read instrument name
    if 'name' in instrument:
        instrument_info['name'] = read_dataset_value(instrument['name'])
    
    # Read instrument XML if available
    if 'instrument_xml' in instrument:
        xml_group = instrument['instrument_xml']
        if 'data' in xml_group:
            instrument_info['instrument_xml_data'] = read_dataset_value(xml_group['data'])
    
    # Read detector info
    if 'detector' in instrument:
        detector = instrument['detector']
        if 'detector_count' in detector:
            counts = read_dataset_value(detector['detector_count'])
            if isinstance(counts, (list, np.ndarray)):
                instrument_info['n_detectors'] = len(counts)
            else:
                instrument_info['n_detectors'] = counts
    
    # Read physical detector positions
    if 'physical_detectors' in instrument:
        phys = instrument['physical_detectors']
        if 'number_of_detectors' in phys:
            instrument_info['n_physical_detectors'] = read_dataset_value(phys['number_of_detectors'])
    
    # Check logs for additional instrument info
    logs_path = f'{workspace_name}/logs'
    if logs_path in h5file:
        logs = h5file[logs_path]
        for log_name in MANTID_INSTRUMENT_LOGS:
            if log_name in logs:
                log_group = logs[log_name]
                if isinstance(log_group, h5py.Group) and 'value' in log_group:
                    instrument_info[log_name] = read_dataset_value(log_group['value'])
    
    return instrument_info


def extract_mantid_sample_info(h5file: h5py.File, workspace_name: str) -> Dict[str, Any]:
    """
    Extract sample information from a Mantid workspace.
    
    Args:
        h5file: Open HDF5 file handle
        workspace_name: Name of the mantid workspace group
        
    Returns:
        Dictionary of sample fields
    """
    sample_path = f'{workspace_name}/sample'
    sample_info = {}
    
    # Read direct sample datasets
    if sample_path in h5file:
        sample = h5file[sample_path]
        for key in sample.keys():
            if isinstance(sample[key], h5py.Dataset):
                value = read_dataset_value(sample[key])
                # Convert lists/arrays to strings for schema compatibility
                if isinstance(value, (list, np.ndarray)):
                    if len(value) == 1:
                        value = value[0]
                    else:
                        value = str(value)
                sample_info[key] = value
    
    # Extract sample info from logs (SNAP uses BL3:CS:ITEMS:* pattern)
    logs_path = f'{workspace_name}/logs'
    if logs_path in h5file:
        logs = h5file[logs_path]
        for log_name in logs.keys():
            for prefix in MANTID_SAMPLE_LOG_PREFIXES:
                if log_name.startswith(prefix):
                    log_group = logs[log_name]
                    if isinstance(log_group, h5py.Group) and 'value' in log_group:
                        # Use the suffix as the field name
                        field_name = log_name.replace(prefix, '').lower()
                        if not field_name:
                            field_name = log_name.split(':')[-1].lower()
                        value = read_dataset_value(log_group['value'])
                        # Convert lists/arrays to strings for schema compatibility
                        if isinstance(value, (list, np.ndarray)):
                            if len(value) == 1:
                                value = value[0]
                            else:
                                value = str(value)
                        sample_info[field_name] = value
    
    return sample_info


def extract_mantid_logs(h5file: h5py.File, workspace_name: str) -> List[Dict[str, Any]]:
    """
    Extract time-series logs from a Mantid workspace.
    
    Converts Mantid /logs/ structure to nexus-processor DASlog format.
    
    Args:
        h5file: Open HDF5 file handle
        workspace_name: Name of the mantid workspace group
        
    Returns:
        List of DASlog records (one per time point per log)
    """
    logs_path = f'{workspace_name}/logs'
    if logs_path not in h5file:
        return []
    
    logs = h5file[logs_path]
    records = []
    
    for log_name in logs.keys():
        log_group = logs[log_name]
        if not isinstance(log_group, h5py.Group):
            continue
        
        # Get time and value arrays
        times = None
        values = None
        
        if 'time' in log_group:
            times = log_group['time'][()]
        if 'value' in log_group:
            values = log_group['value'][()]
        
        if times is None or values is None:
            continue
        
        times = np.atleast_1d(times)
        values = np.atleast_1d(values)
        
        # Handle multi-dimensional values
        if values.ndim > 1:
            # Flatten and convert strings
            if values.dtype.kind in ('S', 'U', 'O'):
                values = np.array([safe_decode(v) for v in values.flatten()])
            else:
                # For 2D numeric arrays, take first column or flatten
                if values.shape[0] == len(times):
                    values = values[:, 0] if values.ndim == 2 else values.flatten()
                else:
                    values = values.flatten()
        
        # Ensure same length
        n_points = min(len(times), len(values))
        times = times[:n_points]
        values = values[:n_points]
        
        # Vectorized record creation for numeric values
        if values.dtype.kind in ('i', 'u', 'f'):
            # Fast path for numeric arrays - create all records at once
            for i in range(n_points):
                records.append({
                    'log_name': log_name,
                    'device_name': None,
                    'device_id': None,
                    'time': float(times[i]),
                    'value': float(values[i]) if np.isfinite(values[i]) else None,
                    'average_value': None,
                    'min_value': None,
                    'max_value': None,
                })
        else:
            # String/object values - need individual decoding
            for i in range(n_points):
                val = values[i]
                if isinstance(val, bytes):
                    val = val.decode('utf-8', errors='replace')
                elif isinstance(val, (np.integer, np.floating)):
                    val = val.item()
                
                records.append({
                    'log_name': log_name,
                    'device_name': None,
                    'device_id': None,
                    'time': float(times[i]),
                    'value': val,
                    'average_value': None,
                    'min_value': None,
                    'max_value': None,
                })
    
    return records


def extract_mantid_logs_vectorized(h5file: h5py.File, workspace_name: str) -> Dict[str, np.ndarray]:
    """
    Extract time-series logs from a Mantid workspace using vectorized operations.
    
    This is a faster alternative to extract_mantid_logs that returns numpy arrays
    instead of a list of dictionaries, suitable for direct PyArrow table creation.
    
    Args:
        h5file: Open HDF5 file handle
        workspace_name: Name of the mantid workspace group
        
    Returns:
        Dictionary with arrays: log_name, time, value (all same length)
    """
    logs_path = f'{workspace_name}/logs'
    if logs_path not in h5file:
        return {'log_name': [], 'time': [], 'value': []}
    
    logs = h5file[logs_path]
    
    # Collect all data first to pre-allocate
    all_log_names = []
    all_times = []
    all_values = []
    
    for log_name in logs.keys():
        log_group = logs[log_name]
        if not isinstance(log_group, h5py.Group):
            continue
        
        if 'time' not in log_group or 'value' not in log_group:
            continue
        
        times = log_group['time'][()]
        values = log_group['value'][()]
        
        times = np.atleast_1d(times)
        values = np.atleast_1d(values)
        
        # Handle multi-dimensional values
        if values.ndim > 1:
            if values.dtype.kind in ('S', 'U', 'O'):
                values = np.array([safe_decode(v) for v in values.flatten()])
            else:
                if values.shape[0] == len(times):
                    values = values[:, 0] if values.ndim == 2 else values.flatten()
                else:
                    values = values.flatten()
        
        n_points = min(len(times), len(values))
        
        # Extend arrays
        all_log_names.extend([log_name] * n_points)
        all_times.extend(times[:n_points].tolist())
        
        # Convert values to strings for schema compatibility
        if values.dtype.kind in ('S', 'U', 'O'):
            for v in values[:n_points]:
                if isinstance(v, bytes):
                    all_values.append(v.decode('utf-8', errors='replace'))
                else:
                    all_values.append(str(v) if v is not None else None)
        else:
            all_values.extend([str(v) if np.isfinite(v) else None for v in values[:n_points]])
    
    return {
        'log_name': all_log_names,
        'time': all_times,
        'value': all_values,
    }


def extract_mantid_events(
    h5file: h5py.File, 
    workspace_name: str,
    max_events: Optional[int] = None,
    chunk_callback: Optional[callable] = None,
    chunk_size: int = 10_000_000,
) -> Dict[str, Any]:
    """
    Extract neutron event data from a Mantid workspace.
    
    Mantid event_workspace stores events differently than standard NeXus:
    - tof: Time-of-flight for each event (microseconds)
    - indices: Cumulative event count per spectrum (indices[i] = first event for spectrum i)
    - weight: Event weight (often 1.0, but can be other values)
    
    Note: Mantid .lite files do NOT contain pulse timing information.
    pulse_time and pulse_index will be None for these files.
    
    Args:
        h5file: Open HDF5 file handle
        workspace_name: Name of the mantid workspace group
        max_events: Maximum number of events to read (None for all)
        chunk_callback: Optional callback function for chunked processing
                       Called with (chunk_records, chunk_number) for each chunk
        chunk_size: Number of events per chunk when using callback
        
    Returns:
        Dictionary with bank data in nexus-processor format:
        {
            'event_workspace': {
                'records': [...],  # List of event records (if no callback)
                'total_counts': N,
                'n_pulses': 0,     # No pulse info in .lite files
            }
        }
    """
    event_path = f'{workspace_name}/event_workspace'
    if event_path not in h5file:
        return {}
    
    event_ws = h5file[event_path]
    
    # Read event data arrays
    if 'tof' not in event_ws or 'indices' not in event_ws:
        print(f"Warning: Missing tof or indices in {event_path}")
        return {}
    
    tof_ds = event_ws['tof']
    indices_ds = event_ws['indices']
    
    n_events_total = tof_ds.shape[0]
    n_spectra = indices_ds.shape[0] - 1  # indices has n_spectra + 1 elements
    
    print(f"    Found {n_events_total:,} events across {n_spectra:,} spectra")
    
    # Limit events if requested
    n_events = n_events_total
    if max_events and n_events > max_events:
        n_events = max_events
        print(f"    Limiting to {n_events:,} events")
    
    # Read indices to build spectrum mapping
    indices = indices_ds[:]
    
    # Check for weights
    has_weights = 'weight' in event_ws
    
    # Process events
    records = []
    chunk_number = 0
    
    # Read in chunks to handle large files
    read_chunk_size = min(chunk_size, n_events)
    
    for start_idx in range(0, n_events, read_chunk_size):
        end_idx = min(start_idx + read_chunk_size, n_events)
        
        # Read TOF chunk
        tof_chunk = tof_ds[start_idx:end_idx]
        
        # Read weights if available
        if has_weights:
            weight_chunk = event_ws['weight'][start_idx:end_idx]
        else:
            weight_chunk = np.ones(len(tof_chunk), dtype=np.float32)
        
        # Map events to spectrum IDs using indices
        # indices[i] = first event index for spectrum i
        # We need to find which spectrum each event belongs to
        event_ids = np.searchsorted(indices, np.arange(start_idx, end_idx), side='right') - 1
        
        # Build records for this chunk
        chunk_records = []
        for i in range(len(tof_chunk)):
            global_idx = start_idx + i
            record = {
                'bank': 'event_workspace',
                'event_idx': global_idx,
                'pulse_index': None,  # No pulse info in .lite files
                'pulse_time': None,   # No pulse info in .lite files
                'event_id': int(event_ids[i]),
                'time_offset': float(tof_chunk[i]),
                'event_weight': float(weight_chunk[i]),
            }
            chunk_records.append(record)
        
        if chunk_callback:
            # Use callback for chunked processing
            chunk_callback(chunk_records, chunk_number)
            chunk_number += 1
        else:
            records.extend(chunk_records)
        
        # Progress reporting
        if end_idx % 50_000_000 == 0 or end_idx == n_events:
            print(f"      Processed {end_idx:,} / {n_events:,} events ({100*end_idx/n_events:.1f}%)")
    
    return {
        'event_workspace': {
            'records': records if not chunk_callback else [],
            'total_counts': n_events_total,
            'n_pulses': 0,  # No pulse info in .lite files
            'n_spectra': n_spectra,
        }
    }


def get_mantid_instrument_id(h5file: h5py.File, workspace_name: str) -> str:
    """
    Determine the instrument ID from a Mantid workspace.
    
    Tries multiple sources:
    1. /instrument/name dataset
    2. Workspace name prefix (e.g., 'SNAP_64413' -> 'SNAP')
    3. Logs (instrument log)
    
    Args:
        h5file: Open HDF5 file handle
        workspace_name: Name of the mantid workspace group
        
    Returns:
        Instrument ID string (e.g., 'SNAP')
    """
    workspace = h5file[workspace_name]
    
    # Try instrument/name
    if 'instrument' in workspace and 'name' in workspace['instrument']:
        name = read_dataset_value(workspace['instrument']['name'])
        if name:
            return str(name).strip()
    
    # Try workspace_name prefix
    if 'workspace_name' in workspace:
        ws_name = read_dataset_value(workspace['workspace_name'])
        if ws_name and '_' in str(ws_name):
            return str(ws_name).split('_')[0]
    
    # Try logs
    if 'logs' in workspace and 'instrument' in workspace['logs']:
        log_group = workspace['logs']['instrument']
        if isinstance(log_group, h5py.Group) and 'value' in log_group:
            return str(read_dataset_value(log_group['value'])).strip()
    
    return 'UNKNOWN'
