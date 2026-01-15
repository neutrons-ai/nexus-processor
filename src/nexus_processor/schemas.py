"""
PyArrow schemas for Iceberg-compatible Parquet files.

This module defines explicit schemas for all parquet file types to ensure:
- Consistent, predictable column types across files
- Compatibility with Apache Iceberg table format
- Proper handling of nested types (lists, structs)
- Self-documenting schemas with field metadata

Schema Design Principles:
- Use nullable types to handle missing data gracefully
- Prefer large_ variants (large_string, large_binary) for variable-length data
- Use struct types for nested metadata instead of string serialization
- Include partition columns (run_number, record_type) for efficient querying
- Embed field descriptions in metadata for discoverability
"""

import pyarrow as pa
from typing import Any, Dict, List, Optional


def _field(name: str, dtype: pa.DataType, description: str, 
           nullable: bool = True) -> pa.Field:
    """Create a PyArrow field with metadata."""
    return pa.field(
        name, 
        dtype, 
        nullable=nullable,
        metadata={"description": description}
    )


# =============================================================================
# Core Schemas for Split File Mode
# =============================================================================

METADATA_SCHEMA = pa.schema([
    _field("run_number", pa.int64(), "Run number identifier"),
    _field("title", pa.large_string(), "Experiment title"),
    _field("start_time", pa.large_string(), "Run start time (ISO format)"),
    _field("end_time", pa.large_string(), "Run end time (ISO format)"),
    _field("duration", pa.float64(), "Run duration in seconds"),
    _field("proton_charge", pa.float64(), "Total proton charge"),
    _field("total_counts", pa.int64(), "Total neutron counts"),
    _field("experiment_identifier", pa.large_string(), "Experiment ID (e.g., IPTS number)"),
    _field("definition", pa.large_string(), "NeXus definition name"),
    _field("source_file", pa.large_string(), "Original filename"),
    _field("source_path", pa.large_string(), "Original file path"),
    _field("ingestion_time", pa.large_string(), "Conversion timestamp (ISO format)"),
    # File and entry attributes stored as key-value maps
    _field("file_attributes", pa.map_(pa.large_string(), pa.large_string()), 
           "HDF5 file-level attributes"),
    _field("entry_attributes", pa.map_(pa.large_string(), pa.large_string()), 
           "HDF5 entry-level attributes"),
])


SAMPLE_SCHEMA = pa.schema([
    _field("run_number", pa.int64(), "Run number (partition key)"),
    _field("name", pa.large_string(), "Sample name"),
    _field("nature", pa.large_string(), "Sample type/nature"),
    _field("chemical_formula", pa.large_string(), "Chemical formula"),
    _field("mass", pa.float64(), "Sample mass"),
    _field("temperature", pa.float64(), "Sample temperature"),
    # Additional fields stored as key-value map for flexibility
    _field("additional_fields", pa.map_(pa.large_string(), pa.large_string()), 
           "Additional sample fields"),
])


INSTRUMENT_SCHEMA = pa.schema([
    _field("run_number", pa.int64(), "Run number (partition key)"),
    _field("name", pa.large_string(), "Instrument name (e.g., REF_L)"),
    _field("beamline", pa.large_string(), "Beamline identifier"),
    _field("instrument_xml_data", pa.large_string(), "Instrument definition XML"),
    # Additional fields stored as key-value map for flexibility
    _field("additional_fields", pa.map_(pa.large_string(), pa.large_string()), 
           "Additional instrument fields"),
])


SOFTWARE_SCHEMA = pa.schema([
    _field("run_number", pa.int64(), "Run number (partition key)"),
    _field("component", pa.large_string(), "Software component name"),
    _field("name", pa.large_string(), "Software name"),
    _field("version", pa.large_string(), "Software version"),
    # Additional fields stored as key-value map
    _field("additional_fields", pa.map_(pa.large_string(), pa.large_string()), 
           "Additional software metadata"),
])


USERS_SCHEMA = pa.schema([
    _field("run_number", pa.int64(), "Run number (partition key)"),
    _field("user_id", pa.large_string(), "User group identifier (user1, user2, etc.)"),
    _field("name", pa.large_string(), "User's full name"),
    _field("facility_user_id", pa.large_string(), "Facility user ID"),
    _field("role", pa.large_string(), "User's role in the experiment"),
    # Additional fields stored as key-value map
    _field("additional_fields", pa.map_(pa.large_string(), pa.large_string()), 
           "Additional user metadata"),
])


DASLOGS_SCHEMA = pa.schema([
    _field("run_number", pa.int64(), "Run number (partition key)"),
    _field("log_name", pa.large_string(), "Name of the DAS log"),
    _field("device_name", pa.large_string(), "Device name"),
    _field("device_id", pa.large_string(), "Device identifier"),
    _field("time", pa.float64(), "Time offset in seconds from run start"),
    _field("value", pa.large_string(), "Log value (string-encoded for mixed types)"),
    _field("value_numeric", pa.float64(), "Numeric value if parseable"),
    _field("average_value", pa.float64(), "Average value over the run"),
    _field("min_value", pa.float64(), "Minimum value over the run"),
    _field("max_value", pa.float64(), "Maximum value over the run"),
])


EVENTS_SCHEMA = pa.schema([
    _field("run_number", pa.int64(), "Run number (partition key)"),
    _field("bank", pa.large_string(), "Detector bank name"),
    _field("event_idx", pa.int64(), "Event index within the bank"),
    _field("event_id", pa.int64(), "Detector pixel ID"),
    _field("time_offset", pa.float64(), "Time offset within pulse (microseconds)"),
])


EVENT_SUMMARY_SCHEMA = pa.schema([
    _field("run_number", pa.int64(), "Run number (partition key)"),
    _field("bank", pa.large_string(), "Detector bank name"),
    _field("total_counts", pa.int64(), "Total counts in the bank"),
    _field("n_pulses", pa.int64(), "Number of neutron pulses"),
    _field("events_extracted", pa.int64(), "Number of events extracted"),
])


# =============================================================================
# Combined Schema for Single File Mode
# =============================================================================

COMBINED_SCHEMA = pa.schema([
    # Partition/discrimination columns
    _field("run_number", pa.int64(), "Run number (partition key)"),
    _field("record_type", pa.large_string(), "Record type: 'daslog' or 'event'"),
    
    # DAS log columns (NULL for events)
    _field("log_name", pa.large_string(), "Name of the DAS log"),
    _field("device_name", pa.large_string(), "Device name"),
    _field("device_id", pa.large_string(), "Device identifier"),
    _field("time", pa.float64(), "Time offset in seconds (daslogs) or microseconds (events)"),
    _field("value", pa.large_string(), "Log value"),
    _field("value_numeric", pa.float64(), "Numeric value if parseable"),
    _field("average_value", pa.float64(), "Average value over the run"),
    _field("min_value", pa.float64(), "Minimum value over the run"),
    _field("max_value", pa.float64(), "Maximum value over the run"),
    
    # Event columns (NULL for daslogs)
    _field("bank", pa.large_string(), "Detector bank name"),
    _field("event_idx", pa.int64(), "Event index within the bank"),
    _field("event_id", pa.int64(), "Detector pixel ID"),
    _field("time_offset", pa.float64(), "Time offset within pulse (microseconds)"),
    
    # Denormalized metadata (struct type for clean organization)
    _field("metadata", pa.struct([
        pa.field("title", pa.large_string()),
        pa.field("start_time", pa.large_string()),
        pa.field("end_time", pa.large_string()),
        pa.field("duration", pa.float64()),
        pa.field("proton_charge", pa.float64()),
        pa.field("total_counts", pa.int64()),
        pa.field("experiment_identifier", pa.large_string()),
        pa.field("definition", pa.large_string()),
        pa.field("source_file", pa.large_string()),
    ]), "Run metadata (nested struct)"),
    
    _field("sample", pa.struct([
        pa.field("name", pa.large_string()),
        pa.field("nature", pa.large_string()),
        pa.field("chemical_formula", pa.large_string()),
        pa.field("mass", pa.float64()),
        pa.field("temperature", pa.float64()),
    ]), "Sample information (nested struct)"),
    
    _field("instrument", pa.struct([
        pa.field("name", pa.large_string()),
        pa.field("beamline", pa.large_string()),
    ]), "Instrument information (nested struct)"),
    
    _field("users", pa.list_(pa.struct([
        pa.field("name", pa.large_string()),
        pa.field("role", pa.large_string()),
    ])), "List of users"),
])


# =============================================================================
# Helper Functions
# =============================================================================

def get_schema_metadata() -> Dict[str, Dict[str, str]]:
    """
    Get schema metadata for documentation.
    
    Returns:
        Dictionary mapping schema name to field descriptions
    """
    schemas = {
        'metadata': METADATA_SCHEMA,
        'sample': SAMPLE_SCHEMA,
        'instrument': INSTRUMENT_SCHEMA,
        'software': SOFTWARE_SCHEMA,
        'users': USERS_SCHEMA,
        'daslogs': DASLOGS_SCHEMA,
        'events': EVENTS_SCHEMA,
        'event_summary': EVENT_SUMMARY_SCHEMA,
        'combined': COMBINED_SCHEMA,
    }
    
    result = {}
    for name, schema in schemas.items():
        result[name] = {}
        for field in schema:
            if field.metadata and b'description' in field.metadata:
                result[name][field.name] = field.metadata[b'description'].decode('utf-8')
            else:
                result[name][field.name] = ""
    
    return result


def try_parse_numeric(value: Any) -> Optional[float]:
    """
    Try to parse a value as a numeric float.
    
    Args:
        value: Any value to attempt to parse
        
    Returns:
        Float value if parseable, None otherwise
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    return None


def normalize_to_string(value: Any) -> Optional[str]:
    """
    Normalize any value to a string representation.
    
    Args:
        value: Any value to convert
        
    Returns:
        String representation, or None if value is None
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, (list, dict)):
        import json
        return json.dumps(value)
    return str(value)


def build_attribute_map(data: Dict[str, Any], prefix: str) -> Dict[str, str]:
    """
    Build a string-to-string map from dictionary with prefixed keys.
    
    Filters keys that start with the given prefix and removes the prefix.
    
    Args:
        data: Source dictionary
        prefix: Prefix to filter and remove
        
    Returns:
        Dictionary with prefix stripped from keys, all values as strings
    """
    result = {}
    for key, value in data.items():
        if key.startswith(prefix):
            clean_key = key[len(prefix):]
            result[clean_key] = normalize_to_string(value)
    return result


def extract_known_fields(data: Dict[str, Any], known_fields: List[str]) -> Dict[str, str]:
    """
    Extract fields not in the known_fields list as a string map.
    
    Args:
        data: Source dictionary
        known_fields: List of field names that are handled separately
        
    Returns:
        Dictionary of additional fields (not in known_fields)
    """
    result = {}
    for key, value in data.items():
        if key not in known_fields and not key.startswith(('file_attr_', 'entry_attr_')):
            result[key] = normalize_to_string(value)
    return result
