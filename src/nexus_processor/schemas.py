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
    _field("instrument_id", pa.large_string(), "Instrument identifier (e.g., REF_L)"),
    _field("run_number", pa.int64(), "Run number identifier"),
    _field("run_id", pa.large_string(), "Unique run identifier (instrument_id:run_number)"),
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
    _field("instrument_id", pa.large_string(), "Instrument identifier (partition key)"),
    _field("run_number", pa.int64(), "Run number (partition key)"),
    _field("run_id", pa.large_string(), "Unique run identifier (instrument_id:run_number)"),
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
    _field("instrument_id", pa.large_string(), "Instrument identifier (partition key)"),
    _field("run_number", pa.int64(), "Run number (partition key)"),
    _field("run_id", pa.large_string(), "Unique run identifier (instrument_id:run_number)"),
    _field("name", pa.large_string(), "Instrument name (e.g., REF_L)"),
    _field("beamline", pa.large_string(), "Beamline identifier"),
    _field("instrument_xml_data", pa.large_string(), "Instrument definition XML"),
    # Additional fields stored as key-value map for flexibility
    _field("additional_fields", pa.map_(pa.large_string(), pa.large_string()), 
           "Additional instrument fields"),
])


SOFTWARE_SCHEMA = pa.schema([
    _field("instrument_id", pa.large_string(), "Instrument identifier (partition key)"),
    _field("run_number", pa.int64(), "Run number (partition key)"),
    _field("run_id", pa.large_string(), "Unique run identifier (instrument_id:run_number)"),
    _field("component", pa.large_string(), "Software component name"),
    _field("name", pa.large_string(), "Software name"),
    _field("version", pa.large_string(), "Software version"),
    # Additional fields stored as key-value map
    _field("additional_fields", pa.map_(pa.large_string(), pa.large_string()), 
           "Additional software metadata"),
])


USERS_SCHEMA = pa.schema([
    _field("instrument_id", pa.large_string(), "Instrument identifier (partition key)"),
    _field("run_number", pa.int64(), "Run number (partition key)"),
    _field("run_id", pa.large_string(), "Unique run identifier (instrument_id:run_number)"),
    _field("user_id", pa.large_string(), "User group identifier (user1, user2, etc.)"),
    _field("name", pa.large_string(), "User's full name"),
    _field("facility_user_id", pa.large_string(), "Facility user ID"),
    _field("role", pa.large_string(), "User's role in the experiment"),
    # Additional fields stored as key-value map
    _field("additional_fields", pa.map_(pa.large_string(), pa.large_string()), 
           "Additional user metadata"),
])


DASLOGS_SCHEMA = pa.schema([
    _field("instrument_id", pa.large_string(), "Instrument identifier (partition key)"),
    _field("run_number", pa.int64(), "Run number (partition key)"),
    _field("run_id", pa.large_string(), "Unique run identifier (instrument_id:run_number)"),
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
    _field("instrument_id", pa.large_string(), "Instrument identifier (partition key)"),
    _field("run_number", pa.int64(), "Run number (partition key)"),
    _field("run_id", pa.large_string(), "Unique run identifier (instrument_id:run_number)"),
    _field("bank", pa.large_string(), "Detector bank name"),
    _field("event_idx", pa.int64(), "Event index within the bank"),
    _field("pulse_index", pa.int64(), "Pulse index (correlates to proton_charge daslog)"),
    _field("event_id", pa.int64(), "Detector pixel ID"),
    _field("time_offset", pa.float64(), "Time offset within pulse (microseconds)"),
])


EVENT_SUMMARY_SCHEMA = pa.schema([
    _field("instrument_id", pa.large_string(), "Instrument identifier (partition key)"),
    _field("run_number", pa.int64(), "Run number (partition key)"),
    _field("run_id", pa.large_string(), "Unique run identifier (instrument_id:run_number)"),
    _field("bank", pa.large_string(), "Detector bank name"),
    _field("total_counts", pa.int64(), "Total counts in the bank"),
    _field("n_pulses", pa.int64(), "Number of neutron pulses"),
    _field("events_extracted", pa.int64(), "Number of events extracted"),
])


# =============================================================================
# Aggregated Schema for Iceberg Tables
# =============================================================================

# Schema for experiment_runs table that aggregates metadata, sample, instrument
# into a single denormalized table. This is the primary table for querying
# experiment information in Iceberg.
EXPERIMENT_RUNS_SCHEMA = pa.schema([
    # Partition columns
    _field("instrument_id", pa.large_string(), "Instrument identifier (partition key)"),
    _field("run_number", pa.int64(), "Run number (partition key)"),
    _field("run_id", pa.large_string(), "Unique run identifier (instrument_id:run_number)"),
    
    # Core metadata
    _field("title", pa.large_string(), "Experiment title"),
    _field("start_time", pa.large_string(), "Run start time (ISO format)"),
    _field("end_time", pa.large_string(), "Run end time (ISO format)"),
    _field("duration", pa.float64(), "Run duration in seconds"),
    _field("proton_charge", pa.float64(), "Total proton charge"),
    _field("total_counts", pa.int64(), "Total neutron counts"),
    _field("experiment_identifier", pa.large_string(), "Experiment ID (e.g., IPTS number)"),
    
    # Nested sample struct
    _field("sample", pa.struct([
        pa.field("name", pa.large_string()),
        pa.field("nature", pa.large_string()),
        pa.field("chemical_formula", pa.large_string()),
        pa.field("mass", pa.float64()),
        pa.field("temperature", pa.float64()),
    ]), "Sample information (nested struct)"),
    
    # Nested instrument struct
    _field("instrument", pa.struct([
        pa.field("name", pa.large_string()),
        pa.field("beamline", pa.large_string()),
    ]), "Instrument information (nested struct)"),
    
    # Nested software list
    _field("software", pa.list_(pa.struct([
        pa.field("component", pa.large_string()),
        pa.field("name", pa.large_string()),
        pa.field("version", pa.large_string()),
    ])), "Software components (list of structs)"),
    
    # Nested users list
    _field("users", pa.list_(pa.struct([
        pa.field("name", pa.large_string()),
        pa.field("role", pa.large_string()),
        pa.field("facility_user_id", pa.large_string()),
    ])), "Experiment users (list of structs)"),
    
    # Provenance
    _field("source_file", pa.large_string(), "Original NeXus filename"),
    _field("ingestion_time", pa.large_string(), "Conversion timestamp (ISO format)"),
])


# =============================================================================
# Helper Functions
# =============================================================================

def get_fields_without_partition(schema: pa.Schema) -> List[pa.Field]:
    """
    Get schema fields excluding partition columns (instrument_id, run_number).
    
    Useful for creating Iceberg tables where partition columns are defined
    separately from data columns.
    
    Args:
        schema: PyArrow schema
        
    Returns:
        List of fields excluding instrument_id and run_number
    """
    partition_cols = {'instrument_id', 'run_number'}
    return [f for f in schema if f.name not in partition_cols]


def schema_to_iceberg_fields(schema: pa.Schema) -> str:
    """
    Convert PyArrow schema to Iceberg SQL field definitions.
    
    This helper generates the column definitions for CREATE TABLE statements.
    
    Args:
        schema: PyArrow schema
        
    Returns:
        SQL column definitions string
    """
    type_map = {
        pa.large_string(): 'STRING',
        pa.int64(): 'BIGINT',
        pa.float64(): 'DOUBLE',
    }
    
    lines = []
    for field in schema:
        if field.type in type_map:
            sql_type = type_map[field.type]
        elif pa.types.is_map(field.type):
            sql_type = 'MAP<STRING, STRING>'
        elif pa.types.is_list(field.type):
            # Simplified - would need recursion for complex nested types
            sql_type = 'ARRAY<STRUCT<...>>'
        elif pa.types.is_struct(field.type):
            sql_type = 'STRUCT<...>'
        else:
            sql_type = str(field.type)
        
        lines.append(f"  {field.name} {sql_type}")
    
    return ',\n'.join(lines)


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
        'experiment_runs': EXPERIMENT_RUNS_SCHEMA,
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
