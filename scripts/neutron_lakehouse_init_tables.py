"""
Neutron Lakehouse Init Script for nexus-processor.

This script defines the nexus_data namespace with Iceberg tables for storing
processed NeXus data from neutron scattering experiments.

Tables (normalized/flat structure matching parquet output):
    - experiment_runs: Core experiment metadata
    - sample: Sample information
    - instrument: Instrument configuration
    - software: Software provenance
    - users: Experiment users
    - daslogs: Time-series DAS log data
    - events: Raw neutron detector events
    - event_summary: Per-bank event statistics

The tables are designed to be partitioned by instrument_id for efficient
queries across different beamlines (e.g., REF_L, VISION, ARCS).

All tables share (instrument_id, run_number, run_id) as join keys.

Usage:
    lakehouse init --script scripts/neutron_lakehouse_init_tables.py

Requirements:
    - neutron-lakehouse package (provides InitScript, TableConfig)
    - pyarrow
"""

import pyarrow as pa

# Import base classes from neutron-lakehouse
import sys
import os
sys.path.insert(0, os.path.expanduser("~/git/neutron-lakehouse/src"))

from neutron_lakehouse.scripts.protocol import InitScript, TableConfig


class NexusDataInit(InitScript):
    """
    Initialize the nexus_data namespace for processed NeXus experiment files.
    
    This namespace stores neutron scattering data extracted from NeXus HDF5 files
    by the nexus-processor package. The schema matches the Parquet files produced
    by `nexus-processor convert`.
    
    Tables use a normalized (flat) structure - each data category has its own table,
    linked by (instrument_id, run_number, run_id).
    """
    
    NAMESPACE = "nexus_data"
    DESCRIPTION = "Processed NeXus neutron scattering experiment data"
    VERSION = "2.0.0"
    
    @classmethod
    def get_tables(cls):
        """Return all table configurations for this namespace."""
        return [
            cls._experiment_runs_table(),
            cls._sample_table(),
            cls._instrument_table(),
            cls._software_table(),
            cls._users_table(),
            cls._daslogs_table(),
            cls._events_table(),
            cls._event_summary_table(),
        ]
    
    @classmethod
    def get_metadata(cls):
        """Return namespace metadata for registry."""
        return {
            "domain": "neutron_scattering",
            "data_source": "nexus_hdf5_files",
            "facility": "ornl_sns",
            "producer_package": "nexus-processor",
            "schema_version": "2.0.0",
            "table_design": "normalized",
        }
    
    @classmethod
    def _key_fields(cls):
        """Return the common key fields used by all tables."""
        return [
            pa.field("instrument_id", pa.large_string(), nullable=False,
                     metadata={"description": "Instrument identifier (e.g., REF_L, VISION)"}),
            pa.field("run_number", pa.int64(), nullable=False,
                     metadata={"description": "Run number identifier"}),
            pa.field("run_id", pa.large_string(), nullable=False,
                     metadata={"description": "Unique run identifier (instrument_id:run_number)"}),
        ]
    
    @classmethod
    def _experiment_runs_table(cls) -> TableConfig:
        """Core experiment metadata table."""
        schema = pa.schema(cls._key_fields() + [
            pa.field("title", pa.large_string(),
                     metadata={"description": "Experiment title"}),
            pa.field("start_time", pa.large_string(),
                     metadata={"description": "Run start time (ISO format)"}),
            pa.field("end_time", pa.large_string(),
                     metadata={"description": "Run end time (ISO format)"}),
            pa.field("duration", pa.float64(),
                     metadata={"description": "Run duration in seconds"}),
            pa.field("proton_charge", pa.float64(),
                     metadata={"description": "Total proton charge (coulombs)"}),
            pa.field("total_counts", pa.int64(),
                     metadata={"description": "Total neutron counts"}),
            pa.field("experiment_identifier", pa.large_string(),
                     metadata={"description": "Experiment ID (e.g., IPTS number)"}),
            pa.field("definition", pa.large_string(),
                     metadata={"description": "NeXus definition (e.g., NXsas, NXrefscan)"}),
            pa.field("source_file", pa.large_string(),
                     metadata={"description": "Original NeXus filename"}),
            pa.field("source_path", pa.large_string(),
                     metadata={"description": "Original NeXus file path"}),
            pa.field("ingestion_time", pa.large_string(),
                     metadata={"description": "Conversion timestamp (ISO format)"}),
            pa.field("file_attributes", pa.large_string(),
                     metadata={"description": "HDF5 file-level attributes (JSON)"}),
            pa.field("entry_attributes", pa.large_string(),
                     metadata={"description": "HDF5 entry-level attributes (JSON)"}),
        ])
        
        return TableConfig(
            name="experiment_runs",
            schema=schema,
            partition_by=["instrument_id"],
            description="Core experiment metadata - join with sample, instrument, software, users tables",
            properties={
                "write.format.default": "parquet",
                "write.parquet.compression-codec": "zstd",
            },
        )
    
    @classmethod
    def _sample_table(cls) -> TableConfig:
        """Sample information table."""
        schema = pa.schema(cls._key_fields() + [
            pa.field("name", pa.large_string(),
                     metadata={"description": "Sample name"}),
            pa.field("nature", pa.large_string(),
                     metadata={"description": "Sample nature (solid, liquid, powder, etc.)"}),
            pa.field("chemical_formula", pa.large_string(),
                     metadata={"description": "Chemical formula"}),
            pa.field("mass", pa.float64(),
                     metadata={"description": "Sample mass in grams"}),
            pa.field("temperature", pa.float64(),
                     metadata={"description": "Sample temperature in Kelvin"}),
            pa.field("additional_fields", pa.large_string(),
                     metadata={"description": "Additional sample fields (JSON)"}),
        ])
        
        return TableConfig(
            name="sample",
            schema=schema,
            partition_by=["instrument_id"],
            description="Sample information for each experiment run",
            properties={
                "write.format.default": "parquet",
                "write.parquet.compression-codec": "zstd",
            },
        )
    
    @classmethod
    def _instrument_table(cls) -> TableConfig:
        """Instrument configuration table."""
        schema = pa.schema(cls._key_fields() + [
            pa.field("name", pa.large_string(),
                     metadata={"description": "Instrument name"}),
            pa.field("beamline", pa.large_string(),
                     metadata={"description": "Beamline identifier"}),
            pa.field("instrument_xml_data", pa.large_string(),
                     metadata={"description": "Instrument XML definition"}),
            pa.field("additional_fields", pa.large_string(),
                     metadata={"description": "Additional instrument fields (JSON)"}),
        ])
        
        return TableConfig(
            name="instrument",
            schema=schema,
            partition_by=["instrument_id"],
            description="Instrument configuration for each experiment run",
            properties={
                "write.format.default": "parquet",
                "write.parquet.compression-codec": "zstd",
            },
        )
    
    @classmethod
    def _software_table(cls) -> TableConfig:
        """Software provenance table (multiple rows per run)."""
        schema = pa.schema(cls._key_fields() + [
            pa.field("component", pa.large_string(),
                     metadata={"description": "Software component (e.g., DAS, reduction)"}),
            pa.field("name", pa.large_string(),
                     metadata={"description": "Software name"}),
            pa.field("version", pa.large_string(),
                     metadata={"description": "Software version"}),
            pa.field("additional_fields", pa.large_string(),
                     metadata={"description": "Additional software fields (JSON)"}),
        ])
        
        return TableConfig(
            name="software",
            schema=schema,
            partition_by=["instrument_id"],
            description="Software provenance - may have multiple rows per run",
            properties={
                "write.format.default": "parquet",
                "write.parquet.compression-codec": "zstd",
            },
        )
    
    @classmethod
    def _users_table(cls) -> TableConfig:
        """Experiment users table (multiple rows per run)."""
        schema = pa.schema(cls._key_fields() + [
            pa.field("user_id", pa.large_string(),
                     metadata={"description": "User identifier"}),
            pa.field("name", pa.large_string(),
                     metadata={"description": "User name"}),
            pa.field("facility_user_id", pa.large_string(),
                     metadata={"description": "Facility-assigned user ID"}),
            pa.field("role", pa.large_string(),
                     metadata={"description": "User role (PI, experimenter, etc.)"}),
            pa.field("additional_fields", pa.large_string(),
                     metadata={"description": "Additional user fields (JSON)"}),
        ])
        
        return TableConfig(
            name="users",
            schema=schema,
            partition_by=["instrument_id"],
            description="Experiment users - may have multiple rows per run",
            properties={
                "write.format.default": "parquet",
                "write.parquet.compression-codec": "zstd",
            },
        )
    
    @classmethod
    def _daslogs_table(cls) -> TableConfig:
        """DAS (Data Acquisition System) time-series log data."""
        schema = pa.schema(cls._key_fields() + [
            pa.field("log_name", pa.large_string(),
                     metadata={"description": "Name of the DAS log"}),
            pa.field("device_name", pa.large_string(),
                     metadata={"description": "Device name"}),
            pa.field("device_id", pa.large_string(),
                     metadata={"description": "Device identifier"}),
            pa.field("time", pa.float64(),
                     metadata={"description": "Time offset in seconds from run start"}),
            pa.field("value", pa.large_string(),
                     metadata={"description": "Log value (string-encoded for mixed types)"}),
            pa.field("value_numeric", pa.float64(),
                     metadata={"description": "Numeric value if parseable"}),
            pa.field("average_value", pa.float64(),
                     metadata={"description": "Average value over the run"}),
            pa.field("min_value", pa.float64(),
                     metadata={"description": "Minimum value over the run"}),
            pa.field("max_value", pa.float64(),
                     metadata={"description": "Maximum value over the run"}),
        ])
        
        return TableConfig(
            name="daslogs",
            schema=schema,
            partition_by=["instrument_id", "run_number"],
            description="Time-series DAS (Data Acquisition System) log data from experiments",
            properties={
                "write.format.default": "parquet",
                "write.parquet.compression-codec": "zstd",
            },
        )
    
    @classmethod
    def _events_table(cls) -> TableConfig:
        """Raw neutron detector events."""
        schema = pa.schema(cls._key_fields() + [
            pa.field("bank", pa.large_string(),
                     metadata={"description": "Detector bank name"}),
            pa.field("event_idx", pa.int64(),
                     metadata={"description": "Event index within the bank"}),
            pa.field("pulse_index", pa.int64(),
                     metadata={"description": "Pulse index (correlates to proton_charge)"}),
            pa.field("pulse_time", pa.float64(),
                     metadata={"description": "Pulse time in seconds from run start"}),
            pa.field("event_id", pa.int64(),
                     metadata={"description": "Detector pixel ID"}),
            pa.field("time_offset", pa.float64(),
                     metadata={"description": "Time offset within pulse (microseconds)"}),
            pa.field("event_weight", pa.float64(),
                     metadata={"description": "Event weight (default 1.0)"}),
        ])
        
        return TableConfig(
            name="events",
            schema=schema,
            partition_by=["instrument_id", "run_number"],
            description="Raw neutron detector events (can be billions of rows per run)",
            properties={
                "write.format.default": "parquet",
                "write.parquet.compression-codec": "zstd",
                "write.target-file-size-bytes": "134217728",
            },
        )
    
    @classmethod
    def _event_summary_table(cls) -> TableConfig:
        """Per-bank event statistics."""
        schema = pa.schema(cls._key_fields() + [
            pa.field("bank", pa.large_string(),
                     metadata={"description": "Detector bank name"}),
            pa.field("total_counts", pa.int64(),
                     metadata={"description": "Total counts in the bank"}),
            pa.field("n_pulses", pa.int64(),
                     metadata={"description": "Number of neutron pulses"}),
            pa.field("events_extracted", pa.int64(),
                     metadata={"description": "Number of events extracted"}),
        ])
        
        return TableConfig(
            name="event_summary",
            schema=schema,
            partition_by=["instrument_id", "run_number"],
            description="Summary statistics per detector bank per run",
            properties={
                "write.format.default": "parquet",
                "write.parquet.compression-codec": "zstd",
            },
        )


if __name__ == "__main__":
    errors = NexusDataInit.validate()
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print(f"Valid init script for namespace: {NexusDataInit.NAMESPACE}")
        print(f"  Description: {NexusDataInit.DESCRIPTION}")
        print(f"  Version: {NexusDataInit.VERSION}")
        print(f"  Tables: {[t.name for t in NexusDataInit.get_tables()]}")
        print(f"  Metadata: {NexusDataInit.get_metadata()}")
