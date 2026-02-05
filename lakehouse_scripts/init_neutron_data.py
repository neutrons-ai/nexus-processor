"""
Neutron Data namespace init script for the lakehouse.

This script defines the neutron_data namespace with tables for storing
neutron scattering experiment data from NeXus files.

Tables:
    - experiment_runs: Primary experiment metadata (sample, instrument, users)
    - daslogs: Time-series DAS log data
    - events: Raw neutron detector events
    - event_summary: Per-bank event statistics

Usage:
    lakehouse init --script examples/neutron_data/init_neutron_data.py
"""

import pyarrow as pa

# Import base classes (available when run via lakehouse init)
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from neutron_lakehouse.scripts.protocol import InitScript, TableConfig


class NeutronDataInit(InitScript):
    """
    Initialize the neutron_data namespace for neutron scattering experiments.
    """

    NAMESPACE = "neutron_data"
    DESCRIPTION = "Neutron scattering experiment data from NeXus files"
    VERSION = "1.0.0"

    @classmethod
    def get_tables(cls):
        return [
            cls._experiment_runs_table(),
            cls._daslogs_table(),
            cls._events_table(),
            cls._event_summary_table(),
        ]

    @classmethod
    def get_metadata(cls):
        return {
            "domain": "neutron_scattering",
            "data_source": "nexus_files",
            "facility": "ornl_sns",
        }

    @classmethod
    def _experiment_runs_table(cls) -> TableConfig:
        """Primary experiment metadata table."""
        schema = pa.schema(
            [
                # Partition columns
                pa.field("instrument_id", pa.large_string(), nullable=False),
                pa.field("run_number", pa.int64(), nullable=False),
                pa.field("run_id", pa.large_string(), nullable=False),
                # Core metadata
                pa.field("title", pa.large_string()),
                pa.field("start_time", pa.large_string()),
                pa.field("end_time", pa.large_string()),
                pa.field("duration", pa.float64()),
                pa.field("proton_charge", pa.float64()),
                pa.field("total_counts", pa.int64()),
                pa.field("experiment_identifier", pa.large_string()),
                # Nested sample struct
                pa.field(
                    "sample",
                    pa.struct(
                        [
                            pa.field("name", pa.large_string()),
                            pa.field("nature", pa.large_string()),
                            pa.field("chemical_formula", pa.large_string()),
                            pa.field("mass", pa.float64()),
                            pa.field("temperature", pa.float64()),
                        ]
                    ),
                ),
                # Nested instrument struct
                pa.field(
                    "instrument",
                    pa.struct(
                        [
                            pa.field("name", pa.large_string()),
                            pa.field("beamline", pa.large_string()),
                        ]
                    ),
                ),
                # Nested software list
                pa.field(
                    "software",
                    pa.list_(
                        pa.struct(
                            [
                                pa.field("component", pa.large_string()),
                                pa.field("name", pa.large_string()),
                                pa.field("version", pa.large_string()),
                            ]
                        )
                    ),
                ),
                # Nested users list
                pa.field(
                    "users",
                    pa.list_(
                        pa.struct(
                            [
                                pa.field("name", pa.large_string()),
                                pa.field("role", pa.large_string()),
                                pa.field("facility_user_id", pa.large_string()),
                            ]
                        )
                    ),
                ),
                # Provenance
                pa.field("source_file", pa.large_string()),
                pa.field("ingestion_time", pa.large_string()),
            ]
        )

        return TableConfig(
            name="experiment_runs",
            schema=schema,
            partition_by=["instrument_id"],
            description="Primary experiment metadata with nested sample, instrument, and user info",
        )

    @classmethod
    def _daslogs_table(cls) -> TableConfig:
        """DAS time-series log data."""
        schema = pa.schema(
            [
                pa.field("instrument_id", pa.large_string(), nullable=False),
                pa.field("run_number", pa.int64(), nullable=False),
                pa.field("run_id", pa.large_string(), nullable=False),
                pa.field("log_name", pa.large_string()),
                pa.field("device_name", pa.large_string()),
                pa.field("device_id", pa.large_string()),
                pa.field("time", pa.float64()),
                pa.field("value", pa.large_string()),
                pa.field("value_numeric", pa.float64()),
                pa.field("average_value", pa.float64()),
                pa.field("min_value", pa.float64()),
                pa.field("max_value", pa.float64()),
            ]
        )

        return TableConfig(
            name="daslogs",
            schema=schema,
            partition_by=["instrument_id", "run_number"],
            description="Time-series DAS (Data Acquisition System) log data",
        )

    @classmethod
    def _events_table(cls) -> TableConfig:
        """Raw neutron detector events (can be very large)."""
        schema = pa.schema(
            [
                pa.field("instrument_id", pa.large_string(), nullable=False),
                pa.field("run_number", pa.int64(), nullable=False),
                pa.field("run_id", pa.large_string(), nullable=False),
                pa.field("bank", pa.large_string()),
                pa.field("event_idx", pa.int64()),
                pa.field("pulse_index", pa.int64()),
                pa.field("event_id", pa.int64()),
                pa.field("time_offset", pa.float64()),
            ]
        )

        return TableConfig(
            name="events",
            schema=schema,
            partition_by=["instrument_id", "run_number"],
            description="Raw neutron detector events (can be billions of rows)",
        )

    @classmethod
    def _event_summary_table(cls) -> TableConfig:
        """Per-bank event statistics."""
        schema = pa.schema(
            [
                pa.field("instrument_id", pa.large_string(), nullable=False),
                pa.field("run_number", pa.int64(), nullable=False),
                pa.field("run_id", pa.large_string(), nullable=False),
                pa.field("bank", pa.large_string()),
                pa.field("total_counts", pa.int64()),
                pa.field("n_pulses", pa.int64()),
                pa.field("events_extracted", pa.int64()),
            ]
        )

        return TableConfig(
            name="event_summary",
            schema=schema,
            partition_by=["instrument_id", "run_number"],
            description="Summary statistics per detector bank per run",
        )
