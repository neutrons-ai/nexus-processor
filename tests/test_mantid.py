"""
Tests for the nexus_processor.mantid module.

These tests use mock HDF5 files to test the Mantid workspace extraction
functions without requiring actual SNAP NeXus files.
"""

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from nexus_processor import mantid


class TestDetectNexusFormat:
    """Tests for the detect_nexus_format function."""

    def test_detects_mantid_format(self, tmp_path):
        filepath = tmp_path / "test_mantid.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("mantid_workspace_1")
        
        with h5py.File(filepath, "r") as f:
            assert mantid.detect_nexus_format(f) == "mantid"

    def test_detects_standard_format(self, tmp_path):
        filepath = tmp_path / "test_standard.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("entry")
        
        with h5py.File(filepath, "r") as f:
            assert mantid.detect_nexus_format(f) == "standard"

    def test_unknown_format_returns_unknown(self, tmp_path):
        filepath = tmp_path / "test_unknown.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("some_other_group")
        
        with h5py.File(filepath, "r") as f:
            assert mantid.detect_nexus_format(f) == "unknown"

    def test_prefers_mantid_over_entry(self, tmp_path):
        """If both exist, mantid format takes precedence."""
        filepath = tmp_path / "test_both.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("entry")
            f.create_group("mantid_workspace_1")
        
        with h5py.File(filepath, "r") as f:
            assert mantid.detect_nexus_format(f) == "mantid"


class TestGetMantidWorkspaceName:
    """Tests for the get_mantid_workspace_name function."""

    def test_finds_workspace(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("mantid_workspace_1")
        
        with h5py.File(filepath, "r") as f:
            assert mantid.get_mantid_workspace_name(f) == "mantid_workspace_1"

    def test_returns_first_workspace_sorted(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("mantid_workspace_2")
            f.create_group("mantid_workspace_1")
        
        with h5py.File(filepath, "r") as f:
            assert mantid.get_mantid_workspace_name(f) == "mantid_workspace_1"

    def test_returns_none_if_not_found(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("entry")
        
        with h5py.File(filepath, "r") as f:
            assert mantid.get_mantid_workspace_name(f) is None


class TestSafeDecode:
    """Tests for the safe_decode function in mantid module."""

    def test_none_returns_none(self):
        assert mantid.safe_decode(None) is None

    def test_bytes_decoded(self):
        assert mantid.safe_decode(b"hello") == "hello"

    def test_numpy_integer_converted(self):
        result = mantid.safe_decode(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_array_single_element(self):
        result = mantid.safe_decode(np.array([3.14]))
        assert result == pytest.approx(3.14)


class TestExtractMantidMetadata:
    """Tests for the extract_mantid_metadata function."""

    def test_extracts_title(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            ws = f.create_group("mantid_workspace_1")
            ws.create_dataset("title", data=b"Test Run")
        
        with h5py.File(filepath, "r") as f:
            metadata = mantid.extract_mantid_metadata(f, "mantid_workspace_1")
            assert metadata["title"] == "Test Run"

    def test_extracts_workspace_name(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            ws = f.create_group("mantid_workspace_1")
            ws.create_dataset("workspace_name", data=b"SNAP_64413")
        
        with h5py.File(filepath, "r") as f:
            metadata = mantid.extract_mantid_metadata(f, "mantid_workspace_1")
            assert metadata["workspace_name"] == "SNAP_64413"

    def test_extracts_run_number_from_logs(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            ws = f.create_group("mantid_workspace_1")
            logs = ws.create_group("logs")
            run_number = logs.create_group("run_number")
            run_number.create_dataset("value", data=np.array([64413]))
        
        with h5py.File(filepath, "r") as f:
            metadata = mantid.extract_mantid_metadata(f, "mantid_workspace_1")
            assert metadata["run_number"] == 64413

    def test_extracts_proton_charge_from_logs(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            ws = f.create_group("mantid_workspace_1")
            logs = ws.create_group("logs")
            pcharge = logs.create_group("proton_charge")
            pcharge.create_dataset("value", data=np.array([123.456]))
        
        with h5py.File(filepath, "r") as f:
            metadata = mantid.extract_mantid_metadata(f, "mantid_workspace_1")
            assert metadata["proton_charge"] == pytest.approx(123.456)

    def test_extracts_run_number_from_workspace_name(self, tmp_path):
        """If run_number not in logs, extract from workspace_name."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            ws = f.create_group("mantid_workspace_1")
            ws.create_dataset("workspace_name", data=b"SNAP_64413")
            ws.create_group("logs")  # Empty logs
        
        with h5py.File(filepath, "r") as f:
            metadata = mantid.extract_mantid_metadata(f, "mantid_workspace_1")
            assert metadata["run_number"] == 64413


class TestExtractMantidInstrumentInfo:
    """Tests for the extract_mantid_instrument_info function."""

    def test_extracts_instrument_name(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            ws = f.create_group("mantid_workspace_1")
            instrument = ws.create_group("instrument")
            instrument.create_dataset("name", data=b"SNAP")
        
        with h5py.File(filepath, "r") as f:
            info = mantid.extract_mantid_instrument_info(f, "mantid_workspace_1")
            assert info["name"] == "SNAP"

    def test_extracts_detector_count(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            ws = f.create_group("mantid_workspace_1")
            instrument = ws.create_group("instrument")
            detector = instrument.create_group("detector")
            detector.create_dataset("detector_count", data=np.arange(18432))
        
        with h5py.File(filepath, "r") as f:
            info = mantid.extract_mantid_instrument_info(f, "mantid_workspace_1")
            assert info["n_detectors"] == 18432

    def test_returns_empty_if_no_instrument(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("mantid_workspace_1")
        
        with h5py.File(filepath, "r") as f:
            info = mantid.extract_mantid_instrument_info(f, "mantid_workspace_1")
            assert info == {}


class TestExtractMantidSampleInfo:
    """Tests for the extract_mantid_sample_info function."""

    def test_extracts_sample_datasets(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            ws = f.create_group("mantid_workspace_1")
            sample = ws.create_group("sample")
            sample.create_dataset("geom_height", data=np.array([1.5]))
            sample.create_dataset("geom_width", data=np.array([2.0]))
        
        with h5py.File(filepath, "r") as f:
            info = mantid.extract_mantid_sample_info(f, "mantid_workspace_1")
            assert info["geom_height"] == pytest.approx(1.5)
            assert info["geom_width"] == pytest.approx(2.0)

    def test_extracts_sample_from_logs(self, tmp_path):
        """SNAP stores sample info in logs with BL3:CS:ITEMS: prefix."""
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            ws = f.create_group("mantid_workspace_1")
            ws.create_group("sample")
            logs = ws.create_group("logs")
            formula = logs.create_group("BL3:CS:ITEMS:Formula")
            formula.create_dataset("value", data=np.array([b"H2O"]))
        
        with h5py.File(filepath, "r") as f:
            info = mantid.extract_mantid_sample_info(f, "mantid_workspace_1")
            assert info["formula"] == "H2O"


class TestExtractMantidLogs:
    """Tests for the extract_mantid_logs function."""

    def test_extracts_time_series_logs(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            ws = f.create_group("mantid_workspace_1")
            logs = ws.create_group("logs")
            
            temp = logs.create_group("temperature")
            temp.create_dataset("time", data=np.array([0.0, 1.0, 2.0]))
            temp.create_dataset("value", data=np.array([300.0, 301.0, 302.0]))
        
        with h5py.File(filepath, "r") as f:
            records = mantid.extract_mantid_logs(f, "mantid_workspace_1")
            
            assert len(records) == 3
            assert records[0]["log_name"] == "temperature"
            assert records[0]["time"] == 0.0
            assert records[0]["value"] == pytest.approx(300.0)
            assert records[2]["value"] == pytest.approx(302.0)

    def test_handles_string_values(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            ws = f.create_group("mantid_workspace_1")
            logs = ws.create_group("logs")
            
            status = logs.create_group("status")
            status.create_dataset("time", data=np.array([0.0]))
            status.create_dataset("value", data=np.array([b"running"]))
        
        with h5py.File(filepath, "r") as f:
            records = mantid.extract_mantid_logs(f, "mantid_workspace_1")
            
            assert len(records) == 1
            assert records[0]["value"] == "running"

    def test_returns_empty_if_no_logs(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("mantid_workspace_1")
        
        with h5py.File(filepath, "r") as f:
            records = mantid.extract_mantid_logs(f, "mantid_workspace_1")
            assert records == []


class TestExtractMantidEvents:
    """Tests for the extract_mantid_events function."""

    def test_extracts_events(self, tmp_path):
        filepath = tmp_path / "test.h5"
        n_events = 100
        n_spectra = 10
        
        with h5py.File(filepath, "w") as f:
            ws = f.create_group("mantid_workspace_1")
            event_ws = ws.create_group("event_workspace")
            
            # Create event data
            tof = np.random.uniform(1000, 50000, n_events)
            event_ws.create_dataset("tof", data=tof)
            
            # Create indices (cumulative event counts per spectrum)
            events_per_spectrum = n_events // n_spectra
            indices = np.arange(0, n_events + 1, events_per_spectrum)
            indices[-1] = n_events  # Ensure last index is total events
            event_ws.create_dataset("indices", data=indices)
            
            # Create weights
            weights = np.ones(n_events)
            event_ws.create_dataset("weight", data=weights)
        
        with h5py.File(filepath, "r") as f:
            result = mantid.extract_mantid_events(f, "mantid_workspace_1")
            
            assert "event_workspace" in result
            data = result["event_workspace"]
            assert data["total_counts"] == n_events
            assert data["n_spectra"] == n_spectra
            assert len(data["records"]) == n_events
            
            # Check first record structure
            record = data["records"][0]
            assert "bank" in record
            assert "event_idx" in record
            assert "event_id" in record
            assert "time_offset" in record
            assert "event_weight" in record
            assert record["pulse_index"] is None  # No pulse info in Mantid .lite
            assert record["pulse_time"] is None

    def test_respects_max_events(self, tmp_path):
        filepath = tmp_path / "test.h5"
        n_events = 1000
        
        with h5py.File(filepath, "w") as f:
            ws = f.create_group("mantid_workspace_1")
            event_ws = ws.create_group("event_workspace")
            event_ws.create_dataset("tof", data=np.random.uniform(1000, 50000, n_events))
            event_ws.create_dataset("indices", data=np.array([0, n_events]))
        
        with h5py.File(filepath, "r") as f:
            result = mantid.extract_mantid_events(f, "mantid_workspace_1", max_events=100)
            
            assert len(result["event_workspace"]["records"]) == 100

    def test_returns_empty_if_no_event_workspace(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("mantid_workspace_1")
        
        with h5py.File(filepath, "r") as f:
            result = mantid.extract_mantid_events(f, "mantid_workspace_1")
            assert result == {}


class TestGetMantidInstrumentId:
    """Tests for the get_mantid_instrument_id function."""

    def test_gets_from_instrument_name(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            ws = f.create_group("mantid_workspace_1")
            instrument = ws.create_group("instrument")
            instrument.create_dataset("name", data=b"SNAP")
        
        with h5py.File(filepath, "r") as f:
            assert mantid.get_mantid_instrument_id(f, "mantid_workspace_1") == "SNAP"

    def test_gets_from_workspace_name(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            ws = f.create_group("mantid_workspace_1")
            ws.create_dataset("workspace_name", data=b"VULCAN_12345")
        
        with h5py.File(filepath, "r") as f:
            assert mantid.get_mantid_instrument_id(f, "mantid_workspace_1") == "VULCAN"

    def test_returns_unknown_if_not_found(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("mantid_workspace_1")
        
        with h5py.File(filepath, "r") as f:
            assert mantid.get_mantid_instrument_id(f, "mantid_workspace_1") == "UNKNOWN"


class TestIntegration:
    """Integration tests for the full Mantid conversion pipeline."""

    def test_full_conversion_pipeline(self, tmp_path):
        """Test that process_mantid_file produces expected outputs."""
        from nexus_processor.parquet import process_nexus_file
        
        # Create a mock Mantid file
        input_file = tmp_path / "SNAP_test.lite.nxs.h5"
        output_dir = tmp_path / "output"
        
        n_events = 1000
        n_spectra = 100
        
        with h5py.File(input_file, "w") as f:
            ws = f.create_group("mantid_workspace_1")
            ws.create_dataset("title", data=b"Test SNAP Run")
            ws.create_dataset("workspace_name", data=b"SNAP_12345")
            
            # Instrument
            instrument = ws.create_group("instrument")
            instrument.create_dataset("name", data=b"SNAP")
            
            # Logs
            logs = ws.create_group("logs")
            run_num = logs.create_group("run_number")
            run_num.create_dataset("time", data=np.array([0.0]))
            run_num.create_dataset("value", data=np.array([12345]))
            
            temp = logs.create_group("temperature")
            temp.create_dataset("time", data=np.array([0.0, 1.0]))
            temp.create_dataset("value", data=np.array([300.0, 301.0]))
            
            # Sample
            sample = ws.create_group("sample")
            sample.create_dataset("geom_height", data=np.array([1.0]))
            
            # Events
            event_ws = ws.create_group("event_workspace")
            event_ws.create_dataset("tof", data=np.random.uniform(1000, 50000, n_events))
            indices = np.linspace(0, n_events, n_spectra + 1, dtype=np.int64)
            event_ws.create_dataset("indices", data=indices)
            event_ws.create_dataset("weight", data=np.ones(n_events))
        
        # Run conversion
        output_files = process_nexus_file(
            str(input_file),
            str(output_dir),
            include_events=True,
            max_events=500,  # Limit for faster test
        )
        
        # Verify outputs
        assert "metadata" in output_files
        assert "daslogs" in output_files
        assert "event_summary" in output_files
        
        # Check metadata file
        import pyarrow.parquet as pq
        metadata_table = pq.read_table(output_files["metadata"])
        assert metadata_table.num_rows == 1
        assert metadata_table["instrument_id"][0].as_py() == "SNAP"
        assert metadata_table["run_number"][0].as_py() == 12345
        
        # Check daslogs file
        daslogs_table = pq.read_table(output_files["daslogs"])
        assert daslogs_table.num_rows >= 2  # At least temperature logs
        
        # Check event summary
        summary_table = pq.read_table(output_files["event_summary"])
        assert summary_table.num_rows == 1
        assert summary_table["bank"][0].as_py() == "event_workspace"
