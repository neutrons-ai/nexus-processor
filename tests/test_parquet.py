"""
Tests for the nexus_processor.parquet module.

These tests use mock HDF5 files created with h5py to test the extraction
functions without requiring actual NeXus files.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pandas as pd
import pytest

from nexus_processor import parquet


class TestSafeDecode:
    """Tests for the safe_decode utility function."""

    def test_none_returns_none(self):
        assert parquet.safe_decode(None) is None

    def test_bytes_decoded_to_string(self):
        assert parquet.safe_decode(b"hello") == "hello"

    def test_bytes_with_invalid_utf8_uses_replacement(self):
        # Invalid UTF-8 sequence
        result = parquet.safe_decode(b"\xff\xfe")
        assert isinstance(result, str)

    def test_numpy_empty_array_returns_none(self):
        arr = np.array([])
        assert parquet.safe_decode(arr) is None

    def test_numpy_single_element_returns_scalar(self):
        arr = np.array([42])
        assert parquet.safe_decode(arr) == 42

    def test_numpy_multiple_elements_returns_list(self):
        arr = np.array([1, 2, 3])
        assert parquet.safe_decode(arr) == [1, 2, 3]

    def test_numpy_string_array_decoded(self):
        arr = np.array([b"hello"])
        assert parquet.safe_decode(arr) == "hello"

    def test_numpy_integer_converted(self):
        val = np.int64(42)
        result = parquet.safe_decode(val)
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float_converted(self):
        val = np.float64(3.14)
        result = parquet.safe_decode(val)
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_list_elements_decoded(self):
        result = parquet.safe_decode([b"a", b"b"])
        assert result == ["a", "b"]

    def test_tuple_elements_decoded(self):
        result = parquet.safe_decode((b"x", b"y"))
        assert result == ["x", "y"]

    def test_plain_string_unchanged(self):
        assert parquet.safe_decode("already string") == "already string"

    def test_plain_int_unchanged(self):
        assert parquet.safe_decode(42) == 42

    def test_2d_array_flattened(self):
        arr = np.array([[1, 2], [3, 4]])
        result = parquet.safe_decode(arr)
        assert result == [1, 2, 3, 4]


class TestReadDatasetValue:
    """Tests for the read_dataset_value function."""

    def test_reads_scalar_dataset(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_dataset("scalar", data=42)

        with h5py.File(filepath, "r") as f:
            result = parquet.read_dataset_value(f["scalar"])
            assert result == 42

    def test_reads_string_dataset(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_dataset("text", data=b"hello world")

        with h5py.File(filepath, "r") as f:
            result = parquet.read_dataset_value(f["text"])
            assert result == "hello world"

    def test_reads_array_dataset(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_dataset("array", data=[1, 2, 3])

        with h5py.File(filepath, "r") as f:
            result = parquet.read_dataset_value(f["array"])
            assert result == [1, 2, 3]

    def test_handles_read_error_gracefully(self, capsys):
        """Test that read errors are handled gracefully."""
        mock_dataset = MagicMock()
        mock_dataset.name = "/broken"
        mock_dataset.__getitem__ = MagicMock(side_effect=OSError("Read error"))

        result = parquet.read_dataset_value(mock_dataset)
        assert result is None
        captured = capsys.readouterr()
        assert "Warning" in captured.out


class TestExtractEntryMetadata:
    """Tests for the extract_entry_metadata function."""

    def test_returns_empty_dict_when_entry_missing(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            pass  # Empty file

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_entry_metadata(f)
            assert result == {}

    def test_extracts_scalar_fields(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            entry = f.create_group("entry")
            entry.create_dataset("title", data=b"Test Experiment")
            entry.create_dataset("run_number", data=12345)
            entry.create_dataset("duration", data=3600.5)

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_entry_metadata(f)
            assert result["title"] == "Test Experiment"
            assert result["run_number"] == 12345
            assert result["duration"] == pytest.approx(3600.5)

    def test_extracts_file_level_attributes(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.attrs["file_name"] = b"test.nxs.h5"
            f.create_group("entry")

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_entry_metadata(f)
            assert result["file_attr_file_name"] == "test.nxs.h5"

    def test_extracts_entry_level_attributes(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            entry = f.create_group("entry")
            entry.attrs["NX_class"] = b"NXentry"

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_entry_metadata(f)
            assert result["entry_attr_NX_class"] == "NXentry"

    def test_custom_entry_name(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            entry = f.create_group("custom_entry")
            entry.create_dataset("title", data=b"Custom")

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_entry_metadata(f, entry_name="custom_entry")
            assert result["title"] == "Custom"


class TestExtractSampleInfo:
    """Tests for the extract_sample_info function."""

    def test_returns_empty_dict_when_sample_missing(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("entry")

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_sample_info(f)
            assert result == {}

    def test_extracts_sample_datasets(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            sample = f.create_group("entry/sample")
            sample.create_dataset("name", data=b"Silicon wafer")
            sample.create_dataset("temperature", data=300.0)

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_sample_info(f)
            assert result["name"] == "Silicon wafer"
            assert result["temperature"] == pytest.approx(300.0)


class TestExtractInstrumentInfo:
    """Tests for the extract_instrument_info function."""

    def test_returns_empty_dict_when_instrument_missing(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("entry")

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_instrument_info(f)
            assert result == {}

    def test_extracts_instrument_datasets(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            instrument = f.create_group("entry/instrument")
            instrument.create_dataset("name", data=b"REF_L")
            instrument.create_dataset("beamline", data=b"BL-4A")

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_instrument_info(f)
            assert result["name"] == "REF_L"
            assert result["beamline"] == "BL-4A"

    def test_extracts_nested_instrument_xml(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            instrument = f.create_group("entry/instrument")
            xml_group = instrument.create_group("instrument_xml")
            xml_group.create_dataset("data", data=b"<instrument/>")

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_instrument_info(f)
            assert result["instrument_xml_data"] == "<instrument/>"


class TestExtractUsers:
    """Tests for the extract_users function."""

    def test_returns_empty_list_when_entry_missing(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            pass

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_users(f)
            assert result == []

    def test_returns_empty_list_when_no_users(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("entry")

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_users(f)
            assert result == []

    def test_extracts_single_user(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            entry = f.create_group("entry")
            user1 = entry.create_group("user1")
            user1.create_dataset("name", data=b"John Doe")
            user1.create_dataset("facility_user_id", data=b"jdoe")

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_users(f)
            assert len(result) == 1
            assert result[0]["user_id"] == "user1"
            assert result[0]["name"] == "John Doe"
            assert result[0]["facility_user_id"] == "jdoe"

    def test_extracts_multiple_users_sorted(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            entry = f.create_group("entry")
            user2 = entry.create_group("user2")
            user2.create_dataset("name", data=b"Jane Smith")
            user1 = entry.create_group("user1")
            user1.create_dataset("name", data=b"John Doe")

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_users(f)
            assert len(result) == 2
            # Should be sorted: user1 before user2
            assert result[0]["user_id"] == "user1"
            assert result[1]["user_id"] == "user2"


class TestExtractDaslogs:
    """Tests for the extract_daslogs function."""

    def test_returns_empty_list_when_daslogs_missing(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("entry")

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_daslogs(f)
            assert result == []

    def test_extracts_simple_log_with_timeseries(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            daslogs = f.create_group("entry/DASlogs")
            temp_log = daslogs.create_group("temperature")
            temp_log.create_dataset("time", data=[0.0, 1.0, 2.0])
            temp_log.create_dataset("value", data=[300.0, 301.0, 302.0])
            temp_log.create_dataset("device_name", data=b"sample_temp")

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_daslogs(f)
            assert len(result) == 3
            assert result[0]["log_name"] == "temperature"
            assert result[0]["device_name"] == "sample_temp"
            assert result[0]["time"] == 0.0
            assert result[0]["value"] == pytest.approx(300.0)
            assert result[2]["value"] == pytest.approx(302.0)

    def test_extracts_log_with_statistics(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            daslogs = f.create_group("entry/DASlogs")
            temp_log = daslogs.create_group("temperature")
            temp_log.create_dataset("time", data=[0.0])
            temp_log.create_dataset("value", data=[300.0])
            temp_log.create_dataset("average_value", data=300.5)
            temp_log.create_dataset("minimum_value", data=299.0)
            temp_log.create_dataset("maximum_value", data=302.0)

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_daslogs(f)
            assert len(result) == 1
            assert result[0]["average_value"] == pytest.approx(300.5)
            assert result[0]["min_value"] == pytest.approx(299.0)
            assert result[0]["max_value"] == pytest.approx(302.0)

    def test_handles_log_without_timeseries(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            daslogs = f.create_group("entry/DASlogs")
            static_log = daslogs.create_group("static_param")
            static_log.create_dataset("average_value", data=42.0)

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_daslogs(f)
            assert len(result) == 1
            assert result[0]["log_name"] == "static_param"
            assert result[0]["time"] is None
            assert result[0]["value"] is None


class TestExtractEvents:
    """Tests for the extract_events function."""

    def test_returns_empty_dict_when_entry_missing(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            pass

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_events(f)
            assert result == {}

    def test_returns_empty_dict_when_no_event_banks(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("entry")

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_events(f)
            assert result == {}

    def test_extracts_event_bank(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            entry = f.create_group("entry")
            bank1 = entry.create_group("bank1_events")
            bank1.create_dataset("event_id", data=[100, 101, 102])
            bank1.create_dataset("event_time_offset", data=[0.1, 0.2, 0.3])
            bank1.create_dataset("event_index", data=[0, 2])
            bank1.create_dataset("total_counts", data=3)

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_events(f)
            assert "bank1_events" in result
            bank_data = result["bank1_events"]
            assert bank_data["total_counts"] == 3
            assert bank_data["n_pulses"] == 2
            assert len(bank_data["records"]) == 3
            assert bank_data["records"][0]["event_id"] == 100
            assert bank_data["records"][0]["time_offset"] == pytest.approx(0.1)

    def test_respects_max_events(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            entry = f.create_group("entry")
            bank1 = entry.create_group("bank1_events")
            bank1.create_dataset("event_id", data=[100, 101, 102, 103, 104])
            bank1.create_dataset("event_time_offset", data=[0.1, 0.2, 0.3, 0.4, 0.5])

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_events(f, max_events=2)
            assert len(result["bank1_events"]["records"]) == 2


class TestExtractSoftwareInfo:
    """Tests for the extract_software_info function."""

    def test_returns_empty_list_when_software_missing(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            f.create_group("entry")

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_software_info(f)
            assert result == []

    def test_extracts_software_components(self, tmp_path):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as f:
            software = f.create_group("entry/Software")
            mantid = software.create_group("mantid")
            mantid.create_dataset("version", data=b"6.5.0")
            mantid.create_dataset("name", data=b"Mantid")

        with h5py.File(filepath, "r") as f:
            result = parquet.extract_software_info(f)
            assert len(result) == 1
            assert result[0]["component"] == "mantid"
            assert result[0]["version"] == "6.5.0"
            assert result[0]["name"] == "Mantid"


class TestProcessNexusFile:
    """Integration tests for the process_nexus_file function."""

    @pytest.fixture
    def mock_nexus_file(self, tmp_path):
        """Create a minimal mock NeXus file for testing."""
        filepath = tmp_path / "test_data.nxs.h5"
        with h5py.File(filepath, "w") as f:
            f.attrs["file_name"] = b"test_data.nxs.h5"

            entry = f.create_group("entry")
            entry.attrs["NX_class"] = b"NXentry"
            entry.create_dataset("title", data=b"Test Run")
            entry.create_dataset("run_number", data=12345)
            entry.create_dataset("start_time", data=b"2025-01-15T10:00:00")

            # Sample
            sample = entry.create_group("sample")
            sample.create_dataset("name", data=b"Test Sample")

            # Instrument
            instrument = entry.create_group("instrument")
            instrument.create_dataset("name", data=b"REF_L")

            # User
            user1 = entry.create_group("user1")
            user1.create_dataset("name", data=b"Test User")

            # DASlogs
            daslogs = entry.create_group("DASlogs")
            temp = daslogs.create_group("temperature")
            temp.create_dataset("time", data=[0.0, 1.0])
            temp.create_dataset("value", data=[300.0, 301.0])

            # Events
            bank = entry.create_group("bank1_events")
            bank.create_dataset("event_id", data=[100, 101])
            bank.create_dataset("event_time_offset", data=[0.1, 0.2])
            bank.create_dataset("event_index", data=[0])
            bank.create_dataset("total_counts", data=2)

            # Software
            software = entry.create_group("Software")
            sw = software.create_group("test_sw")
            sw.create_dataset("version", data=b"1.0")

        return filepath

    def test_creates_output_directory(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=False,
        )
        assert output_dir.exists()

    def test_creates_metadata_parquet(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=False,
        )

        assert "metadata" in result
        df = pd.read_parquet(result["metadata"])
        assert df["title"].iloc[0] == "Test Run"
        assert df["run_number"].iloc[0] == 12345

    def test_creates_sample_parquet(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=False,
        )

        assert "sample" in result
        df = pd.read_parquet(result["sample"])
        assert df["name"].iloc[0] == "Test Sample"

    def test_creates_instrument_parquet(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=False,
        )

        assert "instrument" in result
        df = pd.read_parquet(result["instrument"])
        assert df["name"].iloc[0] == "REF_L"

    def test_creates_daslogs_parquet(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=False,
        )

        assert "daslogs" in result
        df = pd.read_parquet(result["daslogs"])
        assert len(df) == 2
        assert df["log_name"].iloc[0] == "temperature"

    def test_excludes_users_by_default_when_flag_false(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=False,
        )

        assert "users" not in result

    def test_includes_users_when_flag_true(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=True,
        )

        assert "users" in result
        df = pd.read_parquet(result["users"])
        assert df["name"].iloc[0] == "Test User"

    def test_excludes_events_by_default_when_flag_false(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=False,
        )

        # Should not have event bank files
        assert "bank1_events" not in result
        assert "event_summary" not in result

    def test_includes_events_when_flag_true(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=True,
            include_users=False,
        )

        assert "bank1_events" in result
        assert "event_summary" in result
        df = pd.read_parquet(result["bank1_events"])
        assert len(df) == 2

    def test_max_events_limits_output(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=True,
            include_users=False,
            max_events=1,
        )

        df = pd.read_parquet(result["bank1_events"])
        assert len(df) == 1

    def test_adds_ingestion_metadata(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=False,
        )

        df = pd.read_parquet(result["metadata"])
        assert "source_file" in df.columns
        assert "ingestion_time" in df.columns
        assert df["source_file"].iloc[0] == "test_data.nxs.h5"


class TestProcessNexusFileSingleFile:
    """Tests for the single_file option in process_nexus_file."""

    @pytest.fixture
    def mock_nexus_file(self, tmp_path):
        """Create a minimal mock NeXus file for testing."""
        filepath = tmp_path / "test_data.nxs.h5"
        with h5py.File(filepath, "w") as f:
            f.attrs["file_name"] = b"test_data.nxs.h5"

            entry = f.create_group("entry")
            entry.attrs["NX_class"] = b"NXentry"
            entry.create_dataset("title", data=b"Single File Test")
            entry.create_dataset("run_number", data=54321)

            # Sample
            sample = entry.create_group("sample")
            sample.create_dataset("name", data=b"Test Sample")
            sample.create_dataset("temperature", data=300.0)

            # Instrument
            instrument = entry.create_group("instrument")
            instrument.create_dataset("name", data=b"REF_L")

            # User
            user1 = entry.create_group("user1")
            user1.create_dataset("name", data=b"Jane Doe")

            # DASlogs
            daslogs = entry.create_group("DASlogs")
            temp = daslogs.create_group("temperature")
            temp.create_dataset("time", data=[0.0, 1.0, 2.0])
            temp.create_dataset("value", data=[300.0, 301.0, 302.0])

            # Software
            software = entry.create_group("Software")
            sw = software.create_group("test_sw")
            sw.create_dataset("version", data=b"2.0")

        return filepath

    def test_single_file_creates_combined_parquet(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=False,
            single_file=True,
        )

        assert "combined" in result
        assert len(result) == 1  # Only combined file
        
        # Verify run_number partition column is present
        df = pd.read_parquet(result["combined"])
        assert "run_number" in df.columns
        assert df["run_number"].iloc[0] == 54321

    def test_single_file_contains_daslogs(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=False,
            single_file=True,
        )

        df = pd.read_parquet(result["combined"])
        assert len(df) == 3  # 3 daslog records
        assert "log_name" in df.columns
        assert "time" in df.columns
        assert "value" in df.columns
        assert "value_numeric" in df.columns  # Iceberg-compatible numeric column
        assert "record_type" in df.columns
        assert df["record_type"].iloc[0] == "daslog"

    def test_single_file_contains_metadata_columns(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=False,
            single_file=True,
        )

        df = pd.read_parquet(result["combined"])
        # Metadata is now a nested struct column for Iceberg compatibility
        assert "metadata" in df.columns
        metadata = df["metadata"].iloc[0]
        assert metadata["title"] == "Single File Test"
        # run_number is a top-level partition column
        assert df["run_number"].iloc[0] == 54321

    def test_single_file_contains_sample_columns(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=False,
            single_file=True,
        )

        df = pd.read_parquet(result["combined"])
        # Sample is now a nested struct column for Iceberg compatibility
        assert "sample" in df.columns
        sample = df["sample"].iloc[0]
        assert sample["name"] == "Test Sample"
        assert sample["temperature"] == 300.0

    def test_single_file_contains_instrument_columns(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=False,
            single_file=True,
        )

        df = pd.read_parquet(result["combined"])
        # Instrument is now a nested struct column for Iceberg compatibility
        assert "instrument" in df.columns
        instrument = df["instrument"].iloc[0]
        assert instrument["name"] == "REF_L"

    def test_single_file_contains_software_columns(self, mock_nexus_file, tmp_path):
        """Software is not included in combined file (not in schema).
        
        Software metadata is available in split mode via separate software.parquet.
        The combined file focuses on time-series data (daslogs/events) with
        run-level metadata in nested structs.
        """
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=False,
            single_file=True,
        )

        df = pd.read_parquet(result["combined"])
        # Software is not in the combined schema (use split mode for full software info)
        # Just verify the file was created successfully
        assert len(df) == 3  # 3 daslog records

    def test_single_file_includes_users_when_requested(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=True,
            single_file=True,
        )

        df = pd.read_parquet(result["combined"])
        # Users is now a list of structs for Iceberg compatibility
        assert "users" in df.columns
        users = df["users"].iloc[0]
        assert len(users) == 1
        assert users[0]["name"] == "Jane Doe"

    def test_single_file_includes_events(self, mock_nexus_file, tmp_path):
        # Add events to the file
        with h5py.File(mock_nexus_file, "a") as f:
            bank = f["entry"].create_group("bank1_events")
            bank.create_dataset("event_id", data=[100, 101])
            bank.create_dataset("event_time_offset", data=[0.1, 0.2])
            bank.create_dataset("event_index", data=[0])
            bank.create_dataset("total_counts", data=2)

        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=True,
            include_users=False,
            single_file=True,
        )

        # Should only have combined file (no separate event files)
        assert "combined" in result
        assert len(result) == 1
        assert "bank1_events" not in result

        # Combined file should contain both daslogs and events
        df = pd.read_parquet(result["combined"])
        assert "record_type" in df.columns
        assert set(df["record_type"].unique()) == {"daslog", "event"}
        
        # Check daslog records
        daslogs = df[df["record_type"] == "daslog"]
        assert len(daslogs) == 3  # 3 temperature readings
        
        # Check event records
        events = df[df["record_type"] == "event"]
        assert len(events) == 2
        assert "event_id" in df.columns
        assert "bank" in df.columns

    def test_single_file_events_have_metadata(self, mock_nexus_file, tmp_path):
        # Add events to the file
        with h5py.File(mock_nexus_file, "a") as f:
            bank = f["entry"].create_group("bank1_events")
            bank.create_dataset("event_id", data=[100])
            bank.create_dataset("event_time_offset", data=[0.1])
            bank.create_dataset("event_index", data=[0])
            bank.create_dataset("total_counts", data=1)

        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=True,
            include_users=True,
            single_file=True,
        )

        df = pd.read_parquet(result["combined"])
        events = df[df["record_type"] == "event"]
        
        # Event rows should have metadata as nested structs
        metadata = events["metadata"].iloc[0]
        assert metadata["title"] == "Single File Test"
        
        sample = events["sample"].iloc[0]
        assert sample["name"] == "Test Sample"
        
        users = events["users"].iloc[0]
        assert users[0]["name"] == "Jane Doe"

    def test_split_files_separates_events(self, mock_nexus_file, tmp_path):
        # Add events to the file
        with h5py.File(mock_nexus_file, "a") as f:
            bank = f["entry"].create_group("bank1_events")
            bank.create_dataset("event_id", data=[100, 101])
            bank.create_dataset("event_time_offset", data=[0.1, 0.2])
            bank.create_dataset("event_index", data=[0])
            bank.create_dataset("total_counts", data=2)

        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=True,
            include_users=False,
            single_file=False,
        )

        # Should have separate event files
        assert "bank1_events" in result
        assert "event_summary" in result
        assert "metadata" in result

    def test_split_files_is_default(self, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = parquet.process_nexus_file(
            str(mock_nexus_file),
            str(output_dir),
            include_events=False,
            include_users=False,
            # single_file defaults to False
        )

        # Should have separate files
        assert "metadata" in result
        assert "sample" in result
        assert "daslogs" in result
        assert "combined" not in result
