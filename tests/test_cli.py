"""
Tests for the nexus_processor.cli module.
"""

import os
import tempfile
from pathlib import Path

import h5py
import pytest
from click.testing import CliRunner

from nexus_processor.cli import main


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_nexus_file(tmp_path):
    """Create a minimal mock NeXus file for CLI testing."""
    filepath = tmp_path / "test_data.nxs.h5"
    with h5py.File(filepath, "w") as f:
        f.attrs["file_name"] = b"test_data.nxs.h5"

        entry = f.create_group("entry")
        entry.attrs["NX_class"] = b"NXentry"
        entry.create_dataset("title", data=b"CLI Test Run")
        entry.create_dataset("run_number", data=99999)

        # Sample
        sample = entry.create_group("sample")
        sample.create_dataset("name", data=b"CLI Test Sample")

        # Instrument
        instrument = entry.create_group("instrument")
        instrument.create_dataset("name", data=b"TEST_INST")

        # User
        user1 = entry.create_group("user1")
        user1.create_dataset("name", data=b"CLI Test User")

        # DASlogs
        daslogs = entry.create_group("DASlogs")
        temp = daslogs.create_group("temperature")
        temp.create_dataset("time", data=[0.0])
        temp.create_dataset("value", data=[300.0])

        # Events
        bank = entry.create_group("bank1_events")
        bank.create_dataset("event_id", data=[100, 101, 102])
        bank.create_dataset("event_time_offset", data=[0.1, 0.2, 0.3])
        bank.create_dataset("event_index", data=[0])
        bank.create_dataset("total_counts", data=3)

    return filepath


class TestCliBasicUsage:
    """Test basic CLI usage and options."""

    def test_help_option(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Convert NeXus HDF5 files to Parquet format" in result.output
        assert "--include-events" in result.output
        assert "--include-users" in result.output

    def test_missing_input_file(self, runner):
        result = runner.invoke(main, [])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_nonexistent_input_file(self, runner):
        result = runner.invoke(main, ["/nonexistent/file.h5"])
        assert result.exit_code != 0

    def test_basic_conversion(self, runner, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [str(mock_nexus_file), "--output-dir", str(output_dir)],
        )

        assert result.exit_code == 0
        assert "Processing complete!" in result.output
        assert output_dir.exists()

    def test_default_output_directory(self, runner, mock_nexus_file):
        result = runner.invoke(main, [str(mock_nexus_file)])

        assert result.exit_code == 0
        # Default output should be parquet_output next to input file
        expected_dir = mock_nexus_file.parent / "parquet_output"
        assert expected_dir.exists()


class TestCliEventOptions:
    """Test CLI event-related options."""

    def test_events_excluded_by_default(self, runner, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [str(mock_nexus_file), "--output-dir", str(output_dir)],
        )

        assert result.exit_code == 0
        # Check that no event files were created
        output_files = list(output_dir.glob("*_events.parquet"))
        assert len(output_files) == 0
        event_summary = list(output_dir.glob("*_event_summary.parquet"))
        assert len(event_summary) == 0

    def test_include_events_flag(self, runner, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [str(mock_nexus_file), "--output-dir", str(output_dir), "--include-events"],
        )

        assert result.exit_code == 0
        # Check that event files were created
        output_files = list(output_dir.glob("*bank1_events.parquet"))
        assert len(output_files) == 1

    def test_no_events_explicit_flag(self, runner, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [str(mock_nexus_file), "--output-dir", str(output_dir), "--no-events"],
        )

        assert result.exit_code == 0
        output_files = list(output_dir.glob("*_events.parquet"))
        assert len(output_files) == 0

    def test_max_events_option(self, runner, mock_nexus_file, tmp_path):
        import pandas as pd

        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [
                str(mock_nexus_file),
                "--output-dir",
                str(output_dir),
                "--include-events",
                "--max-events",
                "2",
            ],
        )

        assert result.exit_code == 0
        event_files = list(output_dir.glob("*bank1_events.parquet"))
        assert len(event_files) == 1
        df = pd.read_parquet(event_files[0])
        assert len(df) == 2  # Limited to 2 events


class TestCliUserOptions:
    """Test CLI user-related options."""

    def test_users_excluded_by_default(self, runner, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [str(mock_nexus_file), "--output-dir", str(output_dir)],
        )

        assert result.exit_code == 0
        # Check that no users file was created
        users_files = list(output_dir.glob("*_users.parquet"))
        assert len(users_files) == 0

    def test_include_users_flag(self, runner, mock_nexus_file, tmp_path):
        import pandas as pd

        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [str(mock_nexus_file), "--output-dir", str(output_dir), "--include-users"],
        )

        assert result.exit_code == 0
        users_files = list(output_dir.glob("*_users.parquet"))
        assert len(users_files) == 1
        df = pd.read_parquet(users_files[0])
        assert df["name"].iloc[0] == "CLI Test User"

    def test_no_users_explicit_flag(self, runner, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [str(mock_nexus_file), "--output-dir", str(output_dir), "--no-users"],
        )

        assert result.exit_code == 0
        users_files = list(output_dir.glob("*_users.parquet"))
        assert len(users_files) == 0


class TestCliOutputOptions:
    """Test CLI output-related options."""

    def test_short_output_dir_option(self, runner, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "short_output"
        result = runner.invoke(
            main,
            [str(mock_nexus_file), "-o", str(output_dir)],
        )

        assert result.exit_code == 0
        assert output_dir.exists()

    def test_short_max_events_option(self, runner, mock_nexus_file, tmp_path):
        import pandas as pd

        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [
                str(mock_nexus_file),
                "-o",
                str(output_dir),
                "--include-events",
                "-m",
                "1",
            ],
        )

        assert result.exit_code == 0
        event_files = list(output_dir.glob("*bank1_events.parquet"))
        df = pd.read_parquet(event_files[0])
        assert len(df) == 1


class TestCliOutputMessages:
    """Test CLI output messages."""

    def test_displays_file_count(self, runner, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [str(mock_nexus_file), "--output-dir", str(output_dir)],
        )

        assert result.exit_code == 0
        assert "Files created:" in result.output

    def test_displays_output_directory(self, runner, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [str(mock_nexus_file), "--output-dir", str(output_dir)],
        )

        assert result.exit_code == 0
        assert "Output directory:" in result.output
        assert str(output_dir) in result.output

    def test_displays_file_sizes(self, runner, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [str(mock_nexus_file), "--output-dir", str(output_dir)],
        )

        assert result.exit_code == 0
        # Should show KB or MB for file sizes
        assert "KB" in result.output or "MB" in result.output


class TestCliCombinedOptions:
    """Test CLI with multiple options combined."""

    def test_include_both_events_and_users(self, runner, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [
                str(mock_nexus_file),
                "--output-dir",
                str(output_dir),
                "--include-events",
                "--include-users",
            ],
        )

        assert result.exit_code == 0
        event_files = list(output_dir.glob("*bank1_events.parquet"))
        users_files = list(output_dir.glob("*_users.parquet"))
        assert len(event_files) == 1
        assert len(users_files) == 1

    def test_include_events_with_max_limit(self, runner, mock_nexus_file, tmp_path):
        import pandas as pd

        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [
                str(mock_nexus_file),
                "--output-dir",
                str(output_dir),
                "--include-events",
                "--max-events",
                "1",
            ],
        )

        assert result.exit_code == 0
        event_files = list(output_dir.glob("*bank1_events.parquet"))
        df = pd.read_parquet(event_files[0])
        assert len(df) == 1


class TestCliSingleFileOption:
    """Test CLI single-file option."""

    def test_single_file_flag_creates_combined(self, runner, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [str(mock_nexus_file), "--output-dir", str(output_dir), "--single-file"],
        )

        assert result.exit_code == 0
        combined_files = list(output_dir.glob("*_combined.parquet"))
        assert len(combined_files) == 1
        # Should not have separate metadata/sample files
        metadata_files = list(output_dir.glob("*_metadata.parquet"))
        assert len(metadata_files) == 0

    def test_split_files_flag_creates_separate(self, runner, mock_nexus_file, tmp_path):
        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [str(mock_nexus_file), "--output-dir", str(output_dir), "--split-files"],
        )

        assert result.exit_code == 0
        # Should have separate files
        metadata_files = list(output_dir.glob("*_metadata.parquet"))
        assert len(metadata_files) == 1
        combined_files = list(output_dir.glob("*_combined.parquet"))
        assert len(combined_files) == 0

    def test_single_file_with_users(self, runner, mock_nexus_file, tmp_path):
        import pandas as pd

        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [
                str(mock_nexus_file),
                "--output-dir",
                str(output_dir),
                "--single-file",
                "--include-users",
            ],
        )

        assert result.exit_code == 0
        combined_files = list(output_dir.glob("*_combined.parquet"))
        df = pd.read_parquet(combined_files[0])
        assert "users" in df.columns

    def test_single_file_with_events(self, runner, mock_nexus_file, tmp_path):
        import pandas as pd

        output_dir = tmp_path / "output"
        result = runner.invoke(
            main,
            [
                str(mock_nexus_file),
                "--output-dir",
                str(output_dir),
                "--single-file",
                "--include-events",
            ],
        )

        assert result.exit_code == 0
        # Should only have combined file
        combined_files = list(output_dir.glob("*_combined.parquet"))
        assert len(combined_files) == 1
        event_files = list(output_dir.glob("*_events.parquet"))
        assert len(event_files) == 0

        df = pd.read_parquet(combined_files[0])
        assert "record_type" in df.columns

    def test_single_file_help_text(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--single-file" in result.output
        assert "--split-files" in result.output
