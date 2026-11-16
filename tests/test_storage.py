"""Tests for saving the dataset, rebuilding the config, and checking the committed data against the code."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from aiops.simulator import (
    BaseSignalConfig,
    MetricConfig,
    SpikeConfig,
    config_from_dict,
    config_to_dict,
    generate_dataset,
    save_dataset,
)

SEED = 42
COMMITTED_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"	# the dataset in the repo


@pytest.fixture
def saved(tmp_path):
    """A small dataset saved into a fresh temporary folder."""
    dataset = generate_dataset(MetricConfig(), 3, SEED)
    save_dataset(dataset, MetricConfig(), SEED, tmp_path)
    return dataset, tmp_path


def _files_under(root: Path) -> list[Path]:
    return sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())


# Verifies that a config comes back equal after a trip through JSON, including non-default values and dates.
@pytest.mark.parametrize(
    "config",
    [
        MetricConfig(),
        MetricConfig(base=BaseSignalConfig(baseline=30.0, start=datetime(2026, 3, 2, 6, 30)), spike=SpikeConfig((12.0, 30.0), count=5)),
    ],
)
def test_config_survives_a_trip_through_json(config):
    text = json.dumps(config_to_dict(config), default=str)
    assert config_from_dict(json.loads(text)) == config


# Verifies that rebuilding a config runs its validation, so a bad value in saved metadata fails loudly.
def test_config_from_dict_runs_validation():
    data = json.loads(json.dumps(config_to_dict(MetricConfig()), default=str))
    data["base"]["noise_std"] = -1.0
    with pytest.raises(ValueError):
        config_from_dict(data)


# Verifies the folder layout: one CSV per series, plus events.json and metadata.json.
def test_saved_files_have_the_expected_layout(saved):
    _, out_dir = saved
    assert _files_under(out_dir) == [
        Path("events.json"),
        Path("metadata.json"),
        Path("series/series_00.csv"),
        Path("series/series_01.csv"),
        Path("series/series_02.csv"),
    ]


# Verifies that each series CSV holds only timestamps and observed values, in NAB's format.
def test_series_csvs_hold_only_what_detectors_may_see(saved):
    dataset, out_dir = saved
    for series in dataset:
        lines = (out_dir / "series" / f"series_{series.index:02d}.csv").read_text().splitlines()
        assert lines[0] == "timestamp,value"	# no label or expected-value columns for a detector to leak from
        rows = [line.split(",") for line in lines[1:]]
        assert len(rows) == 4032
        assert rows[0][0] == "2026-01-05 00:00:00" and rows[-1][0] == "2026-01-18 23:55:00"
        assert [value for _, value in rows] == [f"{value:.4f}" for value in series.components.observed]


# Verifies that events.json records every event with its window, times, and parameters.
def test_events_json_matches_the_events(saved):
    dataset, out_dir = saved
    saved_events = json.loads((out_dir / "events.json").read_text())
    for series in dataset:
        records = saved_events[f"series_{series.index:02d}"]
        assert [(r["kind"], r["start"], r["end"], r["params"]) for r in records] == [
            (event.kind, event.start, event.end, event.params) for event in series.events
        ]
        for record, event in zip(records, series.events):
            first = np.datetime_as_string(series.components.timestamps[event.start], unit="s").replace("T", " ")
            assert record["first_time"] == first


# Verifies that metadata.json records the seed, the number of series, the NumPy version, and a config that rebuilds exactly.
def test_metadata_records_what_produced_the_data(saved):
    _, out_dir = saved
    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["seed"] == SEED
    assert metadata["n_series"] == 3
    assert metadata["numpy_version"] == np.__version__
    assert config_from_dict(metadata["config"]) == MetricConfig()


# Verifies that every saved file uses "\n" line endings, so files match byte for byte on every OS.
def test_saved_files_use_unix_line_endings(saved):
    _, out_dir = saved
    for path in _files_under(out_dir):
        assert b"\r" not in (out_dir / path).read_bytes()


# Verifies that the committed dataset is exactly what the code produces today from the committed metadata.
def test_committed_dataset_matches_a_fresh_regeneration(tmp_path):
    metadata = json.loads((COMMITTED_DIR / "metadata.json").read_text())
    config = config_from_dict(metadata["config"])
    dataset = generate_dataset(config, metadata["n_series"], metadata["seed"])
    save_dataset(dataset, config, metadata["seed"], tmp_path)

    stale = "data/synthetic is out of date: rerun scripts/generate_dataset.py and commit the result"
    assert _files_under(tmp_path) == _files_under(COMMITTED_DIR), stale
    for path in _files_under(COMMITTED_DIR):
        assert (tmp_path / path).read_bytes() == (COMMITTED_DIR / path).read_bytes(), f"{path} differs. {stale}"