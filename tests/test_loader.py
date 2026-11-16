"""Tests for loading the saved dataset back from disk, and for the checks that keep the data and its answer key in line."""

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from aiops.data import LabeledSeries, MetricSeries, load_synthetic_dataset, read_series_csv
from aiops.simulator import AnomalyEvent, AnomalyKind, MetricConfig, generate_dataset, save_dataset

SEED = 42
COMMITTED_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"	# the dataset in the repo
GOOD_CSV = "timestamp,value\n2026-01-05 00:00:00,23.2646\n2026-01-05 00:05:00,24.0555\n"	# the first two rows of series_00


@pytest.fixture
def saved(tmp_path):
    """A small dataset saved into a fresh temporary folder, which a test is free to corrupt."""
    dataset = generate_dataset(MetricConfig(), 3, SEED)
    save_dataset(dataset, MetricConfig(), SEED, tmp_path)
    return dataset, tmp_path


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="")	# newline="" writes the text's own line endings, so a test can choose them
    return path


def _edit_events(out_dir: Path, change) -> None:
    """Load events.json, let change edit the parsed records, and write them back."""
    path = out_dir / "events.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    change(records)
    path.write_text(json.dumps(records), encoding="utf-8", newline="\n")


# Verifies that a loaded dataset matches the one that was saved: same timestamps, the saved values, the same events and labels.
def test_loaded_dataset_matches_what_was_saved(saved):
    dataset, out_dir = saved
    loaded = load_synthetic_dataset(out_dir)
    assert len(loaded) == len(dataset)
    for item, series in zip(loaded, dataset):
        np.testing.assert_array_equal(item.series.timestamps, series.components.timestamps)	# compares across units: seconds vs minutes
        saved_text = [f"{value:.4f}" for value in series.components.observed]
        assert [f"{value:.4f}" for value in item.series.values] == saved_text	# the file's exact text, so nothing beyond its rounding was lost
        assert item.events == series.events
        np.testing.assert_array_equal(item.labels, series.labels)


# Verifies that series come back in index order, each named after its file.
def test_series_come_back_in_order_named_after_their_files(saved):
    _, out_dir = saved
    assert [item.series.name for item in load_synthetic_dataset(out_dir)] == ["series_00", "series_01", "series_02"]


# Verifies that the committed dataset, the one detectors will read, loads every series and passes every check.
def test_committed_dataset_loads():
    metadata = json.loads((COMMITTED_DIR / "metadata.json").read_text(encoding="utf-8"))
    assert len(load_synthetic_dataset(COMMITTED_DIR)) == metadata["n_series"]


# Verifies that the type detectors receive has no field that could carry the answer key.
def test_metric_series_holds_only_what_a_monitor_records():
    assert [item.name for item in fields(MetricSeries)] == ["name", "timestamps", "values"]


# Verifies that a loaded series can't be edited in place, so one detector can't change the data the next one sees.
def test_loaded_arrays_are_read_only(saved):
    _, out_dir = saved
    series = load_synthetic_dataset(out_dir)[0].series
    for array in (series.timestamps, series.values):
        with pytest.raises(ValueError):
            array[0] = array[1]


# Verifies that harmless differences in how a CSV was saved still give the same series.
@pytest.mark.parametrize(
    "text",
    [
        GOOD_CSV,
        "\ufeff" + GOOD_CSV,	# a byte-order mark, like from a file re-saved in Excel
        GOOD_CSV.replace("\n", "\r\n"),	# Windows line endings
        GOOD_CSV + "\n\n",	# blank lines at the end
    ],
)
def test_harmless_csv_differences_still_load(tmp_path, text):
    series = read_series_csv(_write_csv(tmp_path / "series_99.csv", text))
    assert series.name == "series_99"
    assert series.values.tolist() == [23.2646, 24.0555]
    assert series.timestamps[1] - series.timestamps[0] == np.timedelta64(5, "m")


# Verifies that each kind of malformed CSV is rejected, with an error that starts with the file's name.
@pytest.mark.parametrize(
    "text",
    [
        "",	# an empty file
        "time,cpu\n2026-01-05 00:00:00,23.2646\n",	# the wrong header
        "timestamp,value\n",	# a header but no data
        "timestamp,value\n2026-01-05 00:00:00,23.2646,1\n",	# an extra field
        "timestamp,value\n2026-01-05 00:00:00,high\n",	# a value that isn't a number
        "timestamp,value\nyesterday,23.2646\n",	# a timestamp that isn't a date
        "timestamp,value\n2026-01-05 00:00:00,nan\n",	# NaN parses as a number, but no detector can use it
        "timestamp,value\n2026-01-05 00:00:00,inf\n",	# so does infinity
        "timestamp,value\n,23.2646\n",	# a blank timestamp, which NumPy reads as NaT
        "timestamp,value\n2026-01-05 00:05:00,23.2646\n2026-01-05 00:00:00,24.0555\n",	# rows out of order
        "timestamp,value\n2026-01-05 00:00:00,23.2646\n2026-01-05 00:00:00,24.0555\n",	# a repeated timestamp
    ],
)
def test_malformed_csvs_are_rejected(tmp_path, text):
    with pytest.raises(ValueError, match="^bad"):	# "bad.csv: ..." for problems with the file, "bad: ..." for problems with a sample
        read_series_csv(_write_csv(tmp_path / "bad.csv", text))


# Verifies that a MetricSeries built in code gets the same checks as one read from a file.
@pytest.mark.parametrize(
    "timestamps, values",
    [
        (["2026-01-05 00:00"], [1.0, 2.0]),	# different lengths
        ([], []),	# no samples
        (["2026-01-05 00:00", "2026-01-05 00:05"], [1.0, np.nan]),	# a NaN value
        (["2026-01-05 00:05", "2026-01-05 00:00"], [1.0, 2.0]),	# out of order
    ],
)
def test_metric_series_checks_itself(timestamps, values):
    with pytest.raises(ValueError, match="^built_in_code"):
        MetricSeries("built_in_code", np.array(timestamps, dtype="datetime64[m]"), np.array(values, dtype=np.float64))


# Verifies that pairing a series with events that overlap or run past its end is rejected.
@pytest.mark.parametrize(
    "windows",
    [
        [(1, 3), (2, 4)],	# overlapping
        [(3, 5)],	# past the end of a 4-sample series
    ],
)
def test_labeled_series_rejects_events_that_dont_fit(windows):
    series = MetricSeries("tiny", np.arange(4).astype("datetime64[m]"), np.zeros(4))
    events = [AnomalyEvent(kind=AnomalyKind.LEVEL_SHIFT, start=start, end=end, params={}) for start, end in windows]
    with pytest.raises(ValueError):
        LabeledSeries(series=series, events=events)


# Verifies that an events.json that disagrees with its series is rejected as a ValueError naming the series.
@pytest.mark.parametrize(
    "change",
    [
        lambda records: records["series_00"][0].update(first_time="2026-01-01 00:00:00"),	# the answer key's start time is out of line with the data
        lambda records: records["series_00"][0].update(last_time="2026-01-01 00:00:00"),	# so is its end time
        lambda records: records["series_00"][0].update(kind="dip"),	# a kind the simulator doesn't know
        lambda records: records["series_00"][-1].update(end=4033),	# past the end: a ValueError, not an IndexError from the time check
    ],
)
def test_events_that_disagree_with_their_series_are_rejected(saved, change):
    _, out_dir = saved
    _edit_events(out_dir, change)
    with pytest.raises(ValueError, match=r"^events\.json, series_00"):
        load_synthetic_dataset(out_dir)


# Verifies that a CSV with no events, or events with no CSV, is rejected instead of silently skipped.
@pytest.mark.parametrize(
    "remove",
    [
        lambda out_dir: (out_dir / "series" / "series_02.csv").unlink(),	# events with no CSV
        lambda out_dir: _edit_events(out_dir, lambda records: records.pop("series_02")),	# a CSV with no events
    ],
)
def test_series_and_events_must_cover_the_same_series(saved, remove):
    _, out_dir = saved
    remove(out_dir)
    with pytest.raises(ValueError, match="series_02"):
        load_synthetic_dataset(out_dir)


# Verifies that a folder with no series CSVs is rejected instead of loading as an empty dataset.
def test_folder_without_series_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no series CSVs"):
        load_synthetic_dataset(tmp_path)


# Verifies that the sampling interval is read from the timestamps.
def test_interval_is_read_from_the_timestamps(saved):
    _, out_dir = saved
    assert load_synthetic_dataset(out_dir)[0].series.interval_minutes == 5.0


# Verifies that a series with uneven gaps, or a single sample, has no interval.
@pytest.mark.parametrize(
    "timestamps",
    [
        ["2026-01-05 00:00", "2026-01-05 00:05", "2026-01-05 00:15"],	# a 5-minute gap, then a 10-minute one
        ["2026-01-05 00:00"],	# one sample, so no gap at all
    ],
)
def test_interval_needs_evenly_spaced_samples(timestamps):
    series = MetricSeries("gappy", np.array(timestamps, dtype="datetime64[m]"), np.zeros(len(timestamps)))
    with pytest.raises(ValueError, match="^gappy"):
        series.interval_minutes	# reading the property is what raises