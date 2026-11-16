"""Loading saved datasets: what detectors see, kept apart from the ground truth used to score them."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aiops.simulator import AnomalyEvent, AnomalyKind, labels_from_events

CSV_HEADER = ["timestamp", "value"]	# the same two columns as NAB's files
ZERO_GAP = np.timedelta64(0, "s")	# a unit is required: NumPy is deprecating unitless timedeltas


@dataclass(frozen=True, eq=False)
class MetricSeries:
    """One metric as a monitoring system records it: timestamps and values, with no labels. Detectors only ever get this."""

    name: str	# the file name without .csv, like "series_00"
    timestamps: np.ndarray	# datetime64[s], strictly increasing
    values: np.ndarray	# float64, one per timestamp

    def __post_init__(self):
        if self.values.ndim != 1 or len(self.values) == 0 or self.timestamps.shape != self.values.shape:
            raise ValueError(f"{self.name}: timestamps and values must be non-empty 1-D arrays of the same length")
        problems = {	# each rule, and which samples break it
            "has no timestamp": np.isnat(self.timestamps),
            "has a value that isn't a finite number": ~np.isfinite(self.values),
            "isn't later than the sample before it": np.concatenate([[False], np.diff(self.timestamps) <= ZERO_GAP]),
        }
        for problem, bad in problems.items():
            if bad.any():
                raise ValueError(f"{self.name}: sample {np.argmax(bad)} {problem}")	# argmax finds the first True
        for array in (self.timestamps, self.values):
            array.flags.writeable = False	# every detector reads the same arrays, so no in-place edits


@dataclass(frozen=True, eq=False)
class LabeledSeries:
    """A series paired with its ground-truth events. Only scoring code reads the events."""

    series: MetricSeries
    events: list[AnomalyEvent]	# in time order

    def __post_init__(self):
        labels_from_events(self.events, len(self.series.values))	# run for its checks only: raises if events overlap or one runs past the series

    @property
    def labels(self) -> np.ndarray:
        """True for every sample inside an event's labeled window, computed fresh from the events."""
        return labels_from_events(self.events, len(self.series.values))


def read_series_csv(path: Path) -> MetricSeries:
    """Read a series: load one timestamp,value CSV into a MetricSeries named after the file. Raises if the file is malformed."""
    with path.open(encoding="utf-8-sig", newline="") as file:	# utf-8-sig also accepts a byte-order mark, like from a file re-saved in Excel
        rows = list(csv.reader(file))
    try:
        timestamps, values = _parse_rows(rows)
    except ValueError as error:
        raise ValueError(f"{path.name}: {error}") from error	# name the file, since a dataset has many
    return MetricSeries(name=path.stem, timestamps=timestamps, values=values)	# checks the samples themselves, naming the series in any error


def load_synthetic_dataset(data_dir: Path) -> list[LabeledSeries]:
    """Load the synthetic dataset: pair every series CSV in data_dir/series/ with its events from events.json. Returns them in name order."""
    paths = sorted((data_dir / "series").glob("*.csv"))	# names are zero-padded, so text order is index order
    if not paths:
        raise ValueError(f"no series CSVs in {data_dir / 'series'}")
    records = json.loads((data_dir / "events.json").read_text(encoding="utf-8"))
    names = {path.stem for path in paths}
    if names != records.keys():
        raise ValueError(
            f"events.json and the series folder disagree: CSVs without events {sorted(names - records.keys())}, "
            f"events without a CSV {sorted(records.keys() - names)}"
        )
    return [_labeled_series(path, records[path.stem]) for path in paths]


def _parse_rows(rows: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    if not rows or rows[0] != CSV_HEADER:
        raise ValueError(f"the first line must be {','.join(CSV_HEADER)}")
    body = []
    for line_number, row in enumerate(rows[1:], start=2):	# line 1 is the header
        if not row:
            continue	# a blank line holds no data, like an extra newline at the end of the file
        if len(row) != 2:
            raise ValueError(f"line {line_number} has {len(row)} fields instead of 2")
        body.append(row)
    if not body:
        raise ValueError("no data rows after the header")
    times, values = zip(*body)	# a column of timestamp strings and a column of value strings
    return (
        np.array(times, dtype="datetime64[s]"),	# a blank cell becomes NaT here, which MetricSeries rejects
        np.array(values, dtype=np.float64),	# raises on text that isn't a number; "nan" and "inf" get through, and MetricSeries rejects them
    )


def _labeled_series(path: Path, records: list[dict]) -> LabeledSeries:
    series = read_series_csv(path)
    try:
        events = [
            AnomalyEvent(kind=AnomalyKind(record["kind"]), start=record["start"], end=record["end"], params=record["params"])
            for record in records
        ]
        labeled = LabeledSeries(series=series, events=events)	# checks that every window fits the series and none overlap
        for record, event in zip(records, events):
            _check_event_times(record, event, series.timestamps)
    except ValueError as error:
        raise ValueError(f"events.json, {series.name}: {error}") from error
    return labeled


def _check_event_times(record: dict, event: AnomalyEvent, timestamps: np.ndarray) -> None:
    """Raise unless the event's saved first and last times match the series at its window, so the answer key lines up with the data."""
    for key, position in (("first_time", event.start), ("last_time", event.end - 1)):
        if np.datetime64(record[key], "s") != timestamps[position]:
            raise ValueError(
                f"the {event.kind} starting at sample {event.start} has {key} {record[key]}, "
                f"but the series has {timestamps[position]} there"
            )