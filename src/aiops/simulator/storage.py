"""Saving the dataset: what detectors may see, the ground truth, and what produced them, each in its own files."""

import json
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from aiops.simulator.anomalies import AnomalyEvent
from aiops.simulator.config import MetricConfig
from aiops.simulator.dataset import SimulatedSeries

VALUE_DECIMALS = 4	# rounding error of at most 0.00005, far below the noise


def save_dataset(dataset: list[SimulatedSeries], config: MetricConfig, seed: int, out_dir: Path) -> None:
    """Write the series CSVs, events.json, and metadata.json into out_dir."""
    series_dir = out_dir / "series"
    series_dir.mkdir(parents=True, exist_ok=True)
    for series in dataset:
        _write_series_csv(series, series_dir / f"{_series_name(series.index)}.csv")

    events = {
        _series_name(series.index): [_event_to_dict(event, series.components.timestamps) for event in series.events]
        for series in dataset
    }
    _write_json(out_dir / "events.json", events)
    metadata = {
        "seed": seed,
        "n_series": len(dataset),
        "numpy_version": np.__version__,	# the random streams depend on it, so it's part of what produced the data
        "config": config_to_dict(config),
    }
    _write_json(out_dir / "metadata.json", metadata)


def config_to_dict(config: MetricConfig) -> dict:
    """Turn the config into plain dicts, ready for JSON."""
    return asdict(config)


def config_from_dict(data: dict) -> MetricConfig:
    """Rebuild a MetricConfig from what config_to_dict produced, after a trip through JSON."""
    sections = {}
    for section in fields(MetricConfig):	# base, spike, level_shift, ... with their classes in section.type
        values = dict(data[section.name])
        for item in fields(section.type):
            if item.type is datetime:
                values[item.name] = datetime.fromisoformat(values[item.name])	# JSON has no dates, so they were saved as text
            elif isinstance(values[item.name], list):
                values[item.name] = tuple(values[item.name])	# JSON has no tuples, so they came back as lists
        sections[section.name] = section.type(**values)
    return MetricConfig(**sections)


def _series_name(index: int) -> str:
    return f"series_{index:02d}"	# series_00, series_01, ...


def _write_series_csv(series: SimulatedSeries, path: Path) -> None:
    times = np.datetime_as_string(series.components.timestamps, unit="s")	# like 2026-01-05T00:00:00
    rows = [f"{time.replace('T', ' ')},{value:.{VALUE_DECIMALS}f}" for time, value in zip(times, series.components.observed)]
    _write_text(path, "\n".join(["timestamp,value", *rows]) + "\n")


def _event_to_dict(event: AnomalyEvent, timestamps: np.ndarray) -> dict:
    return {
        "kind": event.kind,
        "start": event.start,
        "end": event.end,
        "first_time": _format_time(timestamps[event.start]),	# first labeled sample
        "last_time": _format_time(timestamps[event.end - 1]),	# last labeled sample
        "params": event.params,
    }


def _format_time(timestamp: np.datetime64) -> str:
    return np.datetime_as_string(timestamp, unit="s").replace("T", " ")


def _write_json(path: Path, data: dict) -> None:
    _write_text(path, json.dumps(data, indent=2, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    raise TypeError(f"can't save {type(value).__name__} to JSON")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")	# "\n" on every OS, so files match byte for byte everywhere