"""Tests for per-point labels and dataset assembly."""

from collections import Counter

import numpy as np
import pytest

from aiops.simulator import (
    AnomalyEvent,
    AnomalyKind,
    MetricConfig,
    generate_dataset,
    labels_from_events,
)

SEED = 42
N_SERIES = 20


@pytest.fixture
def dataset():
    return generate_dataset(MetricConfig(), N_SERIES, SEED)


def _event(kind: AnomalyKind, start: int, end: int) -> AnomalyEvent:
    return AnomalyEvent(kind=kind, start=start, end=end, params={})


# Verifies that labels are True exactly inside each event's window and False everywhere else.
def test_labels_mark_exactly_each_events_window():
    events = [_event(AnomalyKind.SPIKE, 10, 11), _event(AnomalyKind.LEVEL_SHIFT, 20, 30)]
    labels = labels_from_events(events, 50)
    assert labels.dtype == bool
    assert np.flatnonzero(labels).tolist() == [10] + list(range(20, 30))


# Verifies that a series with no events is labeled normal everywhere.
def test_no_events_gives_all_normal():
    assert not labels_from_events([], 50).any()


# Verifies that events whose labeled windows overlap are rejected.
@pytest.mark.parametrize(
    "first, second",
    [
        ((20, 30), (25, 35)),	# partial overlap
        ((20, 30), (20, 30)),	# identical windows
        ((20, 30), (22, 24)),	# one inside the other
    ],
)
def test_overlapping_events_are_rejected(first, second):
    events = [_event(AnomalyKind.LEVEL_SHIFT, *first), _event(AnomalyKind.LEVEL_SHIFT, *second)]
    with pytest.raises(ValueError):
        labels_from_events(events, 50)


# Verifies that an event running past the end of the series is rejected.
def test_event_past_the_series_is_rejected():
    with pytest.raises(ValueError):
        labels_from_events([_event(AnomalyKind.LEVEL_SHIFT, 40, 51)], 50)


# Verifies that the dataset has the requested number of series, indexed 0 to n - 1.
def test_dataset_has_n_series_indexed_in_order(dataset):
    assert [series.index for series in dataset] == list(range(N_SERIES))


# Verifies that each series' labels property matches the labels built from its events.
def test_labels_property_matches_its_events(dataset):
    for series in dataset:
        np.testing.assert_array_equal(series.labels, labels_from_events(series.events, len(series.components.level)))
        assert series.labels.sum() == sum(event.end - event.start for event in series.events)


# Verifies that every series in the dataset follows the scheduler's rules.
def test_every_series_follows_the_schedule_rules(dataset):
    for series in dataset:
        counts = Counter(event.kind for event in series.events)
        assert counts == {
            AnomalyKind.SPIKE: 3,
            AnomalyKind.LEVEL_SHIFT: 2,
            AnomalyKind.SEASONAL_SUPPRESSION: 1,
            AnomalyKind.DRIFT: 1,
        }
        assert series.events[0].changed_start >= 288	# nothing in the warm-up
        for earlier, later in zip(series.events, series.events[1:]):
            assert later.changed_start - earlier.end >= 24	# the 2-hour gap


# Verifies that the same seed gives an identical dataset, data and events both.
def test_same_seed_gives_identical_dataset(dataset):
    again = generate_dataset(MetricConfig(), N_SERIES, SEED)
    for first, second in zip(dataset, again):
        np.testing.assert_array_equal(first.components.observed, second.components.observed)
        assert first.events == second.events


# Verifies that the series in one dataset differ from each other.
def test_series_differ_from_each_other(dataset):
    for first, second in zip(dataset, dataset[1:]):
        assert not np.array_equal(first.components.noise, second.components.noise)
        assert first.events != second.events


# Verifies that a different seed gives a different dataset.
def test_different_seeds_give_different_datasets(dataset):
    other = generate_dataset(MetricConfig(), N_SERIES, SEED + 1)
    assert not np.array_equal(dataset[0].components.observed, other[0].components.observed)


# Verifies that each series' stream is independent: a smaller dataset is exactly the start of a larger one.
def test_smaller_dataset_matches_the_start_of_a_larger_one():
    small = generate_dataset(MetricConfig(), 3, SEED)
    large = generate_dataset(MetricConfig(), 5, SEED)
    for first, second in zip(small, large):	# zip stops after the 3 series in small
        np.testing.assert_array_equal(first.components.observed, second.components.observed)
        assert first.events == second.events


# Verifies that asking for fewer than one series is rejected.
@pytest.mark.parametrize("n_series", [0, -1])
def test_n_series_below_one_is_rejected(n_series):
    with pytest.raises(ValueError):
        generate_dataset(MetricConfig(), n_series, SEED)