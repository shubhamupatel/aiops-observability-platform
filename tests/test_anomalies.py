"""Tests for the anomaly injectors and the anomaly-related config checks."""

import numpy as np
import pytest

from aiops.simulator import (
    AnomalyEvent,
    AnomalyKind,
    MetricConfig,
    SpikeConfig,
    generate_base_signal,
    inject_spike,
)

SEED = 42
N_STEPS = 4032	# default series length: 14 days*288 samples per day


@pytest.fixture
def series():
    return generate_base_signal(MetricConfig().base, np.random.default_rng(SEED))


# Verifies that a spike changes only its target sample by the recorded magnitude.
@pytest.mark.parametrize("start", [0, 100, N_STEPS - 1])	# both edges of the series and a middle position
def test_spike_changes_exactly_one_sample_by_its_magnitude(series, start):
    spiked, event = inject_spike(series, start, SpikeConfig(), np.random.default_rng(SEED))
    changed = np.flatnonzero(spiked.observed != series.observed)
    assert changed.tolist() == [start]
    jump = spiked.expected[start] - series.expected[start]
    assert jump == pytest.approx(event.params["magnitude"])


# Verifies that injecting a spike does not mutate the input series.
def test_input_series_is_left_untouched(series):
    level_before = series.level.copy()
    inject_spike(series, 100, SpikeConfig(), np.random.default_rng(SEED))
    np.testing.assert_array_equal(series.level, level_before)


# Verifies that the generated event records the spike kind, window, and magnitude.
def test_event_records_the_spike(series):
    _, event = inject_spike(series, 100, SpikeConfig(), np.random.default_rng(SEED))
    assert event.kind is AnomalyKind.SPIKE
    assert event.kind == "spike"	# a StrEnum member is also a plain string
    assert (event.start, event.end) == (100, 101)
    assert set(event.params) == {"magnitude"}


# Verifies that the same random seed produces the same spike magnitude.
def test_same_seed_gives_same_magnitude(series):
    _, first = inject_spike(series, 100, SpikeConfig(), np.random.default_rng(SEED))
    _, second = inject_spike(series, 100, SpikeConfig(), np.random.default_rng(SEED))
    assert first.params == second.params


# Verifies that random spike magnitudes remain within and cover the configured range.
def test_magnitudes_stay_in_range_and_cover_it(series):
    rng = np.random.default_rng(SEED)
    magnitudes = [inject_spike(series, 100, SpikeConfig(), rng)[1].params["magnitude"] for _ in range(200)]
    assert 10.0 <= min(magnitudes) and max(magnitudes) <= 40.0
    assert min(magnitudes) < 15.0 and max(magnitudes) > 35.0	# spread across the range, not bunched at one end


# Verifies that equal range bounds produce the requested fixed magnitude.
def test_fixed_range_pins_the_magnitude(series):
    _, event = inject_spike(series, 100, SpikeConfig((30.0, 30.0)), np.random.default_rng(SEED))
    assert event.params["magnitude"] == 30.0


@pytest.mark.parametrize("start", [-1, N_STEPS])	# one before the first sample, one past the last
# Verifies that spike positions outside the series are rejected.
def test_start_outside_series_is_rejected(series, start):
    with pytest.raises(ValueError):
        inject_spike(series, start, SpikeConfig(), np.random.default_rng(SEED))


@pytest.mark.parametrize("start, end", [(5, 5), (5, 3), (-1, 2)])	# empty, backwards, negative start
# Verifies that anomaly events reject empty, reversed, and negative windows.
def test_event_rejects_bad_windows(start, end):
    with pytest.raises(ValueError):
        AnomalyEvent(AnomalyKind.SPIKE, start, end, {})


@pytest.mark.parametrize("magnitude_range", [(0.0, 10.0), (20.0, 10.0)])	# zero low end, reversed order
# Verifies that spike magnitude ranges reject invalid lower bounds and ordering.
def test_spike_config_rejects_bad_ranges(magnitude_range):
    with pytest.raises(ValueError):
        SpikeConfig(magnitude_range)


# Verifies that spike settings exceeding the metric's headroom are rejected.
def test_metric_config_rejects_spikes_without_headroom():
    with pytest.raises(ValueError):
        MetricConfig(spike=SpikeConfig((10.0, 45.0)))	# 35 + 15 + 9 + 45 = 104, past the 100% bound