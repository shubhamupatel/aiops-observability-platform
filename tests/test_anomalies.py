"""Tests for the anomaly injectors and the anomaly-related config checks."""

import numpy as np
import pytest

from aiops.simulator import (
    AnomalyEvent,
    AnomalyKind,
    LevelShiftConfig,
    MetricConfig,
    SpikeConfig,
    generate_base_signal,
    inject_level_shift,
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


# Verifies that spike positions outside the series are rejected.
@pytest.mark.parametrize("start", [-1, N_STEPS])	# one before the first sample, one past the last
def test_start_outside_series_is_rejected(series, start):
    with pytest.raises(ValueError):
        inject_spike(series, start, SpikeConfig(), np.random.default_rng(SEED))


# Verifies that anomaly events reject empty, reversed, and negative windows.
@pytest.mark.parametrize("start, end", [(5, 5), (5, 3), (-1, 2)])	# empty, backwards, negative start
def test_event_rejects_bad_windows(start, end):
    with pytest.raises(ValueError):
        AnomalyEvent(AnomalyKind.SPIKE, start, end, {})


# Verifies that spike magnitude ranges reject invalid lower bounds and ordering.
@pytest.mark.parametrize("magnitude_range", [(0.0, 10.0), (20.0, 10.0)])	# zero low end, reversed order
def test_spike_config_rejects_bad_ranges(magnitude_range):
    with pytest.raises(ValueError):
        SpikeConfig(magnitude_range)


# Verifies that spike settings exceeding the metric's headroom are rejected.
def test_metric_config_rejects_spikes_without_headroom():
    with pytest.raises(ValueError):
        MetricConfig(spike=SpikeConfig((10.0, 45.0)))	# 35 + 15 + 9 + 45 = 104, past the 100% bound


# Verifies that a level-shift raises exactly its window, every sample by the same magnitude.
@pytest.mark.parametrize("start", [0, 1000, N_STEPS - 96])	# first sample, middle, last start with room for the longest shift
def test_level_shift_raises_exactly_its_window_by_its_magnitude(series, start):
    shifted, event = inject_level_shift(series, start, LevelShiftConfig(), np.random.default_rng(SEED))
    changed = np.flatnonzero(shifted.observed != series.observed)
    assert changed.tolist() == list(range(event.start, event.end))
    jumps = shifted.expected[event.start:event.end] - series.expected[event.start:event.end]
    np.testing.assert_allclose(jumps, event.params["magnitude"])


# Verifies that injecting a level-shift doesn't modify the original series.
def test_level_shift_leaves_input_untouched(series):
    level_before = series.level.copy()
    inject_level_shift(series, 1000, LevelShiftConfig(), np.random.default_rng(SEED))
    np.testing.assert_array_equal(series.level, level_before)


# Verifies the event's type, start, duration, and recorded parameters.
def test_level_shift_event_records_the_shift(series):
    _, event = inject_level_shift(series, 1000, LevelShiftConfig(), np.random.default_rng(SEED))
    assert event.kind is AnomalyKind.LEVEL_SHIFT
    assert event.kind == "level_shift"
    assert event.start == 1000
    assert 12 <= event.end - event.start <= 96	# 60 to 480 minutes at 5 minutes per sample
    assert set(event.params) == {"magnitude"}


# Verifies that the same seed gives the same magnitude and duration.
def test_level_shift_same_seed_gives_same_event(series):
    _, first = inject_level_shift(series, 1000, LevelShiftConfig(), np.random.default_rng(SEED))
    _, second = inject_level_shift(series, 1000, LevelShiftConfig(), np.random.default_rng(SEED))
    assert first == second


# Verifies that drawn magnitudes and durations stay inside their configured ranges.
def test_level_shift_draws_stay_in_range(series):
    rng = np.random.default_rng(SEED)
    events = [inject_level_shift(series, 1000, LevelShiftConfig(), rng)[1] for _ in range(200)]
    magnitudes = [event.params["magnitude"] for event in events]
    durations = [event.end - event.start for event in events]
    assert 5.0 <= min(magnitudes) and max(magnitudes) <= 25.0
    assert 12 <= min(durations) and max(durations) <= 96


# Verifies that both the shortest and the longest duration can actually be drawn.
def test_level_shift_duration_can_hit_both_ends(series):
    config = LevelShiftConfig(duration_range_minutes=(60, 70))	# 12 to 14 samples, so 100 draws hit every value
    rng = np.random.default_rng(SEED)
    durations = set()
    for _ in range(100):
        _, event = inject_level_shift(series, 1000, config, rng)
        durations.add(event.end - event.start)
    assert durations == {12, 13, 14}	# 14 only shows up because of endpoint=True


# Verifies that single-value ranges pin the magnitude and the minutes-to-samples conversion.
def test_fixed_ranges_pin_the_level_shift(series):
    config = LevelShiftConfig(magnitude_range=(20.0, 20.0), duration_range_minutes=(120, 120))
    _, event = inject_level_shift(series, 1000, config, np.random.default_rng(SEED))
    assert event.params["magnitude"] == 20.0
    assert (event.start, event.end) == (1000, 1024)	# 120 minutes = 24 samples


# Verifies that starts outside the series, or too close to its end, are rejected.
@pytest.mark.parametrize("start", [-1, N_STEPS - 95])	# before the series; one past the last start with room
def test_level_shift_rejects_bad_starts(series, start):
    with pytest.raises(ValueError):
        inject_level_shift(series, start, LevelShiftConfig(), np.random.default_rng(SEED))


# Verifies that the level-shift config rejects invalid ranges.
@pytest.mark.parametrize(
    "bad_field",
    [
        {"magnitude_range": (0.0, 5.0)},	# zero low end
        {"duration_range_minutes": (480, 60)},	# reversed order
    ],
)
def test_level_shift_config_rejects_bad_ranges(bad_field):
    with pytest.raises(ValueError):
        LevelShiftConfig(**bad_field)


# Verifies the cross-section rules MetricConfig enforces for level-shifts.
@pytest.mark.parametrize(
    "level_shift",
    [
        LevelShiftConfig(magnitude_range=(5.0, 45.0)),	# 59 + 45 = 104, past the 100% bound
        LevelShiftConfig(duration_range_minutes=(62, 480)),	# 62 isn't a multiple of 5
        LevelShiftConfig(duration_range_minutes=(60, 20165)),	# longer than the 14-day series (20,160 minutes)
    ],
)
def test_metric_config_rejects_bad_level_shift_settings(level_shift):
    with pytest.raises(ValueError):
        MetricConfig(level_shift=level_shift)