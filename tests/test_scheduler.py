"""Tests for the scheduler and the placement settings in the config."""

from collections import Counter

import numpy as np
import pytest

from aiops.simulator import (
    AnomalyEvent,
    AnomalyKind,
    DriftConfig,
    LevelShiftConfig,
    MetricConfig,
    PlacementError,
    ScheduleConfig,
    SeasonalSuppressionConfig,
    SpikeConfig,
    generate_base_signal,
    inject_anomalies,
    inject_level_shift,
)
from aiops.simulator.scheduler import _place_one

SEEDS = range(20)	# placement is random, so the rules are checked across many different schedules
WARMUP = 288	# 1440 minutes at 5 minutes per sample
GAP = 24	# 120 minutes at 5 minutes per sample


def _schedule(config: MetricConfig, seed: int):
    rng = np.random.default_rng(seed)
    base = generate_base_signal(config.base, rng)
    injected, events = inject_anomalies(base, config, rng)
    return base, injected, events


# Verifies that every series gets exactly the configured number of each anomaly type.
def test_each_type_gets_its_configured_count():
    for seed in SEEDS:
        _, _, events = _schedule(MetricConfig(), seed)
        counts = Counter(event.kind for event in events)
        assert counts == {
            AnomalyKind.SPIKE: 3,
            AnomalyKind.LEVEL_SHIFT: 2,
            AnomalyKind.SEASONAL_SUPPRESSION: 1,
            AnomalyKind.DRIFT: 1,
        }


# Verifies that events come back in time order with at least the minimum gap between them.
def test_anomalies_keep_their_gap():
    for seed in SEEDS:
        _, _, events = _schedule(MetricConfig(), seed)
        for earlier, later in zip(events, events[1:]):	# each event paired with the one after it
            assert later.changed_start - earlier.end >= GAP


# Verifies that no anomaly changes any data inside the warm-up.
def test_nothing_lands_in_the_warmup():
    for seed in SEEDS:
        _, _, events = _schedule(MetricConfig(), seed)
        assert min(event.changed_start for event in events) >= WARMUP


# Verifies that the data changed exactly where the events say, and nowhere else.
def test_data_changes_exactly_where_the_events_say():
    for seed in SEEDS:
        base, injected, events = _schedule(MetricConfig(), seed)
        changed = (injected.level != base.level) | (injected.seasonal != base.seasonal)
        claimed = np.zeros(len(changed), dtype=bool)
        for event in events:
            claimed[event.changed_start:event.end] = True
        np.testing.assert_array_equal(changed, claimed)
        np.testing.assert_array_equal(injected.noise, base.noise)


# Verifies that the same seed gives the same placements.
def test_same_seed_gives_same_schedule():
    _, _, first = _schedule(MetricConfig(), 42)
    _, _, second = _schedule(MetricConfig(), 42)
    assert first == second


# Verifies that different seeds give different placements.
def test_different_seeds_give_different_schedules():
    _, _, first = _schedule(MetricConfig(), 42)
    _, _, second = _schedule(MetricConfig(), 43)
    assert [event.changed_start for event in first] != [event.changed_start for event in second]


# Verifies that scheduling doesn't modify the original series.
def test_input_series_is_left_untouched():
    config = MetricConfig()
    rng = np.random.default_rng(42)
    base = generate_base_signal(config.base, rng)
    level_before, seasonal_before = base.level.copy(), base.seasonal.copy()
    inject_anomalies(base, config, rng)
    np.testing.assert_array_equal(base.level, level_before)
    np.testing.assert_array_equal(base.seasonal, seasonal_before)


# Verifies that a count of zero turns a type off without affecting the others.
def test_count_zero_turns_a_type_off():
    _, _, events = _schedule(MetricConfig(spike=SpikeConfig(count=0)), 42)
    kinds = {event.kind for event in events}
    assert AnomalyKind.SPIKE not in kinds
    assert kinds == {AnomalyKind.LEVEL_SHIFT, AnomalyKind.SEASONAL_SUPPRESSION, AnomalyKind.DRIFT}


# Verifies that a schedule too crowded to place raises PlacementError instead of looping forever.
def test_crowded_schedule_raises_placement_error():
    config = MetricConfig(
        level_shift=LevelShiftConfig(duration_range_minutes=(480, 480), count=30),	# passes the fit check but fills 96% of the room
        spike=SpikeConfig(count=0),
        seasonal_suppression=SeasonalSuppressionConfig(count=0),
        drift=DriftConfig(count=0),
    )
    with pytest.raises(PlacementError):
        _schedule(config, 42)


# Verifies that every retry reuses the same magnitude and duration, so collisions can't bias placed anomalies toward short ones.
def test_retries_reuse_the_same_parameters():
    base = generate_base_signal(MetricConfig().base, np.random.default_rng(42))
    blocker = AnomalyEvent(kind=AnomalyKind.LEVEL_SHIFT, start=WARMUP, end=3500, params={"magnitude": 10.0})	# leaves room only near the end, so most starts collide
    attempts: list[AnomalyEvent] = []

    def recording_inject(components, start, config, rng):	# the real injector, but it records every attempt
        candidate, event = inject_level_shift(components, start, config, rng)
        attempts.append(event)
        return candidate, event

    attempt_counts = []
    for seed in SEEDS:
        attempts.clear()
        _place_one(base, recording_inject, LevelShiftConfig(), ScheduleConfig(), [blocker], np.random.default_rng(seed))
        assert len({(event.params["magnitude"], event.end - event.start) for event in attempts}) == 1	# one magnitude and duration across every attempt
        attempt_counts.append(len(attempts))
    assert max(attempt_counts) > 1	# the check above only means something if the scheduler actually retried


# Verifies that changed_start is the onset for drift and the start for every other type.
def test_changed_start_is_onset_for_drift_and_start_otherwise():
    drift = AnomalyEvent(kind=AnomalyKind.DRIFT, start=1038, end=1220, params={"magnitude": 28.8, "onset": 1000})
    spike = AnomalyEvent(kind=AnomalyKind.SPIKE, start=500, end=501, params={"magnitude": 25.6})
    assert drift.changed_start == 1000
    assert spike.changed_start == 500


# Verifies that every anomaly config rejects a negative count.
@pytest.mark.parametrize("config_class", [SpikeConfig, LevelShiftConfig, SeasonalSuppressionConfig, DriftConfig])
def test_negative_count_is_rejected(config_class):
    with pytest.raises(ValueError):
        config_class(count=-1)


# Verifies that the schedule config rejects negative times and a zero attempt limit.
@pytest.mark.parametrize(
    "bad_field",
    [
        {"warmup_minutes": -5},
        {"min_gap_minutes": -5},
        {"max_attempts": 0},
    ],
)
def test_schedule_config_rejects_bad_settings(bad_field):
    with pytest.raises(ValueError):
        ScheduleConfig(**bad_field)


# Verifies the cross-section rules MetricConfig enforces for placement.
@pytest.mark.parametrize(
    "overrides",
    [
        {"schedule": ScheduleConfig(warmup_minutes=1442)},	# not a multiple of 5
        {"schedule": ScheduleConfig(min_gap_minutes=122)},	# not a multiple of 5
        {"spike": SpikeConfig(count=200)},	# the worst case needs 30,760 minutes; the series has 20,160
    ],
)
def test_metric_config_rejects_bad_schedules(overrides):
    with pytest.raises(ValueError):
        MetricConfig(**overrides)