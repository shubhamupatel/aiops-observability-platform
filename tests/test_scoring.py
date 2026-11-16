"""Tests for scoring: which samples count, how flags are matched to events, and how counts become metrics."""

from pathlib import Path

import numpy as np
import pytest

from aiops.data import LabeledSeries, MetricSeries, load_synthetic_dataset
from aiops.evaluation import EventResult, Score, ScoringConfig, pool, score_series, scored_mask, split_tuning_and_test
from aiops.simulator import AnomalyEvent, AnomalyKind

COMMITTED_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"	# the dataset in the repo
STRICT = ScoringConfig(warmup_minutes=0, grace_minutes=0)	# every sample scored, so a test can check one rule at a time


def _labeled(n_steps: int, events: list[AnomalyEvent]) -> LabeledSeries:
    """A tiny all-zero series at 5-minute intervals with the given events, small enough to score by hand."""
    timestamps = np.datetime64("2026-01-05T00:00") + np.arange(n_steps) * np.timedelta64(5, "m")
    return LabeledSeries(series=MetricSeries("tiny", timestamps, np.zeros(n_steps)), events=events)


def _event(kind: AnomalyKind, start: int, end: int, **params) -> AnomalyEvent:
    return AnomalyEvent(kind=kind, start=start, end=end, params=params)


def _flags(n_steps: int, positions: list[int]) -> np.ndarray:
    flags = np.zeros(n_steps, dtype=bool)
    flags[positions] = True
    return flags


def _score(**counts) -> Score:
    """A Score built directly, with every count zero unless given."""
    defaults = dict(true_positives=0, false_positives=0, false_negatives=0, true_negatives=0, false_alarms=0, scored_minutes=0.0, events=())
    return Score(**(defaults | counts))


# A level shift at samples 4 to 7 and a spike at 12, in 20 samples: 5 labeled, 15 normal.
HAND_EVENTS = [_event(AnomalyKind.LEVEL_SHIFT, 4, 8), _event(AnomalyKind.SPIKE, 12, 13)]
HAND_FLAGS = [5, 6, 10, 12, 15, 16]	# hits 5, 6, 12; misses 4 and 7; flags 10, 15, 16 wrongly


# Verifies the counts on a series scored by hand: 3 hits, 3 wrong flags, 2 misses, 12 normal samples left alone.
def test_counts_match_a_hand_scored_series():
    score = score_series(_labeled(20, HAND_EVENTS), _flags(20, HAND_FLAGS), STRICT)
    assert (score.true_positives, score.false_positives, score.false_negatives, score.true_negatives) == (3, 3, 2, 12)
    assert score.false_alarms == 2	# 10 alone, and 15 to 16 as one run; 5 to 6 and 12 touch labels
    assert score.scored_minutes == 100.0	# 20 samples of 5 minutes


# Verifies every metric on the hand-scored series against its formula worked out by hand.
def test_metrics_match_the_hand_scored_series():
    score = score_series(_labeled(20, HAND_EVENTS), _flags(20, HAND_FLAGS), STRICT)
    assert score.precision == pytest.approx(3 / 6)
    assert score.recall == pytest.approx(3 / 5)
    assert score.f1 == pytest.approx(2 * 0.5 * 0.6 / (0.5 + 0.6))
    assert score.false_positive_rate == pytest.approx(3 / 15)
    assert score.event_recall == 1.0	# both events have a flag inside their window
    assert score.composite_f1 == pytest.approx(2 * 0.5 * 1.0 / (0.5 + 1.0))
    assert score.median_delay_minutes == 2.5	# the shift was caught 5 minutes in, the spike at once


# Verifies that an event counts as detected only with a flag inside its window, and that delay runs from the window's start.
@pytest.mark.parametrize(
    "position, detected, delay",
    [
        (4, True, 0.0),	# the first labeled sample
        (7, True, 15.0),	# the last labeled sample, 3 samples in
        (3, False, None),	# just before the window
        (8, False, None),	# just after it; end is one past the last labeled sample
    ],
)
def test_detection_needs_a_flag_inside_the_window(position, detected, delay):
    score = score_series(_labeled(20, HAND_EVENTS[:1]), _flags(20, [position]), STRICT)
    assert score.events == (EventResult(kind=AnomalyKind.LEVEL_SHIFT, detected=detected, delay_minutes=delay),)


# Verifies the scored mask with every rule at once: warm-up, a drift's unlabeled ramp, and the grace after each event.
def test_scored_mask_leaves_out_warmup_ramp_and_grace():
    events = [_event(AnomalyKind.DRIFT, 7, 10, onset=5), _event(AnomalyKind.SPIKE, 15, 16)]
    config = ScoringConfig(warmup_minutes=10, grace_minutes=10)	# 2 samples each
    mask = scored_mask(_labeled(20, events), config)
    assert np.flatnonzero(~mask).tolist() == [0, 1, 5, 6, 10, 11, 16, 17]	# warm-up, ramp, grace after the drift, grace after the spike


# Verifies that no exclusion rule can remove a labeled sample, even where the rules overlap one.
@pytest.mark.parametrize(
    "events, config",
    [
        ([_event(AnomalyKind.SPIKE, 1, 2)], ScoringConfig(warmup_minutes=20, grace_minutes=0)),	# a spike inside the warm-up
        (
            [_event(AnomalyKind.LEVEL_SHIFT, 2, 4), _event(AnomalyKind.SPIKE, 5, 6)],
            ScoringConfig(warmup_minutes=0, grace_minutes=15),
        ),	# a spike inside the shift's 3-sample grace
    ],
)
def test_labeled_samples_are_always_scored(events, config):
    labeled = _labeled(20, events)
    assert scored_mask(labeled, config)[labeled.labels].all()


# Verifies that flags in the warm-up count neither for nor against the detector.
def test_warmup_flags_count_neither_way():
    config = ScoringConfig(warmup_minutes=25, grace_minutes=0)	# samples 0 to 4
    score = score_series(_labeled(20, []), _flags(20, [0, 1, 2, 3, 4, 10]), config)
    assert (score.false_positives, score.true_negatives, score.false_alarms) == (1, 14, 1)	# only the flag at 10 counts
    assert score.scored_minutes == 75.0


# Verifies the grace after an event: flags there are free, but a run that continues past it is a false alarm unless it touched the event.
@pytest.mark.parametrize(
    "positions, false_positives, false_alarms",
    [
        ([8, 9], 0, 0),	# only the drop and the sample after it
        ([8, 9, 10], 1, 1),	# runs one sample past the grace
        ([7, 8, 9, 10], 1, 0),	# the same, but the run started inside the event, so it's one true alarm that lingered
    ],
)
def test_grace_after_an_event(positions, false_positives, false_alarms):
    config = ScoringConfig(warmup_minutes=0, grace_minutes=10)	# samples 8 and 9
    score = score_series(_labeled(20, HAND_EVENTS[:1]), _flags(20, positions), config)
    assert (score.false_positives, score.false_alarms) == (false_positives, false_alarms)


# Verifies that flags on a drift's unlabeled ramp are neither false positives nor detections.
def test_drift_ramp_flags_count_neither_way():
    events = [_event(AnomalyKind.DRIFT, 6, 10, onset=3)]	# ramp at 3 to 5, labeled from 6
    score = score_series(_labeled(20, events), _flags(20, [3, 4, 5]), STRICT)
    assert (score.false_positives, score.false_alarms, score.event_recall) == (0, 0, 0.0)


# Verifies that a run of false flags is one false alarm however long it is.
def test_a_long_false_run_is_one_false_alarm():
    score = score_series(_labeled(20, []), _flags(20, list(range(2, 12))), STRICT)
    assert (score.false_positives, score.false_alarms) == (10, 1)


# Verifies that pooling adds counts rather than averaging metrics: precisions of 1.0 and 0.25 pool to 2/5, not their mean of 0.625.
def test_pool_adds_counts():
    first = _score(true_positives=1, scored_minutes=60.0, events=(EventResult("spike", True, 0.0),))
    second = _score(true_positives=1, false_positives=3, scored_minutes=30.0, events=(EventResult("spike", False, None),))
    pooled = pool([first, second])
    assert (pooled.true_positives, pooled.false_positives, pooled.scored_minutes) == (2, 3, 90.0)
    assert pooled.precision == pytest.approx(2 / 5)
    assert pooled.events == first.events + second.events


# Verifies that pooling nothing is rejected instead of producing an empty score.
def test_pool_rejects_an_empty_list():
    with pytest.raises(ValueError):
        pool([])


# Verifies that every 0/0 metric comes out as 0, and that delay is None with nothing detected.
def test_empty_score_gives_zeros():
    score = _score()
    metrics = [score.precision, score.recall, score.f1, score.false_positive_rate, score.event_recall, score.composite_f1, score.false_alarms_per_day]
    assert metrics == [0.0] * 7
    assert score.median_delay_minutes is None


# Verifies that false alarms are reported per day of scored time.
def test_false_alarms_per_day():
    assert _score(false_alarms=3, scored_minutes=2880.0).false_alarms_per_day == 1.5	# 3 alarms over 2 days


# Verifies the per-kind counts, sorted by kind, and that the median delay uses detected events only.
def test_event_recall_by_kind_and_median_delay():
    events = (
        EventResult("spike", True, 0.0),
        EventResult("spike", False, None),
        EventResult("drift", True, 30.0),
        EventResult("level_shift", True, 10.0),
    )
    score = _score(events=events)
    assert list(score.event_recall_by_kind.items()) == [("drift", (1, 1)), ("level_shift", (1, 1)), ("spike", (1, 2))]
    assert score.median_delay_minutes == 10.0	# median of 0, 30, and 10; the missed spike has no delay


# Verifies that flags must be a True/False array with one entry per sample.
@pytest.mark.parametrize(
    "flags",
    [
        np.zeros(20, dtype=int),	# 0s and 1s instead of True/False
        np.zeros(19, dtype=bool),	# one entry short
    ],
)
def test_score_series_rejects_bad_flags(flags):
    with pytest.raises(ValueError, match="^tiny"):
        score_series(_labeled(20, []), flags)


# Verifies that a series with uneven gaps between samples is rejected, since minutes can't be turned into sample counts.
def test_uneven_spacing_is_rejected():
    timestamps = np.array(["2026-01-05T00:00", "2026-01-05T00:05", "2026-01-05T00:15"], dtype="datetime64[m]")
    labeled = LabeledSeries(series=MetricSeries("gappy", timestamps, np.zeros(3)), events=[])
    with pytest.raises(ValueError):
        score_series(labeled, np.zeros(3, dtype=bool))


# Verifies that the scoring config rejects negative times.
@pytest.mark.parametrize("warmup_minutes, grace_minutes", [(-5, 60), (1440, -5)])
def test_scoring_config_rejects_negative_times(warmup_minutes, grace_minutes):
    with pytest.raises(ValueError):
        ScoringConfig(warmup_minutes=warmup_minutes, grace_minutes=grace_minutes)


# Verifies the locked split: series_00 to 09 choose thresholds, series_10 to 19 report results.
def test_split_is_series_00_to_09_and_10_to_19():
    tuning, test = split_tuning_and_test(load_synthetic_dataset(COMMITTED_DIR))
    assert [item.series.name for item in tuning] == [f"series_{index:02d}" for index in range(10)]
    assert [item.series.name for item in test] == [f"series_{index:02d}" for index in range(10, 20)]


# Verifies that a dataset too small to split is rejected.
def test_split_rejects_a_single_series():
    with pytest.raises(ValueError):
        split_tuning_and_test(load_synthetic_dataset(COMMITTED_DIR)[:1])


# Verifies on the real dataset that flagging exactly the labels scores perfectly, so no rule clashes with a real event layout.
def test_perfect_flags_score_perfectly_on_every_series():
    pooled = pool([score_series(item, item.labels) for item in load_synthetic_dataset(COMMITTED_DIR)])
    assert (pooled.precision, pooled.recall, pooled.event_recall, pooled.false_alarms) == (1.0, 1.0, 1.0, 0)
    assert all(event.delay_minutes == 0.0 for event in pooled.events)


# Verifies on the real dataset that flagging everything gives no false alarms, and that only point precision exposes it.
def test_flagging_everything_is_exposed_only_by_precision():
    dataset = load_synthetic_dataset(COMMITTED_DIR)
    pooled = pool([score_series(item, np.ones(len(item.series.values), dtype=bool)) for item in dataset])
    labeled_share = sum(item.labels.sum() for item in dataset) / sum(scored_mask(item).sum() for item in dataset)
    assert (pooled.recall, pooled.event_recall, pooled.false_alarms) == (1.0, 1.0, 0)	# one giant alarm touches every event
    assert pooled.precision == pytest.approx(labeled_share)