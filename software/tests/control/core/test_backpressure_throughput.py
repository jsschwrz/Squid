"""Tests for the throughput-driven additions to backpressure:

- cumulative captured/written counters + RateSampler, and
- the adaptive effective byte cap.
"""

import pytest

from control.core.backpressure import BackpressureController, RateSampler


class TestRateSampler:
    def test_first_sample_seeds_silently(self):
        s = RateSampler()
        s.update(1000, 500, now=10.0)
        assert s.capture_mb_s == 0.0
        assert s.write_mb_s == 0.0

    def test_positive_rate_after_second_sample(self):
        s = RateSampler(alpha=1.0)  # pure instantaneous -> easy to assert
        s.update(0, 0, now=0.0)
        s.update(1024 * 1024, 512 * 1024, now=1.0)  # 1 MiB / 0.5 MiB over 1 s
        assert s.capture_mb_s == pytest.approx(1.0, rel=1e-3)
        assert s.write_mb_s == pytest.approx(0.5, rel=1e-3)

    def test_dt_guard_ignores_too_frequent_samples(self):
        s = RateSampler(alpha=1.0, min_dt_s=0.05)
        s.update(0, 0, now=0.0)
        s.update(10 * 1024 * 1024, 0, now=0.01)  # dt below guard -> ignored
        assert s.capture_mb_s == 0.0

    def test_negative_delta_clamped(self):
        s = RateSampler(alpha=1.0)
        s.update(10 * 1024 * 1024, 10 * 1024 * 1024, now=0.0)
        s.update(0, 0, now=1.0)  # counters reset -> no negative rate
        assert s.capture_mb_s == 0.0
        assert s.write_mb_s == 0.0

    def test_reset_clears_state(self):
        s = RateSampler(alpha=1.0)
        s.update(0, 0, now=0.0)
        s.update(1024 * 1024, 1024 * 1024, now=1.0)
        assert s.write_mb_s > 0
        s.reset()
        assert s.write_mb_s == 0.0
        s.update(5 * 1024 * 1024, 5 * 1024 * 1024, now=2.0)  # seeds silently again
        assert s.write_mb_s == 0.0


class TestCumulativeCounters:
    def test_job_dispatched_increments_captured(self):
        c = BackpressureController(max_jobs=100, max_mb=1000.0)
        c.job_dispatched(2 * 1024 * 1024)
        stats = c.get_stats()
        assert stats.captured_bytes == 2 * 1024 * 1024
        assert stats.captured_count == 1
        assert stats.written_bytes == 0
        c.close()

    def test_job_completed_increments_written_and_drains_pending(self):
        c = BackpressureController(max_jobs=100, max_mb=1000.0)
        c.job_dispatched(3 * 1024 * 1024)
        c.job_completed(3 * 1024 * 1024)
        stats = c.get_stats()
        assert stats.written_bytes == 3 * 1024 * 1024
        assert stats.written_count == 1
        assert stats.pending_jobs == 0
        assert stats.pending_bytes_mb == pytest.approx(0.0)
        c.close()

    def test_reset_clears_cumulative(self):
        c = BackpressureController(max_jobs=100, max_mb=1000.0)
        c.job_dispatched(1024 * 1024)
        c.job_completed(1024 * 1024)
        c.reset()
        stats = c.get_stats()
        assert stats.captured_bytes == 0
        assert stats.written_bytes == 0
        c.close()

    def test_closed_stats_have_zeroed_new_fields(self):
        c = BackpressureController(max_jobs=1, max_mb=1.0)
        c.close()
        stats = c.get_stats()
        assert stats.captured_bytes == 0
        assert stats.written_bytes == 0
        assert stats.write_mb_s == 0.0


class TestAdaptiveCap:
    def test_static_when_target_backlog_zero(self):
        c = BackpressureController(max_jobs=10, max_mb=500.0, target_backlog_s=0.0)
        assert c._effective_max_bytes() == int(500.0 * 1024 * 1024)
        c.close()

    def test_cap_tracks_write_rate(self):
        c = BackpressureController(max_jobs=10, max_mb=10000.0, target_backlog_s=30.0, floor_mb=100.0)
        c._rate_sampler._write_mb_s = 200.0  # 200 * 30 = 6000 MB, within [100, 10000]
        assert c._effective_max_bytes() == int(6000.0 * 1024 * 1024)
        c.close()

    def test_cap_respects_floor(self):
        c = BackpressureController(max_jobs=10, max_mb=10000.0, target_backlog_s=30.0, floor_mb=500.0)
        c._rate_sampler._write_mb_s = 0.0
        assert c._effective_max_bytes() == int(500.0 * 1024 * 1024)
        c.close()

    def test_cap_respects_ceiling(self):
        c = BackpressureController(max_jobs=10, max_mb=1000.0, target_backlog_s=30.0, floor_mb=100.0)
        c._rate_sampler._write_mb_s = 1000.0  # 1000 * 30 = 30000 > ceiling
        assert c._effective_max_bytes() == int(1000.0 * 1024 * 1024)
        c.close()

