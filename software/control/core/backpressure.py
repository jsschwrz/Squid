"""Backpressure controller for acquisition throttling.

Prevents RAM exhaustion by tracking pending jobs/bytes and throttling
acquisition when limits are exceeded.

The controller also maintains cumulative captured/written byte counters and a
live throughput estimate (RateSampler). That single write-rate estimate drives:
  - the adaptive byte cap (should_throttle bounds the backlog to ~target_backlog_s
    of measured write throughput),
  - the dynamic end-of-run drain timeout (computed by the worker from pending
    bytes / write rate), and
  - the live status-bar monitor (capture MB/s vs write MB/s vs ratio).

Ownership Model:
    BackpressureValues (the tuple of multiprocessing primitives) can be created
    either by create_backpressure_values() or internally by BackpressureController.

    The tuple layout is (7 elements):
        (pending_jobs, pending_bytes, capacity_event,
         captured_bytes, written_bytes, captured_count, written_count)
    The first three preserve the historical layout; the four cumulative counters
    are appended. Only this module and the controller unpack the tuple positionally;
    everything else should use the named accessor properties.

    The values are shared between:
    - JobRunner subprocess (increments/decrements counters)
    - BackpressureController in main process (checks throttling)

    Cleanup: multiprocessing.Value and Event don't have close() methods.
    They're garbage collected when all references are dropped. Ensure:
    1. JobRunner.shutdown() is called (terminates subprocess, releases its refs)
    2. BackpressureController.close() is called (clears main process refs)

    After both, the primitives will be GC'd and underlying semaphores released.
"""

import multiprocessing
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import squid.logging

log = squid.logging.get_logger(__name__)

__all__ = [
    "BackpressureController",
    "BackpressureStats",
    "BackpressureValues",
    "RateSampler",
    "create_backpressure_values",
]

# Conversion constant: 1 MiB = 1,048,576 bytes (binary prefix, not SI megabyte)
_BYTES_PER_MB = 1024 * 1024

# Type alias for the backpressure values tuple (7 elements):
#   (pending_jobs, pending_bytes, capacity_event,
#    captured_bytes, written_bytes, captured_count, written_count)
BackpressureValues = Tuple[
    multiprocessing.Value,
    multiprocessing.Value,
    multiprocessing.Event,
    multiprocessing.Value,
    multiprocessing.Value,
    multiprocessing.Value,
    multiprocessing.Value,
]


@dataclass
class BackpressureStats:
    """Current backpressure statistics for monitoring."""

    pending_jobs: int
    pending_bytes_mb: float
    max_pending_jobs: int
    max_pending_mb: float  # current EFFECTIVE cap (adaptive), not the static ceiling
    is_throttled: bool
    # Cumulative, increment-only run totals (reset once at acquisition start).
    captured_bytes: int = 0
    written_bytes: int = 0
    captured_count: int = 0
    written_count: int = 0
    # Smoothed live throughput (MB/s). ratio < ~1 means the backlog is growing.
    capture_mb_s: float = 0.0
    write_mb_s: float = 0.0


class RateSampler:
    """Turns cumulative byte counters into smoothed capture/write MB/s rates.

    Not thread-safe on its own; the owning BackpressureController serializes access
    on the main thread. Uses an exponential moving average so the ratio and the
    adaptive cap don't oscillate on a single slow FOV.
    """

    def __init__(self, alpha: float = 0.3, min_dt_s: float = 0.05):
        self._alpha = alpha
        self._min_dt_s = min_dt_s
        self.reset()

    def reset(self) -> None:
        self._last_t: Optional[float] = None
        self._last_captured = 0
        self._last_written = 0
        self._capture_mb_s = 0.0
        self._write_mb_s = 0.0

    @property
    def capture_mb_s(self) -> float:
        return self._capture_mb_s

    @property
    def write_mb_s(self) -> float:
        return self._write_mb_s

    def update(self, captured_bytes: int, written_bytes: int, now: float) -> None:
        """Fold a new counter sample into the smoothed rates.

        First call seeds the baseline silently (rates stay 0). Samples closer than
        min_dt_s are ignored (protects against timer jitter). Negative deltas are
        clamped to 0 (defends against the run-start reset and teardown drop-to-zero).
        """
        if self._last_t is None:
            self._last_t = now
            self._last_captured = captured_bytes
            self._last_written = written_bytes
            return

        dt = now - self._last_t
        if dt <= self._min_dt_s:
            return  # too soon; keep the last smoothed value

        d_captured = max(0, captured_bytes - self._last_captured)
        d_written = max(0, written_bytes - self._last_written)
        inst_capture = (d_captured / _BYTES_PER_MB) / dt
        inst_write = (d_written / _BYTES_PER_MB) / dt

        self._capture_mb_s = (1.0 - self._alpha) * self._capture_mb_s + self._alpha * inst_capture
        self._write_mb_s = (1.0 - self._alpha) * self._write_mb_s + self._alpha * inst_write

        self._last_t = now
        self._last_captured = captured_bytes
        self._last_written = written_bytes


def create_backpressure_values() -> BackpressureValues:
    """Create multiprocessing primitives for cross-process backpressure tracking.

    Returns:
        7-tuple (pending_jobs, pending_bytes, capacity_event,
                 captured_bytes, written_bytes, captured_count, written_count)

    These values should be:
    1. Passed to JobRunner at construction (subprocess uses them)
    2. Passed to BackpressureController at construction (main process uses them)

    The values are process-safe and can be shared between main process and subprocess.

    Cleanup: The returned values don't need explicit cleanup. They're garbage collected
    when all references are dropped (after JobRunner.shutdown() and BackpressureController.close()).
    """
    pending_jobs = multiprocessing.Value("i", 0)
    pending_bytes = multiprocessing.Value("q", 0)
    capacity_event = multiprocessing.Event()
    captured_bytes = multiprocessing.Value("q", 0)
    written_bytes = multiprocessing.Value("q", 0)
    captured_count = multiprocessing.Value("q", 0)
    written_count = multiprocessing.Value("q", 0)
    return (
        pending_jobs,
        pending_bytes,
        capacity_event,
        captured_bytes,
        written_bytes,
        captured_count,
        written_count,
    )


class BackpressureController:
    """Manages backpressure across multiple job runners.

    Uses multiprocessing-safe shared values for cross-process tracking.

    Usage:
        # Option 1: Let controller create its own values (no pre-warming)
        controller = BackpressureController(max_jobs=10, max_mb=500)
        runner = JobRunner(
            bp_pending_jobs=controller.pending_jobs_value,
            bp_pending_bytes=controller.pending_bytes_value,
            bp_capacity_event=controller.capacity_event,
            bp_captured_bytes=controller.captured_bytes_value,
            bp_written_bytes=controller.written_bytes_value,
            bp_captured_count=controller.captured_count_value,
            bp_written_count=controller.written_count_value,
        )

        # Option 2: Use pre-created values (for pre-warming)
        bp_values = create_backpressure_values()
        runner = JobRunner(bp_pending_jobs=bp_values[0], ...)
        runner.start()  # Pre-warm
        # Later...
        controller = BackpressureController(max_jobs=10, bp_values=bp_values)

    Adaptive cap:
        When target_backlog_s > 0, the effective byte cap is
        clamp(write_mb_s * target_backlog_s, floor_mb, max_mb). This bounds the
        RAM backlog to ~target_backlog_s of measured write throughput; max_mb
        becomes an absolute safety ceiling. When target_backlog_s == 0 the cap is
        the static max_mb (legacy behavior).

    Thread Safety:
        - All public methods are thread-safe (use locks on shared values)
        - close() can be called from any thread and wakes threads in wait_for_capacity()
    """

    # How often (wall-clock) to re-read the cumulative counters and refresh the rate
    # estimate. Bounds the cost of should_throttle() being called in a tight loop.
    _RATE_REFRESH_INTERVAL_S = 0.25

    def __init__(
        self,
        max_jobs: int = 10,
        max_mb: float = 500.0,
        timeout_s: float = 30.0,
        enabled: bool = True,
        # Adaptive cap params. target_backlog_s == 0 disables adaptation (static cap).
        target_backlog_s: float = 0.0,
        floor_mb: float = 0.0,
        # Pre-created backpressure values for sharing with pre-warmed JobRunner.
        # If provided, uses these instead of creating new ones.
        bp_values: Optional[BackpressureValues] = None,
    ):
        self._enabled = enabled
        self._max_jobs = max_jobs
        self._max_bytes = int(max_mb * _BYTES_PER_MB)  # absolute ceiling
        self._timeout_s = timeout_s
        self._target_backlog_s = target_backlog_s
        self._floor_bytes = int(floor_mb * _BYTES_PER_MB)
        self._closed = False  # Lifecycle tracking

        # Live throughput estimate (main-thread only).
        self._rate_sampler = RateSampler()
        self._last_refresh = 0.0

        # Use provided values or create new ones
        if bp_values is not None:
            (
                self._pending_jobs,
                self._pending_bytes,
                self._capacity_event,
                self._captured_bytes,
                self._written_bytes,
                self._captured_count,
                self._written_count,
            ) = bp_values
        else:
            self._pending_jobs = multiprocessing.Value("i", 0)
            self._pending_bytes = multiprocessing.Value("q", 0)
            self._capacity_event = multiprocessing.Event()
            self._captured_bytes = multiprocessing.Value("q", 0)
            self._written_bytes = multiprocessing.Value("q", 0)
            self._captured_count = multiprocessing.Value("q", 0)
            self._written_count = multiprocessing.Value("q", 0)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def is_closed(self) -> bool:
        """True if close() has been called."""
        return self._closed

    def _warn_if_closed(self, method_name: str) -> bool:
        """Log warning if controller is closed. Returns True if closed."""
        if self._closed:
            log.warning(f"BackpressureController.{method_name}() called after close()")
            return True
        return False

    @property
    def pending_jobs_value(self) -> Optional[multiprocessing.Value]:
        """Shared value for pending jobs (pass to JobRunner). None after close()."""
        return self._pending_jobs

    @property
    def pending_bytes_value(self) -> Optional[multiprocessing.Value]:
        """Shared value for pending bytes (pass to JobRunner). None after close()."""
        return self._pending_bytes

    @property
    def capacity_event(self) -> Optional[multiprocessing.Event]:
        """Event signaled when capacity becomes available. None after close()."""
        return self._capacity_event

    @property
    def captured_bytes_value(self) -> Optional[multiprocessing.Value]:
        """Cumulative captured-bytes counter (pass to JobRunner). None after close()."""
        return self._captured_bytes

    @property
    def written_bytes_value(self) -> Optional[multiprocessing.Value]:
        """Cumulative written-bytes counter (pass to JobRunner). None after close()."""
        return self._written_bytes

    @property
    def captured_count_value(self) -> Optional[multiprocessing.Value]:
        """Cumulative captured-job counter (pass to JobRunner). None after close()."""
        return self._captured_count

    @property
    def written_count_value(self) -> Optional[multiprocessing.Value]:
        """Cumulative written-job counter (pass to JobRunner). None after close()."""
        return self._written_count

    @staticmethod
    def _read_value(v: Optional[multiprocessing.Value]) -> int:
        """Read a shared Value under its lock; 0 if the ref is None (closed)."""
        if v is None:
            return 0
        with v.get_lock():
            return v.value

    def get_pending_jobs(self) -> int:
        pending_jobs = self._pending_jobs
        if pending_jobs is None:
            return 0
        with pending_jobs.get_lock():
            return pending_jobs.value

    def get_pending_mb(self) -> float:
        pending_bytes = self._pending_bytes
        if pending_bytes is None:
            return 0.0
        with pending_bytes.get_lock():
            return pending_bytes.value / _BYTES_PER_MB

    @property
    def write_mb_s(self) -> float:
        """Latest smoothed write throughput (MB/s). Refreshes lazily."""
        self._refresh_rates()
        return self._rate_sampler.write_mb_s

    @property
    def capture_mb_s(self) -> float:
        """Latest smoothed capture throughput (MB/s). Refreshes lazily."""
        self._refresh_rates()
        return self._rate_sampler.capture_mb_s

    def _refresh_rates(self) -> None:
        """Re-read cumulative counters and fold them into the rate sampler.

        Wall-clock throttled to _RATE_REFRESH_INTERVAL_S so a tight should_throttle()
        loop doesn't hammer the shared-Value locks.
        """
        now = time.monotonic()
        if now - self._last_refresh < self._RATE_REFRESH_INTERVAL_S:
            return
        self._last_refresh = now
        captured = self._captured_bytes
        written = self._written_bytes
        if captured is None or written is None:
            return
        self._rate_sampler.update(self._read_value(captured), self._read_value(written), now)

    def _effective_max_bytes(self) -> int:
        """Current adaptive byte cap.

        Bounds the backlog to ~target_backlog_s of measured write throughput,
        clamped between the warmup floor and the absolute ceiling. With
        target_backlog_s == 0 (adaptation off) this is just the static ceiling.
        """
        if self._target_backlog_s <= 0:
            return self._max_bytes
        target = self._rate_sampler.write_mb_s * _BYTES_PER_MB * self._target_backlog_s
        return int(min(self._max_bytes, max(self._floor_bytes, target)))

    def should_throttle(self) -> bool:
        """Check if acquisition should wait (jobs limit or effective byte cap exceeded)."""
        if not self._enabled:
            return False

        # Capture references to avoid race with close()
        pending_jobs = self._pending_jobs
        pending_bytes = self._pending_bytes

        # Guard against closed state (values set to None)
        if pending_jobs is None or pending_bytes is None:
            return False

        self._refresh_rates()
        effective_max_bytes = self._effective_max_bytes()

        with pending_jobs.get_lock():
            jobs_over = pending_jobs.value >= self._max_jobs
        with pending_bytes.get_lock():
            bytes_over = pending_bytes.value >= effective_max_bytes

        return jobs_over or bytes_over

    def wait_for_capacity(self, should_abort: Optional[Callable[[], bool]] = None) -> bool:
        """Wait until capacity available or timeout. Returns True if got capacity.

        Args:
            should_abort: Optional predicate polled while waiting. If it returns True, the wait
                exits immediately (returns False) instead of blocking for the full timeout. Used
                so a stop/abort request doesn't sit through a throttle pause waiting for capacity
                to launch a frame we're about to abandon anyway.
        """
        if self._warn_if_closed("wait_for_capacity"):
            return True  # Don't block on closed controller
        if not self._enabled or not self.should_throttle():
            return True
        if should_abort is not None and should_abort():
            return False

        log.info(
            f"Backpressure throttling: jobs={self.get_pending_jobs()}/{self._max_jobs}, "
            f"MB={self.get_pending_mb():.1f}/{self._effective_max_bytes() / _BYTES_PER_MB:.1f} "
            f"(write={self._rate_sampler.write_mb_s:.1f} MB/s)"
        )

        deadline = time.monotonic() + self._timeout_s
        while self.should_throttle():
            if should_abort is not None and should_abort():
                log.info("Backpressure wait aborted by stop request")
                return False
            if time.monotonic() > deadline:
                log.warning(f"Backpressure timeout after {self._timeout_s}s, continuing")
                return False
            # Capture reference to avoid race with close()
            event = self._capacity_event
            if event is None:
                break  # Controller was closed
            # Clear stale signals, then re-check condition before waiting.
            # If capacity frees between clear() and wait(), should_throttle()
            # returns False and we exit without blocking.
            event.clear()
            if self.should_throttle():
                event.wait(timeout=0.1)

        log.debug("Backpressure released")
        return True

    def job_dispatched(self, image_bytes: int) -> None:
        """Manually increment backpressure counters (pending + cumulative captured).

        Primarily for testing. In production, JobRunner automatically increments
        counters when dispatch() is called.

        No-op if controller is disabled or closed.
        """
        if not self._enabled:
            return
        # Capture references to avoid race with close()
        pending_jobs = self._pending_jobs
        pending_bytes = self._pending_bytes
        if pending_jobs is None or pending_bytes is None:
            return
        with pending_jobs.get_lock():
            pending_jobs.value += 1
        with pending_bytes.get_lock():
            pending_bytes.value += image_bytes
        if self._captured_bytes is not None:
            with self._captured_bytes.get_lock():
                self._captured_bytes.value += image_bytes
        if self._captured_count is not None:
            with self._captured_count.get_lock():
                self._captured_count.value += 1

    def job_completed(self, image_bytes: int) -> None:
        """Manually decrement pending and increment cumulative written counters.

        Primarily for testing (mirrors JobRunner's completion finally block).
        No-op if controller is disabled or closed.
        """
        if not self._enabled:
            return
        pending_jobs = self._pending_jobs
        pending_bytes = self._pending_bytes
        if pending_jobs is None or pending_bytes is None:
            return
        with pending_jobs.get_lock():
            pending_jobs.value = max(0, pending_jobs.value - 1)
        with pending_bytes.get_lock():
            pending_bytes.value = max(0, pending_bytes.value - image_bytes)
        if self._written_bytes is not None:
            with self._written_bytes.get_lock():
                self._written_bytes.value += image_bytes
        if self._written_count is not None:
            with self._written_count.get_lock():
                self._written_count.value += 1

    def get_stats(self) -> BackpressureStats:
        """Get a snapshot of backpressure state (pending, effective cap, cumulative, rates)."""
        # Capture references to avoid race with close()
        pending_jobs = self._pending_jobs
        pending_bytes = self._pending_bytes

        if pending_jobs is None or pending_bytes is None:
            # Controller is closed, return zeroed stats
            return BackpressureStats(
                pending_jobs=0,
                pending_bytes_mb=0.0,
                max_pending_jobs=self._max_jobs,
                max_pending_mb=self._max_bytes / _BYTES_PER_MB,
                is_throttled=False,
                captured_bytes=0,
                written_bytes=0,
                captured_count=0,
                written_count=0,
                capture_mb_s=0.0,
                write_mb_s=0.0,
            )

        self._refresh_rates()
        effective_max_bytes = self._effective_max_bytes()

        # Acquire the pending pair for an atomic pending snapshot.
        # Lock ordering: pending_jobs before pending_bytes (consistent throughout module).
        with pending_jobs.get_lock():
            jobs = pending_jobs.value
            jobs_over = jobs >= self._max_jobs
            with pending_bytes.get_lock():
                bytes_val = pending_bytes.value
                bytes_over = bytes_val >= effective_max_bytes

        # Cumulative counters are independent monotonic values; read each under its own
        # lock OUTSIDE the pending pair so we don't introduce a new nested lock edge.
        captured_bytes = self._read_value(self._captured_bytes)
        written_bytes = self._read_value(self._written_bytes)
        captured_count = self._read_value(self._captured_count)
        written_count = self._read_value(self._written_count)

        return BackpressureStats(
            pending_jobs=jobs,
            pending_bytes_mb=bytes_val / _BYTES_PER_MB,
            max_pending_jobs=self._max_jobs,
            max_pending_mb=effective_max_bytes / _BYTES_PER_MB,
            is_throttled=self._enabled and (jobs_over or bytes_over),
            captured_bytes=captured_bytes,
            written_bytes=written_bytes,
            captured_count=captured_count,
            written_count=written_count,
            capture_mb_s=self._rate_sampler.capture_mb_s,
            write_mb_s=self._rate_sampler.write_mb_s,
        )

    def reset(self) -> None:
        """Reset counters and the rate estimate (call at acquisition start).

        Resets both the net pending counters and the cumulative run-total counters,
        plus the rate sampler, so throughput is measured fresh per run.

        WARNING: Only call when no jobs are pending. If jobs complete after reset,
        counters will go negative, which breaks throttling logic.
        """
        if self._warn_if_closed("reset"):
            return
        # Capture references to avoid race with close()
        pending_jobs = self._pending_jobs
        pending_bytes = self._pending_bytes
        if pending_jobs is None or pending_bytes is None:
            return

        # Check for pending jobs - warn if resetting with jobs in flight
        with pending_jobs.get_lock():
            current_jobs = pending_jobs.value
            if current_jobs > 0:
                log.warning(
                    f"Backpressure reset() called with {current_jobs} jobs pending. "
                    f"This may cause counter underflow."
                )
            pending_jobs.value = 0
        with pending_bytes.get_lock():
            pending_bytes.value = 0

        # Reset cumulative run totals (done once at run start, before any jobs, so the
        # rate sampler measures per-run throughput rather than lifetime totals).
        for counter in (self._captured_bytes, self._written_bytes, self._captured_count, self._written_count):
            if counter is not None:
                with counter.get_lock():
                    counter.value = 0
        self._rate_sampler.reset()
        self._last_refresh = 0.0

    def close(self) -> None:
        """Release references to multiprocessing resources.

        Signals the capacity event to wake any threads blocked in wait_for_capacity(),
        then clears local references to allow garbage collection.

        Thread Safety: This method is safe to call from any thread. It captures
        references before use to avoid TOCTOU races with concurrent calls.

        This method is idempotent - safe to call multiple times.
        """
        if self._closed:
            return  # Already closed

        self._closed = True

        # Capture references first to avoid TOCTOU race
        pending_jobs = self._pending_jobs
        capacity_event = self._capacity_event

        if pending_jobs is None:
            return  # Values already cleared

        # Signal capacity event to wake any threads blocked in wait_for_capacity().
        # This must happen BEFORE clearing references to prevent AttributeError
        # when the woken thread calls should_throttle().
        if capacity_event is not None:
            try:
                capacity_event.set()
            except Exception as e:
                log.debug(f"Could not set capacity event during close (may be invalid): {e}")

        # Clear local references to allow GC
        self._pending_jobs = None
        self._pending_bytes = None
        self._capacity_event = None
        self._captured_bytes = None
        self._written_bytes = None
        self._captured_count = None
        self._written_count = None
