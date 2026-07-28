# Acquisition write path — state and next steps

Working notes for the acquisition write-path work (PR #2). Written 2026-07-28, after
four faults were fixed and validated on hardware. Read this before picking the work back
up.

---

## Where things stand

The acquisition loop is **camera-bound and healthy**. The save path is **not**, and there
is one known data-loss path left.

Measured on a Toupcam system (6224×4168 sensor → 51.9 MB/frame captured, cropped to
4168×4168 ≈ 34.7 MB saved), 4 FOVs × 32 z-levels, `acquisition_writer_processes = 4`:

| | Before the fixes | Now |
|---|---|---|
| Time blocked on backpressure | 302.7 s (68% of a 443 s run) | 0 s |
| Per-image acquisition time | — | 0.396 s (0.327 s of it camera exposure) |
| Writers actually doing work | 1 of 4 | 4 of 4 |
| Blank planes in output | 34 of 148 | 0 of 128 |

Fixed already (see PR #2 for the detail): writer warmup blocking the frame-delivery path,
the adaptive cap starving its own parallel writers, `crc32` routing leaving half the
writers idle, and the Qt6/Windows dark-mode contrast regression. Upstream's
progress-based drain is merged.

---

## Next task: bound the writer input queue

**Goal:** make `dispatch()` wait when a writer is behind, instead of buffering without
limit inside the parent process.

### Why

`JobRunner` builds `multiprocessing.Queue()` with no `maxsize`
(`control/core/job_processing.py`, queue construction in `__init__`). `put_nowait()` only
appends to an in-process deque and hands the payload to a feeder thread — it does not
wait for the child. Measured at the real payload size:

```
20 × 52 MB put_nowait()  ->  all returned in 0.36 s
                             child had actually received: 0 / 20
                             ~1 GB sitting in the parent, invisible to the child
```

Reproduce with a consumer process that `get()`s with a ~120 ms per-item cost, and compare
how long the puts take against how many items the child has actually received.

Two consequences:

1. **`pending_count()` overstates deliverable work.** It counts dispatched-minus-completed,
   which includes images that never left the parent. The drain's stall detector waits for
   *completions*, so a runner whose data is merely undelivered looks stalled and is killed
   at `stall_timeout_s`. The signature is a runner that logs **nothing at all** — not even
   "Running job" — because it is idle on an empty pipe, not hung.
2. **`pending_bytes` is not bounding the RAM it claims to.** The bytes are in the parent's
   feeder buffers, not where the accounting implies. Note the `min_span_images` fix
   deliberately raises the cap, which *enlarges* this window; re-measure peak RSS once the
   queue is bounded.

This is also the leading explanation for the remaining throughput gap.
`tools/benchmark_write_paths.py` measures **449 MB/s at 4 writers** on the same volume the
acquisition writes to, while an acquisition sustains roughly **20 MB/s** — a ~20× gap that
is not the disk. Individual write jobs range from 96 ms to 15.8 s for identical operations
on identically-sized files; that spread is queuing behaviour, not a disk at its limit.

### The trap

`dispatch()` currently treats a failed enqueue as an error, and the caller in
`multi_point_worker._image_callback` responds with `request_abort_fn()`. A naive `maxsize`
therefore converts "writer briefly behind" into "aborted acquisition".

Dispatch must **block with a timeout** (or backpressure must be guaranteed to throttle
before the queue can fill) — not fail fast. Note the counter increments and their rollback
path in `dispatch()` assume the enqueue is instantaneous; revisit them together with the
blocking behaviour.

### Suggested verification

- Re-run the probe above with the bound in place; the puts should now pace to the consumer.
- Compare peak RSS before/after on the same acquisition shape.
- Confirm end-to-end MB/s moves toward the benchmark figure for the same path and writer
  count.
- Confirm a slow writer throttles the acquisition instead of aborting it.

---

## Other open items

| Item | Notes |
|---|---|
| `mosaic_mosaic_0um.ome.tiff` written entirely blank | 5662×4855, all planes zero. Unrelated to the write path; suspect the performance-mode deferred render never flushes. Not investigated. |
| 6D Zarr with `writer_processes > 1` never run on hardware | The region-only routing path is unit-tested but has not been exercised against real Zarr output. This is the case where wrong routing corrupts a shared array, so it deserves a real run before anyone relies on it. |
| Dark theme selectable but not usable | ~15 buttons hardcode light backgrounds with no text colour. `GUI_COLOR_SCHEME` in `_def.py` is pinned to `light`; making `dark` genuinely usable means giving those an explicit colour. |
| `gui_hcs.py` sets `QT_API` after its first imports | Works only because `main_hcs.py` sets it at line 6 first. Anything importing `gui_hcs` directly binds whatever Qt it finds. Worth hardening. |
| MCU log noise | Hundreds of `received ack for command N, but waiting for N+1` per run. Self-resolving, but not understood. |

---

## Running and testing

The GUI runs on **Python 3.14 / PyQt6** on the validation machine:

```powershell
cd software
py -3.14 main_hcs.py
```

`py -3.10` fails with `QtBindingsNotFoundError` on any branch carrying the PyQt6
migration — 3.10 still has PyQt5 and is kept deliberately as the rollback environment.
Do not install PyQt6 into it; napari breaks when both bindings are present.

Test command, and the expected baseline on Windows:

```powershell
py -3.14 -m pytest tests/control/core/ tests/control/test_performance_mode.py -q `
  --deselect tests/control/core/test_backpressure.py::TestJobRunnerBackpressureTracking
```

- Expect **273 passed, 1 failed**. `test_has_pending_with_none_result_job` fails on
  unmodified `master` too.
- The deselected `TestJobRunnerBackpressureTracking` (5 tests) **hangs** under Windows
  `spawn`, on unmodified `master` as well. It runs normally in CI on Linux.
- `test_has_pending_tracks_multiple_jobs` flakes under full-suite load only; it passes in
  isolation.

Confirm any *new* failure against `origin/master` before treating it as a regression.

---

## Tuning on a new machine

Run the benchmark before assuming anything about writer count or save path:

```powershell
py -3.14 tools\benchmark_write_paths.py --width <crop_w> --height <crop_h> --no-apply
```

On the validation machine it showed the system NVMe at 201 MB/s (1 writer) rising to
449 MB/s (4 writers), with 8 writers buying nothing, while the attached HDD and USB
volumes managed 6–26 MB/s and got *worse* with more writers. Do not assume more writers
or a bigger drive is faster — measure. See `docs/write-path-benchmark.md`.
