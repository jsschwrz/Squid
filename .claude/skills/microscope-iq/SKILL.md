---
name: microscope-iq
description: Analyse a Squid/SquidXplorer acquisition output folder for installation-qualification (IQ) and performance metrics — laser reflection AF lock rate and focus accuracy, throughput and slowdown detection, and target-vs-actual XYZ stage discrepancy. Use whenever asked to run a diagnostic, QC, IQ, OQ, soak-test, or performance analysis on an acquisition dataset, acquisition.log, or coordinates.csv; or to check autofocus performance, frame rate, acquisition timing, focus drift, focus-map quality, or stage positioning accuracy.
---

# Microscope IQ / performance analysis

Turns a Squid acquisition output folder into a defensible IQ report. The bundled
script does the arithmetic; this file tells you how to read the result and where
the traps are.

## Run it

```bash
python <skill_dir>/scripts/analyze_acquisition.py <dataset_dir> -o <outdir>
```

Writes `summary.json` plus `af_per_fov.csv`, `timing_per_fov.csv`,
`discrepancy.csv`, `focus_quality.csv`. Prints `summary.json` to stdout (`--quiet`
to suppress). Needs numpy + pandas; pyyaml optional.

**Interpreter:** use a venv that has numpy + pandas — on Windows, bare `python`
often resolves to the Microsoft Store stub and fails. The SquidXplorer checkout's
`.venv\Scripts\python.exe` (Python 3.12) works. Any 3.9+ environment with numpy
and pandas will do; pyyaml is optional but gives richer config parsing.

Expected dataset layout:

```
<dataset>/
  acquisition.log              # the primary source; everything else is corroboration
  acquisition.yaml             # authoritative config (objective, NA, channels, z-stack, regions)
  acquisition parameters.json  # older/redundant config
  coordinates.csv              # TARGET x,y per FOV  (z column is often EMPTY)
  0/coordinates.csv            # ACTUAL x,y,z per (region, fov, z_level) — one dir per timepoint
  0/*.tiff
```

## Reporting

Write a markdown report next to the CSVs. Structure it around whatever the
requester actually asked; the default three sections are AF performance,
throughput, positioning discrepancy. Lead each with the headline number, then
the evidence, then the caveat. Always state which numbers came from the log
versus the CSVs. Finish with a severity-ranked findings table.

Reference examples (repo-relative), both worth reading before writing a new one:
- `QCMetric/IQ_SoakTest_Results.md` — the 2026-07-30 four-corners soak test this
  skill came from
- `QCMetric/IQ_NewAF_Results.md` — the same grid re-run after the connected-components
  AF merge, showing how to structure a **comparison** report and how to be explicit
  about confounded attribution when more than one thing changed between runs

## How to read the results

### Laser AF — do not stop at the lock rate

The AF sequence is **measure → move → verify → accept or revert**
(`software/control/core/laser_auto_focus_controller.py`, `move_to_target`).
On verification failure the code calls `_move_z(-um_to_move)` and **backs the
stage out**, then `MultiPointWorker` images at the uncorrected focus-map z.

So a "failure" is not simply a missed lock — it can be a *successful* lock that
was thrown away. The script separates these:

- **`false_rejections`** — post-move spot centroid returned to within 1 µm of the
  reference, but the correlation gate vetoed it. The AF worked; the software
  discarded it. These leave the FOV *worse* than accepting would have.
- **`genuine_failures`** — the spot never converged. Look at the spot centroid:
  the AF spot should translate **only in x**. A y-shift of more than ~2 px means a
  corrupted reflection (bubble, debris, meniscus), not a focus error.

The gate is `correlation_threshold` in the objective's `laser_af_settings.json`.
Docs default is 0.9; the repo's own backup configs ship 0.7. The script reports
`threshold_bracket` — the observed [worst pass, best fail] window — which
recovers the active value without reading the config.

Judge focus quality by **`focus_error_um`** (residual after correction for locked
FOV, full uncorrected defocus for failed FOV), not by the lock rate. Compare it to:
- half the depth of field (`dof_full_um / 2`) — the in-focus criterion
- the full z-stack range (`z_stack_range_um`) — beyond this the stack contains
  no in-focus plane at all, which is the finding that actually matters

### Tilt vs. drift — the trap

Rising defocus across a region has two candidate causes: **residual focus-map
tilt** (fixed to position) or **thermal drift** (fixed to time). In a raster scan
these are collinear, and R² alone cannot separate them. The script reports both
fits, adjusted R², and a `verdict` that is honest about the confound.

The discriminator that does work: compare the fitted gradient's **isotropy**
against the **visit-rate anisotropy**. Within a row, x-steps take one FOV cycle;
row-to-row, y-steps take a whole row. If defocus were pure time drift, the
apparent dz/dx and dz/dy would differ by that same ratio. If they are comparable
while the visit rates differ several-fold, it is tilt.

When defocus varies only along the slow axis, the data genuinely cannot decide —
say so rather than guessing.

**The cheapest way to resolve it: re-acquire the same geometry at a different time
scale.** Drop to 1 channel × 1 z and the run gets ~15-20x faster while every FOV
position stays identical — laser AF fires once per FOV regardless of channels or z,
so you lose no AF data. Tilt is fixed to position and reproduces the same
peak-to-peak; drift is fixed to time and shrinks in proportion to the duration.
This settled A1/A6/D6 on this instrument: 374 s -> 22 s per region, tilt unchanged
at 15.2 -> 15.6 um. Compare `tilt_pp_um` and `drift_total_um_over_region` between
the two runs, not `drift_um_per_min` -- the per-minute rate is an extrapolation and
is inflated on short runs.

(Re-acquiring in reverse FOV order also works -- tilt reproduces, drift inverts --
but it needs a config change the GUI may not expose, so try the time-scale test first.)

A region with low adj R² on *both* fits is neither tilt nor drift — that is a flat
focus-map *offset* plus scatter, and a tilt correction will not help it. The script
emits an explicit `neither (both fits weak...)` verdict below
`MIN_ADJ_R2_FOR_VERDICT` rather than declaring whichever model happened to fit
marginally better; do not read a winner out of two failing fits.

### Throughput

`acquire_camera_image` is per frame; `acquire_at_position` is per z-position and
covers the whole channel loop. Frame count = z-positions × channels; the
`Acquiring image: ID=` lines are logged once per **z-position**, not per frame, so
never use their count as the frame count.

Check for slowdown three ways, all reported under `slowdown`: first-half vs
second-half `acquire_at_position`, a linear fit of per-frame time across the run,
and intra-FOV image intervals against their own median. Runs often get *faster*
(caches warm, display throttles) — report that plainly rather than hunting for a
regression that is not there.

Compare `duty_cycle_pct` against configured exposure. A large fixed per-frame
overhead that does **not** scale with the exposure setting points at the
trigger/readout path, not at the camera. Check whether the overhead is uniform
across channel slots — uniformity across differing exposure times is what proves
it is fixed cost.

Isolate a slow FOV by summing timers inside its window. Elevated `_image_callback`
with nominal camera time means write-side backpressure, not an optical problem.

### Positioning

`coordinates.csv` (targets) frequently ships with an **empty z column** — take Z
targets from the log's `moving to coordinate` entries instead, and say so in the
report.

XY errors that take only a handful of discrete values are **stage step
quantisation**, not positioning error. Express the worst case in pixels via
`pixel_size_um` to make it interpretable.

Z discrepancy inverts the usual reading. Split it by AF outcome:
- **AF locked → large dz is correct.** dz *is* the correction applied against the
  focus map, and should equal −(measured defocus).
- **AF failed → dz ≈ 0.** No correction survived; the stack sits on the raw
  focus-map plane carrying its full defocus.

Reporting an undifferentiated mean |dz| hides both facts. Always split.

Also check z-step fidelity within stacks (should be exactly the encoder quantum,
zero variance) and XY stability within a stack (should be exactly zero).

## Extending the suite

Keep each new probe as a separate script under `scripts/` with the same contract:
take a dataset dir, write CSVs plus a JSON block, print nothing else. Candidate
additions:

- **Reverse-order rescan** — resolves tilt vs. drift (see above)
- **Image-based focus scoring** — Brenner/Tenengrad per z-slice to find the true
  best-focus plane, validating the laser AF against the images rather than trusting
  its own displacement number
- **Illumination uniformity / flat-field** — per-channel corner-vs-centre intensity
- **Channel registration** — inter-channel XY shift via phase correlation
- **Stage repeatability** — revisit the same FOV N times, measure spread
- **Photobleaching** — intensity vs. z-index and vs. time within a channel
- **Long-soak trending** — run this script across many datasets, plot lock rate,
  duty cycle, and focus error over days

## Gotchas

- The log is UTF-8 with µ/μ characters; read with `errors="replace"`. Never match
  on the micro sign.
- `squid.Timer` and `squid.Microcontroller` DEBUG lines dominate (~78% of a 20 MB
  log). Filter by logger before doing anything expensive.
- `Cross correlation check failed` is logged **twice** per failure (lines 555 and
  379). Count FOV outcomes, not log lines.
- Per-region AF offsets (`applying per-region laser AF offset`) are applied only
  after a successful lock — a failed FOV misses its offset too.
- Multiple timepoints appear as sibling numeric dirs (`0/`, `1/`, …). The script
  reads config and coordinates from the first; extend it for time-series work.
