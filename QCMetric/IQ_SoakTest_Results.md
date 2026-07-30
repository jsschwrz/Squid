# IQ Soak Test — Four Corners, 5 Channel, 10 Z

**Dataset:** `SetUp_SoakTest_FourCorners_5Channel_10Z_2026-07-30_16-20-13.274866`
**Acquired:** 2026-07-30 16:20:13 → 16:43:26
**Analysed:** 2026-07-30 · `microscope-iq` skill, `analyze_acquisition.py`
**Log lines parsed:** 136,683

## Configuration

| | |
|---|---|
| Objective | 10x / 0.3 NA, 0.376 µm/px (3.76 µm sensor) |
| Sample | 24-well plate, regions A1 / A6 / D6 / D1 |
| Grid | 56 FOV (A1 16, A6 12, D6 12, D1 16) |
| Channels | 5 — 730/638/561/488 nm @ 150 ms, 405 nm @ 100 ms, all filter pos. 1 |
| Z-stack | Nz = 10, dz = 0.469 µm, FROM BOTTOM (range 4.69 µm) |
| Timepoints | 1 |
| Autofocus | Reflection (laser) AF **on**, contrast AF **off**, manual focus map **on** |
| Output | 2,800 TIFF, 97.29 GB |

---

## 1. Laser AF performance

### Headline

**38 of 56 FOV confirmed lock — 67.9%.** 18 failed (32.1%). Every FOV attempted AF; no skips.

The pass/fail gate is a normalised cross-correlation of the post-move spot crop against the stored reference crop (`_verify_spot_alignment`, `laser_auto_focus_controller.py:549`). The empirically observed threshold brackets **[0.896, 0.901]**, consistent with `correlation_threshold: 0.9` documented in `software/docs/configuration-system.md:462`.

| Region | FOV | Locked | Rate | mean xcorr | mean defocus | region offset |
|---|---|---|---|---|---|---|
| A1 | 16 | 12 | 75.0% | 0.924 | +7.09 µm | +0.21 µm |
| A6 | 12 | 7 | 58.3% | 0.857 | −5.82 µm | −1.02 µm |
| D6 | 12 | 10 | 83.3% | 0.939 | −2.51 µm | −1.39 µm |
| D1 | 16 | 9 | 56.3% | 0.904 | −5.25 µm | −2.75 µm |

### Primary finding — 17 of 18 failures are false rejections

The AF sequence is *measure → move → verify → accept or revert*. On a correlation failure the code calls `_move_z(-um_to_move)` and backs the stage out (`laser_auto_focus_controller.py:378-382`).

Reconstructing the post-move spot centroid for every failed FOV shows the stage **had already converged in 17 of 18 cases** — the verification spot returned to the reference x within ±0.4 px (±0.8 µm) — and the correlation gate vetoed it anyway.

| | n | post-move residual (mean abs) | z delivered |
|---|---|---|---|
| Locked | 38 | 0.24 µm | corrected |
| Failed but converged | 17 | 0.20 µm | **reverted; left 4.75 µm off, max 11.3 µm** |
| Failed, genuine | 1 | 67.6 µm | reverted (correct behaviour) |

Spot-position scale derived from the data: **1.968 µm per pixel** of spot x-shift; reference spot x = 774.9 px.

### The one genuine failure — A6 fov 1 (110.09, 8.23)

Spot found at (751.8, 133.8), then (740.6, 138.5) after the move, against a reference of (774.9, 128.2). The **+10 px y-shift is diagnostic**: a reflection AF spot should translate only in x, so this is a corrupted or mis-identified reflection — bubble, debris, or meniscus. Reported −45.5 µm at xcorr 0.135. The gate did its job here.

### Consequence for the data

| Metric | Value |
|---|---|
| Residual focus error, locked FOV | **0.24 µm mean abs** (max 1.38) |
| Uncorrected defocus, failed FOV | **7.02 µm mean abs** (max 45.5) |
| Depth of field, 10x/0.3 (λ≈0.55 µm) | ±1.24 µm (full 2.48 µm) |
| Failed FOV exceeding half-DOF | **15 / 18** |
| Failed FOV exceeding the full 4.69 µm z-stack | **11 / 18** |

Those 11 stacks likely contain **no in-focus plane at all** — a worse outcome than accepting the AF correction would have produced.

### Defocus structure: tilt vs. drift

Fitting measured defocus against position (plane) and against elapsed time (linear), per region, excluding the A6 spot failure:

| Region | dz/dx | dz/dy | adj R² plane | drift | adj R² time | p-p | verdict |
|---|---|---|---|---|---|---|---|
| A1 | +0.956 µm/mm | +0.997 µm/mm | **0.921** | +1.82 µm/min | 0.680 | 15.2 µm | **tilt** |
| A6 | +0.043 | +0.639 | 0.962 | +1.24 µm/min | 0.917 | 6.0 µm | collinear |
| D6 | +0.067 | +0.778 | 0.977 | +1.58 µm/min | 0.946 | 7.3 µm | collinear |
| D1 | −0.054 | +0.371 | 0.203 | +0.54 µm/min | 0.212 | 4.1 µm | neither |

**A1 is decisively residual tilt.** Its gradient is isotropic — ~1 µm/mm in both x and y — but the visit rate is not: x-steps take ~25 s and y-steps ~100 s. Pure thermal drift would produce a 4:1 anisotropy in the apparent gradient; the measurement shows 1:1. The grid makes it plain, rising monotonically in both axes:

```
A1 measured defocus (µm)
 y \ x   10.99   13.59   16.19   18.80
 16.04     6.5     9.6    14.2    14.3
 13.44     3.7     6.8    11.3    11.1
 10.83     1.8     4.0     8.6     8.1
  8.23    -0.1     1.9     6.1     5.6
```

**A6 and D6 cannot be resolved from this run.** Their defocus varies almost entirely along y (dz/dx ≈ 0.04–0.07 µm/mm), and y is the slow raster axis, so tilt and drift are collinear. Both models fit well (adj R² 0.92–0.98); the data cannot separate them. Distinguishing these requires a dedicated test — see the recommendations.

**D1 is neither.** Both models are near-worthless (adj R² ≈ 0.2). D1 is a roughly constant −5.3 µm offset plus ~1.6 µm scatter, i.e. a focus-map *offset* error rather than a gradient.

A1 is also the only region whose per-region AF offset carries the opposite sign (+0.21 µm vs −1.02 / −1.39 / −2.75), while having the largest residual tilt.

---

## 2. Throughput

### Headline

**No slowdown. The run got marginally faster over time.**

| Metric | Value |
|---|---|
| Acquisition time (worker-reported) | 1391.99 s (23.2 min) |
| Wall clock incl. teardown | 1393.13 s |
| Frames | 2,800 → **2.01 fps, 497 ms/frame** |
| Per z-position (5 channels) | 2.486 s |
| Per FOV (10 z × 5 ch) | 24.86 s |
| Data written | 97.29 GB → **69.9 MB/s** |

### Slowdown checks — all clean or negative

| Check | Result |
|---|---|
| `acquire_at_position`, first half vs second | 24.58 s → 23.86 s (**−2.9%**) |
| Per-frame camera time, linear fit over run | **−22.9 ms** across the entire run |
| Intra-FOV image intervals (504 samples) | median 2.379 s, max 2.851 s |
| Intervals > 3× median | **0** |
| Memory footprint | flat ~2430 MB start→end; RSS *fell* 1253 → 794 MB; peak 2536 MB. No leak. |

### The one hiccup — FOV 26 (A6 fov 10, t = 653 s)

27.238 s against a 24.156 s median (**+12.8%**). Cause is write-side, not optical: `_image_callback` totalled 6.42 s for that FOV against 3.95 s typical, while camera and exposure times stayed nominal. Transient disk/queue backpressure. Isolated, no recurrence.

The only other >2 s entries are the three inter-region hops (`move_to_coordinate` 3.57 / 2.28 / 4.05 s). Intra-region moves ran 0.34–0.82 s.

### Time budget per z-position (2.486 s)

| Stage | s | % of run |
|---|---|---|
| Camera acquire, 5 ch | 2.307 | 92.8% |
| — exposure wait | 1.603 | 64.5% |
| — image callback | 0.423 | 17.0% |
| — trigger | 0.013 | 0.5% |
| XY moves (amortised over z) | 0.063 | 2.5% |
| z moves, AF, filter, sync | 0.121 | 4.9% |

**Duty cycle: 28.2%.** Configured exposure totals 0.700 s per z-position (4 × 150 ms + 100 ms), but the exposure-wait timer averages 1.603 s — a **2.3× per-frame overhead**. Critically, that overhead is uniform across all five channel slots (0.316–0.324 s each) regardless of whether the channel is set to 150 ms or 100 ms, which identifies it as a **fixed per-frame cost, not exposure-proportional**.

An exposure-only floor for this run is **6.5 min** against the actual 23.2 min.

`_image_callback` at 17% of wall clock (236.8 s) is the second-largest lever, expected given 97 GB written.

### Firmware chatter

**752** `Read thread is stale` warnings from `microcontroller.py`, gaps 0.100–0.206 s (mean 0.113 s), spanning t = 20 s to t = 1380 s and denser in the first half. None coincide with a measured stall, so they are cosmetic in this run — but at roughly one packet gap every 1.9 s they are worth tracking across longer soaks.

Full warning/error inventory: 752 stale reads + 36 correlation failures (2 lines each × 18) + 18 autofocus-failed errors. Nothing else.

---

## 3. Target vs. actual position discrepancy

> Note: the top-level `coordinates.csv` has an **empty `z (mm)` column**, so Z targets were taken from the `moving to coordinate` log entries (the focus-map interpolated value). X/Y targets come from the CSV as supplied.

### XY — at the encoder floor

| | mean signed | mean abs | p95 abs | max abs |
|---|---|---|---|---|
| dx | +0.043 µm | 0.185 µm | 0.345 | 0.345 µm |
| dy | −0.014 µm | 0.207 µm | 0.285 | 0.285 µm |
| radial | — | 0.294 µm | 0.428 | **0.447 µm** |

Per region, mean |dx| runs 0.156–0.207 µm and mean |dy| is 0.207 µm everywhere — no region dependence, no bias. The errors take only four discrete values per axis, identifying this as stage step quantisation rather than positioning error. At 0.376 µm/px the worst case is **1.19 pixels**.

XY is also perfectly constant within each z-stack: max span **0.0000 µm** across all 56 stacks.

**Verdict: pass, with no margin concern.**

### Z — the raw number is misleading; split it by AF outcome

| | n | mean signed | mean abs | max abs |
|---|---|---|---|---|
| All FOV | 56 | −0.94 µm | 2.57 µm | 14.16 µm |
| AF locked | 38 | −1.36 µm | **3.76 µm** | 14.16 µm |
| AF failed | 18 | −0.05 µm | **0.047 µm** | 0.094 µm |

This inverts the usual reading: **large dz means the AF worked.** For a locked FOV, dz *is* the correction applied against the focus map — it should be non-zero, and it equals −(measured defocus) to within 0.24 µm. For a failed FOV, dz ≈ 0 because no correction survived; the stack sits exactly on the raw focus-map plane carrying its full defocus.

The meaningful Z metric is therefore residual focus error at the imaged plane — **0.24 µm on locked FOV** (well inside ±1.24 µm DOF) versus **7.02 µm on failed FOV**.

Per-region mean |dz| — A1 5.51, D6 1.91, D1 1.27, A6 1.04 µm — tracks residual tilt magnitude, as expected.

### Z-step fidelity — exact

All **504** intra-stack steps measured **0.46875 µm**, standard deviation **0.000000**, −0.25 nm from the 0.469 µm nominal (pure representation rounding). No missed or doubled steps.

---

## Findings and recommendations

| # | Finding | Severity | Action |
|---|---|---|---|
| 1 | Correlation gate rejects 17 good locks vs 1 real failure — 94% false-positive rate | **High** | Lower `correlation_threshold` in `laser_af_settings.json` from 0.9 toward 0.75. Note the repo's own backup configs already ship **0.7** for 4x/10x/20x. |
| 2 | Failure fallback reverts to a *worse* z than the AF had reached | **High** | Reverting is wrong when the move demonstrably converged. Either keep the correction and flag it, or gate on post-move centroid residual instead of crop correlation. 11 of 18 failed FOV ended up outside their own z-stack. |
| 3 | A1 carries 15.2 µm p-p residual tilt (adj R² 0.921 on a plane fit) | **Medium** | Rebuild the focus map for A1; check its +0.21 µm per-region offset, which is the only positive one of the four. |
| 4 | A6/D6 defocus varies only along the slow raster axis — tilt and drift indistinguishable | **Medium** | Run the serpentine-vs-raster discriminator (see below) before deciding whether this is a focus-map or a thermal problem. |
| 5 | D1 shows a flat −5.3 µm focus-map offset, not a gradient | **Medium** | Re-zero the D1 focus map; a tilt correction will not help. |
| 6 | 0.32 s fixed overhead per frame, independent of exposure setting; 64.5% of run | **Medium** | Largest available throughput lever — a ~3.5× speedup is on the table. Investigate the `exposure_time_done_sleep_hw or wait_for_image_sw` path. |
| 7 | 752 microcontroller stale-read warnings | **Low** | No measured impact here. Track across longer soaks. |
| 8 | Transient +12.8% write-side stall at FOV 26 | **Low** | Isolated. Watch `_image_callback` on longer/faster runs. |
| 9 | A6 fov 1 optical anomaly at (110.09, 8.23) | **Low** | Inspect physically for bubble or debris. |

### Suggested follow-up test — separating tilt from drift

Acquire the same region twice back to back, the second pass in **reverse FOV order**. Tilt is fixed to position and will reproduce the same spatial map both times; thermal drift is fixed to time and will invert relative to position. This resolves finding #4 unambiguously and is worth adding to the standing IQ suite.

---

## Artefacts

| File | Contents |
|---|---|
| `summary.json` | All computed metrics, machine-readable |
| `af_per_fov.csv` | Per-FOV AF: target xyz, displacement, xcorr, lock, spot centroids, residual, false-rejection flag |
| `timing_per_fov.csv` | Per-FOV move / acquire / cycle times, relative timestamps |
| `discrepancy.csv` | Per-position target vs actual x/y/z with dx/dy/dz/dr |
| `focus_quality.csv` | Per-position delivered focus error, joined to AF outcome |

Regenerate with:

```
python ~/.claude/skills/microscope-iq/scripts/analyze_acquisition.py <dataset_dir> -o <outdir>
```
