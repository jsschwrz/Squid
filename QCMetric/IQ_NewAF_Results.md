# IQ Run 2 — Connected-Components Laser AF

**Dataset:** `SetUp_NewAF_2026-07-30_19-50-44.181790`
**Acquired:** 2026-07-30 19:50:44 → 19:52:07
**Code:** `feat/laser-af-connected-components` @ `029c304d` — squashed import of
Alpaca233/octopi-research `af-scan` plus legacy-config shim
**Baseline:** [IQ_SoakTest_Results.md](IQ_SoakTest_Results.md) (16:20, same four regions, same 4×4 grids)

## Purpose

Like-for-like laser AF comparison against the 16:20 soak test. Same regions, same
grids, same 56 positions, same 10x objective, same correlation threshold (0.9).
Reduced to **1 channel × 1 z-slice** because free disk was down to 6.6 GB — laser AF
runs once per FOV regardless of channels or z, so dropping both preserves every AF
measurement at 1.81 GB instead of 97 GB.

| | |
|---|---|
| Objective | 10x / 0.3 NA, recalibrated 19:29:41 (`pixel_to_um` 1.9708 → 1.8924) |
| Regions / FOV | A1 16, A6 12, D6 12, D1 16 = **56** (identical to baseline) |
| Channels × Z | 1 × 1 (baseline: 5 × 10) |
| Output | 56 TIFF, 1.81 GB |
| Duration | **82.6 s** (baseline: 1392.0 s) |

---

## 1. Laser AF — 92.9%, up from 67.9%

| | Baseline 16:20 | This run 19:50 |
|---|---|---|
| **Lock rate** | 38/56 — **67.9%** | 52/56 — **92.9%** |
| False rejections | **17** | **3** |
| Genuine failures | 1 | 1 |
| Mean residual, locked | 0.243 µm | **0.127 µm** |
| xcorr median | 0.941 | **0.980** |
| xcorr 5th pct / min | — / 0.135 | 0.883 / **0.608** |
| Threshold bracket | [0.896, 0.901] | [0.884, 0.901] |

Per region, every one improved:

| Region | Baseline | This run | mean xcorr | mean defocus | region offset |
|---|---|---|---|---|---|
| A1 | 12/16 — 75.0% | **16/16 — 100%** | 0.982 (was 0.924) | +0.09 µm (was +7.09) | +3.28 µm (was +0.21) |
| A6 | 7/12 — 58.3% | **10/12 — 83.3%** | 0.942 (was 0.857) | +0.27 µm (was −5.82) | +1.00 µm (was −1.02) |
| D6 | 10/12 — 83.3% | **12/12 — 100%** | 0.976 (was 0.939) | +1.22 µm (was −2.51) | +0.84 µm (was −1.39) |
| D1 | 9/16 — 56.3% | **14/16 — 87.5%** | 0.951 (was 0.904) | +0.24 µm (was −5.25) | +1.80 µm (was −2.75) |

### The revert path works

All four failures logged `Restoring z position: moving -8.1 / -3.0 / 23.6 / 2.9 µm`,
and delivered dz for failed FOV was **exactly 0.000 µm** (baseline: 0.047 µm mean,
because the old code undid only its last relative move). The absolute-position restore
introduced by `_restore_to_position` does what it claims.

### Attribution is mixed — read this before crediting the detector

Three things changed between runs, not one:

1. Connected-components spot detection replaced line-profile peak finding
2. The 10x objective was **recalibrated** at 19:29:41 with a fresh reference image
3. The focus map / per-region offsets were **re-centred** — mean defocus per region
   moved from −5.8…+7.1 µm to +0.09…+1.22 µm

Point 3 alone makes AF's job easier: mean |defocus| across A1 halved, 7.1 → 3.5 µm.

The counter-evidence is that the *spread* is unchanged. A1's defocus span is 14.3 µm
this run against 14.4 µm at baseline — the tilt is identical, only its centre moved.
So AF faced the same range of correction magnitudes and succeeded 16/16 where it
previously managed 12/16.

**Fair summary: the new detector is clearly not worse, residual focus error halved,
and some share of the lock-rate gain belongs to the fresh calibration.** Isolating the
detector alone would need a run against the *old* focus map.

---

## 2. Tilt vs. drift — settled

The baseline report could not separate residual focus-map tilt from thermal drift for
A6 and D6, because in a raster scan position and elapsed time are collinear.

**This run breaks that confound**, by changing the time scale ~17× while holding the
geometry fixed:

| Region | Baseline duration | This run | Baseline tilt p-p | This run tilt p-p |
|---|---|---|---|---|
| A1 | 374.2 s | **21.6 s** | 15.24 µm | **15.55 µm** |
| A6 | 281.3 s | 17.2 s | 6.01 µm | 10.07 µm |
| D6 | 271.6 s | 15.9 s | 7.32 µm | 6.95 µm |
| D1 | 367.2 s | 23.3 s | 4.14 µm | 4.25 µm |

Thermal drift across a 17×-shorter run would produce roughly 17× less defocus. It
produced **the same**. A1's fitted gradients reproduce within ~10% — dz/dx +0.887 vs
+0.956 µm/mm, dz/dy +1.095 vs +0.997 µm/mm — across runs an order of magnitude apart
in duration.

**This is residual tilt in the focus map, fixed to position. It is not thermal drift.**
A6 and D6, previously "collinear — cannot tell", now also resolve as tilt.

The fix is rebuilding those focus maps. Pursuing thermal stability would be wasted effort.

A1's defocus map, showing the same gradient with a shifted centre:

```
BASELINE 16:20                          THIS RUN 19:50
 y \ x   10.99  13.59  16.19  18.80      y \ x   10.99  13.59  16.19  18.80
 16.04    6.5    9.6   14.2   14.3       16.04    0.1    2.9    7.5    7.6
 13.44    3.7    6.8   11.3   11.1       13.44   -3.1   -0.1    4.5    3.9
 10.83    1.8    4.0    8.6    8.1       10.83   -5.3   -2.6    1.5    0.9
  8.23   -0.1    1.9    6.1    5.6        8.23   -6.7   -5.5   -1.8   -2.3
        span 14.4 µm                             span 14.3 µm
```

**D1 is still not a gradient.** Both fits are weak in both runs (adj R² ≈ 0.20/0.27) —
it is a flat offset plus ~1.6 µm scatter. A tilt correction will not help it.

---

## 3. Threshold: 0.85 is free

Sweeping the correlation threshold against this run's data, and checking whether each
admitted FOV had actually converged (post-move spot within 1 µm of reference):

| Threshold | Lock rate | Non-converged admitted |
|---|---|---|
| **0.90** (current) | 52/56 — 92.9% | 0 |
| **0.85** | **54/56 — 96.4%** | **0** |
| 0.80 | 54/56 — 96.4% | 0 |
| 0.75 | 54/56 — 96.4% | 0 |
| 0.70 | 55/56 — 98.2% | **1** |

The three false rejections sat at xcorr **0.880**, **0.884** and 0.608. Dropping the
threshold to 0.85 recovers the first two at **zero cost in bad admissions**, and still
correctly rejects the genuine failure at 0.743. Below 0.75 it starts admitting
non-converged results.

Set `correlation_threshold: 0.85` in
`software/user_profiles/<profile>/laser_af_configs/10x.yaml`.

Note the other objectives disagree: 10x and 4x are at 0.9, 20x and 40x at 0.7. Worth
reconciling.

---

## 4. The four failures

| Region | FOV | Position | defocus | xcorr | post-move residual | classification |
|---|---|---|---|---|---|---|
| A6 | 1 | (110.09, 8.23) | −8.0 µm | 0.608 | 0.19 µm | false rejection |
| A6 | 2 | (112.69, 8.23) | −3.0 µm | 0.880 | 0.00 µm | false rejection |
| D1 | 13 | (13.59, 73.94) | +2.9 µm | 0.884 | 0.38 µm | false rejection |
| D1 | 6 | (16.19, 68.73) | +23.6 µm | 0.743 | **−37.79 µm** | **genuine failure** |

**A6 fov 1 failed in both runs at the same physical position.** Baseline: −45.5 µm at
xcorr 0.135. Now: −8.0 µm at xcorr 0.608, and the move did converge (0.19 µm residual)
— detection is much improved, but correlation is still poor. That signature points at
a persistent optical problem at that location (bubble, debris, meniscus), not software.
**Worth physically inspecting.**

**D1 fov 6** is the single genuine failure: measured +23.6 µm, moved, and ended 37.8 µm
away. Correctly rejected and reverted.

---

## 5. Still untested: the z spot search

**Zero spot-search events across all 56 FOV.** The spot was found on the first attempt
every time, so the headline feature of this merge has still never executed on hardware.

It only triggers when *no* spot is detected. D1 fov 6 found a *wrong* spot rather than
none, so even the worst case did not engage it. Exercising it deliberately would mean
starting an acquisition well outside focus.

---

## 6. Throughput and positioning

Not comparable to baseline on throughput — 1 frame per FOV against 50.

| | Baseline | This run |
|---|---|---|
| Duration | 1392.0 s | 82.6 s |
| s/FOV | 24.86 | 1.48 |
| Frames | 2800 | 56 |
| Stale MCU reads | 752 | 6 |

No slowdown: `acquire_at_position` first half 0.788 s vs second half 0.780 s (−1.0%),
one FOV at 0.941 s against a 0.780 s median, per-frame drift +24.9 ms across the run.

**XY positioning is identical to the baseline to three decimal places:**

| | Baseline | This run |
|---|---|---|
| mean abs dx | 0.185 µm | **0.185 µm** |
| mean abs dy | 0.207 µm | **0.207 µm** |
| max radial | 0.4474 µm | **0.4474 µm** |

Stage positioning is at the encoder quantisation floor and perfectly reproducible.

Delivered dz: locked FOV mean abs 3.15 µm (the AF correction), failed FOV **exactly
0.000 µm** (reverted).

---

## Findings

| # | Finding | Severity | Action |
|---|---|---|---|
| 1 | Lock rate 67.9% → 92.9%; residual halved to 0.127 µm | — | Merge is a clear improvement; keep |
| 2 | Threshold 0.85 gives 96.4% at zero cost in bad admissions | **High** | One-line config change |
| 3 | Defocus is residual focus-map **tilt**, confirmed by 17× time-scale change | **High** | Rebuild focus maps for A1/A6/D6. Do not chase thermal |
| 4 | D1 is a flat offset plus scatter, not a gradient | Medium | Re-zero D1's focus map; tilt correction won't help |
| 5 | A6 fov 1 (110.09, 8.23) fails in both runs | Medium | Physically inspect |
| 6 | Z spot search never executed | Medium | Test deliberately from a defocused start |
| 7 | Cross-correlation still reverts converged moves | Medium | Unchanged by this merge; separate fix |
| 8 | Config recalibration is one-way (new format unreadable by master) | Medium | Snapshots on Desktop; see `029c304d` message |

## Artefacts

`run2_newaf_2026-07-30_1950/` — `summary.json`, `af_per_fov.csv`, `timing_per_fov.csv`,
`discrepancy.csv`, `focus_quality.csv`.

Regenerate:

```
python .claude/skills/microscope-iq/scripts/analyze_acquisition.py \
    <dataset_dir> -o QCMetric/run2_newaf_2026-07-30_1950
```

Both runs' baseline artefacts sit alongside in `QCMetric/` for direct diffing.
