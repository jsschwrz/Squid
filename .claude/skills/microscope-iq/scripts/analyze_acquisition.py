#!/usr/bin/env python3
"""
Squid / SquidXplorer acquisition IQ analyser.

Parses an acquisition output folder (acquisition.log + coordinates CSVs + config)
and reports:
  1. Laser reflection AF performance  (lock rate, false rejections, residual focus error)
  2. Throughput                       (frame rate, time budget, slowdown / drift detection)
  3. Target-vs-actual XYZ discrepancy (stage positioning fidelity)

Usage:
    python analyze_acquisition.py <dataset_dir> [-o OUTDIR] [--md REPORT.md] [--quiet]

Writes af_per_fov.csv, timing_per_fov.csv, discrepancy.csv, focus_quality.csv,
summary.json into OUTDIR (default: <dataset_dir>/qc_analysis).

Requires: numpy, pandas.  pyyaml optional (falls back to regex config parsing).
"""

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# log grammar
# --------------------------------------------------------------------------- #
RE_LINE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) - (\d+) - (\S+) - (\w+) - (.*)$"
)
RE_MOVE = re.compile(
    r"moving to coordinate \(np\.float64\(([-\d.e+]+)\), "
    r"np\.float64\(([-\d.e+]+)\), np\.float64\(([-\d.e+]+)\)\)"
)
RE_DISP = re.compile(r"Current laser AF displacement: ([-\d.]+)")
RE_XCORR = re.compile(r"Cross correlation with reference: ([-\d.]+)")
RE_SPOT = re.compile(
    r"Spot centroid found at \(([-\d.]+), ([-\d.]+)\) from (\d+) detections"
)
RE_AFERR = re.compile(
    r"Autofocus failed in acquire_at_position.*?z=([-\d.]+)"
)
RE_OFFSET = re.compile(
    r"applying per-region laser AF offset for region '(\w+)': ([-+\d.]+)"
)
RE_ACQIMG = re.compile(
    r"Acquiring image: ID=(\w+?)_(\d+)_(\d+), Metadata=\{'x': ([-\d.e+]+), "
    r"'y': ([-\d.e+]+), 'z': ([-\d.e+]+)\}"
)
RE_TSTOP = re.compile(r"Stopping name=(.+?) with elapsed=([\d.eE\-+]+) \[s\]")
RE_OOR = re.compile(
    r"Measured displacement \(([-\d.]+) \S+\) is unreasonably large"
)
RE_STALE = re.compile(r"it has been ([\d.]+) \[s\] since a valid packet")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config(root):
    """Read acquisition.yaml / acquisition parameters.json. Everything optional."""
    cfg = {
        "experiment_id": os.path.basename(os.path.normpath(root)),
        "objective": None, "NA": None, "pixel_size_um": None,
        "nz": None, "dz_um": None, "nt": 1,
        "laser_af": None, "contrast_af": None,
        "channels": [], "regions": [],
    }
    ypath = os.path.join(root, "acquisition.yaml")
    if os.path.exists(ypath):
        data = None
        try:
            import yaml
            with open(ypath, "r", encoding="utf-8", errors="replace") as fh:
                data = yaml.safe_load(fh)
        except Exception:
            data = None
        if isinstance(data, dict):
            cfg["experiment_id"] = data.get("acquisition", {}).get("experiment_id", cfg["experiment_id"])
            obj = data.get("objective") or {}
            cfg["objective"] = obj.get("name")
            cfg["NA"] = obj.get("NA")
            cfg["pixel_size_um"] = obj.get("pixel_size_um")
            zs = data.get("z_stack") or {}
            cfg["nz"] = zs.get("nz")
            if zs.get("delta_z_mm") is not None:
                cfg["dz_um"] = zs["delta_z_mm"] * 1000.0
            cfg["nt"] = (data.get("time_series") or {}).get("nt", 1)
            af = data.get("autofocus") or {}
            cfg["laser_af"] = af.get("laser_af")
            cfg["contrast_af"] = af.get("contrast_af")
            for ch in data.get("channels") or []:
                if ch.get("enabled", True):
                    cfg["channels"].append({
                        "name": ch.get("name"),
                        "exposure_ms": (ch.get("camera_settings") or {}).get("exposure_time_ms"),
                        "z_offset_um": ch.get("z_offset_um"),
                    })
            for rg in (data.get("wellplate_scan") or {}).get("regions") or []:
                cfg["regions"].append(rg.get("name"))

    jpath = os.path.join(root, "acquisition parameters.json")
    if os.path.exists(jpath):
        try:
            with open(jpath, "r", encoding="utf-8", errors="replace") as fh:
                j = json.load(fh)
            cfg["nz"] = cfg["nz"] or j.get("Nz")
            cfg["dz_um"] = cfg["dz_um"] or j.get("dz(um)")
            if cfg["laser_af"] is None:
                cfg["laser_af"] = j.get("with reflection AF")
            if cfg["objective"] is None:
                cfg["objective"] = (j.get("objective") or {}).get("name")
            if cfg["NA"] is None:
                cfg["NA"] = (j.get("objective") or {}).get("NA")
        except Exception:
            pass
    return cfg


# --------------------------------------------------------------------------- #
# log parsing
# --------------------------------------------------------------------------- #
def parse_log(logpath):
    rows = []
    with open(logpath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RE_LINE.match(line.rstrip("\n"))
            if m:
                rows.append((
                    datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f"),
                    m.group(3), m.group(4), m.group(5),
                ))
    if not rows:
        raise SystemExit(f"No parseable log lines in {logpath}")
    return rows


def extract(rows):
    """Walk the log once, bucketing everything by FOV visit."""
    fovs, timers, imgs, stale, warns = [], [], [], [], []
    cur = None
    for t, logger, lvl, msg in rows:
        m = RE_MOVE.search(msg)
        if m and "MultiPointWorker" in logger:
            cur = dict(idx=len(fovs), t_move=t,
                       x_t=_f(m.group(1)), y_t=_f(m.group(2)), z_t=_f(m.group(3)),
                       disp=np.nan, xcorr=np.nan, locked=None, out_of_range=False,
                       region_offset_um=np.nan, z_fallback=np.nan, spots=[])
            fovs.append(cur)
        elif cur is not None:
            m = RE_SPOT.search(msg)
            if m:
                cur["spots"].append((_f(m.group(1)), _f(m.group(2)), int(m.group(3))))
            m = RE_DISP.search(msg)
            if m:
                cur["disp"] = _f(m.group(1))
            m = RE_XCORR.search(msg)
            if m:
                cur["xcorr"] = _f(m.group(1))
            if "Cross correlation check passed" in msg:
                cur["locked"] = True
            elif "Cross correlation check failed" in msg:
                cur["locked"] = False
            if RE_OOR.search(msg):
                cur["out_of_range"] = True
                cur["locked"] = False
            m = RE_AFERR.search(msg)
            if m:
                cur["z_fallback"] = _f(m.group(1))
            m = RE_OFFSET.search(msg)
            if m:
                cur["region_offset_um"] = _f(m.group(2))

        m = RE_ACQIMG.search(msg)
        if m:
            imgs.append(dict(t=t, region=m.group(1), fov=int(m.group(2)),
                             z_level=int(m.group(3)), x=_f(m.group(4)),
                             y=_f(m.group(5)), z=_f(m.group(6)),
                             fov_idx=len(fovs) - 1))
        m = RE_TSTOP.search(msg)
        if m and logger == "squid.Timer":
            timers.append((t, m.group(1), _f(m.group(2))))
        m = RE_STALE.search(msg)
        if m:
            stale.append((t, _f(m.group(1))))
        if lvl in ("WARNING", "ERROR", "CRITICAL"):
            warns.append((t, lvl, msg))
    return fovs, timers, imgs, stale, warns


# --------------------------------------------------------------------------- #
# section 1: laser AF
# --------------------------------------------------------------------------- #
def analyse_af(fovs, imgs_df, cfg, out):
    af = pd.DataFrame(fovs)
    if af.empty:
        return af, {}
    af["n_spot_reads"] = af.spots.apply(len)
    af["meas_x"] = af.spots.apply(lambda s: s[0][0] if len(s) > 0 else np.nan)
    af["meas_y"] = af.spots.apply(lambda s: s[0][1] if len(s) > 0 else np.nan)
    af["ver_x"] = af.spots.apply(lambda s: s[1][0] if len(s) > 1 else np.nan)
    af["ver_y"] = af.spots.apply(lambda s: s[1][1] if len(s) > 1 else np.nan)
    af = af.drop(columns=["spots"])

    if not imgs_df.empty:
        ident = imgs_df.groupby("fov_idx").agg(
            region_id=("region", "first"), fov_id=("fov", "first"),
            n_z=("z_level", "nunique"))
        af = af.join(ident, on="idx")
    for c in ("region_id", "fov_id"):
        if c not in af:
            af[c] = np.nan

    attempted = af[af.locked.notna()]
    n = len(attempted)
    if n == 0:
        return af, {"attempted": 0}

    # Empirical threshold: between the worst pass and the best fail.
    thr_lo = attempted.loc[attempted.locked == False, "xcorr"].max()   # noqa: E712
    thr_hi = attempted.loc[attempted.locked == True, "xcorr"].min()    # noqa: E712

    # Spot reference = median verification position of successful locks.
    ref_x = attempted.loc[attempted.locked == True, "ver_x"].median()  # noqa: E712
    af["res_px"] = af.ver_x - ref_x

    # px -> um scale for the AF spot, fitted from (measured spot x, reported displacement).
    good = af[(af.disp.notna()) & (af.meas_x.notna()) & (af.disp.abs() < 500)]
    scale = np.nan
    if len(good) >= 3 and good.meas_x.std() > 1e-6:
        scale = float(np.polyfit(good.meas_x, good.disp, 1)[0])
    af["residual_um"] = af.res_px * scale

    # A "false rejection" = correlation gate vetoed a move that had actually converged.
    conv_tol = 1.0  # um
    af["false_rejection"] = (af.locked == False) & (af.residual_um.abs() < conv_tol)  # noqa: E712
    af["genuine_failure"] = (af.locked == False) & ~af.false_rejection                # noqa: E712

    # Focus error actually delivered at the imaged plane.
    af["focus_error_um"] = np.where(af.locked == True, af.residual_um, af.disp)       # noqa: E712

    # Re-slice now that the derived columns exist.
    attempted = af[af.locked.notna()]
    locked = attempted[attempted.locked == True]           # noqa: E712
    failed = attempted[attempted.locked == False]          # noqa: E712

    dof_full = np.nan
    if cfg.get("NA"):
        dof_full = 0.55 * 0.405 / (float(cfg["NA"]) ** 2)  # rough wave-optics DOF, um
    stack_um = (cfg.get("nz") or 0) * (cfg.get("dz_um") or 0)

    s = {
        "attempted": int(n),
        "locked": int(len(locked)),
        "failed": int(len(failed)),
        "lock_rate_pct": round(100.0 * len(locked) / n, 1),
        "threshold_bracket": [None if math.isnan(thr_lo) else round(thr_lo, 3),
                              None if math.isnan(thr_hi) else round(thr_hi, 3)],
        "spot_reference_x_px": None if math.isnan(ref_x) else round(ref_x, 2),
        "um_per_px": None if math.isnan(scale) else round(scale, 4),
        "false_rejections": int(af.false_rejection.sum()),
        "genuine_failures": int(af.genuine_failure.sum()),
        "mean_abs_residual_locked_um": round(float(locked.residual_um.abs().mean()), 3) if len(locked) else None,
        "mean_abs_defocus_failed_um": round(float(failed.disp.abs().mean()), 3) if len(failed) else None,
        "dof_full_um": None if math.isnan(dof_full) else round(dof_full, 2),
        "z_stack_range_um": round(stack_um, 3) if stack_um else None,
    }
    if len(failed) and stack_um:
        s["failed_exceeding_stack_range"] = int((failed.disp.abs() > stack_um).sum())
    if len(failed) and not math.isnan(dof_full):
        s["failed_exceeding_half_dof"] = int((failed.disp.abs() > dof_full / 2).sum())

    # Per-region + tilt-vs-drift discrimination.
    per_region, tilt = [], []
    for r, g in attempted.groupby("region_id"):
        gg = g[g.disp.abs() < 500]
        row = dict(region=r, n=len(g), locked=int((g.locked == True).sum()),  # noqa: E712
                   lock_rate_pct=round(100.0 * (g.locked == True).sum() / len(g), 1),  # noqa: E712
                   mean_xcorr=round(float(g.xcorr.mean()), 3),
                   mean_disp_um=round(float(g.disp.mean()), 2),
                   region_offset_um=float(g.region_offset_um.dropna().iloc[0])
                   if g.region_offset_um.notna().any() else None)
        per_region.append(row)
        # Plane fit (spatial tilt) vs linear-in-time fit (thermal drift).
        # Exclude genuine spot-detection failures: their reported displacement is meaningless.
        gg = gg[~gg.get("genuine_failure", pd.Series(False, index=gg.index)).fillna(False)]
        if len(gg) >= 5 and gg.x_t.std() > 0 and gg.y_t.std() > 0:
            y = gg.disp.values
            sst = max(float(np.sum((y - y.mean()) ** 2)), 1e-12)
            t_rel = (gg.t_move - gg.t_move.min()).dt.total_seconds().values
            xc, yc, tc = gg.x_t - gg.x_t.mean(), gg.y_t - gg.y_t.mean(), t_rel - t_rel.mean()

            def fit(cols):
                A = np.column_stack(cols + [np.ones(len(gg))])
                c = np.linalg.lstsq(A, y, rcond=None)[0]
                r2 = 1 - float(np.sum((y - A @ c) ** 2)) / sst
                p = len(cols)
                adj = 1 - (1 - r2) * (len(gg) - 1) / max(len(gg) - p - 1, 1)
                return c, r2, adj

            cxy, r2xy, adjxy = fit([xc, yc])
            ct, r2t, adjt = fit([tc])
            _, _, adjall = fit([xc, yc, tc])

            # In a raster scan, position and elapsed time are collinear, so R2 alone
            # cannot separate them. The decisive test: pure time drift would make the
            # apparent gradient scale with SECONDS per step, not MILLIMETRES per step.
            # Compare the fitted x/y gradients against that prediction.
            verdict = "collinear (raster order confounds space and time)"
            if adjxy - adjt > 0.05:
                verdict = "tilt"
            elif adjt - adjxy > 0.05:
                verdict = "drift"
            else:
                # tie-break on gradient isotropy: seconds-per-mm differs hugely between
                # the fast (within-row) and slow (row-to-row) axis
                try:
                    sx = float(np.polyfit(gg.x_t, t_rel, 1)[0])  # s per mm along x
                    sy = float(np.polyfit(gg.y_t, t_rel, 1)[0])  # s per mm along y
                    if abs(sx) > 1e-6 and abs(sy) > 1e-6:
                        grad_ratio = abs(cxy[0]) / max(abs(cxy[1]), 1e-9)
                        time_ratio = abs(sx) / max(abs(sy), 1e-9)
                        # drift predicts grad_ratio ~= time_ratio; tilt predicts ~= 1
                        if abs(math.log((grad_ratio + 1e-9) / (time_ratio + 1e-9))) > \
                           abs(math.log(grad_ratio + 1e-9)):
                            verdict = "tilt (gradient isotropic, visit rate is not)"
                except Exception:
                    pass

            span = float(np.hypot(gg.x_t.max() - gg.x_t.min(), gg.y_t.max() - gg.y_t.min()))
            tilt.append(dict(region=r, n_fit=int(len(gg)),
                             dz_dx_um_per_mm=round(float(cxy[0]), 3),
                             dz_dy_um_per_mm=round(float(cxy[1]), 3),
                             r2_plane=round(float(r2xy), 3), adj_r2_plane=round(float(adjxy), 3),
                             drift_um_per_min=round(float(ct[0]) * 60, 3),
                             r2_time=round(float(r2t), 3), adj_r2_time=round(float(adjt), 3),
                             adj_r2_xyt=round(float(adjall), 3),
                             tilt_pp_um=round(float(np.hypot(cxy[0], cxy[1]) * span), 2),
                             verdict=verdict))
    s["per_region"] = per_region
    s["tilt_vs_drift"] = tilt
    af.to_csv(os.path.join(out, "af_per_fov.csv"), index=False)
    return af, s


# --------------------------------------------------------------------------- #
# section 2: throughput
# --------------------------------------------------------------------------- #
def analyse_throughput(rows, fovs, timers, imgs_df, stale, cfg, root, out):
    t0, t1 = rows[0][0], rows[-1][0]
    span = (t1 - t0).total_seconds()

    tdf = pd.DataFrame(timers, columns=["t", "name", "s"])
    # authoritative duration if the worker logged it
    dur = span
    for nm in ("run_single_time_point", "run_coordinate_acquisition"):
        v = tdf.loc[tdf.name == nm, "s"]
        if len(v):
            dur = float(v.max())
            break

    n_frames = int((tdf.name == "acquire_camera_image").sum())
    n_zpos = int(len(imgs_df)) if not imgs_df.empty else 0
    n_fov = len(fovs)
    n_ch = len(cfg.get("channels") or []) or (round(n_frames / n_zpos) if n_zpos else None)

    # bytes on disk
    tot_bytes, n_files = 0, 0
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if os.path.isdir(d) and sub.isdigit():
            for fn in os.listdir(d):
                if fn.lower().endswith((".tiff", ".tif", ".png", ".bmp", ".npy")):
                    n_files += 1
                    try:
                        tot_bytes += os.path.getsize(os.path.join(d, fn))
                    except OSError:
                        pass

    s = {
        "wall_time_s": round(span, 1),
        "acquisition_time_s": round(dur, 2),
        "n_fov": n_fov, "n_z_positions": n_zpos, "n_frames": n_frames,
        "n_channels": n_ch, "n_image_files": n_files,
        "bytes_on_disk_gb": round(tot_bytes / 1e9, 2),
        "frames_per_s": round(n_frames / dur, 3) if dur else None,
        "ms_per_frame": round(1000 * dur / n_frames, 1) if n_frames else None,
        "s_per_z_position": round(dur / n_zpos, 3) if n_zpos else None,
        "s_per_fov": round(dur / n_fov, 2) if n_fov else None,
        "data_rate_MB_s": round(tot_bytes / 1e6 / dur, 1) if dur else None,
    }

    # time budget
    budget = {}
    for nm in ("acquire_camera_image", "exposure_time_done_sleep_hw or wait_for_image_sw",
               "_image_callback", "send_trigger", "move_to_coordinate",
               "image_to_display*.emit", "job creation and dispatch"):
        v = tdf.loc[tdf.name == nm, "s"]
        if len(v):
            budget[nm] = dict(n=int(len(v)), mean_s=round(float(v.mean()), 4),
                              p95_s=round(float(v.quantile(.95)), 4),
                              max_s=round(float(v.max()), 4),
                              total_s=round(float(v.sum()), 1),
                              pct_of_run=round(100 * float(v.sum()) / dur, 1) if dur else None)
    s["time_budget"] = budget

    exp_total_ms = sum(c["exposure_ms"] for c in cfg.get("channels") or []
                       if c.get("exposure_ms")) or 0
    if exp_total_ms and n_zpos:
        s["exposure_per_z_position_s"] = round(exp_total_ms / 1000.0, 3)
        s["duty_cycle_pct"] = round(100 * (exp_total_ms / 1000.0) / (dur / n_zpos), 1)
        s["floor_if_exposure_only_min"] = round(exp_total_ms / 1000.0 * n_zpos / 60, 1)

    # per-FOV timing: assign each timer stop to the FOV window it falls in
    per_fov = pd.DataFrame([dict(idx=f["idx"], t_move=f["t_move"]) for f in fovs])
    if len(per_fov):
        bounds = list(per_fov.t_move) + [t1]
        for nm, col in (("move_to_coordinate", "move_s"), ("acquire_at_position", "acquire_s")):
            v = tdf[tdf.name == nm]
            assign = np.full(len(per_fov), np.nan)
            for _, r in v.iterrows():
                k = int(np.searchsorted(bounds, r.t, side="right") - 1)
                if 0 <= k < len(per_fov) and math.isnan(assign[k]):
                    assign[k] = r.s
            per_fov[col] = assign
        per_fov["cycle_s"] = per_fov.t_move.shift(-1).sub(per_fov.t_move).dt.total_seconds()
        per_fov["t_rel_s"] = (per_fov.t_move - t0).dt.total_seconds().round(1)

    # slowdown / drift detection
    slow = {}
    if "acquire_s" in per_fov and per_fov.acquire_s.notna().sum() >= 4:
        a = per_fov.acquire_s.dropna()
        h = len(a) // 2
        slow["acquire_first_half_s"] = round(float(a[:h].mean()), 3)
        slow["acquire_second_half_s"] = round(float(a[h:].mean()), 3)
        slow["acquire_delta_pct"] = round(100 * (a[h:].mean() - a[:h].mean()) / a[:h].mean(), 1)
        med = float(a.median())
        outl = per_fov[per_fov.acquire_s > med * 1.10].sort_values("acquire_s", ascending=False)
        slow["acquire_median_s"] = round(med, 3)
        slow["fovs_over_110pct_of_median"] = outl[["idx", "t_rel_s", "acquire_s"]].round(3).to_dict("records")
    fr = tdf[tdf.name == "acquire_camera_image"]
    if len(fr) > 10:
        tr = (fr.t - t0).dt.total_seconds().values
        k = float(np.polyfit(tr, fr.s.values, 1)[0])
        slow["per_frame_drift_ms_over_run"] = round(k * dur * 1000, 1)
    if not imgs_df.empty:
        ii = imgs_df.sort_values("t").copy()
        ii["dt"] = ii.t.diff().dt.total_seconds()
        intra = ii[ii.fov_idx == ii.fov_idx.shift()]
        if len(intra):
            m = float(intra.dt.median())
            slow["intra_fov_interval_median_s"] = round(m, 4)
            slow["intra_fov_interval_max_s"] = round(float(intra.dt.max()), 4)
            slow["intra_fov_stalls_over_3x_median"] = int((intra.dt > 3 * m).sum())
    s["slowdown"] = slow

    if stale:
        sv = pd.Series([x[1] for x in stale])
        s["microcontroller_stale_reads"] = dict(
            n=int(len(sv)), mean_gap_s=round(float(sv.mean()), 3),
            max_gap_s=round(float(sv.max()), 3),
            first_t_rel_s=round((stale[0][0] - t0).total_seconds(), 1),
            last_t_rel_s=round((stale[-1][0] - t0).total_seconds(), 1))

    if len(per_fov):
        per_fov.to_csv(os.path.join(out, "timing_per_fov.csv"), index=False)
    return per_fov, s


# --------------------------------------------------------------------------- #
# section 3: target vs actual
# --------------------------------------------------------------------------- #
def analyse_coords(root, af, cfg, out):
    exp_p = os.path.join(root, "coordinates.csv")
    tp = None
    for sub in sorted(os.listdir(root)):
        if os.path.isdir(os.path.join(root, sub)) and sub.isdigit():
            c = os.path.join(root, sub, "coordinates.csv")
            if os.path.exists(c):
                tp = c
                break
    if not (os.path.exists(exp_p) and tp):
        return pd.DataFrame(), {"error": "coordinates.csv not found (expected and/or acquired)"}

    exp = pd.read_csv(exp_p).rename(columns={"x (mm)": "x_t", "y (mm)": "y_t", "z (mm)": "z_t_csv"})
    act = pd.read_csv(tp).rename(columns={"x (mm)": "x_a", "y (mm)": "y_a",
                                          "z (um)": "z_a", "fov": "fov_id"})
    exp["fov_id"] = exp.groupby("region").cumcount()
    a0 = act[act.z_level == act.z_level.min()]

    mg = exp.merge(a0[["region", "fov_id", "x_a", "y_a", "z_a"]],
                   on=["region", "fov_id"], how="outer", indicator=True)

    have_log_z = "region_id" in af and af.region_id.notna().any()
    if have_log_z:
        zt = af[["region_id", "fov_id", "z_t", "locked", "disp", "xcorr",
                 "residual_um", "focus_error_um", "false_rejection"]].rename(
            columns={"region_id": "region", "z_t": "z_t_log"})
        mg = mg.merge(zt, on=["region", "fov_id"], how="left")
    elif "z_t_csv" in mg:
        mg["z_t_log"] = mg.z_t_csv

    mg["dx_um"] = (mg.x_a - mg.x_t) * 1000.0
    mg["dy_um"] = (mg.y_a - mg.y_t) * 1000.0
    mg["dr_um"] = np.hypot(mg.dx_um, mg.dy_um)
    if "z_t_log" in mg:
        mg["dz_um"] = mg.z_a - mg.z_t_log * 1000.0

    def stat(col):
        v = mg[col].dropna()
        if not len(v):
            return None
        return dict(n=int(len(v)), mean_signed=round(float(v.mean()), 4),
                    mean_abs=round(float(v.abs().mean()), 4),
                    std=round(float(v.std()), 4),
                    p95_abs=round(float(v.abs().quantile(.95)), 4),
                    max_abs=round(float(v.abs().max()), 4))

    s = {"merge": mg._merge.value_counts().to_dict() if "_merge" in mg else {},
         "dx_um": stat("dx_um"), "dy_um": stat("dy_um"), "dr_um": stat("dr_um")}
    if cfg.get("pixel_size_um"):
        s["max_xy_error_px"] = round(float(mg.dr_um.max()) / float(cfg["pixel_size_um"]), 2)
    if "dz_um" in mg:
        s["dz_um_all"] = stat("dz_um")
        if "locked" in mg and mg.locked.notna().any():
            s["dz_um_af_locked"] = stat_sub(mg, "dz_um", mg.locked == True)     # noqa: E712
            s["dz_um_af_failed"] = stat_sub(mg, "dz_um", mg.locked == False)    # noqa: E712
        if "focus_error_um" in mg:
            s["residual_focus_error_um"] = stat("focus_error_um")

    # z-step fidelity + XY stability inside stacks
    a = act.sort_values(["region", "fov_id", "z_level"]).copy()
    a["dz_step"] = a.groupby(["region", "fov_id"])["z_a"].diff()
    st = a.dz_step.dropna()
    if len(st):
        s["z_step_um"] = dict(n=int(len(st)), mean=round(float(st.mean()), 6),
                              std=round(float(st.std()), 6),
                              min=round(float(st.min()), 6), max=round(float(st.max()), 6),
                              nominal=cfg.get("dz_um"),
                              err_vs_nominal=round(float(st.mean()) - float(cfg["dz_um"]), 6)
                              if cfg.get("dz_um") else None)
    span = act.groupby(["region", "fov_id"]).agg(
        x_span=("x_a", lambda v: (v.max() - v.min()) * 1000),
        y_span=("y_a", lambda v: (v.max() - v.min()) * 1000))
    s["xy_drift_within_stack_um"] = dict(max_x=round(float(span["x_span"].max()), 5),
                                         max_y=round(float(span["y_span"].max()), 5))

    per_region = []
    for r, g in mg.groupby("region"):
        row = dict(region=r, n=len(g),
                   mean_abs_dx_um=round(float(g.dx_um.abs().mean()), 3),
                   mean_abs_dy_um=round(float(g.dy_um.abs().mean()), 3),
                   max_dr_um=round(float(g.dr_um.max()), 3))
        if "dz_um" in g:
            row["mean_abs_dz_um"] = round(float(g.dz_um.abs().mean()), 3)
            row["max_abs_dz_um"] = round(float(g.dz_um.abs().max()), 3)
        per_region.append(row)
    s["per_region"] = per_region

    mg.drop(columns=["_merge"], errors="ignore").to_csv(
        os.path.join(out, "discrepancy.csv"), index=False)
    if "focus_error_um" in mg:
        mg[[c for c in ("region", "fov_id", "x_t", "y_t", "z_t_log", "z_a", "dz_um",
                        "locked", "disp", "xcorr", "residual_um", "focus_error_um",
                        "false_rejection") if c in mg]].to_csv(
            os.path.join(out, "focus_quality.csv"), index=False)
    return mg, s


def stat_sub(df, col, mask):
    v = df.loc[mask, col].dropna()
    if not len(v):
        return None
    return dict(n=int(len(v)), mean_signed=round(float(v.mean()), 4),
                mean_abs=round(float(v.abs().mean()), 4),
                std=round(float(v.std()), 4), max_abs=round(float(v.abs().max()), 4))


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def emit(summary, quiet=False):
    if not quiet:
        print(json.dumps(summary, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.dataset)
    if not os.path.isdir(root):
        raise SystemExit(f"not a directory: {root}")
    logpath = os.path.join(root, "acquisition.log")
    if not os.path.exists(logpath):
        raise SystemExit(f"no acquisition.log in {root}")

    out = args.outdir or os.path.join(root, "qc_analysis")
    os.makedirs(out, exist_ok=True)

    cfg = load_config(root)
    rows = parse_log(logpath)
    fovs, timers, imgs, stale, warns = extract(rows)
    imgs_df = pd.DataFrame(imgs)

    af, s_af = analyse_af(fovs, imgs_df, cfg, out)
    per_fov, s_tp = analyse_throughput(rows, fovs, timers, imgs_df, stale, cfg, root, out)
    mg, s_co = analyse_coords(root, af, cfg, out)

    wt = {}
    for _, lvl, msg in warns:
        k = re.sub(r"[-+]?\d*\.?\d+", "#", msg)[:160]
        wt[k] = wt.get(k, 0) + 1

    summary = {
        "dataset": root,
        "config": cfg,
        "log_lines_parsed": len(rows),
        "laser_af": s_af,
        "throughput": s_tp,
        "positioning": s_co,
        "warning_error_templates": dict(sorted(wt.items(), key=lambda x: -x[1])[:20]),
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    emit(summary, args.quiet)
    if not args.quiet:
        print(f"\n--> wrote CSVs + summary.json to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
