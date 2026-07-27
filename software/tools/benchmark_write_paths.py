#!/usr/bin/env python3
"""Benchmark real disk write throughput for candidate save paths and writer counts.

The acquisition write path (uncompressed OME-TIFF via memmap) can bottleneck a run when a
single writer subprocess can't keep up with capture. This tool measures how fast each
candidate save location can actually absorb representative image writes, and how that scales
with N parallel writer processes, so you can pick:

  - the fastest save path,
  - a rational number of writer processes (ACQUISITION_WRITER_PROCESSES), and
  - a sane adaptive backlog target (ACQUISITION_TARGET_BACKLOG_S).

It writes representative uint16 OME-TIFF stacks (create + memmap-assign + flush, mirroring
SaveOMETiffJob) into a temporary scratch folder under each candidate path, then deletes them.

Usage:
    cd software
    python tools/benchmark_write_paths.py
    python tools/benchmark_write_paths.py --width 4096 --height 4096 --writers 1,2,4,8 --duration 3
    python tools/benchmark_write_paths.py --paths "C:/Users/me/Downloads,D:/"
"""

import argparse
import concurrent.futures
import glob
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import tifffile

# Add software dir to path so we can read the configured save path / resolution rule.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import squid.logging  # noqa: E402

log = squid.logging.get_logger("benchmark_write_paths")

_BYTES_PER_MB = 1024 * 1024


def _write_worker(scratch_dir, worker_id, duration_s, height, width, planes):
    """Write representative OME-TIFF stacks for duration_s; return (bytes_written, elapsed_s).

    Module-level so it is picklable under Windows 'spawn'. Mirrors SaveOMETiffJob: create the
    stack file with tifffile.imwrite(..., ome=True), then memmap-assign + flush each plane.
    """
    image = np.random.randint(0, 4096, (height, width), dtype=np.uint16)
    plane_bytes = image.nbytes
    bytes_written = 0
    file_idx = 0
    start = time.monotonic()
    deadline = start + duration_s
    while time.monotonic() < deadline:
        path = os.path.join(scratch_dir, f"w{worker_id}_{file_idx:05d}.ome.tiff")
        try:
            # tifffile.memmap squeezes singleton dims, so create a plain (planes, H, W) stack
            # and assign per plane. Uncompressed memmap write + flush matches SaveOMETiffJob.
            tifffile.imwrite(path, shape=(planes, height, width), dtype=np.uint16, ome=True)
            stack = tifffile.memmap(path, mode="r+")
            for p in range(planes):
                stack[p] = image
                stack.flush()
                bytes_written += plane_bytes
                if time.monotonic() >= deadline:
                    break
            del stack
        except Exception as e:  # noqa: BLE001 - a failing path shouldn't abort the whole benchmark
            log.warning(f"writer {worker_id} error on {path}: {e}")
            break
        file_idx += 1
    return bytes_written, time.monotonic() - start


def benchmark_path(path, writer_counts, duration_s, height, width, planes):
    """Return {num_writers: mb_per_s} for one candidate path, or None if unusable."""
    scratch_root = None
    try:
        scratch_root = tempfile.mkdtemp(prefix="squid_wbench_", dir=path)
    except Exception as e:  # noqa: BLE001
        log.warning(f"Cannot write to {path}: {e}")
        return None

    results = {}
    try:
        for n in writer_counts:
            worker_dirs = []
            for w in range(n):
                d = os.path.join(scratch_root, f"n{n}_w{w}")
                os.makedirs(d, exist_ok=True)
                worker_dirs.append(d)

            # Run n writer processes concurrently, each for duration_s.
            with concurrent.futures.ProcessPoolExecutor(max_workers=n) as ex:
                futures = [
                    ex.submit(_write_worker, worker_dirs[w], w, duration_s, height, width, planes) for w in range(n)
                ]
                per_worker = [f.result() for f in futures]

            total_bytes = sum(b for b, _ in per_worker)
            # Use the max per-worker elapsed to exclude process-spawn skew from the rate.
            elapsed = max((e for _, e in per_worker), default=duration_s) or duration_s
            mb_per_s = (total_bytes / _BYTES_PER_MB) / elapsed
            results[n] = mb_per_s
            log.info(f"  {path}  writers={n:<2d}  {mb_per_s:8.1f} MB/s")

            # Free the space between writer-count trials.
            for d in worker_dirs:
                shutil.rmtree(d, ignore_errors=True)
    finally:
        if scratch_root:
            shutil.rmtree(scratch_root, ignore_errors=True)
    return results


def discover_paths():
    """Candidate save paths: configured default + writable disk partitions."""
    paths = []

    def add(p):
        if p and os.path.isdir(p) and p not in paths:
            paths.append(p)

    # Configured default saving path (already resolved, e.g. /Downloads -> ~/Downloads).
    try:
        from control._def import DEFAULT_SAVING_PATH

        add(DEFAULT_SAVING_PATH)
    except Exception as e:  # noqa: BLE001
        log.debug(f"Could not read DEFAULT_SAVING_PATH: {e}")

    # Last-used saving path, if cached.
    try:
        cache_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "last_saving_path.txt"
        )
        if os.path.isfile(cache_file):
            with open(cache_file) as fh:
                add(fh.read().strip())
    except Exception as e:  # noqa: BLE001
        log.debug(f"Could not read last saving path: {e}")

    # Writable disk partitions.
    try:
        import psutil

        for part in psutil.disk_partitions(all=False):
            add(part.mountpoint)
    except Exception as e:  # noqa: BLE001
        log.debug(f"Could not enumerate partitions: {e}")

    return paths


def free_gb(path):
    try:
        return shutil.disk_usage(path).free / (1024**3)
    except Exception:  # noqa: BLE001
        return float("nan")


def physical_cores():
    try:
        import psutil

        return psutil.cpu_count(logical=False) or (os.cpu_count() or 1)
    except Exception:  # noqa: BLE001
        return os.cpu_count() or 1


def recommend_writers(rates, cap):
    """Smallest writer count reaching >= 90% of this path's best throughput, capped at `cap`."""
    if not rates:
        return 1
    best = max(rates.values())
    for n in sorted(rates):
        if rates[n] >= 0.9 * best:
            return min(n, cap)
    return min(max(rates, key=rates.get), cap)


def find_config_file():
    """Locate the machine configuration*.ini, mirroring _def.py's discovery.

    Returns the path, or None if not found. When multiple exist, prefer the cached one
    (cache/config_file_path.txt), else the first alphabetically.
    """
    software_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = sorted(glob.glob(os.path.join(software_dir, "configuration*.ini")))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    cache_file = os.path.join(software_dir, "cache", "config_file_path.txt")
    if os.path.isfile(cache_file):
        try:
            with open(cache_file) as fh:
                cached = fh.read().strip()
            for c in candidates:
                if os.path.abspath(c) == os.path.abspath(cached):
                    return c
        except Exception:  # noqa: BLE001
            pass
    log.warning(f"Multiple config files found; defaulting to {candidates[0]}")
    return candidates[0]


def update_ini_keys(config_path, updates):
    """Surgically set key=value pairs inside the [GENERAL] section, in place.

    Preserves comments, ordering and formatting (unlike a configparser rewrite): existing
    keys are updated in place; missing keys are appended to the end of the [GENERAL] section.
    Returns a list of (key, old_value, new_value) describing what changed.
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    # Find the bounds of the [GENERAL] section.
    general_start = None
    general_end = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if general_start is None:
                if stripped.lower() == "[general]":
                    general_start = i
            else:
                general_end = i  # next section header ends GENERAL
                break

    changes = []
    remaining = {k.lower(): str(v) for k, v in updates.items()}

    if general_start is None:
        # No [GENERAL] section: create one at end of file.
        lines.append("\n[GENERAL]\n")
        general_start = len(lines) - 1
        general_end = len(lines)

    key_re = re.compile(r"^(\s*)([A-Za-z0-9_]+)(\s*=\s*)(.*?)(\s*)$")
    for i in range(general_start + 1, general_end):
        m = key_re.match(lines[i])
        if not m:
            continue
        key = m.group(2).lower()
        if key in remaining:
            old = m.group(4)
            new = remaining.pop(key)
            if old != new:
                changes.append((key, old, new))
            lines[i] = f"{m.group(2)} = {new}\n"

    # Append any keys that weren't already present, at the end of the GENERAL section.
    if remaining:
        insert = [f"{k} = {v}\n" for k, v in remaining.items()]
        for k, v in remaining.items():
            changes.append((k, "(absent)", v))
        lines[general_end:general_end] = insert

    with open(config_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    return changes


def prompt_yes_no(message):
    """Prompt for Y/N on stdin; default No. Returns True only on an explicit yes."""
    try:
        answer = input(message).strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def main(args):
    if args.verbose:
        import logging

        squid.logging.set_stdout_log_level(logging.DEBUG)

    writer_counts = sorted({int(x) for x in args.writers.split(",") if x.strip()})
    if args.paths:
        paths = [p.strip() for p in args.paths.split(",") if p.strip() and os.path.isdir(p.strip())]
    else:
        paths = discover_paths()

    if not paths:
        log.error("No candidate save paths found. Pass --paths explicitly.")
        return 1

    frame_mb = (args.height * args.width * 2) / _BYTES_PER_MB
    cores = physical_cores()
    print("=" * 72)
    print("Squid write-path benchmark")
    print(
        f"frame {args.width}x{args.height} uint16 (~{frame_mb:.1f} MB), planes/file={args.planes}, "
        f"{args.duration:.1f}s/trial, writers={writer_counts}, physical cores={cores}"
    )
    print("=" * 72)

    all_rates = {}
    for path in paths:
        print(f"\n{path}   (free {free_gb(path):.1f} GB)")
        rates = benchmark_path(path, writer_counts, args.duration, args.height, args.width, args.planes)
        if rates:
            all_rates[path] = rates

    if not all_rates:
        log.error("No path was writable/benchmarkable.")
        return 1

    # Results table.
    print("\n" + "=" * 72)
    print("Results (MB/s)")
    header = "  path".ljust(40) + "".join(f"{n:>8d}w" for n in writer_counts)
    print(header)
    for path, rates in all_rates.items():
        row = ("  " + path)[:40].ljust(40) + "".join(f"{rates.get(n, float('nan')):8.0f} " for n in writer_counts)
        print(row)

    # Recommendation: path with best peak throughput.
    fastest = max(all_rates, key=lambda p: max(all_rates[p].values()))
    best_rates = all_rates[fastest]
    best_mbps = max(best_rates.values())
    rec_writers = recommend_writers(best_rates, cores)
    single = best_rates.get(min(best_rates), best_mbps)
    speedup = best_mbps / single if single else 1.0
    suggested_backlog_mb = best_rates.get(rec_writers, best_mbps) * 30.0  # ~30s of write throughput

    print("\n" + "=" * 72)
    print("Recommendation")
    print(f"  Fastest path        : {fastest}")
    print(f"  Peak throughput     : {best_mbps:.0f} MB/s (single writer {single:.0f} MB/s, {speedup:.1f}x scaling)")
    print(f"  Writer processes    : {rec_writers}   -> set acquisition_writer_processes = {rec_writers}")
    print(f"  Save path           : set default_saving_path = {fastest}")
    print(
        f"  Adaptive cap        : keep acquisition_target_backlog_s = 30  "
        f"(~{suggested_backlog_mb / 1024:.1f} GB backlog at {best_rates.get(rec_writers, best_mbps):.0f} MB/s)"
    )
    print("=" * 72)

    # Offer to write the recommendation into the machine config, as two independent choices:
    # (1) the writer/throttle knobs, and (2) the save path.
    if args.no_apply:
        return 0

    config_path = find_config_file()
    if config_path is None:
        log.warning("No configuration*.ini found; cannot apply recommendation automatically.")
        return 0

    perf_updates = {
        "acquisition_writer_processes": rec_writers,
        "acquisition_target_backlog_s": 30.0,
    }
    path_updates = {"default_saving_path": fastest}

    updates = {}

    # Q1: writer / throttle settings.
    print(f"\nRecommended writer/throttle settings for {config_path}:")
    for key, value in perf_updates.items():
        print(f"    {key} = {value}")
    if args.yes or prompt_yes_no("Apply these writer/throttle settings to the .ini? [y/N] "):
        updates.update(perf_updates)

    # Q2: save path (separate decision).
    print(f"\nRecommended save path: default_saving_path = {fastest}")
    if args.yes or prompt_yes_no("Change the save path in the .ini too? [y/N] "):
        updates.update(path_updates)

    if not updates:
        print("No changes written.")
        return 0

    try:
        changes = update_ini_keys(config_path, updates)
    except Exception as e:  # noqa: BLE001
        log.error(f"Failed to update {config_path}: {e}")
        return 1

    if changes:
        print(f"Updated {config_path}:")
        for key, old, new in changes:
            print(f"    {key}: {old} -> {new}")
    else:
        print(f"{config_path} already matched the recommendation; nothing to change.")
    print("Restart the application to apply the new settings.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark disk write throughput for candidate save paths.")
    ap.add_argument("--writers", type=str, default="1,2,4,8", help="Comma-separated writer counts to test")
    ap.add_argument("--duration", type=float, default=3.0, help="Seconds per (path, writers) trial")
    ap.add_argument("--width", type=int, default=2048, help="Frame width (px)")
    ap.add_argument("--height", type=int, default=2048, help="Frame height (px)")
    ap.add_argument("--planes", type=int, default=4, help="Planes per OME-TIFF stack file")
    ap.add_argument("--paths", type=str, default="", help="Comma-separated paths to test (default: auto-detect)")
    ap.add_argument("--yes", action="store_true", help="Apply the recommendation to the .ini without prompting")
    ap.add_argument("--no-apply", action="store_true", help="Never offer to modify the .ini (report only)")
    ap.add_argument("--verbose", action="store_true", help="Turn on debug logging")
    args = ap.parse_args()
    sys.exit(main(args))
