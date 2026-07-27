# Write-Path Benchmark Tool — Guide

A short guide to `tools/benchmark_write_paths.py`: what it's for, and exactly how to run it
(written for someone new to the command line).

---

## 1. Why this tool exists

During an acquisition the microscope captures images faster than they can be written to disk.
Unwritten images pile up in RAM (a "backlog"). If that backlog grows too large, two bad things
happen:

1. The whole system **slows down** as RAM fills up (in one run it climbed to ~27 GB and throughput
   dropped ~4x toward the end).
2. At the end of the run the software only waits a limited time for the leftover images to finish
   saving — and if they can't, it **throws them away** (silent data loss).

The root cause was that images were written by a **single writer process**, one at a time. The disk
itself is usually fast — the bottleneck was doing all the writing serially.

The software now supports **multiple parallel writer processes** and an **adaptive cap** that keeps
the RAM backlog in check. To use them well you need two numbers:

- **How many writer processes** to use (more isn't always better — they compete for CPU).
- **Which save location** is fastest (if you have more than one drive).

This tool measures both on your actual machine, recommends settings, and can write them straight
into the configuration file.

---

## 2. How to run it

### Step 1 — Open a terminal

Press the **Windows key**, type **PowerShell**, and press **Enter**. A window with a text prompt
appears. This is where you type commands.

### Step 2 — Go to the `software` folder

Type `cd `, then drag the `software` folder into the window (which pastes its path), then press
**Enter**. It will look something like:

```powershell
cd "<path-to-your-Squid-checkout>\software"
```

### Step 3 — Run the benchmark

```powershell
py -3.10 tools\benchmark_write_paths.py
```

- `py -3.10` tells Windows to use Python 3.10 (the version the microscope software uses).
- The rest is the path to the tool.

On Linux, use `python3 tools/benchmark_write_paths.py`.

> **Tip — match your camera's image size for the most accurate result.** The default test uses a
> 2048x2048 image. If your camera produces larger frames, add `--width` and `--height`, e.g. for a
> 4096x4096 sensor:
>
> ```powershell
> py -3.10 tools\benchmark_write_paths.py --width 4096 --height 4096
> ```

The test takes roughly 15–30 seconds. It writes temporary files to each drive and then deletes
them — it does **not** touch your real acquisition data.

### Step 4 — Read the results

You'll see a table and a recommendation, like this:

```
Results (MB/s)
  path                       1w       2w       4w       8w
  C:\Data                   213      340      465      470
  D:\FastScratch            402      780      1120     1140

========================================================================
Recommendation
  Fastest path        : D:\FastScratch
  Peak throughput     : 1120 MB/s (single writer 402 MB/s, 2.8x scaling)
  Writer processes    : 4   -> set acquisition_writer_processes = 4
  Save path           : set default_saving_path = D:\FastScratch
  Adaptive cap        : keep acquisition_target_backlog_s = 30
========================================================================
```

- **Results table** — write speed (MB/s, higher is better) for each save location and each number of
  parallel writers (`1w`, `2w`, `4w`, `8w`).
- **Recommendation** — the fastest location and the number of writer processes worth using (the
  point where adding more stops helping much).

### Step 5 — Apply the recommendation (optional)

The tool asks **two separate yes/no questions**. For each, type `y` and press **Enter** to accept,
or just press **Enter** to skip:

```
Apply these writer/throttle settings to the .ini? [y/N]
Change the save path in the .ini too? [y/N]
```

- The **first** sets the number of writers and the RAM-backlog target.
- The **second** changes where images are saved (skip if you're happy with your current folder).

If you say yes it edits the configuration file in place, keeping all your other settings and
comments, and prints exactly what it changed.

### Step 6 — Restart the software

Config changes only take effect on the next launch. Close the microscope software and start it
again.

---

## 3. Options

| Flag | What it does | Example |
|---|---|---|
| `--width`, `--height` | Test image size in pixels (match your camera) | `--width 4096 --height 4096` |
| `--writers` | Which writer counts to test | `--writers 1,2,4` |
| `--duration` | Seconds spent on each test (longer = more accurate) | `--duration 5` |
| `--paths` | Test specific folders instead of auto-detecting | `--paths "C:\Data,D:\"` |
| `--yes` | Apply the recommendation without asking | `--yes` |
| `--no-apply` | Only report results; never offer to edit the file | `--no-apply` |
| `--verbose` | Show extra detail (for troubleshooting) | `--verbose` |

Full example:

```powershell
py -3.10 tools\benchmark_write_paths.py --width 4096 --height 4096 --duration 5
```

---

## 4. What "good" looks like

- If **more writers = clearly faster** (e.g. 1 writer 213 MB/s -> 4 writers 465 MB/s), set the
  recommended writer count and you'll capture faster without the RAM backlog piling up.
- If your acquisition's capture rate stays **below** the write speed the tool reports, the disk can
  keep up and you shouldn't see slowdowns or lost images.
- While a real acquisition runs, watch the status-bar readout showing capture vs write MB/s and a
  ratio — if the ratio stays around 1.0 or above, writes are keeping pace.

---

*Related config keys (section `[GENERAL]` of your `configuration*.ini`):*
`acquisition_writer_processes`, `acquisition_target_backlog_s`, `default_saving_path`.
