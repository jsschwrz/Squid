---
name: build-portable
description: Build the self-contained portable Windows bundle of the Squid control software, so the folder can be copied to another machine and run without installing Python or any dependencies. Use when asked to build/rebuild the portable bundle, make Squid portable or self-contained, package Squid for another machine, produce the Squid-Portable zip, or when a copied Squid folder fails on another machine with ModuleNotFoundError.
---

# Build the portable Squid bundle

Produces `Squid-Portable-<yyyyMMdd>.zip` (~287 MB) containing the Python
interpreter, every dependency, and the app. The target machine unzips it and
double-clicks `Squid.cmd` — nothing to install.

## Background: why the repo is not portable on its own

Do not try to fix this by copying the folder or re-pointing a `.lnk`. It will
fail with `ModuleNotFoundError`.

- There is no venv, no lockfile, and no pinned Python in the repo. Every
  dependency lives in a user-scoped Python 3.12 install *outside* the repo.
- `control/_def.py` globs `./configuration*.ini` and reads
  `cache/config_file_path.txt` relative to the working directory, so the app
  **must** be launched with CWD = `software\`.
- On the build machine `C:\Python314\python.exe` shadows 3.12 on PATH and has
  none of the dependencies, so a bare `python main_hcs.py` fails even there.

The build script handles all three. Read its header comment before changing it.

## Run it

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File software\tools\build_portable_windows.ps1
```

Takes roughly 10 minutes; run it in the background and wait for the
notification rather than polling. Useful switches:

| Switch | Effect |
| --- | --- |
| `-KeepStaging` | leave the staging dir for inspection |
| `-SkipZip` | stop after staging |
| `-IncludeGit` | +365 MB, makes the target folder `git pull`-able |
| `-IncludeFirmware` | +50 MB, PlatformIO sources |
| `-SourcePython <dir>` | use a different interpreter as the source |

The script self-verifies: it smoke-tests the staged runtime's imports and
asserts the staged interpreter no longer resolves to the source install.

## Verify after building

1. Unzip to a path different from the repo, then run the bundle's
   `CHECK-DRIVERS.cmd`. Expect OK for the Toupcam main camera
   (ITR3CMOS26000KMA), the Daheng laser-AF camera (MER2-630-60U3M), and the
   microcontroller's COM port.
2. Confirm no reach-back into the source install:

   ```
   <bundle>\runtime\python312\python.exe -c "import sys, numpy, napari; print([p for p in sys.path if 'Programs\\Python' in p]); print(numpy.__file__)"
   ```

   The list must be empty and `numpy.__file__` must be inside the bundle.

**Do not launch the GUI to verify.** Startup can home the stage on this config.
Ask the user to run `Squid-Console.cmd` themselves with the stage clear.

## Two things that are not in the bundle, by design

- **The Daheng Galaxy SDK.** `control/gxipy/gxwrapper.py` loads `GxIAPI.dll` by
  bare name off the system PATH, and the SDK ships a kernel-mode USB driver, so
  it cannot travel in a zip. The target machine needs it installed, plus a
  reboot. The launchers prepend the SDK directories to PATH defensively.
- **Per-instrument calibration.** `software/user_profiles/default/` carries the
  channel configs and laser-AF calibration of the machine the bundle was built
  on. If the target drives a *different physical microscope*, say so — those
  values must be redone before an acquisition can be trusted.

## Gotchas worth remembering

- `default_saving_path` is rewritten to a bare `Downloads`. `_def.py` strips
  only slashes, not a `C:` drive prefix, so a foreign absolute path would be
  concatenated into garbage rather than falling back.
- Config files must stay BOM-less; `configparser` would otherwise choke on the
  first section header.
- `requirements-windows.txt` lives at the **repo root** on purpose —
  `software/.gitignore` ignores `*.txt`, so it would vanish silently there.
- The bundle carries git-ignored machine state (`configuration_Squid+.ini`,
  `cache/`, `user_profiles/`, `machine_configs/*.yaml`) that a fresh clone would
  not produce. That is why the build copies the working tree rather than cloning.
