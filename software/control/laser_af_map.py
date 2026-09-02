"""Laser AF Map - a Z-sweep characterization run for troubleshooting laser autofocus.

For each XYZ point in a list, this drops the stage half a Z-range below the recorded
focus and steps back up in `delta Z` increments. At every step it captures a main-camera
image, a laser-AF-camera image, and the laser AF displacement readout. Afterwards it
optionally exercises the closed-loop AF (`move_to_target(0.0)`) N times at that point to
measure where AF actually lands and how much it scatters run to run.

The output is a pair of CSVs plus the image stacks. `displacement_um` plotted against
`z_offset_um` should be a straight line of slope 1 through the origin; deviations, NaNs,
or a `focus_measure` peak that disagrees with the AF zero-crossing are the failure
signatures this tool exists to find.

Entry point is the "Laser AF Map" button on the Laser Autofocus Settings panel, which
opens LaserAFMapDialog.
"""

import csv
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import imageio
import numpy as np
import yaml

# Match control/widgets.py's binding choice, but don't clobber a deliberate override: by the
# time this module is imported (lazily, from the button handler) qtpy is normally already bound.
os.environ.setdefault("QT_API", "pyqt5")

from qtpy.QtCore import *
from qtpy.QtWidgets import *
from qtpy.QtGui import *

import control.utils as utils
import control.utils_acquisition as utils_acquisition
import squid.logging
from control._def import (
    FocusMeasureOperator,
    SCAN_STABILIZATION_TIME_MS_X,
    SCAN_STABILIZATION_TIME_MS_Y,
    SCAN_STABILIZATION_TIME_MS_Z,
    SOFTWARE_POS_LIMIT,
)

# Distance the stage backs off below a target Z so every approach is made in the same
# (upward) direction, taking up leadscrew backlash consistently. _def.py has no general
# backlash constant, so this is local to the diagnostic.
LASER_AF_MAP_BACKLASH_CLEARANCE_MM = 0.020

SWEEP_CSV_NAME = "sweep.csv"
CLOSED_LOOP_CSV_NAME = "closed_loop.csv"
METADATA_NAME = "run_metadata.yaml"
IMAGES_DIR_NAME = "images"

SWEEP_COLUMNS = [
    "region_id",
    "point_index",
    "x_mm",
    "y_mm",
    "z_nominal_mm",
    "z_index",
    "z_commanded_mm",
    "z_actual_mm",
    "z_offset_um",
    "displacement_um",
    "spot_x_px",
    "spot_y_px",
    "spot_found",
    "focus_measure",
    "main_image",
    "af_image",
    "timestamp",
]

CLOSED_LOOP_COLUMNS = [
    "region_id",
    "point_index",
    "trial",
    "z_start_mm",
    "success",
    "z_final_mm",
    "z_error_um",
    "residual_displacement_um",
    "timestamp",
]


@dataclass
class LaserAFMapPoint:
    region_id: str
    x_mm: float
    y_mm: float
    z_mm: float


@dataclass
class LaserAFMapConfig:
    points: List[LaserAFMapPoint]
    channel_name: str
    delta_z_um: float
    z_range_um: float
    closed_loop_repeats: int
    base_path: str
    run_name: str
    save_main_images: bool = True
    save_af_images: bool = True

    @property
    def num_z_steps(self) -> int:
        if self.delta_z_um <= 0:
            return 1
        return int(round(self.z_range_um / self.delta_z_um)) + 1

    @property
    def num_points(self) -> int:
        return len(self.points)


@dataclass
class LaserAFMapProgress:
    point_index: int
    num_points: int
    region_id: str
    stage_name: str  # "sweep" or "closed loop"
    step_index: int
    num_steps: int
    z_offset_um: float = float("nan")
    displacement_um: float = float("nan")
    message: str = ""


@dataclass
class LaserAFMapCallbacks:
    """Plain callables, following control/core/multi_point_utils.MultiPointControllerFunctions.

    The worker runs on a plain thread and must never touch Qt widgets directly; the
    progress dialog supplies bound methods on a QObject that re-emit these as signals.
    """

    on_progress: Callable[[LaserAFMapProgress], None] = lambda _p: None
    on_finished: Callable[[str], None] = lambda _path: None
    on_error: Callable[[str], None] = lambda _msg: None


class _CsvAppender:
    """Row-at-a-time CSV writer that flushes after every row.

    Written incrementally (rather than built up and dumped with pandas at the end) so an
    aborted or crashed run still leaves usable data on disk.
    """

    def __init__(self, path: str, columns: List[str]):
        self._path = path
        self._columns = columns
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=columns)
        self._writer.writeheader()
        self._file.flush()

    def append(self, row: dict) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


def _git_revision(repo_dir: str) -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL, timeout=5)
            .decode()
            .strip()
        )
    except Exception:
        return None


class LaserAFMapWorker:
    """Runs the Z sweep + closed-loop trials. Blocking `run()`, driven on a plain thread.

    Mirrors how MultiPointWorker is driven (threading.Thread, plain callbacks) rather than
    QThread/moveToThread.
    """

    def __init__(
        self,
        microscope,
        laser_af_controller,
        config: LaserAFMapConfig,
        callbacks: Optional[LaserAFMapCallbacks] = None,
    ):
        self._log = squid.logging.get_logger(self.__class__.__name__)
        self._microscope = microscope
        self._laser_af = laser_af_controller
        self._config = config
        self._callbacks = callbacks or LaserAFMapCallbacks()

        self._stage = microscope.stage
        self._live_controller = microscope.live_controller
        self._live_controller_focus = getattr(microscope, "live_controller_focus", None)
        self._microcontroller = microscope.low_level_drivers.microcontroller

        self._abort = threading.Event()
        self._channel = None
        self._was_live = False
        self._was_focus_live = False
        self._skipped_points: List[str] = []

    # ------------------------------------------------------------------ control

    def request_abort(self) -> None:
        self._log.info("Laser AF Map abort requested")
        self._abort.set()

    @property
    def aborted(self) -> bool:
        return self._abort.is_set()

    # ------------------------------------------------------------------ run

    def run(self) -> None:
        output_dir = None
        sweep_csv = None
        closed_loop_csv = None
        try:
            output_dir = self._prepare_output_dir()
            images_dir = os.path.join(output_dir, IMAGES_DIR_NAME)
            utils.ensure_directory_exists(images_dir)

            self._enter_measurement_state()
            try:
                self._write_metadata(output_dir)
                sweep_csv = _CsvAppender(os.path.join(output_dir, SWEEP_CSV_NAME), SWEEP_COLUMNS)
                closed_loop_csv = _CsvAppender(os.path.join(output_dir, CLOSED_LOOP_CSV_NAME), CLOSED_LOOP_COLUMNS)

                for point_index, point in enumerate(self._config.points):
                    if self._abort.is_set():
                        break
                    self._run_point(point_index, point, images_dir, sweep_csv, closed_loop_csv)
            finally:
                if sweep_csv is not None:
                    sweep_csv.close()
                if closed_loop_csv is not None:
                    closed_loop_csv.close()
                self._exit_measurement_state()

            utils.create_done_file(output_dir)
            self._log.info(f"Laser AF Map finished, results in {output_dir}")
            self._callbacks.on_finished(output_dir)
        except Exception as e:
            self._log.exception("Laser AF Map run failed")
            self._callbacks.on_error(f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ setup / teardown

    def _prepare_output_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = self._config.run_name.strip() or "laser_af_map"
        output_dir = os.path.join(self._config.base_path, f"{name}_{timestamp}")
        utils.ensure_directory_exists(output_dir)
        return output_dir

    def _enter_measurement_state(self) -> None:
        self._was_live = bool(getattr(self._live_controller, "is_live", False))
        if self._was_live:
            self._live_controller.stop_live()

        if self._live_controller_focus is not None:
            self._was_focus_live = bool(getattr(self._live_controller_focus, "is_live", False))
            if self._was_focus_live:
                self._live_controller_focus.stop_live()

        objective = self._microscope.objective_store.current_objective
        channel = self._live_controller.get_channel_by_name(objective, self._config.channel_name)
        if channel is None:
            available = [ch.name for ch in self._live_controller.get_channels(objective)]
            raise ValueError(
                f"Channel '{self._config.channel_name}' not found for objective '{objective}'. Available: {available}"
            )
        self._channel = channel
        self._live_controller.set_microscope_mode(channel)
        self._microcontroller.wait_till_operation_is_completed()
        self._microscope.camera.start_streaming()

    def _exit_measurement_state(self) -> None:
        # The laser is toggled around every measurement, but make certain it is off if we
        # bailed out mid-measurement.
        try:
            self._laser_af._turn_off_laser()
        except Exception:
            self._log.exception("Failed to turn the AF laser off during teardown")

        # LaserAutofocusController._get_laser_spot_centroid() disables focus-camera
        # callbacks and never re-enables them. Without this the focus live view stays dead
        # after the run.
        try:
            self._laser_af.camera.enable_callbacks(True)
        except Exception:
            self._log.exception("Failed to re-enable focus camera callbacks")

        if self._was_focus_live and self._live_controller_focus is not None:
            try:
                self._live_controller_focus.start_live()
            except Exception:
                self._log.exception("Failed to restart focus camera live view")

        if self._was_live:
            try:
                self._live_controller.start_live()
            except Exception:
                self._log.exception("Failed to restart live view")

    def _write_metadata(self, output_dir: str) -> None:
        properties = self._laser_af.laser_af_properties
        # reference_image is a base64 blob; excluding it keeps the metadata readable.
        properties_dump = properties.model_dump(exclude={"reference_image"})

        metadata = {
            "run_name": self._config.run_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f"),
            "objective": self._microscope.objective_store.current_objective,
            "channel": {
                "name": self._channel.name,
                "exposure_time_ms": self._channel.exposure_time,
                "analog_gain": self._channel.analog_gain,
                "illumination_intensity": self._channel.illumination_intensity,
            },
            "sweep": {
                "delta_z_um": self._config.delta_z_um,
                "z_range_um": self._config.z_range_um,
                "num_z_steps": self._config.num_z_steps,
                "closed_loop_repeats": self._config.closed_loop_repeats,
                "backlash_clearance_mm": LASER_AF_MAP_BACKLASH_CLEARANCE_MM,
            },
            "points": [
                {"region_id": p.region_id, "x_mm": p.x_mm, "y_mm": p.y_mm, "z_mm": p.z_mm} for p in self._config.points
            ],
            "laser_af_properties": properties_dump,
            "laser_af_is_initialized": self._laser_af.is_initialized,
            "git_revision": _git_revision(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        }

        with open(os.path.join(output_dir, METADATA_NAME), "w", encoding="utf-8") as f:
            yaml.dump(utils.serialize_for_yaml(metadata), f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------ per point

    def _quantized_delta_z_mm(self) -> float:
        """Round delta Z to a whole number of Z microsteps.

        Same idiom as FlexibleMultiPointWidget.set_deltaZ. This build is stepper-only with
        no objective piezo, so all Z motion is stage motion.
        """
        delta_z_mm = self._config.delta_z_um / 1000.0
        try:
            mm_per_ustep = 1.0 / self._stage.get_config().Z_AXIS.convert_real_units_to_ustep(1.0)
        except Exception:
            self._log.exception("Could not read Z microstep size; using unquantized delta Z")
            return delta_z_mm
        if mm_per_ustep <= 0:
            return delta_z_mm
        return round(delta_z_mm / mm_per_ustep) * mm_per_ustep

    def _approach_z_from_below(self, z_mm: float) -> None:
        below = max(z_mm - LASER_AF_MAP_BACKLASH_CLEARANCE_MM, SOFTWARE_POS_LIMIT.Z_NEGATIVE)
        self._stage.move_z_to(below)
        self._stage.move_z_to(z_mm)
        time.sleep(SCAN_STABILIZATION_TIME_MS_Z / 1000.0)

    def _run_point(
        self,
        point_index: int,
        point: LaserAFMapPoint,
        images_dir: str,
        sweep_csv: _CsvAppender,
        closed_loop_csv: _CsvAppender,
    ) -> None:
        num_z = self._config.num_z_steps
        delta_z_mm = self._quantized_delta_z_mm()
        z_start = point.z_mm - (self._config.z_range_um / 2.0) / 1000.0
        z_end = z_start + delta_z_mm * (num_z - 1)

        if z_start < SOFTWARE_POS_LIMIT.Z_NEGATIVE or z_end > SOFTWARE_POS_LIMIT.Z_POSITIVE:
            message = (
                f"Skipping {point.region_id}: sweep spans {z_start:.4f}-{z_end:.4f} mm, outside the "
                f"software Z limits ({SOFTWARE_POS_LIMIT.Z_NEGATIVE}-{SOFTWARE_POS_LIMIT.Z_POSITIVE} mm)"
            )
            self._log.warning(message)
            self._skipped_points.append(point.region_id)
            self._callbacks.on_progress(
                LaserAFMapProgress(
                    point_index=point_index,
                    num_points=self._config.num_points,
                    region_id=point.region_id,
                    stage_name="skipped",
                    step_index=0,
                    num_steps=num_z,
                    message=message,
                )
            )
            return

        self._stage.move_x_to(point.x_mm)
        time.sleep(SCAN_STABILIZATION_TIME_MS_X / 1000.0)
        self._stage.move_y_to(point.y_mm)
        time.sleep(SCAN_STABILIZATION_TIME_MS_Y / 1000.0)

        try:
            self._approach_z_from_below(z_start)

            for z_index in range(num_z):
                if self._abort.is_set():
                    return
                z_commanded = z_start + delta_z_mm * z_index
                self._record_sweep_step(point_index, point, z_index, z_commanded, images_dir, sweep_csv)
                if z_index < num_z - 1:
                    self._stage.move_z(delta_z_mm)
                    time.sleep(SCAN_STABILIZATION_TIME_MS_Z / 1000.0)

            if not self._abort.is_set() and self._config.closed_loop_repeats > 0:
                self._run_closed_loop(point_index, point, z_start, closed_loop_csv)
        finally:
            # Always leave the stage at the point's nominal focus, including on abort.
            try:
                self._approach_z_from_below(point.z_mm)
            except Exception:
                self._log.exception(f"Failed to restore Z to {point.z_mm} mm after {point.region_id}")

    def _record_sweep_step(
        self,
        point_index: int,
        point: LaserAFMapPoint,
        z_index: int,
        z_commanded: float,
        images_dir: str,
        sweep_csv: _CsvAppender,
    ) -> None:
        z_actual = self._stage.get_pos().z_mm
        file_id = f"{point.region_id}_z{z_index:03d}"

        main_image_path = ""
        focus_measure = float("nan")
        main_image = self._microscope.acquire_image()
        if main_image is not None:
            try:
                focus_measure = float(utils.calculate_focus_measure(main_image, FocusMeasureOperator.LAPE))
            except Exception:
                self._log.exception("Failed to compute focus measure")
            if self._config.save_main_images:
                main_image_path = utils_acquisition.get_image_filepath(
                    images_dir, file_id, self._channel.name, main_image.dtype
                )
                utils_acquisition.save_image(
                    image=main_image,
                    file_id=file_id,
                    save_directory=images_dir,
                    config=self._channel,
                    is_color=len(main_image.shape) > 2,
                )

        displacement_um, spot_x, spot_y, af_image = self._measure_laser_af()

        af_image_path = ""
        if self._config.save_af_images and af_image is not None:
            af_image_path = os.path.join(images_dir, f"{file_id}_laser_af.bmp")
            imageio.imwrite(af_image_path, af_image)

        sweep_csv.append(
            {
                "region_id": point.region_id,
                "point_index": point_index,
                "x_mm": point.x_mm,
                "y_mm": point.y_mm,
                "z_nominal_mm": point.z_mm,
                "z_index": z_index,
                "z_commanded_mm": z_commanded,
                "z_actual_mm": z_actual,
                "z_offset_um": (z_actual - point.z_mm) * 1000.0,
                "displacement_um": displacement_um,
                "spot_x_px": spot_x,
                "spot_y_px": spot_y,
                "spot_found": not math.isnan(spot_x),
                "focus_measure": focus_measure,
                "main_image": os.path.basename(main_image_path) if main_image_path else "",
                "af_image": os.path.basename(af_image_path) if af_image_path else "",
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f"),
            }
        )

        self._callbacks.on_progress(
            LaserAFMapProgress(
                point_index=point_index,
                num_points=self._config.num_points,
                region_id=point.region_id,
                stage_name="sweep",
                step_index=z_index,
                num_steps=self._config.num_z_steps,
                z_offset_um=(z_actual - point.z_mm) * 1000.0,
                displacement_um=displacement_um,
            )
        )

    def _measure_laser_af(self) -> Tuple[float, float, float, Optional[np.ndarray]]:
        """One laser-on window yielding displacement, raw spot pixels, and the AF frame.

        Deliberately does NOT use LaserAutofocusController.measure_displacement(): with its
        default search_for_spot=True, a failed detection triggers a +/- laser_af_range stage
        scan and returns from the new Z *without restoring it*, which would silently corrupt
        the map. This is equivalent to measure_displacement(search_for_spot=False) plus the
        centroid and the image.
        """
        laser_af = self._laser_af
        # _get_laser_spot_centroid stores each frame it reads on .image; clear it first so we
        # never save a stale frame from a previous step when every read fails.
        laser_af.image = None

        centroid = None
        try:
            laser_af._turn_on_laser()
        except TimeoutError:
            self._log.exception("Turning on the AF laser timed out")
            return float("nan"), float("nan"), float("nan"), None

        try:
            centroid = laser_af._get_laser_spot_centroid()
        except Exception:
            self._log.exception("Laser spot detection raised")
        finally:
            try:
                laser_af._turn_off_laser()
            except TimeoutError:
                self._log.exception("Turning off the AF laser timed out! The laser may still be on.")

        af_image = laser_af.image
        if centroid is None:
            return float("nan"), float("nan"), float("nan"), af_image

        displacement_um = laser_af._get_displacement_from_centroid(centroid)
        return displacement_um, float(centroid[0]), float(centroid[1]), af_image

    def _run_closed_loop(
        self, point_index: int, point: LaserAFMapPoint, z_start: float, closed_loop_csv: _CsvAppender
    ) -> None:
        for trial in range(self._config.closed_loop_repeats):
            if self._abort.is_set():
                return

            # Every trial pulls in from the same starting offset so the trials are comparable.
            self._approach_z_from_below(z_start)

            success = False
            try:
                success = bool(self._laser_af.move_to_target(0.0))
            except Exception:
                self._log.exception(f"move_to_target failed on {point.region_id} trial {trial}")

            z_final = self._stage.get_pos().z_mm
            try:
                residual = self._laser_af.measure_displacement(search_for_spot=False)
            except Exception:
                self._log.exception("Residual displacement measurement failed")
                residual = float("nan")

            closed_loop_csv.append(
                {
                    "region_id": point.region_id,
                    "point_index": point_index,
                    "trial": trial,
                    "z_start_mm": z_start,
                    "success": success,
                    "z_final_mm": z_final,
                    "z_error_um": (z_final - point.z_mm) * 1000.0,
                    "residual_displacement_um": residual,
                    "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f"),
                }
            )

            self._callbacks.on_progress(
                LaserAFMapProgress(
                    point_index=point_index,
                    num_points=self._config.num_points,
                    region_id=point.region_id,
                    stage_name="closed loop",
                    step_index=trial,
                    num_steps=self._config.closed_loop_repeats,
                    z_offset_um=(z_final - point.z_mm) * 1000.0,
                    displacement_um=residual,
                )
            )


class _WorkerSignals(QObject):
    """Bridges the worker's plain callbacks onto the GUI thread."""

    progress = Signal(object)
    finished = Signal(str)
    error = Signal(str)


class LaserAFMapProgressDialog(QDialog):
    def __init__(self, worker: LaserAFMapWorker, parent=None):
        super().__init__(parent)
        self._log = squid.logging.get_logger(self.__class__.__name__)
        self._worker = worker
        self._thread: Optional[threading.Thread] = None
        self._done = False

        self.setWindowTitle("Laser AF Map")
        self.setModal(True)
        self.setMinimumWidth(460)

        self.label_point = QLabel("Starting...")
        self.progress_points = QProgressBar()
        self.progress_points.setRange(0, max(1, worker._config.num_points))
        self.progress_steps = QProgressBar()
        self.progress_steps.setRange(0, max(1, worker._config.num_z_steps))
        self.label_readout = QLabel("")
        self.label_readout.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        self.btn_abort = QPushButton("Abort")
        self.btn_close = QPushButton("Close")
        self.btn_close.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.btn_abort)
        buttons.addWidget(self.btn_close)

        layout = QVBoxLayout()
        layout.addWidget(self.label_point)
        layout.addWidget(QLabel("Points"))
        layout.addWidget(self.progress_points)
        layout.addWidget(QLabel("Steps at this point"))
        layout.addWidget(self.progress_steps)
        layout.addWidget(self.label_readout)
        layout.addLayout(buttons)
        self.setLayout(layout)

        self._signals = _WorkerSignals()
        self._signals.progress.connect(self._on_progress)
        self._signals.finished.connect(self._on_finished)
        self._signals.error.connect(self._on_error)

        self._worker._callbacks = LaserAFMapCallbacks(
            on_progress=self._signals.progress.emit,
            on_finished=self._signals.finished.emit,
            on_error=self._signals.error.emit,
        )

        self.btn_abort.clicked.connect(self._on_abort)
        self.btn_close.clicked.connect(self.accept)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker.run, name="Laser AF Map", daemon=True)
        self._thread.start()

    def _on_abort(self) -> None:
        self.btn_abort.setEnabled(False)
        self.label_point.setText("Aborting after the current step...")
        self._worker.request_abort()

    def _on_progress(self, progress: LaserAFMapProgress) -> None:
        self.label_point.setText(
            f"Point {progress.point_index + 1}/{progress.num_points} ({progress.region_id}) - {progress.stage_name}"
        )
        self.progress_points.setValue(progress.point_index)
        self.progress_steps.setRange(0, max(1, progress.num_steps))
        self.progress_steps.setValue(progress.step_index + 1)
        if progress.message:
            self.label_readout.setText(progress.message)
        else:
            self.label_readout.setText(
                f"z offset {progress.z_offset_um:+.2f} μm    displacement {progress.displacement_um:+.2f} μm"
            )

    def _finish(self) -> None:
        self._done = True
        self.btn_abort.setEnabled(False)
        self.btn_close.setEnabled(True)

    def _on_finished(self, output_dir: str) -> None:
        self.progress_points.setValue(self.progress_points.maximum())
        self.label_point.setText("Aborted." if self._worker.aborted else "Finished.")
        skipped = self._worker._skipped_points
        skipped_note = f"\nSkipped (outside Z limits): {', '.join(skipped)}" if skipped else ""
        self.label_readout.setText(f"Results saved to:\n{output_dir}{skipped_note}")
        self._finish()

    def _on_error(self, message: str) -> None:
        self.label_point.setText("Run failed.")
        self.label_readout.setText(message)
        self._finish()
        QMessageBox.critical(self, "Laser AF Map Failed", message)

    def closeEvent(self, event):
        if not self._done:
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self):
        # Don't let Esc close the dialog out from under a running sweep.
        if not self._done:
            return
        super().reject()


class LaserAFMapDialog(QDialog):
    """Confirm-and-configure window for a Laser AF Map run."""

    def __init__(
        self,
        laser_af_controller,
        microscope,
        objective_store,
        live_controller,
        get_flexible_points: Optional[Callable[[], Tuple[np.ndarray, np.ndarray]]] = None,
        default_base_path: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._log = squid.logging.get_logger(self.__class__.__name__)
        self._laser_af = laser_af_controller
        self._microscope = microscope
        self._objective_store = objective_store
        self._live_controller = live_controller
        self._get_flexible_points = get_flexible_points

        self._points: List[LaserAFMapPoint] = []
        self._points_source = "Flexible Multipoint tab"
        self.config: Optional[LaserAFMapConfig] = None

        self.setWindowTitle("Laser AF Map")
        self.setModal(True)
        self.setMinimumWidth(560)

        self._build_ui(default_base_path)
        self._load_points_from_flexible_multipoint()
        self._refresh_reference_banner()
        self._update_summary()

        if hasattr(self._laser_af, "signal_reference_changed"):
            self._laser_af.signal_reference_changed.connect(self._on_reference_changed)

    # ------------------------------------------------------------------ UI

    def _build_ui(self, default_base_path: str) -> None:
        layout = QVBoxLayout()

        self.label_reference_warning = QLabel()
        self.label_reference_warning.setWordWrap(True)
        self.label_reference_warning.setStyleSheet("color: #B00020; font-weight: bold;")
        layout.addWidget(self.label_reference_warning)

        # --- coordinate source ---
        points_group = QGroupBox("XYZ coordinates")
        points_layout = QVBoxLayout()
        self.label_points_source = QLabel()
        points_layout.addWidget(self.label_points_source)

        self.table_points = QTableWidget(0, 4)
        self.table_points.setHorizontalHeaderLabels(["x (mm)", "y (mm)", "z (μm)", "ID"])
        self.table_points.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_points.setSelectionMode(QAbstractItemView.NoSelection)
        self.table_points.setMaximumHeight(160)
        points_layout.addWidget(self.table_points)

        points_buttons = QHBoxLayout()
        self.btn_reload_points = QPushButton("Reload from Flexible Multipoint")
        self.btn_import_points = QPushButton("Import CSV...")
        self.btn_export_points = QPushButton("Export CSV...")
        points_buttons.addWidget(self.btn_reload_points)
        points_buttons.addWidget(self.btn_import_points)
        points_buttons.addWidget(self.btn_export_points)
        points_layout.addLayout(points_buttons)
        points_group.setLayout(points_layout)
        layout.addWidget(points_group)

        # --- channel ---
        channel_group = QGroupBox("Main camera channel")
        channel_layout = QHBoxLayout()
        self.dropdown_channel = QComboBox()
        self.label_channel_details = QLabel()
        channel_layout.addWidget(self.dropdown_channel)
        channel_layout.addWidget(self.label_channel_details)
        channel_layout.addStretch()
        channel_group.setLayout(channel_layout)
        layout.addWidget(channel_group)
        self._populate_channels()

        # --- sweep parameters ---
        sweep_group = QGroupBox("Z sweep")
        sweep_layout = QGridLayout()

        self.entry_delta_z = QDoubleSpinBox()
        self.entry_delta_z.setKeyboardTracking(False)
        self.entry_delta_z.setRange(0.01, 100.0)
        self.entry_delta_z.setDecimals(3)
        self.entry_delta_z.setSingleStep(0.1)
        self.entry_delta_z.setValue(1.0)
        self.entry_delta_z.setSuffix(" μm")

        self.entry_z_range = QDoubleSpinBox()
        self.entry_z_range.setKeyboardTracking(False)
        self.entry_z_range.setRange(0.1, 2000.0)
        self.entry_z_range.setDecimals(2)
        self.entry_z_range.setSingleStep(1.0)
        self.entry_z_range.setValue(20.0)
        self.entry_z_range.setSuffix(" μm")

        self.entry_repeats = QSpinBox()
        self.entry_repeats.setKeyboardTracking(False)
        self.entry_repeats.setRange(0, 100)
        self.entry_repeats.setValue(3)

        sweep_layout.addWidget(QLabel("Delta Z"), 0, 0)
        sweep_layout.addWidget(self.entry_delta_z, 0, 1)
        sweep_layout.addWidget(QLabel("Z range"), 1, 0)
        sweep_layout.addWidget(self.entry_z_range, 1, 1)
        sweep_layout.addWidget(QLabel("Closed-loop repeats per point"), 2, 0)
        sweep_layout.addWidget(self.entry_repeats, 2, 1)
        sweep_group.setLayout(sweep_layout)
        layout.addWidget(sweep_group)

        # --- output ---
        output_group = QGroupBox("Output")
        output_layout = QGridLayout()
        self.lineEdit_base_path = QLineEdit(default_base_path)
        self.lineEdit_base_path.setReadOnly(True)
        self.btn_browse = QPushButton("Browse...")
        self.lineEdit_run_name = QLineEdit("laser_af_map")
        output_layout.addWidget(QLabel("Save directory"), 0, 0)
        output_layout.addWidget(self.lineEdit_base_path, 0, 1)
        output_layout.addWidget(self.btn_browse, 0, 2)
        output_layout.addWidget(QLabel("Run name"), 1, 0)
        output_layout.addWidget(self.lineEdit_run_name, 1, 1, 1, 2)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        self.label_summary = QLabel()
        self.label_summary.setWordWrap(True)
        layout.addWidget(self.label_summary)

        self.label_range_warning = QLabel()
        self.label_range_warning.setWordWrap(True)
        self.label_range_warning.setStyleSheet("color: #A05000;")
        layout.addWidget(self.label_range_warning)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.btn_proceed = QPushButton("Proceed")
        self.btn_proceed.setDefault(True)
        self.btn_cancel = QPushButton("Cancel")
        buttons.addWidget(self.btn_proceed)
        buttons.addWidget(self.btn_cancel)
        layout.addLayout(buttons)

        self.setLayout(layout)

        self.btn_reload_points.clicked.connect(self._load_points_from_flexible_multipoint)
        self.btn_import_points.clicked.connect(self._import_points)
        self.btn_export_points.clicked.connect(self._export_points)
        self.btn_browse.clicked.connect(self._browse_base_path)
        self.entry_delta_z.valueChanged.connect(self._update_summary)
        self.entry_z_range.valueChanged.connect(self._update_summary)
        self.entry_repeats.valueChanged.connect(self._update_summary)
        self.dropdown_channel.currentTextChanged.connect(self._update_channel_details)
        self.btn_proceed.clicked.connect(self._on_proceed)
        self.btn_cancel.clicked.connect(self.reject)

    def _populate_channels(self) -> None:
        self.dropdown_channel.blockSignals(True)
        self.dropdown_channel.clear()
        try:
            objective = self._objective_store.current_objective
            for channel in self._live_controller.get_channels(objective):
                self.dropdown_channel.addItem(channel.name)
            current = getattr(self._live_controller, "currentConfiguration", None)
            if current is not None:
                index = self.dropdown_channel.findText(current.name)
                if index >= 0:
                    self.dropdown_channel.setCurrentIndex(index)
        except Exception:
            self._log.exception("Failed to populate the channel list")
        self.dropdown_channel.blockSignals(False)
        self._update_channel_details()

    def _update_channel_details(self) -> None:
        name = self.dropdown_channel.currentText()
        if not name:
            self.label_channel_details.setText("")
            return
        try:
            channel = self._live_controller.get_channel_by_name(self._objective_store.current_objective, name)
        except Exception:
            channel = None
        if channel is None:
            self.label_channel_details.setText("")
            return
        self.label_channel_details.setText(
            f"{channel.exposure_time:g} ms, gain {channel.analog_gain:g}, intensity {channel.illumination_intensity:g}"
        )

    # ------------------------------------------------------------------ points

    def _set_points(self, points: List[LaserAFMapPoint], source: str) -> None:
        self._points = points
        self._points_source = source
        self.label_points_source.setText(f"Source: {source} — {len(points)} point(s)")
        self.table_points.setRowCount(0)
        for point in points:
            row = self.table_points.rowCount()
            self.table_points.insertRow(row)
            self.table_points.setItem(row, 0, QTableWidgetItem(f"{point.x_mm:.3f}"))
            self.table_points.setItem(row, 1, QTableWidgetItem(f"{point.y_mm:.3f}"))
            self.table_points.setItem(row, 2, QTableWidgetItem(f"{point.z_mm * 1000:.1f}"))
            self.table_points.setItem(row, 3, QTableWidgetItem(point.region_id))
        self._update_summary()

    def _load_points_from_flexible_multipoint(self) -> None:
        if self._get_flexible_points is None:
            self._set_points([], "Flexible Multipoint tab unavailable — import a CSV")
            return
        try:
            location_list, location_ids = self._get_flexible_points()
        except Exception:
            self._log.exception("Failed to read the Flexible Multipoint location list")
            self._set_points([], "Flexible Multipoint tab unavailable — import a CSV")
            return

        points = []
        for i, row in enumerate(np.atleast_2d(np.asarray(location_list, dtype=float))):
            if row.size < 3:
                continue
            region_id = str(location_ids[i]) if location_ids is not None and i < len(location_ids) else f"R{i}"
            points.append(
                LaserAFMapPoint(region_id=region_id, x_mm=float(row[0]), y_mm=float(row[1]), z_mm=float(row[2]))
            )
        self._set_points(points, "Flexible Multipoint tab")

    def _import_points(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import coordinates", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            import pandas as pd

            frame = pd.read_csv(path)
            required = ["x (mm)", "y (mm)", "z (mm)"]
            missing = [column for column in required if column not in frame.columns]
            if missing:
                raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")
            points = []
            for i, row in frame.iterrows():
                region_id = str(row["ID"]) if "ID" in frame.columns else f"R{i}"
                points.append(
                    LaserAFMapPoint(
                        region_id=region_id,
                        x_mm=float(row["x (mm)"]),
                        y_mm=float(row["y (mm)"]),
                        z_mm=float(row["z (mm)"]),
                    )
                )
            self._set_points(points, os.path.basename(path))
        except Exception as e:
            QMessageBox.warning(self, "Import Failed", f"Could not import '{path}':\n{e}")

    def _export_points(self) -> None:
        if not self._points:
            QMessageBox.information(self, "Nothing to Export", "There are no coordinates to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export coordinates", "coordinates.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            import pandas as pd

            pd.DataFrame(
                [{"x (mm)": p.x_mm, "y (mm)": p.y_mm, "z (mm)": p.z_mm, "ID": p.region_id} for p in self._points]
            ).to_csv(path, index=False)
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", f"Could not export to '{path}':\n{e}")

    def _browse_base_path(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select save directory", self.lineEdit_base_path.text())
        if path:
            self.lineEdit_base_path.setText(path)
            self._update_summary()

    # ------------------------------------------------------------------ state

    def _has_reference(self) -> bool:
        try:
            return bool(self._laser_af.laser_af_properties.has_reference)
        except Exception:
            return False

    def _on_reference_changed(self, _has_reference) -> None:
        self._refresh_reference_banner()
        self._update_summary()

    def _refresh_reference_banner(self) -> None:
        if self._has_reference():
            self.label_reference_warning.setText("")
            self.label_reference_warning.setVisible(False)
        else:
            self.label_reference_warning.setVisible(True)
            self.label_reference_warning.setText(
                "No laser AF reference is set, so every displacement reading would be NaN. "
                "Click 'Set Reference' on the Laser AF tab first."
            )

    def _estimated_seconds(self, num_points: int, num_z: int, repeats: int) -> float:
        try:
            main_exposure_s = self._microscope.camera.get_exposure_time() / 1000.0
        except Exception:
            main_exposure_s = 0.05
        try:
            properties = self._laser_af.laser_af_properties
            af_step_s = properties.laser_af_averaging_n * properties.focus_camera_exposure_time_ms / 1000.0
        except Exception:
            af_step_s = 0.05
        per_step_s = main_exposure_s + af_step_s + 0.25 + SCAN_STABILIZATION_TIME_MS_Z / 1000.0
        per_point_s = (
            num_z * per_step_s
            + repeats * 3.0
            + (SCAN_STABILIZATION_TIME_MS_X + SCAN_STABILIZATION_TIME_MS_Y) / 1000.0
            + 2.0
        )
        return num_points * per_point_s

    def _estimated_bytes_per_step(self) -> int:
        try:
            width, height = self._microscope.camera.get_resolution()
            pixel_format = str(self._microscope.camera.get_pixel_format())
            bytes_per_pixel = 1 if "8" in pixel_format else 2
            channels = 3 if self._microscope.camera.is_color else 1
            main_bytes = width * height * bytes_per_pixel * channels
        except Exception:
            main_bytes = 5 * 1024 * 1024
        try:
            properties = self._laser_af.laser_af_properties
            af_bytes = int(properties.width) * int(properties.height)
        except Exception:
            af_bytes = 512 * 1024
        return main_bytes + af_bytes

    def _update_summary(self) -> None:
        delta_z = self.entry_delta_z.value()
        z_range = self.entry_z_range.value()
        repeats = self.entry_repeats.value()
        num_points = len(self._points)
        num_z = int(round(z_range / delta_z)) + 1 if delta_z > 0 else 1

        total_images = num_points * num_z * 2
        total_bytes = num_points * num_z * self._estimated_bytes_per_step()
        seconds = self._estimated_seconds(num_points, num_z, repeats)

        self.label_summary.setText(
            f"{num_z} Z steps per point, spanning {-z_range / 2:+.2f} to {z_range / 2:+.2f} μm "
            f"around each point's recorded Z.\n"
            f"{num_points} point(s) → {total_images:,} images, ~{total_bytes / 1024 / 1024:,.0f} MB, "
            f"~{seconds / 60:.1f} min, plus {num_points * repeats} closed-loop trial(s)."
        )

        # _get_laser_spot_centroid rejects spots farther than displacement_success_window_pixels
        # from the reference, so wide sweeps read NaN at the extremes by design.
        warning = ""
        try:
            properties = self._laser_af.laser_af_properties
            window_um = properties.displacement_success_window_pixels * properties.pixel_to_um
            if z_range / 2.0 > window_um:
                warning = (
                    f"Z range exceeds the laser AF displacement window (±{window_um:.0f} μm). "
                    f"Steps beyond that will read NaN by design — raise 'Displacement Success Window' "
                    f"or shorten the range if you need readings across the whole sweep."
                )
        except Exception:
            pass
        self.label_range_warning.setText(warning)
        self.label_range_warning.setVisible(bool(warning))

        self.btn_proceed.setEnabled(num_points > 0 and self._has_reference() and bool(self.lineEdit_base_path.text()))

    # ------------------------------------------------------------------ accept

    def _on_proceed(self) -> None:
        base_path = self.lineEdit_base_path.text().strip()
        if not base_path or not os.path.isdir(base_path):
            QMessageBox.warning(self, "Invalid Save Directory", "Choose an existing directory to save results into.")
            return
        if not self._points:
            QMessageBox.warning(self, "No Coordinates", "There are no XYZ coordinates to visit.")
            return
        if not self.dropdown_channel.currentText():
            QMessageBox.warning(self, "No Channel", "Select the imaging channel for the main camera.")
            return

        self.config = LaserAFMapConfig(
            points=list(self._points),
            channel_name=self.dropdown_channel.currentText(),
            delta_z_um=self.entry_delta_z.value(),
            z_range_um=self.entry_z_range.value(),
            closed_loop_repeats=self.entry_repeats.value(),
            base_path=base_path,
            run_name=self.lineEdit_run_name.text().strip() or "laser_af_map",
        )
        self.accept()
