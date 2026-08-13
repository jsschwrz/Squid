import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
from scipy.ndimage import gaussian_filter
from datetime import datetime
import math
import numpy as np
from qtpy.QtCore import QObject, Signal

from control import utils
import control._def
from control.core.config import ConfigRepository
from control.core.live_controller import LiveController
from control.core.objective_store import ObjectiveStore
from control.microcontroller import Microcontroller
from control.piezo import PiezoStage
from control.models import LaserAFConfig
from squid.abc import AbstractCamera, AbstractStage
from squid.camera.utils import SimulatedCamera
import squid.logging


# How many candidates may fail the motion confirm before the search gives up. A reflection visible
# at every z would otherwise cost a confirm step at every position. A loop bound, not a physical
# property of an objective, so it is not a per-objective config field.
_CONFIRM_MAX_FAILURES = 3


@dataclass
class SweepSample:
    """One z position of a diagnostic AF sweep.

    Holds every candidate in frame, not just the one the spot detection mode selects, because
    the point of a sweep is to compare them: across z the sample reflection translates and a
    static back-reflection does not.
    """

    z_um: float  # absolute z visited (piezo um, or stage um)
    dz_um: float  # offset from the z the sweep started at
    candidates: List[Dict[str, Any]] = field(default_factory=list)  # crop-relative, left to right
    selected_x: Optional[float] = None  # what the configured mode would have picked, if anything


class LaserAutofocusController(QObject):
    image_to_display = Signal(np.ndarray)
    signal_displacement_um = Signal(float)
    signal_cross_correlation = Signal(float)
    signal_piezo_position_update = Signal()  # Signal to emit piezo position updates
    signal_reference_changed = Signal(bool)  # emitted with new has_reference state
    signal_af_sweep_sample = Signal(object)  # SweepSample, emitted per z position
    signal_af_sweep_finished = Signal(object)  # List[SweepSample]; partial if cancelled or aborted

    def __init__(
        self,
        microcontroller: Microcontroller,
        camera: AbstractCamera,
        liveController: LiveController,
        stage: AbstractStage,
        piezo: Optional[PiezoStage] = None,
        objectiveStore: Optional[ObjectiveStore] = None,
    ):
        QObject.__init__(self)
        self._log = squid.logging.get_logger(__class__.__name__)
        self.microcontroller = microcontroller
        self.camera: AbstractCamera = camera
        self.liveController: LiveController = liveController
        self.stage = stage
        self.piezo = piezo
        self.objectiveStore = objectiveStore
        self.characterization_mode = control._def.LASER_AF_CHARACTERIZATION_MODE

        self.is_initialized = False

        self.laser_af_properties = LaserAFConfig()
        self.reference_crop = None

        self.spot_spacing_pixels = None  # spacing between the spots from the two interfaces (unit: pixel)

        self.image = None  # for saving the focus camera image for debugging when centroid cannot be found

        # Capture the sensor size before load_cached_configuration() narrows the ROI to the
        # stored crop. get_resolution() reports WidthMax/HeightMax, which under GenICam is
        # allowed to mean SensorWidth - OffsetX; reading it here, while the offset is still
        # at the driver default, avoids under-reporting. The max() is belt-and-braces for a
        # driver that has already applied an offset of its own.
        try:
            roi_x, roi_y, roi_w, roi_h = camera.get_region_of_interest()
            max_w, max_h = camera.get_resolution()
            self._sensor_size = (max(int(max_w), roi_x + roi_w), max(int(max_h), roi_y + roi_h))
        except Exception:
            self._sensor_size = (3088, 2064)
            self._log.warning("Could not query focus camera sensor size; assuming 3088x2064.", exc_info=True)

        # Load configurations if available
        self.load_cached_configuration()

    @property
    def _config_repo(self) -> ConfigRepository:
        """Access ConfigRepository via LiveController's microscope."""
        return self.liveController.microscope.config_repo

    @property
    def _current_profile(self) -> Optional[str]:
        """Get current profile from ConfigRepository."""
        return self._config_repo.current_profile

    def get_sensor_size(self) -> Tuple[int, int]:
        """Focus camera sensor size as (width, height), captured before any crop was applied."""
        return self._sensor_size

    def initialize_manual(self, config: LaserAFConfig) -> None:
        """Initialize laser autofocus with manual parameters."""
        # x_reference needs adjustment only if set
        x_ref_adjusted = config.x_reference - config.x_offset if config.x_reference is not None else None
        adjusted_config = config.model_copy(
            update={
                "x_reference": x_ref_adjusted,  # self.x_reference is relative to the cropped region
                "x_offset": int((config.x_offset // 8) * 8),
                "y_offset": int((config.y_offset // 2) * 2),
                "width": int((config.width // 8) * 8),
                "height": int((config.height // 2) * 2),
            }
        )

        self.laser_af_properties = adjusted_config

        if self.laser_af_properties.has_reference:
            self.reference_crop = self.laser_af_properties.reference_image_cropped

            # Invalidate reference if crop image is missing
            if self.reference_crop is None:
                self._log.warning("Loaded laser AF profile is missing reference image. Please re-set reference.")
                self.laser_af_properties = self.laser_af_properties.model_copy(update={"has_reference": False})
                self.reference_crop = None

        self.camera.set_region_of_interest(
            self.laser_af_properties.x_offset,
            self.laser_af_properties.y_offset,
            self.laser_af_properties.width,
            self.laser_af_properties.height,
        )

        self.is_initialized = True

        # Update cache if objective store and profile is available
        if self.objectiveStore and self._current_profile and self.objectiveStore.current_objective:
            updated_config = LaserAFConfig(**config.model_dump(warnings=False))
            self._config_repo.save_laser_af_config(
                self._current_profile, self.objectiveStore.current_objective, updated_config
            )

        # Re-emit has_reference so listeners stay in sync after config reloads
        # (load_cached_configuration / on_settings_changed). Duplicate emits are fine —
        # the signal is idempotent.
        self.signal_reference_changed.emit(self.laser_af_properties.has_reference)

    def load_cached_configuration(self):
        """Load configuration from the cache if available."""
        if not self._current_profile:
            return

        current_objective = self.objectiveStore.current_objective if self.objectiveStore else None
        if not current_objective:
            return

        config = self._config_repo.get_laser_af_config(current_objective)
        if config is None:
            return

        # Update camera settings
        self.camera.set_exposure_time(config.focus_camera_exposure_time_ms)
        try:
            self.camera.set_analog_gain(config.focus_camera_analog_gain)
        except NotImplementedError:
            # Some camera drivers don't support analog gain; continue with existing gain
            self._log.debug(
                f"Focus camera does not support setting analog gain; "
                f"continuing with existing gain (requested: {config.focus_camera_analog_gain})"
            )

        # Initialize with loaded config
        self.initialize_manual(config)

    def initialize_auto(self, search_within_current_crop: bool = False) -> bool:
        """Automatically initialize laser autofocus by finding the spot and calibrating.

        This method:
        1. Finds the laser spot
        2. Sets up ROI around the spot
        3. Calibrates pixel-to-um conversion using two z positions

        By default the search covers a window centered on the sensor, and the crop is then
        placed around whatever was found. That window cannot be moved off-center, so when
        the sensor shows more than one reflection it may well pick the wrong one -- and it
        overwrites any crop the operator had placed by hand.

        With search_within_current_crop, the existing crop *is* the search region and is
        left exactly as it was. This is the mode to use once the operator has framed the
        reflection they want and excluded the spurious ones: the crop is the answer to
        "which spot", so initialization has nothing left to guess.

        Either way the reference is cleared and pixel-to-um is re-calibrated.

        Returns:
            bool: True if initialization successful, False if any step fails
        """
        if search_within_current_crop:
            # Apply the configured crop before searching, so the region searched is exactly
            # the region the caller framed. Without this, a crop that had been edited but not
            # yet applied would be searched at its old position. apply_crop also clamps it and
            # clears the reference, and is a no-op on the camera if it is already in effect.
            self.apply_crop(
                self.laser_af_properties.x_offset,
                self.laser_af_properties.y_offset,
                self.laser_af_properties.width,
                self.laser_af_properties.height,
            )
        else:
            self.camera.set_region_of_interest(0, 0, *self._sensor_size)

        # update camera settings
        self.camera.set_exposure_time(self.laser_af_properties.focus_camera_exposure_time_ms)
        try:
            self.camera.set_analog_gain(self.laser_af_properties.focus_camera_analog_gain)
        except NotImplementedError:
            pass

        # Find initial spot position
        self.microcontroller.turn_on_AF_laser()
        self.microcontroller.wait_till_operation_is_completed()

        result = self._get_laser_spot_centroid(
            remove_background=True,
            # Without a center crop the search covers the current ROI, which in this mode is
            # the operator's crop.
            use_center_crop=(
                None
                if search_within_current_crop
                else (
                    self.laser_af_properties.initialize_crop_width,
                    self.laser_af_properties.initialize_crop_height,
                )
            ),
            # The search region is the constraint on which spot is picked, so the row filter
            # would only add a second, invisible constraint on top of it.
            ignore_row_tolerance=True,
        )
        if result is None:
            self._log.error("Failed to find laser spot during initialization")
            self.microcontroller.turn_off_AF_laser()
            self.microcontroller.wait_till_operation_is_completed()
            return False
        x, y = result

        self.microcontroller.turn_off_AF_laser()
        self.microcontroller.wait_till_operation_is_completed()

        if search_within_current_crop:
            # The crop is deliberate, so keep it: re-centering on the spot could pull a
            # reflection the operator excluded back into frame. apply_crop above already
            # cleared the reference; drop the stale reference position with it, since it means
            # nothing without the reference image and carrying it is what corrupts the stored
            # value on the next save.
            self._log.info(
                f"Laser spot found at crop-relative ({int(x)}, {int(y)}) within the crop "
                f"({int(self.laser_af_properties.x_offset)}, {int(self.laser_af_properties.y_offset)}, "
                f"{int(self.laser_af_properties.width)}, {int(self.laser_af_properties.height)}); crop left as is."
            )
            self.laser_af_properties = self.laser_af_properties.model_copy(update={"x_reference": None})

            if not self._calibrate_pixel_to_um():
                self._log.error("Failed to calibrate pixel-to-um conversion")
                return False
            return True

        # Set up ROI around spot and clear reference.
        #
        # The centered ROI is clamped to the sensor: a spot close enough to an edge would
        # otherwise produce an offset that runs off the sensor (a spot at x=2819 with
        # width=1536 asks for offset 2051, and 2051 + 1536 > 3088), which the camera
        # rejects. Clamping trades centering for an ROI that actually applies -- the spot
        # then sits off-center with less travel room on the near side, which the crop
        # controls in the laser AF settings widget let the user rebalance.
        requested_x = x - self.laser_af_properties.width / 2
        requested_y = y - self.laser_af_properties.height / 2
        crop_x, crop_y, crop_w, crop_h = utils.clamp_roi(
            requested_x,
            requested_y,
            self.laser_af_properties.width,
            self.laser_af_properties.height,
            *self._sensor_size,
        )
        # Only warn when the clamp actually bit -- an offset that merely got truncated to the
        # camera's 8/2 px alignment grid is normal and would otherwise warn on every init.
        if not (0 <= requested_x <= self._sensor_size[0] - crop_w and 0 <= requested_y <= self._sensor_size[1] - crop_h):
            self._log.warning(
                f"Laser spot at ({x:.1f}, {y:.1f}) cannot be centered in a {crop_w}x{crop_h} crop on a "
                f"{self._sensor_size[0]}x{self._sensor_size[1]} sensor; using offset ({crop_x}, {crop_y}). "
                f"The spot sits {x - crop_x:.0f} px from the left crop edge and {crop_x + crop_w - x:.0f} px "
                f"from the right, limiting focus travel on the near side. Narrow the crop or move the spot."
            )
        config = self.laser_af_properties.model_copy(
            update={
                "x_offset": crop_x,
                "y_offset": crop_y,
                "width": crop_w,
                "height": crop_h,
                "has_reference": False,
            }
        )
        self.reference_crop = None
        config.set_reference_image(None)
        self.signal_reference_changed.emit(False)
        self._log.info(f"Laser spot location on the full sensor is ({int(x)}, {int(y)})")

        self.initialize_manual(config)

        # Calibrate pixel-to-um conversion
        if not self._calibrate_pixel_to_um():
            self._log.error("Failed to calibrate pixel-to-um conversion")
            return False

        # Save configuration
        if self._current_profile:
            self._config_repo.save_laser_af_config(
                self._current_profile, self.objectiveStore.current_objective, self.laser_af_properties
            )

        return True

    def apply_crop(self, x_offset: float, y_offset: float, width: int, height: int) -> Tuple[int, int, int, int]:
        """Re-program the focus camera ROI without re-running spot search or calibration.

        This is the manual counterpart to initialize_auto's automatic crop placement, for
        when the spot the operator wants is not the one automatic initialization picks, or
        sits too close to a sensor edge to be centered. Because pixel_to_um and
        calibration_timestamp are untouched, the crop can be nudged repeatedly while
        watching the live stream.

        The requested ROI is snapped to the camera's alignment grid and clamped to the
        sensor, so an out-of-range request cannot reach set_region_of_interest().

        x_reference is carried across the shift in the full-sensor frame so it keeps
        pointing at the same physical pixel. The cross-correlation reference *image* is
        always dropped, including for an x-only shift: set_reference() anchors that crop at
        the ROI's vertical center, so any change to y_offset or height silently
        desynchronizes the template from the sensor rows it was taken from. The operator
        has to press Set Reference again.

        Returns the ROI actually applied, as (x_offset, y_offset, width, height).
        """
        new_x, new_y, new_w, new_h = utils.clamp_roi(x_offset, y_offset, width, height, *self._sensor_size)
        requested = (x_offset, y_offset, width, height)
        if (new_x, new_y, new_w, new_h) != requested:
            self._log.info(f"Requested laser AF crop {requested} adjusted to {(new_x, new_y, new_w, new_h)}.")

        # laser_af_properties.x_reference is crop-relative; initialize_manual expects the
        # full-sensor value and subtracts the new offset itself.
        old_x_reference = self.laser_af_properties.x_reference
        x_reference_full = None if old_x_reference is None else old_x_reference + self.laser_af_properties.x_offset

        config = self.laser_af_properties.model_copy(
            update={
                "x_offset": new_x,
                "y_offset": new_y,
                "width": new_w,
                "height": new_h,
                "x_reference": x_reference_full,
                "has_reference": False,
            }
        )
        config.set_reference_image(None)
        self.reference_crop = None

        # initialize_manual applies the ROI, converts x_reference back to crop-relative,
        # keeps is_initialized True and persists. Its own 8/2 px truncation is a no-op here
        # because clamp_roi already snapped the values.
        self.initialize_manual(config)
        self.signal_reference_changed.emit(False)

        return new_x, new_y, new_w, new_h

    def center_crop_on_point(
        self,
        x_in_crop: float,
        y_in_crop: float,
        width: Optional[int] = None,
        height: Optional[int] = None,
        source_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[int, int, int, int]:
        """Shift the crop so a crop-relative point sits at the center of the crop.

        Used to re-center on a spot found by manual spot detection, giving it equal focus
        travel room on both sides.

        source_roi is the ROI the coordinates were measured in; it defaults to the camera's
        current ROI. Pass it explicitly when the measurement and this call are separated in
        time, so a crop change in between cannot cause the coordinates to be misread.
        """
        if source_roi is None:
            source_roi = self.camera.get_region_of_interest()
        source_x_offset, source_y_offset = source_roi[0], source_roi[1]

        width = self.laser_af_properties.width if width is None else width
        height = self.laser_af_properties.height if height is None else height

        return self.apply_crop(
            source_x_offset + x_in_crop - width / 2,
            source_y_offset + y_in_crop - height / 2,
            width,
            height,
        )

    def _calibrate_pixel_to_um(self) -> bool:
        """Calibrate pixel-to-um conversion.

        Returns:
            bool: True if calibration successful, False otherwise
        """
        # Calibrate pixel-to-um conversion
        try:
            self.microcontroller.turn_on_AF_laser()
            self.microcontroller.wait_till_operation_is_completed()
        except TimeoutError:
            self._log.exception("Faield to turn on AF laser before pixel to um calibration, cannot continue!")
            return False

        # Move to first position and measure
        self._move_z(-self.laser_af_properties.pixel_to_um_calibration_distance / 2)
        if self.piezo is not None:
            time.sleep(control._def.MULTIPOINT_PIEZO_DELAY_MS / 1000)

        result = self._get_laser_spot_centroid()
        if result is None:
            self._log.error("Failed to find laser spot during calibration (position 1)")
            try:
                self.microcontroller.turn_off_AF_laser()
                self.microcontroller.wait_till_operation_is_completed()
            except TimeoutError:
                self._log.exception("Error turning off AF laser after spot calibration failure (position 1)")
                # Just fall through since we are already on a failure path.
            return False
        x0, y0 = result

        # Move to second position and measure
        self._move_z(self.laser_af_properties.pixel_to_um_calibration_distance)
        if self.piezo is not None:
            time.sleep(control._def.MULTIPOINT_PIEZO_DELAY_MS / 1000)

        result = self._get_laser_spot_centroid()
        if result is None:
            self._log.error("Failed to find laser spot during calibration (position 2)")
            try:
                self.microcontroller.turn_off_AF_laser()
                self.microcontroller.wait_till_operation_is_completed()
            except TimeoutError:
                self._log.exception("Error turning off AF laser after spot calibration failure (position 2)")
                # Just fall through since we are already on a failure path.
            return False
        x1, y1 = result

        try:
            self.microcontroller.turn_off_AF_laser()
            self.microcontroller.wait_till_operation_is_completed()
        except TimeoutError:
            self._log.exception(
                "Error turning off AF laser after spot calibration acquisition.  Continuing in unknown state"
            )

        # move back to initial position
        self._move_z(-self.laser_af_properties.pixel_to_um_calibration_distance / 2)
        if self.piezo is not None:
            time.sleep(control._def.MULTIPOINT_PIEZO_DELAY_MS / 1000)

        # Calculate conversion factor
        displacement_px = x1 - x0
        if isinstance(self.camera, SimulatedCamera):
            # The simulated focus camera renders a static spot, so there is no displacement
            # to divide by. Gate this on the camera actually being simulated rather than on
            # the measured displacement -- keying it off "the spot did not move" is exactly
            # the real-hardware failure below, and would mask it.
            pixel_to_um = 0.4
            self._log.warning("Using simulation value for pixel_to_um conversion")
        elif abs(displacement_px) < control._def.LASER_AF_MIN_CALIBRATION_DISPLACEMENT_PX:
            self._log.error(
                f"Calibration failed: the spot moved {displacement_px:.3f} px "
                f"(x0={x0:.2f}, x1={x1:.2f}) over a "
                f"{self.laser_af_properties.pixel_to_um_calibration_distance} um z move, below the "
                f"{control._def.LASER_AF_MIN_CALIBRATION_DISPLACEMENT_PX} px minimum. A reflection that "
                f"does not translate with defocus is usually a static back-reflection rather than the "
                f"sample reflection; check which spot is being detected before recalibrating."
            )
            return False
        else:
            pixel_to_um = self.laser_af_properties.pixel_to_um_calibration_distance / displacement_px
        self._log.info(f"Pixel to um conversion factor is {pixel_to_um:.3f} um/pixel")
        if abs(pixel_to_um) > control._def.LASER_AF_MAX_PLAUSIBLE_PIXEL_TO_UM:
            self._log.warning(
                f"Calibrated pixel_to_um of {pixel_to_um:.3f} um/pixel is implausibly large "
                f"(> {control._def.LASER_AF_MAX_PLAUSIBLE_PIXEL_TO_UM}); the detected spot barely moved and "
                f"may not be the sample reflection. Autofocus will be unreliable until this is re-done."
            )
        calibration_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Update config with new calibration values
        self.laser_af_properties = self.laser_af_properties.model_copy(
            update={"pixel_to_um": pixel_to_um, "calibration_timestamp": calibration_timestamp}
        )

        # Update cache
        if self.objectiveStore and self._current_profile:
            self._config_repo.save_laser_af_config(
                self._current_profile, self.objectiveStore.current_objective, self.laser_af_properties
            )

        return True

    def set_laser_af_properties(self, updates: dict) -> None:
        """Update laser autofocus properties. Used for updating settings from GUI."""
        self.laser_af_properties = self.laser_af_properties.model_copy(update=updates)
        self.is_initialized = False

    def update_threshold_properties(self, updates: dict) -> None:
        """Update threshold properties. Save settings without re-initializing."""
        self.laser_af_properties = self.laser_af_properties.model_copy(update=updates)
        if self._current_profile and self.objectiveStore:
            self._config_repo.save_laser_af_config(
                self._current_profile, self.objectiveStore.current_objective, self.laser_af_properties
            )
        self._log.info("Updated threshold properties")

    def _turn_on_laser(self) -> None:
        """Turn on AF laser. Raises TimeoutError on failure."""
        self.microcontroller.turn_on_AF_laser()
        self.microcontroller.wait_till_operation_is_completed()

    def _turn_off_laser(self) -> None:
        """Turn off AF laser. Raises TimeoutError on failure."""
        self.microcontroller.turn_off_AF_laser()
        self.microcontroller.wait_till_operation_is_completed()

    def _get_displacement_from_centroid(self, centroid: tuple) -> float:
        """Calculate displacement in um from centroid coordinates."""
        if self.laser_af_properties.x_reference is None:
            self._log.warning("Cannot calculate displacement - reference position not set")
            return float("nan")
        x, y = centroid
        return (x - self.laser_af_properties.x_reference) * self.laser_af_properties.pixel_to_um

    def _confirm_spot_moves_with_z(self, x_before: float) -> Tuple[bool, str]:
        """Take one small z step and check the spot translated by the predicted amount.

        This is the only runtime test of the property that actually distinguishes the sample
        reflection from a static back-reflection. The intensity, area, aspect and row filters all
        pass a static blob that happens to sit on the same row, and the spot detection mode
        selects positionally rather than rejecting anything. Calibration tests this once, at
        Initialize, and nothing re-checks it afterwards.

        Assumes the AF laser is already on. Always restores z before returning.

        Fails OPEN -- returns True with a reason -- whenever the test could not discriminate:
        an untrustworthy pixel_to_um, a step too small to produce measurable motion, or no room
        to move. Failing closed in those cases would break autofocus on objectives where the
        check simply does not apply, which is worse than not checking.

        Returns (accepted, reason).
        """
        pixel_to_um = self.laser_af_properties.pixel_to_um
        if (
            not math.isfinite(pixel_to_um)
            or pixel_to_um == 0
            or abs(pixel_to_um) > control._def.LASER_AF_MAX_PLAUSIBLE_PIXEL_TO_UM
        ):
            reason = f"skipped: pixel_to_um ({pixel_to_um}) is not trustworthy"
            self._log.warning(f"Motion confirm {reason}; accepting the candidate unchecked.")
            return True, reason

        dz_um = self.laser_af_properties.confirm_step_um
        # Signed on purpose. pixel_to_um carries the direction the spot travels with defocus, and
        # a spot moving the wrong way is exactly what this check exists to catch.
        predicted_dx_px = dz_um / pixel_to_um

        if abs(predicted_dx_px) < control._def.LASER_AF_CONFIRM_MIN_PREDICTED_PX:
            reason = (
                f"skipped: a {dz_um} um step predicts only {predicted_dx_px:.2f} px of motion at "
                f"{pixel_to_um:.4f} um/px, below the {control._def.LASER_AF_CONFIRM_MIN_PREDICTED_PX} px "
                f"floor. Increase the confirm step for this objective."
            )
            self._log.warning(f"Motion confirm {reason}")
            return True, reason

        if self.piezo is not None:
            z_before = self.piezo.position
            # Prefer stepping up; fall back to down if the piezo has no headroom that way.
            if z_before + dz_um > self.piezo.range_um:
                dz_um = -dz_um
                predicted_dx_px = -predicted_dx_px
            if z_before + dz_um < 0 or z_before + dz_um > self.piezo.range_um:
                reason = "skipped: no piezo headroom for the confirm step in either direction"
                self._log.warning(f"Motion confirm {reason}; accepting the candidate unchecked.")
                return True, reason
        else:
            z_before = self.stage.get_pos().z_mm * 1000

        try:
            self._move_z(dz_um)
            if self.piezo is not None:
                time.sleep(control._def.MULTIPOINT_PIEZO_DELAY_MS / 1000)
            result = self._get_laser_spot_centroid()
        finally:
            # Absolute restore, not _move_z(-dz_um): a clamped or partial move would otherwise
            # leave z quietly drifted, and this runs once per candidate during an acquisition.
            self._restore_to_position(z_before)

        if result is None:
            reason = "rejected: spot lost during the confirm step"
            self._log.info(f"Motion confirm {reason}")
            return False, reason

        observed_dx_px = result[0] - x_before
        tolerance_px = max(
            self.laser_af_properties.confirm_tolerance_px,
            control._def.LASER_AF_CONFIRM_TOLERANCE_FRACTION * abs(predicted_dx_px),
        )
        accepted = abs(observed_dx_px - predicted_dx_px) <= tolerance_px

        self._log.info(
            f"Motion confirm: z step {dz_um:+.2f} um predicted {predicted_dx_px:+.2f} px, "
            f"observed {observed_dx_px:+.2f} px, tolerance {tolerance_px:.2f} px -> "
            f"{'accepted' if accepted else 'REJECTED'}"
        )
        if accepted:
            return True, "confirmed: spot translated with z"
        return False, (
            f"rejected: spot moved {observed_dx_px:+.2f} px where {predicted_dx_px:+.2f} px was expected"
        )

    def _build_search_positions(
        self, range_um: Optional[float] = None, step_um: Optional[float] = None
    ) -> Tuple[float, float, List[float]]:
        """Build the absolute z positions the spot-search visits.

        Shared by the search itself and by run_af_sweep, so the diagnostic sweep samples exactly
        the grid the real search uses. A sweep that visited different positions than the thing it
        is diagnosing would be worse than no sweep at all.

        Returns (current_z_um, step_um_used, positions_um). Positions are ordered by
        LASER_AF_SEARCH_DOWN_FIRST and clamped to the piezo's travel when a piezo is present.
        """
        if range_um is None:
            range_um = self.laser_af_properties.laser_af_search_range_um
        if step_um is None:
            step_um = self.laser_af_properties.laser_af_search_step_um

        # The loops below build positions by repeated subtraction, so a zero or negative step
        # never terminates -- and this runs on the GUI thread inside measure_displacement, so it
        # would hang the application. LaserAFConfig constrains the field to > 0; this catches
        # anything that reaches us by another route.
        step_um = max(float(step_um), 0.05)
        range_um = abs(float(range_um))

        if self.piezo is not None:
            current_z_um = self.piezo.position
            # For piezo, clamp bounds to valid piezo range (0 to range_um)
            lower_bound_um = max(0, current_z_um - range_um)
            upper_bound_um = min(self.piezo.range_um, current_z_um + range_um)
        else:
            current_z_um = self.stage.get_pos().z_mm * 1000
            lower_bound_um = current_z_um - range_um
            upper_bound_um = current_z_um + range_um

        # Generate positions going downward (from current to lower_bound)
        downward_positions = []
        pos = current_z_um - step_um
        while pos >= lower_bound_um:
            downward_positions.append(pos)
            pos -= step_um

        # Generate positions going upward (from current to upper_bound)
        upward_positions = []
        pos = current_z_um + step_um
        while pos <= upper_bound_um:
            upward_positions.append(pos)
            pos += step_um

        # Order positions based on search direction preference
        if control._def.LASER_AF_SEARCH_DOWN_FIRST:
            # Search downward first, then upward
            positions_um = downward_positions + [current_z_um] + upward_positions
        else:
            # Search upward first, then downward
            positions_um = upward_positions + [current_z_um] + downward_positions

        if len(positions_um) == 1:
            self._log.warning(
                f"Z search step ({step_um} um) is larger than the search range ({range_um} um), so the "
                f"search will only re-check the position that already failed. Reduce the step or widen "
                f"the range."
            )
        elif self.piezo is not None and (
            lower_bound_um > current_z_um - range_um or upper_bound_um < current_z_um + range_um
        ):
            self._log.info(
                f"Z search span clamped by piezo travel: requested +/-{range_um:.1f} um around "
                f"{current_z_um:.1f}, searching {lower_bound_um:.1f}..{upper_bound_um:.1f} um."
            )

        return current_z_um, step_um, positions_um

    def measure_displacement(self, search_for_spot: bool = True) -> float:
        """Measure the displacement of the laser spot from the reference position.

        Args:
            search_for_spot: If True, search for spot if not found at current position

        Returns:
            float: Displacement in micrometers, or float('nan') if measurement fails
        """

        def finish_with(um: float) -> float:
            self.signal_displacement_um.emit(um)
            return um

        try:
            self._turn_on_laser()
        except TimeoutError:
            self._log.exception("Turning on AF laser timed out, failed to measure displacement.")
            return finish_with(float("nan"))

        # get laser spot location
        result = self._get_laser_spot_centroid()

        if result is not None and self.laser_af_properties.confirm_motion_mode == (
            control._def.LaserAFConfirmMotionMode.ALWAYS
        ):
            # This path runs at every FOV of an acquisition, so the extra z step is only taken
            # when explicitly asked for. A rejection here is not a failure -- it means whatever is
            # at this z is not the sample reflection, which is a reason to go looking for it.
            confirmed, reason = self._confirm_spot_moves_with_z(result[0])
            if not confirmed:
                self._log.warning(f"First-try candidate {reason}; falling through to the z search.")
                result = None

        if result is not None:
            # Spot found on first try
            try:
                self._turn_off_laser()
            except TimeoutError:
                self._log.exception("Turning off AF laser timed out! Laser may still be on.")
            return finish_with(self._get_displacement_from_centroid(result))

        self._log.error("Failed to detect laser spot during displacement measurement")

        if not search_for_spot:
            try:
                self._turn_off_laser()
            except TimeoutError:
                self._log.exception("Turning off AF laser timed out! Laser may still be on.")
            return finish_with(float("nan"))

        # Search for spot by scanning through z range (laser stays on during search)
        current_z_um, search_step_um, search_positions_um = self._build_search_positions()

        self._log.info(
            f"Starting spot search ({'downward' if control._def.LASER_AF_SEARCH_DOWN_FIRST else 'upward'} first): "
            f"positions {search_positions_um} um"
        )

        # A candidate whose displacement is further from the reference than one search step could
        # not have been reached by the step that just happened, so it is a different spot.
        accept_window_um = search_step_um * (1.0 + control._def.LASER_AF_SEARCH_ACCEPT_TOLERANCE_FRACTION)
        confirm_mode = self.laser_af_properties.confirm_motion_mode
        confirm_failures = 0

        current_pos_um = current_z_um  # Track where we are

        for target_pos_um in search_positions_um:
            # Move to target position
            move_um = target_pos_um - current_pos_um
            if move_um != 0:
                self._log.info(f"Z search: moving to {target_pos_um:.1f} um (delta: {move_um:+.1f} um)")
                self._move_z(move_um)
                current_pos_um = target_pos_um
                # Wait for piezo to settle
                if self.piezo is not None:
                    time.sleep(control._def.MULTIPOINT_PIEZO_DELAY_MS / 1000)
            else:
                self._log.info(f"Z search: checking current position {target_pos_um:.1f} um")

            # Attempt spot detection
            result = self._get_laser_spot_centroid()

            if result is None:
                self._log.info(f"Z search: no valid spot at {target_pos_um:.1f} um")
                continue

            displacement_um = self._get_displacement_from_centroid(result)
            if abs(displacement_um) > accept_window_um:
                self._log.info(
                    f"Z search: spot at {target_pos_um:.1f} um has displacement {displacement_um:.1f} um (out of range)"
                )
                continue

            if confirm_mode in (
                control._def.LaserAFConfirmMotionMode.SEARCH_ONLY,
                control._def.LaserAFConfirmMotionMode.ALWAYS,
            ):
                confirmed, reason = self._confirm_spot_moves_with_z(result[0])
                if not confirmed:
                    confirm_failures += 1
                    self._log.warning(f"Z search: candidate at {target_pos_um:.1f} um {reason}")
                    if confirm_failures >= _CONFIRM_MAX_FAILURES:
                        self._log.error(
                            f"Candidates were found at {confirm_failures} z positions but none translated "
                            f"with z. That is the signature of a static back-reflection rather than the "
                            f"sample reflection. Run Test AF Sweep to see which reflections are in frame."
                        )
                        # Fall through to the shared restore-and-NaN tail below.
                        break
                    continue

            self._log.info(f"Z search: spot found at {target_pos_um:.1f} um, displacement {displacement_um:.1f} um")
            try:
                self._turn_off_laser()
            except TimeoutError:
                self._log.exception("Turning off AF laser timed out! Laser may still be on.")
            return finish_with(displacement_um)

        # Spot not found - move back to original position
        self._restore_to_position(current_z_um)
        self._log.warning("Spot not found during z search")

        try:
            self._turn_off_laser()
        except TimeoutError:
            self._log.exception("Turning off AF laser timed out! Laser may still be on.")
        return finish_with(float("nan"))

    def _spot_detection_params(self, row_tolerance: Optional[float] = None) -> Dict[str, Any]:
        """The cc_* parameter dict passed to the detection functions."""
        return {
            "threshold": self.laser_af_properties.cc_threshold,
            "min_area": self.laser_af_properties.cc_min_area,
            "max_area": self.laser_af_properties.cc_max_area,
            "row_tolerance": (
                self.laser_af_properties.cc_row_tolerance if row_tolerance is None else row_tolerance
            ),
            "max_aspect_ratio": self.laser_af_properties.cc_max_aspect_ratio,
        }

    def run_af_sweep(
        self,
        range_um: Optional[float] = None,
        step_um: Optional[float] = None,
        keep_running: Optional[threading.Event] = None,
    ) -> List[SweepSample]:
        """Step z across the search range, recording every candidate spot at each position.

        A diagnostic, not a measurement: it writes no configuration, sets no reference, and
        restores z when it finishes. What it produces is the one piece of evidence the rest of
        the system cannot supply -- how each reflection in frame behaves as a function of z. The
        sample reflection traces a line whose slope is 1/pixel_to_um; a static back-reflection
        traces a flat one. Nothing in the normal detection path can tell them apart.

        Returns the samples collected, which may be partial if cancelled via keep_running or
        aborted because the crop or objective changed underneath it.
        """
        samples: List[SweepSample] = []

        # Latch what the coordinates are relative to. apply_crop() reprograms the camera ROI, and
        # changing objective reloads the whole config; either landing mid-sweep would splice two
        # coordinate frames into one plot without any visible sign.
        try:
            source_roi = self.camera.get_region_of_interest()
        except Exception:
            source_roi = None
        source_objective = self.objectiveStore.current_objective if self.objectiveStore else None

        if self.piezo is not None:
            start_z_um = self.piezo.position
        else:
            start_z_um = self.stage.get_pos().z_mm * 1000

        _, step_used_um, positions_um = self._build_search_positions(range_um, step_um)
        # The search orders positions by LASER_AF_SEARCH_DOWN_FIRST so it can find a spot sooner.
        # A sweep visits all of them regardless, and wants a monotone z axis to plot against.
        positions_um = sorted(positions_um)

        self._log.info(
            f"Starting AF sweep: {len(positions_um)} positions, step {step_used_um} um, "
            f"{positions_um[0]:.1f}..{positions_um[-1]:.1f} um."
        )

        self.camera.enable_callbacks(False)
        try:
            self._turn_on_laser()
        except TimeoutError:
            self._log.exception("Turning on AF laser timed out, cannot run AF sweep.")
            self.signal_af_sweep_finished.emit(samples)
            return samples

        current_pos_um = start_z_um
        try:
            for target_pos_um in positions_um:
                if keep_running is not None and not keep_running.is_set():
                    self._log.info("AF sweep cancelled.")
                    break

                if source_roi is not None:
                    try:
                        if self.camera.get_region_of_interest() != source_roi:
                            self._log.warning("Camera ROI changed during AF sweep; aborting, results are partial.")
                            break
                    except Exception:
                        pass
                if source_objective is not None and self.objectiveStore is not None:
                    if self.objectiveStore.current_objective != source_objective:
                        self._log.warning("Objective changed during AF sweep; aborting, results are partial.")
                        break

                move_um = target_pos_um - current_pos_um
                if move_um != 0:
                    self._move_z(move_um)
                    current_pos_um = target_pos_um
                    if self.piezo is not None:
                        time.sleep(control._def.MULTIPOINT_PIEZO_DELAY_MS / 1000)

                # One frame per position, not laser_af_averaging_n. Candidates from different
                # frames do not correspond to one another, so there is nothing to average -- and
                # averaging would trade a three-fold slower sweep for no extra information.
                image = self.get_new_frame()
                if image is None:
                    image = self.get_new_frame()
                if image is None:
                    self._log.warning(f"AF sweep: no frame at {target_pos_um:.1f} um")
                    samples.append(SweepSample(z_um=target_pos_um, dz_um=target_pos_um - start_z_um))
                    self.signal_af_sweep_sample.emit(samples[-1])
                    continue

                self.image = image.copy()
                candidates = utils.find_all_spot_locations(
                    image,
                    params=self._spot_detection_params(),
                    filter_sigma=self.laser_af_properties.filter_sigma,
                )

                selected_x = None
                if candidates:
                    try:
                        selected_x = utils.select_spot_by_mode(
                            candidates, self.laser_af_properties.get_spot_detection_mode()
                        )["x"]
                    except (ValueError, NotImplementedError):
                        # e.g. SINGLE mode with several candidates. The candidates are still worth
                        # recording -- that the mode cannot choose is itself the finding.
                        selected_x = None

                sample = SweepSample(
                    z_um=target_pos_um,
                    dz_um=target_pos_um - start_z_um,
                    candidates=candidates,
                    selected_x=selected_x,
                )
                samples.append(sample)
                self.signal_af_sweep_sample.emit(sample)
        finally:
            try:
                self._turn_off_laser()
            except TimeoutError:
                self._log.exception("Turning off AF laser timed out! Laser may still be on.")
            self._restore_to_position(start_z_um)
            self._log.info(f"AF sweep finished: {len(samples)} positions sampled, z restored.")
            self.signal_af_sweep_finished.emit(samples)

        return samples

    def move_to_target(self, target_um: float) -> bool:
        """Move the stage to reach a target displacement from reference position.

        Args:
            target_um: Target displacement in micrometers

        Returns:
            bool: True if move was successful, False if measurement failed or displacement was out of range
        """
        if not self.laser_af_properties.has_reference:
            self._log.warning("Cannot move to target - reference not set")
            return False

        # Record original z position so we can restore it on failure
        if self.piezo is not None:
            original_z_um = self.piezo.position
        else:
            original_z_um = self.stage.get_pos().z_mm * 1000

        current_displacement_um = self.measure_displacement()
        self._log.info(f"Current laser AF displacement: {current_displacement_um:.1f} μm")

        if math.isnan(current_displacement_um):
            self._log.error("Cannot move to target: failed to measure current displacement")
            # measure_displacement already restores position on search failure
            return False

        if abs(current_displacement_um) > self.laser_af_properties.laser_af_range:
            self._log.warning(f"Measured displacement ({current_displacement_um:.1f} μm) is unreasonably large")
            self._restore_to_position(original_z_um)
            return False

        um_to_move = target_um - current_displacement_um
        self._move_z(um_to_move)
        if self.piezo is not None:
            time.sleep(control._def.MULTIPOINT_PIEZO_DELAY_MS / 1000)

        # Verify using cross-correlation that spot is in same location as reference
        cc_result, correlation = self._verify_spot_alignment()
        self.signal_cross_correlation.emit(correlation)
        if not cc_result:
            self._log.warning("Cross correlation check failed - spots not well aligned")
            # Restore to original position (not just undo last move)
            self._restore_to_position(original_z_um)
            return False
        else:
            self._log.info("Cross correlation check passed - spots are well aligned")
            return True

    def _restore_to_position(self, target_z_um: float) -> None:
        """Restore z position to a specific absolute position."""
        if self.piezo is not None:
            current_z_um = self.piezo.position
        else:
            current_z_um = self.stage.get_pos().z_mm * 1000

        move_um = target_z_um - current_z_um
        if abs(move_um) > 0.01:  # Only move if difference is significant
            self._log.info(f"Restoring z position: moving {move_um:.1f} μm")
            self._move_z(move_um)

    def _move_z(self, um_to_move: float) -> None:
        if self.piezo is not None:
            # TODO: check if um_to_move is in the range of the piezo
            self.piezo.move_relative(um_to_move)
            self.signal_piezo_position_update.emit()
        else:
            self.stage.move_z(um_to_move / 1000)

    def apply_relative_offset_um(self, offset_um: float) -> None:
        """Open-loop relative Z move of ``offset_um`` (displacement µm, 1:1 with Z), with NO
        spot re-verification.

        Intended to run right after ``move_to_target(0.0)`` has anchored — and verified — the
        spot at the reference plane, to then place the sample at a target displacement from
        that reference. Spot-alignment verification (``_verify_spot_alignment``) always crops
        at ``x_reference`` and would fail for a deliberately-displaced spot, so it must NOT be
        used to reach a nonzero target; this method deliberately skips it. No-op for offset 0.
        """
        if offset_um:
            self._move_z(offset_um)

    def set_reference(self) -> bool:
        """Set the current spot position as the reference position.

        Captures and stores both the spot position and a cropped reference image
        around the spot for later alignment verification.

        Returns:
            bool: True if reference was set successfully, False if spot detection failed
        """
        if not self.is_initialized:
            self._log.error("Laser autofocus is not initialized, cannot set reference")
            return False

        # Reset image so we only use image from successful detection
        self.image = None

        # turn on the laser
        try:
            self.microcontroller.turn_on_AF_laser()
            self.microcontroller.wait_till_operation_is_completed()
        except TimeoutError:
            self._log.exception("Failed to turn on AF laser for reference setting!")
            return False

        # get laser spot location and image
        result = self._get_laser_spot_centroid()
        reference_image = self.image

        # turn off the laser
        try:
            self.microcontroller.turn_off_AF_laser()
            self.microcontroller.wait_till_operation_is_completed()
        except TimeoutError:
            self._log.exception("Failed to turn off AF laser after setting reference, laser is in an unknown state!")
            # Continue on since we got our reading, but the system is potentially in a weird state!

        if result is None or reference_image is None:
            self._log.error("Failed to detect laser spot while setting reference")
            return False

        x, y = result

        # Store cropped and normalized reference image
        center_y = int(reference_image.shape[0] / 2)
        x_start = max(0, int(x) - self.laser_af_properties.spot_crop_size // 2)
        x_end = min(
            reference_image.shape[1],
            int(x) + self.laser_af_properties.spot_crop_size // 2,
        )
        y_start = max(0, center_y - self.laser_af_properties.spot_crop_size // 2)
        y_end = min(
            reference_image.shape[0],
            center_y + self.laser_af_properties.spot_crop_size // 2,
        )

        reference_crop = reference_image[y_start:y_end, x_start:x_end].astype(np.float32)
        if self.laser_af_properties.filter_sigma is not None and self.laser_af_properties.filter_sigma > 0:
            reference_crop = gaussian_filter(reference_crop, sigma=self.laser_af_properties.filter_sigma)
        self.reference_crop = (reference_crop - np.mean(reference_crop)) / np.max(reference_crop)

        self._log.info(
            f"Reference crop updated: shape={self.reference_crop.shape}, "
            f"crop region=[{x_start}:{x_end}, {y_start}:{y_end}]"
        )

        self.signal_displacement_um.emit(0)
        self._log.info(f"Set reference position to ({x:.1f}, {y:.1f})")

        self.laser_af_properties = self.laser_af_properties.model_copy(update={"x_reference": x, "has_reference": True})
        # Update the reference image in laser_af_properties
        # so that self.laser_af_properties.reference_image_cropped stays in sync with self.reference_crop
        self.laser_af_properties.set_reference_image(self.reference_crop)

        # Update cached file
        if self._current_profile and self.objectiveStore:
            # Create config for saving with reference image encoded
            save_config = self.laser_af_properties.model_copy(
                update={"x_reference": x + self.laser_af_properties.x_offset, "has_reference": True}
            )
            save_config.set_reference_image(self.reference_crop)
            self._config_repo.save_laser_af_config(
                self._current_profile, self.objectiveStore.current_objective, save_config
            )

        self._log.info("Reference spot position set")

        self.signal_reference_changed.emit(True)
        return True

    def on_settings_changed(self) -> None:
        """Handle objective change or profile load event.

        This method is called when the objective changes. It resets the initialization
        status and loads the cached configuration for the new objective.
        """
        self.is_initialized = False
        self.load_cached_configuration()

    def _verify_spot_alignment(self) -> Tuple[bool, np.array]:
        """Verify laser spot alignment using cross-correlation with reference image.

        Captures current laser spot image and compares it with the reference image
        using normalized cross-correlation. Images are cropped around the expected
        spot location and normalized by maximum intensity before comparison.

        Returns:
            bool: True if spots are well aligned (correlation > CORRELATION_THRESHOLD), False otherwise
        """
        failure_return_value = False, float("nan")
        # Reset image so CC verification uses its own frame, not the earlier measurement image
        self.image = None

        # Get current spot image
        try:
            self.microcontroller.turn_on_AF_laser()
            self.microcontroller.wait_till_operation_is_completed()
        except TimeoutError:
            self._log.exception("Failed to turn on AF laser for verifying spot alignment.")
            return failure_return_value

        # TODO: create a function to get the current image (taking care of trigger mode checking and laser on/off switching)
        """
        self.camera.send_trigger()
        current_image = self.camera.read_frame()
        """
        centroid_result = self._get_laser_spot_centroid()
        current_image = self.image

        try:
            self.microcontroller.turn_off_AF_laser()
            self.microcontroller.wait_till_operation_is_completed()
        except TimeoutError:
            self._log.exception("Failed to turn off AF laser after verifying spot alignment, laser in unknown state!")
            # Continue on because we got a reading, but the system is in a potentially weird and unknown state here.

        if self.reference_crop is None:
            self._log.warning("No reference crop stored")
            return failure_return_value

        if current_image is None:
            self._log.error("Failed to get images for cross-correlation check")
            return failure_return_value

        if centroid_result is None:
            self._log.error("Failed to detect spot centroid for cross-correlation check")
            return failure_return_value

        # Crop current image around the reference position to detect off-position spots
        # If the spot moved to the wrong location (e.g., debris), it will appear off-center
        # in this crop, resulting in low correlation and failing the CC check
        current_peak_x, current_peak_y = centroid_result
        center_x = int(self.laser_af_properties.x_reference)
        center_y = int(current_image.shape[0] / 2)

        # Log if detected spot is far from reference (potential debris/contamination)
        spot_offset = abs(current_peak_x - self.laser_af_properties.x_reference)
        if spot_offset > 20:  # pixels
            self._log.warning(
                f"Detected spot at x={current_peak_x:.1f} is {spot_offset:.1f} pixels from reference "
                f"x={self.laser_af_properties.x_reference:.1f} - possible debris/contamination"
            )

        x_start = max(0, center_x - self.laser_af_properties.spot_crop_size // 2)
        x_end = min(current_image.shape[1], center_x + self.laser_af_properties.spot_crop_size // 2)
        y_start = max(0, center_y - self.laser_af_properties.spot_crop_size // 2)
        y_end = min(current_image.shape[0], center_y + self.laser_af_properties.spot_crop_size // 2)

        current_crop = current_image[y_start:y_end, x_start:x_end].astype(np.float32)
        if self.laser_af_properties.filter_sigma is not None and self.laser_af_properties.filter_sigma > 0:
            current_crop = gaussian_filter(current_crop, sigma=self.laser_af_properties.filter_sigma)
        current_norm = (current_crop - np.mean(current_crop)) / np.max(current_crop)

        # Calculate normalized cross correlation
        correlation = np.corrcoef(current_norm.ravel(), self.reference_crop.ravel())[0, 1]

        self._log.info(f"Cross correlation with reference: {correlation:.3f}")

        if False:  # Set to True to enable debug plot
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # Reference crop
            axes[0].imshow(self.reference_crop, cmap="gray")
            axes[0].set_title(f"Reference Crop\n(x={self.laser_af_properties.x_reference:.1f})")
            axes[0].axis("off")

            # Current crop (centered on reference position)
            axes[1].imshow(current_norm, cmap="gray")
            axes[1].set_title(
                f"Current Crop @ Reference\n(detected x={current_peak_x:.1f}, crop x={self.laser_af_properties.x_reference:.1f})"
            )
            axes[1].axis("off")

            # Difference image
            diff = current_norm - self.reference_crop
            axes[2].imshow(diff, cmap="RdBu", vmin=-0.5, vmax=0.5)
            axes[2].set_title("Difference\n(Current - Reference)")
            axes[2].axis("off")

            passed = correlation >= self.laser_af_properties.correlation_threshold
            status = "PASS" if passed else "FAIL"
            color = "green" if passed else "red"
            peak_diff = current_peak_x - self.laser_af_properties.x_reference
            fig.suptitle(
                f"Cross-Correlation: {correlation:.3f} (threshold={self.laser_af_properties.correlation_threshold}) [{status}]\n"
                f"Peak shift: {peak_diff:.1f} pixels",
                fontsize=11,
                color=color,
            )

            plt.tight_layout()
            plt.show()

        # Check if correlation exceeds threshold
        if correlation < self.laser_af_properties.correlation_threshold:
            self._log.warning("Cross correlation check failed - spots not well aligned")
            return False, correlation

        return True, correlation

    def get_new_frame(self):
        # IMPORTANT: This assumes that the autofocus laser is already on!
        self.camera.send_trigger(self.camera.get_exposure_time())
        return self.camera.read_frame()

    def _get_laser_spot_centroid(
        self,
        remove_background: bool = False,
        use_center_crop: Optional[Tuple[int, int]] = None,
        ignore_row_tolerance: bool = False,
    ) -> Optional[Tuple[float, float]]:
        """Get the centroid location of the laser spot.

        Averages multiple measurements to improve accuracy. The number of measurements
        is controlled by LASER_AF_AVERAGING_N.

        Args:
            remove_background: Apply background removal using top-hat filter
            use_center_crop: (width, height) to crop around center before detection
            ignore_row_tolerance: If True, disable row tolerance filtering (for initialization)

        Returns:
            Optional[Tuple[float, float]]: (x,y) coordinates of spot centroid, or None if detection fails
        """
        # disable camera callback
        self.camera.enable_callbacks(False)

        successful_detections = 0
        tmp_x = 0
        tmp_y = 0

        image = None
        for i in range(self.laser_af_properties.laser_af_averaging_n):
            try:
                image = self.get_new_frame()
                if image is None:
                    self._log.warning(f"Failed to read frame {i + 1}/{self.laser_af_properties.laser_af_averaging_n}")
                    continue

                self.image = image.copy()  # Always store latest frame for error debugging
                full_height, full_width = image.shape[:2]

                if use_center_crop is not None:
                    image = utils.crop_image(image, use_center_crop[0], use_center_crop[1])

                if remove_background:
                    # remove background using top hat filter
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))  # TODO: tmp hard coded value
                    image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

                # calculate centroid using connected components parameters
                # Use large row_tolerance during initialization when spot location is unknown
                row_tolerance = image.shape[0] if ignore_row_tolerance else self.laser_af_properties.cc_row_tolerance
                spot_detection_params = {
                    "threshold": self.laser_af_properties.cc_threshold,
                    "min_area": self.laser_af_properties.cc_min_area,
                    "max_area": self.laser_af_properties.cc_max_area,
                    "row_tolerance": row_tolerance,
                    "max_aspect_ratio": self.laser_af_properties.cc_max_aspect_ratio,
                }

                result = utils.find_spot_location(
                    image,
                    mode=self.laser_af_properties.get_spot_detection_mode(),
                    params=spot_detection_params,
                    filter_sigma=self.laser_af_properties.filter_sigma,
                )
                if result is None:
                    self._log.warning(
                        f"No spot detected in frame {i + 1}/{self.laser_af_properties.laser_af_averaging_n}"
                    )
                    continue

                # Unpack result: (centroid_x, centroid_y)
                spot_x, spot_y = result

                if use_center_crop is not None:
                    x, y = (
                        spot_x + (full_width - use_center_crop[0]) // 2,
                        spot_y + (full_height - use_center_crop[1]) // 2,
                    )
                else:
                    x, y = spot_x, spot_y

                # Check if displacement from reference exceeds the success window (in pixels)
                if (
                    self.laser_af_properties.has_reference
                    and self.laser_af_properties.x_reference is not None
                    and abs(x - self.laser_af_properties.x_reference)
                    > self.laser_af_properties.displacement_success_window_pixels
                ):
                    self._log.warning(
                        f"Spot detected at ({x:.1f}, {y:.1f}) is outside displacement window "
                        f"({abs(x - self.laser_af_properties.x_reference):.1f} > "
                        f"{self.laser_af_properties.displacement_success_window_pixels:.0f} pixels), skipping it."
                    )
                    continue

                tmp_x += x
                tmp_y += y
                successful_detections += 1

            except Exception as e:
                self._log.error(
                    f"Error processing frame {i + 1}/{self.laser_af_properties.laser_af_averaging_n}: {str(e)}"
                )
                continue

        # optionally display the image
        if control._def.LASER_AF_DISPLAY_SPOT_IMAGE:
            self.image_to_display.emit(image)

        # Check if we got enough successful detections
        if successful_detections <= 0:
            self._log.error(f"No successful detections")
            return None

        # Calculate average position from successful detections
        x = tmp_x / successful_detections
        y = tmp_y / successful_detections

        self._log.debug(f"Spot centroid found at ({x:.1f}, {y:.1f}) from {successful_detections} detections")
        return (x, y)

    def get_image(self) -> Optional[np.ndarray]:
        """Capture and display a single image from the laser autofocus camera.

        Turns the laser on, captures an image, displays it, then turns the laser off.

        Returns:
            Optional[np.ndarray]: The captured image, or None if capture failed
        """
        # turn on the laser
        try:
            self.microcontroller.turn_on_AF_laser()
            self.microcontroller.wait_till_operation_is_completed()
        except TimeoutError:
            self._log.exception("Failed to turn on laser AF laser before get_image, cannot get image.")
            return None

        try:
            # send trigger, grab image and display image
            self.camera.send_trigger()
            image = self.camera.read_frame()

            if image is None:
                self._log.error("Failed to read frame in get_image")
                return None

            self.image_to_display.emit(image)
            return image

        except Exception as e:
            self._log.error(f"Error capturing image: {str(e)}")
            return None

        finally:
            # turn off the laser
            try:
                self.microcontroller.turn_off_AF_laser()
                self.microcontroller.wait_till_operation_is_completed()
            except TimeoutError:
                self._log.exception("Failed to turn off AF laser after get_image!")
