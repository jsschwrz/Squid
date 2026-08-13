"""
Laser autofocus configuration models.

These models define per-objective laser autofocus settings, including
calibration data and detection parameters.
"""

import base64
from typing import Any, List, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator

import control._def as _def
from control._def import LaserAFConfirmMotionMode, SpotDetectionMode
import squid.logging

_log = squid.logging.get_logger(__name__)

# Fields written by the line-profile spot detector, which connected-components detection
# replaced. They have no equivalent in the new schema and are dropped on load.
#
# displacement_success_window_um is NOT convertible to displacement_success_window_pixels:
# the old field was a convergence tolerance for averaged measurements, the new one is a
# maximum accepted distance from the reference x. Different quantities, so the new default
# from _def is used rather than a fabricated conversion.
_LEGACY_LINE_PROFILE_FIELDS = frozenset(
    {
        "displacement_success_window_um",
        "y_window",
        "x_window",
        "min_peak_width",
        "min_peak_distance",
        "min_peak_prominence",
        "spot_spacing",
    }
)

_migration_warned: set = set()


class LaserAFConfig(BaseModel):
    """
    Laser autofocus configuration (per objective, YAML format).

    Stores calibration data, detection parameters, and reference images
    for the laser autofocus system.
    """

    version: int = Field(1, description="Configuration format version")

    # Crop region
    x_offset: float = Field(0, description="X offset for crop region")
    y_offset: float = Field(0, description="Y offset for crop region")
    width: int = Field(default_factory=lambda: _def.LASER_AF_CROP_WIDTH, description="Width of crop region")
    height: int = Field(default_factory=lambda: _def.LASER_AF_CROP_HEIGHT, description="Height of crop region")

    # Calibration
    pixel_to_um: float = Field(1.0, description="Pixels to micrometers conversion factor")
    x_reference: Optional[float] = Field(None, description="X reference position")
    has_reference: bool = Field(False, description="Whether a reference image exists")
    calibration_timestamp: str = Field("", description="Timestamp of last calibration")
    pixel_to_um_calibration_distance: float = Field(
        default_factory=lambda: _def.PIXEL_TO_UM_CALIBRATION_DISTANCE,
        description="Distance used for pixel-to-um calibration",
    )

    # Detection parameters
    laser_af_range: float = Field(
        default_factory=lambda: float(_def.LASER_AF_RANGE), description="Autofocus search range in um"
    )
    laser_af_averaging_n: int = Field(
        default_factory=lambda: _def.LASER_AF_AVERAGING_N, description="Number of measurements to average"
    )
    laser_af_search_range_um: float = Field(
        default_factory=lambda: float(_def.LASER_AF_RANGE),
        gt=0,
        description="Half-span of the z spot-search, in um. Bounds only the search; laser_af_range "
        "remains the ceiling on an accepted displacement.",
    )
    laser_af_search_step_um: float = Field(
        default_factory=lambda: float(_def.LASER_AF_SEARCH_STEP_UM),
        gt=0,
        description="Z step of the spot-search, in um. Must be > 0: the search builds its positions "
        "by repeated subtraction, so zero would not terminate.",
    )
    confirm_motion_mode: LaserAFConfirmMotionMode = Field(
        default=LaserAFConfirmMotionMode.OFF,
        description="When to verify that a detected spot translates with defocus",
    )
    confirm_step_um: float = Field(
        2.0,
        gt=0,
        description="Extra z step used to confirm the spot translates with z. Must be large enough "
        "that dz/pixel_to_um is measurable, which differs by an order of magnitude between objectives.",
    )
    confirm_tolerance_px: float = Field(
        4.0, gt=0, description="Absolute slack on the predicted translation, in pixels"
    )
    # A plain bool rather than a mode enum: unlike confirm_motion_mode there is only one place
    # this can hook in, so there is no second variant to name.
    iterative_correction_enabled: bool = Field(
        False,
        description="Re-measure after a large correction and move again until the residual settles, "
        "instead of trusting a single linear correction",
    )
    iterative_correction_tolerance_um: float = Field(
        default_factory=lambda: float(_def.LASER_AF_ITERATIVE_CORRECTION_TOLERANCE_UM),
        gt=0,
        description="Stop iterating once the residual displacement is within this",
    )
    iterative_correction_min_displacement_um: float = Field(
        default_factory=lambda: float(_def.LASER_AF_ITERATIVE_CORRECTION_MIN_DISPLACEMENT_UM),
        gt=0,
        description="Only iterate when the initial correction is at least this large. Below it a "
        "single linear move is accurate, and re-measuring would cost a frame grab per FOV for nothing.",
    )
    spot_detection_mode: SpotDetectionMode = Field(
        default_factory=lambda: SpotDetectionMode(_def.LASER_AF_SPOT_DETECTION_MODE),
        description="Spot detection mode",
    )
    displacement_success_window_pixels: float = Field(
        default_factory=lambda: float(_def.DISPLACEMENT_SUCCESS_WINDOW_PIXELS),
        description="Max displacement from reference x to accept detection (pixels)",
    )

    # Spot detection
    spot_crop_size: int = Field(default_factory=lambda: _def.SPOT_CROP_SIZE, description="Size of spot crop region")
    correlation_threshold: float = Field(
        default_factory=lambda: _def.CORRELATION_THRESHOLD, description="Correlation threshold"
    )
    # Connected component spot detection parameters
    cc_threshold: float = Field(
        default_factory=lambda: float(_def.LASER_AF_CC_THRESHOLD), description="Intensity threshold for binarization"
    )
    cc_min_area: int = Field(
        default_factory=lambda: _def.LASER_AF_CC_MIN_AREA, description="Minimum component area in pixels"
    )
    cc_max_area: int = Field(
        default_factory=lambda: _def.LASER_AF_CC_MAX_AREA, description="Maximum component area in pixels"
    )
    cc_row_tolerance: float = Field(
        default_factory=lambda: float(_def.LASER_AF_CC_ROW_TOLERANCE),
        description="Allowed deviation from expected row",
    )
    cc_max_aspect_ratio: float = Field(
        default_factory=lambda: float(_def.LASER_AF_CC_MAX_ASPECT_RATIO),
        description="Maximum aspect ratio for valid spot",
    )
    filter_sigma: Optional[float] = Field(
        default_factory=lambda: _def.LASER_AF_FILTER_SIGMA, description="Gaussian filter sigma (None to disable)"
    )

    # Camera settings
    focus_camera_exposure_time_ms: float = Field(
        default_factory=lambda: float(_def.FOCUS_CAMERA_EXPOSURE_TIME_MS),
        description="Focus camera exposure time in ms",
    )
    focus_camera_analog_gain: float = Field(
        default_factory=lambda: float(_def.FOCUS_CAMERA_ANALOG_GAIN), description="Focus camera analog gain"
    )

    # Initialization
    initialize_crop_width: int = Field(
        default_factory=lambda: _def.LASER_AF_INITIALIZE_CROP_WIDTH, description="Initial crop width"
    )
    initialize_crop_height: int = Field(
        default_factory=lambda: _def.LASER_AF_INITIALIZE_CROP_HEIGHT, description="Initial crop height"
    )

    # Reference image (base64 encoded)
    reference_image: Optional[str] = Field(None, description="Base64-encoded reference image data")
    reference_image_shape: Optional[List[int]] = Field(None, description="Shape of reference image array")
    reference_image_dtype: Optional[str] = Field(None, description="Data type of reference image array")

    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def _clamp_unsatisfiable_correlation_threshold(cls, data: Any) -> Any:
        """Pull a stored correlation_threshold back below 1.0.

        The check is `correlation >= threshold`, and a live frame correlated against a stored
        template never reaches exactly 1.0 -- camera noise alone keeps genuine matches in the
        0.75-0.99 band. A threshold of 1.0 therefore rejects every measurement, including perfect
        ones, and the failure looks like a misaligned spot rather than a bad setting.

        Clamped rather than rejected: LaserAFConfig is loaded through ConfigRepository._load_yaml,
        which swallows ValidationError and returns None, so a `le=` constraint here would silently
        discard the whole objective's calibration over one out-of-range number.
        """
        if isinstance(data, dict):
            threshold = data.get("correlation_threshold")
            if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
                if threshold > _def.MAX_CORRELATION_THRESHOLD:
                    _log.warning(
                        "Laser AF correlation_threshold was %r, which no real measurement can reach, "
                        "so every cross-correlation check would fail. Clamping to %r.",
                        threshold,
                        _def.MAX_CORRELATION_THRESHOLD,
                    )
                    data = dict(data)
                    data["correlation_threshold"] = _def.MAX_CORRELATION_THRESHOLD
        return data

    @model_validator(mode="before")
    @classmethod
    def _default_search_span_from_laser_af_range(cls, data: Any) -> Any:
        """Back-fill the z-search span from laser_af_range for configs written before the split.

        laser_af_range used to bound the search as well as cap an accepted displacement. Configs
        saved then carry a deliberately chosen value -- the 40x objective is set to 40 um, not the
        100 um default -- so taking the field default here would silently widen that objective's
        search on the first launch after upgrade. Carrying the old value across means an existing
        profile behaves identically until someone edits the new setting.

        Deliberately a separate validator from _migrate_legacy_line_profile_config: that one
        early-returns when no legacy keys are present, which is the common case here. The two touch
        disjoint keys, so the order pydantic runs them in does not matter.
        """
        if isinstance(data, dict) and "laser_af_search_range_um" not in data and "laser_af_range" in data:
            data = dict(data)
            data["laser_af_search_range_um"] = data["laser_af_range"]
        return data

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_line_profile_config(cls, data: Any) -> Any:
        """Allow configs written by the line-profile detector to load.

        ``extra="forbid"`` would otherwise reject them. Because
        ``ConfigRepository._load_yaml`` swallows ValidationError and returns None, that
        rejection is silent: the objective would come up with no laser AF config at all,
        losing its calibration and reference image with only a log warning. Dropping the
        dead keys here keeps every real calibration field (pixel_to_um, x_reference,
        reference_image, correlation_threshold, ...) intact.

        The cc_* detection parameters are deliberately NOT translated -- they are a
        different parameterisation of a different algorithm, so they fall back to the
        _def defaults and the objective needs re-tuning.

        ``filter_sigma`` is the one exception, rewritten because leaving it would run the
        new detector unfiltered; see the comment on that branch below.

        Unknown keys that are not on the legacy list still raise, so genuine typos are
        still caught.
        """
        if not isinstance(data, dict):
            return data

        present = _LEGACY_LINE_PROFILE_FIELDS.intersection(data)
        if not present:
            return data

        data = {k: v for k, v in data.items() if k not in _LEGACY_LINE_PROFILE_FIELDS}

        # Legacy configs carry filter_sigma = -1 (or None), the "filtering off" sentinel the
        # line-profile detector ran with. Connected-components detection was developed with
        # the Gaussian pre-filter enabled, so a migrating config adopts the new _def default
        # rather than running the new detector in a regime it was never tuned for.
        #
        # Only applied on this legacy path: a config that has already been migrated (no
        # legacy keys) keeps whatever filter_sigma it was deliberately given, including 0.
        legacy_sigma = data.get("filter_sigma")
        # Guard the comparison: this runs before pydantic coercion, so a malformed value
        # must fall through to normal validation rather than raising TypeError here.
        filtering_disabled = legacy_sigma is None or (
            isinstance(legacy_sigma, (int, float)) and not isinstance(legacy_sigma, bool) and legacy_sigma <= 0
        )
        if filtering_disabled:
            data["filter_sigma"] = _def.LASER_AF_FILTER_SIGMA
            if legacy_sigma != _def.LASER_AF_FILTER_SIGMA:
                _log.info(
                    "Laser AF filter_sigma was %r (off); adopting cc default of %r.",
                    legacy_sigma,
                    _def.LASER_AF_FILTER_SIGMA,
                )

        # One warning per distinct field set, so repeated loads don't spam the log.
        key = tuple(sorted(present))
        if key not in _migration_warned:
            _migration_warned.add(key)
            _log.warning(
                "Laser AF config uses legacy line-profile fields %s; they were dropped. "
                "Connected-components detection parameters (cc_threshold, cc_min_area, "
                "cc_max_area, cc_row_tolerance, cc_max_aspect_ratio) now use defaults from "
                "_def and this objective should be re-tuned. Re-save the config to remove "
                "this warning.",
                ", ".join(sorted(present)),
            )
        return data

    def get_spot_detection_mode(self) -> SpotDetectionMode:
        """Get the SpotDetectionMode enum value."""
        return self.spot_detection_mode

    def set_spot_detection_mode(self, mode: SpotDetectionMode) -> None:
        """Set the spot detection mode from enum."""
        self.spot_detection_mode = mode

    @property
    def reference_image_cropped(self) -> Optional[np.ndarray]:
        """Convert stored base64 data back to numpy array."""
        if self.reference_image is None:
            return None
        data = base64.b64decode(self.reference_image.encode("utf-8"))
        return np.frombuffer(data, dtype=np.dtype(self.reference_image_dtype)).reshape(self.reference_image_shape)

    def set_reference_image(self, image: Optional[np.ndarray]) -> None:
        """Convert numpy array to base64 encoded string or clear reference if None."""
        if image is None:
            self.reference_image = None
            self.reference_image_shape = None
            self.reference_image_dtype = None
            self.has_reference = False
            return
        self.reference_image = base64.b64encode(image.tobytes()).decode("utf-8")
        self.reference_image_shape = list(image.shape)
        self.reference_image_dtype = str(image.dtype)
        self.has_reference = True
