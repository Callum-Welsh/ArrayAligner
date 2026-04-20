import awgcontrol as ac
import numpy as np
from instrumental.drivers.cameras import uc480
import time
from pathlib import Path
import tweezer.interface as ti
from tweezer.logger import Logger
from dataclasses import dataclass
import tifffile


@dataclass
class CalibrationParams:
    vertical_alignment_delta: float = 0.0
    vertical_alignment_scale: float = 0.0
    horizontal_alignment_delta: float = 0.0


_MERGE_DISTANCE = 0.5
_CHANNELS = [0, 1, 2, 3]
_AMPLITUDES = [720, 540, 620, 540]


def _shift(params, freqDelta, freqScale):
    p = params.copy()
    for k in range(len(p[:, 1])):
        p[:, 1][k] = (p[:, 1][k] + 2 * np.pi * (k * freqScale)) + 2 * np.pi * freqDelta
        p[:, 1][k] = round(p[:, 1][k] / (2 * np.pi), 3)
        p[:, 1][k] = 2 * np.pi * p[:, 1][k]
    return p


class CalibrationSession:
    """Manages hardware lifecycle for one auto-calibration run."""

    def __init__(self):
        log = Logger()
        self._int_10 = log.load_rf_config("2026-1-30-int_10_4MHz")
        self._main_10 = log.load_rf_config("2026-02-02-main_4MHz_10_corrected")

        aod = ac.Aod(_CHANNELS, _AMPLITUDES)
        self._awg = ac.AwgController(400, aod)
        self._awg.initialize_card()
        self._aod = aod

        instruments = uc480.list_instruments()
        print(instruments)
        self._cam = uc480.UC480_Camera(instruments[0])
        self._cam.pixelclock = '40MHz'

    def _capture_average(self, exposure_ms: float = 0.01, num_shots: int = 20, total_duration: float = 10.0) -> np.ndarray:
        interval = total_duration / num_shots
        accumulator = None
        original_dtype = None
        print(f"Capturing: {num_shots} shots over {total_duration}s...")
        for i in range(num_shots):
            start = time.time()
            img = self._cam.grab_image(timeout='10s', copy=True, exposure_time=f"{exposure_ms}ms")
            if accumulator is None:
                original_dtype = img.dtype
                accumulator = img.astype(np.float64)
            else:
                accumulator += img.astype(np.float64)
            print(f"  Shot {i + 1}/{num_shots}")
            if i < num_shots - 1:
                sleep_time = interval - (time.time() - start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print("Warning: capture overran interval")
        return np.round(accumulator / num_shots).astype(original_dtype)

    def run_cycle(self, params: CalibrationParams, phase: str, output_dir: Path = Path('.')) -> tuple[Path, Path]:
        """Program AWG with params, capture and save interlace + main images for the given phase."""
        int_10_merge = _shift(
            self._int_10,
            params.vertical_alignment_delta + _MERGE_DISTANCE,
            params.vertical_alignment_scale,
        )
        int_10_horizontal = ti.generate_rf_params(np.array([32.5 * 200]), np.array([100]), np.array([0]))
        int_10_horizontal_off = ti.generate_rf_params(np.array([32.5 * 200]), np.array([100]), np.array([0]))

        main_10_merge = _shift(self._main_10, -0.5, 0)
        main_10_horizontal = ti.generate_rf_params(np.array([32.5 * 100]), np.array([100]), np.array([0]))
        main_10_horizontal_off = ti.generate_rf_params(np.array([0]), np.array([100]), np.array([0]))

        rf_params_interlace = [int_10_merge, int_10_horizontal, main_10_merge, main_10_horizontal_off]
        rf_params_main = [int_10_merge, int_10_horizontal_off, main_10_merge, main_10_horizontal]

        self._aod.add_array(rf_params_interlace, duration=1000, trigger=True, moving_flag=False)
        self._aod.add_array(rf_params_main, duration=1000, trigger=True, moving_flag=False)
        self._awg.generate_tweezer_arrays()
        self._awg.upload_tweezer_arrays()
        self._awg.start_hardware()

        interlace_path = output_dir / f'interlace_{phase}.tif'
        img = self._capture_average()
        tifffile.imwrite(str(interlace_path), img)
        print(f"Saved {interlace_path}")

        self._awg.force_trigger()

        main_path = output_dir / f'main_{phase}.tif'
        img = self._capture_average()
        tifffile.imwrite(str(main_path), img)
        print(f"Saved {main_path}")

        return interlace_path, main_path

    def close(self):
        try:
            self._cam.close()
        except Exception:
            pass


if __name__ == '__main__':
    session = CalibrationSession()
    try:
        session.run_cycle(CalibrationParams(), 'start')
    finally:
        session.close()
