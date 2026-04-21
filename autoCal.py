import tweezer.awgcontrol as ac
import tweezer.servers as ts
import numpy as np
from instrumental.drivers.cameras import uc480
import threading
import time
from pathlib import Path
from multiprocessing import shared_memory as _mp_shm
import tweezer.interface as ti
from tweezer.logger import Logger
from dataclasses import dataclass
import tifffile


@dataclass
class CalibrationParams:
    vertical_alignment_delta: float = 0.0
    vertical_alignment_scale: float = 0.0
    horizontal_alignment_delta: float = 0.0


# ── AWG / scheduler parameters ───────────────────────────────────────────────
_MERGE_DISTANCE = 0.5
_CHANNELS   = [0, 1, 2, 3]
_AMPLITUDES = [720, 540, 620, 540]

SCHEDULER_IP   = "169.254.208.242"
SCHEDULER_PORT = 5014
NUM_TWEEZERS   = 39

# Shared memory segments created by TweezerScheduler.__init__ that its stop()
# method closes but never unlinks — leaving them alive between cycles on Windows.
_SCHEDULER_SHM_NAMES = ["tweezer-rt-status"]
# ─────────────────────────────────────────────────────────────────────────────


def _force_unlink_shm() -> None:
    """Unlink shared memory segments on POSIX (no-op on Windows, but harmless).

    On Windows, unlink() does nothing — the OS destroys the segment only when
    every handle is closed. Use _join_image_receiver() first to ensure all
    subprocess handles are released before the next cycle tries to create it.
    On Linux/macOS this removes the /dev/shm/ file directly.
    """
    for name in _SCHEDULER_SHM_NAMES:
        try:
            seg = _mp_shm.SharedMemory(name=name, create=False)
            seg.close()
            seg.unlink()
            print(f"[cleanup] Unlinked leftover shared memory: {name!r}")
        except FileNotFoundError:
            pass  # already gone — nothing to do


def _join_image_receiver(scheduler, timeout: float = 5.0) -> None:
    """Wait for the ImageReceiver subprocess to exit and release its shm handle.

    On Windows, unlink() is a no-op — the named mapping persists until EVERY
    process closes its handle.  ImageReceiver is a separate Process that holds
    its own handle; we must join() it after scheduler.stop() signals it to
    exit, otherwise the mapping is still alive when the next cycle tries to
    create it with create=True and Windows raises FileExistsError.
    """
    ir = getattr(scheduler, 'image_receiver', None)
    if ir is None or not hasattr(ir, 'join'):
        return
    ir.join(timeout=timeout)
    if ir.is_alive():
        print("Warning: ImageReceiver did not exit within timeout — terminating")
        ir.terminate()
        ir.join(timeout=2)


def _shift(params, freqDelta, freqScale):
    p = params.copy()
    for k in range(len(p[:, 1])):
        p[:, 1][k] = (p[:, 1][k] + 2 * np.pi * (k * freqScale)) + 2 * np.pi * freqDelta
        p[:, 1][k] = round(p[:, 1][k] / (2 * np.pi), 3)
        p[:, 1][k] = 2 * np.pi * p[:, 1][k]
    return p


class CalibrationSession:
    """Manages hardware lifecycle for one auto-calibration run.

    The camera is initialised once and kept warm across cycles.
    The AWG and scheduler are created fresh inside every run_cycle call and
    torn down completely before the call returns, so nothing holds the hardware
    between cycle 1 and cycle 2.
    """

    def __init__(self):
        self._cancel = threading.Event()

        log = Logger()
        self._int_10  = log.load_rf_config("2026-1-30-int_10_4MHz")
        self._main_10 = log.load_rf_config("2026-02-02-main_4MHz_10_corrected")

        # Camera stays alive for the full calibration session
        instruments = uc480.list_instruments()
        print(instruments)
        self._cam = uc480.UC480_Camera(instruments[0])
        self._cam.pixelclock = '40MHz'

    def cancel(self) -> None:
        """Signal the session to abort at the next safe point (between shots)."""
        self._cancel.set()

    def _capture_average(self, exposure_ms: float = 0.01,
                         num_shots: int = 20, total_duration: float = 10.0) -> np.ndarray:
        interval = total_duration / num_shots
        accumulator = None
        original_dtype = None
        print(f"Capturing: {num_shots} shots over {total_duration}s...")
        for i in range(num_shots):
            if self._cancel.is_set():
                raise InterruptedError("Calibration cancelled by user")
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

    def run_cycle(self, params: CalibrationParams, phase: str,
                  output_dir: Path = Path('.')) -> tuple[Path, Path]:
        """Program AWG with params, capture images, release AWG, return paths.

        The AWG and scheduler are created at the start of this method and
        unconditionally torn down in the finally block, so every exit path
        (normal, exception, or cancel) leaves the hardware free.
        """
        # ── Build RF parameter arrays ─────────────────────────────────────────
        int_10_merge = _shift(
            self._int_10,
            params.vertical_alignment_delta + _MERGE_DISTANCE,
            params.vertical_alignment_scale,
        )
        int_10_horizontal     = ti.generate_rf_params(np.array([32.5 * 200]), np.array([100]), np.array([0]))
        int_10_horizontal_off = ti.generate_rf_params(np.array([32.5 * 200]), np.array([100]), np.array([0]))
        main_10_merge         = _shift(self._main_10, -0.5, 0)
        main_10_horizontal    = ti.generate_rf_params(np.array([32.5 * 100]), np.array([100]), np.array([0]))
        main_10_horizontal_off = ti.generate_rf_params(np.array([0]),         np.array([100]), np.array([0]))

        rf_params_interlace = [int_10_merge, int_10_horizontal,     main_10_merge, main_10_horizontal_off]
        rf_params_main      = [int_10_merge, int_10_horizontal_off, main_10_merge, main_10_horizontal]

        # ── Fresh AWG initialisation for this cycle ───────────────────────────
        aod = ti.Aod(_CHANNELS, _AMPLITUDES)
        awg = ac.AwgController(400, aod)
        awg.initialize_card()
        aod.add_array(rf_params_interlace, duration=1000, trigger=True, moving_flag=False)
        aod.add_array(rf_params_main,      duration=1000, trigger=True, moving_flag=False)

        # Defensive: unlink any shared memory left by a previous crashed session
        # before TweezerScheduler.__init__ tries to create it with create=True.
        _force_unlink_shm()

        scheduler = ts.TweezerScheduler(awg, NUM_TWEEZERS, SCHEDULER_IP, SCHEDULER_PORT)

        # scheduler.run() is a blocking event loop — run it in a background thread
        # so image capture can proceed. scheduler.stop() in the finally block
        # signals it to exit.
        run_thread = threading.Thread(target=scheduler.run, daemon=True, name="tweezer-scheduler")

        try:
            scheduler.upload_waveforms()
            scheduler.awg.start_hardware()
            scheduler.start_servers()
            run_thread.start()
            print("Image Capture Running!")

            # Capture interlace image (AWG in interlace configuration)
            interlace_path = output_dir / f'interlace_{phase}.tif'
            img = self._capture_average()
            tifffile.imwrite(str(interlace_path), img)
            print(f"Saved {interlace_path}")

            awg.force_trigger()

            # Capture main image (AWG in main configuration)
            main_path = output_dir / f'main_{phase}.tif'
            img = self._capture_average()
            tifffile.imwrite(str(main_path), img)
            print(f"Saved {main_path}")

            return interlace_path, main_path

        finally:
            # ── Unconditional AWG teardown ────────────────────────────────────
            # scheduler.stop() signals run_thread to exit, then we wait for it.
            try:
                scheduler.stop()          # sets stop_irq; closes (but doesn't unlink) shm handle
            except Exception as exc:
                print(f"Warning: scheduler.stop() raised: {exc}")

            # Wait for ImageReceiver subprocess to exit and release its handle.
            # On Windows this is the ONLY way to free the named mapping — unlink()
            # is a no-op there, so the segment persists until all handles are closed.
            _join_image_receiver(scheduler)

            # POSIX: unlink the segment. Windows: no-op, but harmless.
            _force_unlink_shm()

            for fn, label in [
                (awg.stop_card,  "awg.stop_card()"),
                (awg.close_card, "awg.close_card()"),
            ]:
                try:
                    fn()
                except Exception as exc:
                    print(f"Warning: cleanup step '{label}' raised: {exc}")

            run_thread.join(timeout=5)

    def close(self) -> None:
        """Release the camera."""
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
