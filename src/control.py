"""
motor_controller.py — Arduino motor controller for Rubik's-cube solver
=====================================================================

This module provides two main classes:

* ArduinoConnector — utility to find and open a serial port connected to
  an Arduino-like device (VID/PID + descriptive keyword matching).
* MotorController — higher-level interface for sending move sequences,
  coalescing `set_move_duration` requests and providing simulation mode
  when no serial device is found.

Protocol (firmware expected):
 - "T\n"          -> test -> Arduino replies "OK\n" when finished
 - "S{ms}\n"      -> set move duration to ms milliseconds -> no reply
 - "<sequence>\n" -> space-separated move tokens -> Arduino replies "OK\n" when finished

Design notes:
 - `send_sequence()` is blocking and waits for `OK` by default; integrate into
   your GUI on a background thread or use the non-blocking simulation mode.
 - `set_move_duration(ms)` schedules sending the `S{ms}` command via a short-
   lived background worker which coalesces rapid updates to avoid spamming serial.

-------------------------------------------------------------------------------

Copyright (c) 2025 Facundo Gauna & Ulises Carnevale. Licensed under MIT License.
"""

from __future__ import annotations
import time
import logging
import threading
from typing import Optional, List, Callable, Sequence, Tuple

import serial
import serial.tools.list_ports

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 1.0


class ArduinoConnectorError(Exception):
    """Raised for connector failures (port not found, cannot open)."""
    pass


class ArduinoConnector:
    """Discover and open serial ports that look like Arduino-like devices.

    Parameters
    ----------
    baudrate: serial baud
    accepted_vid_pid: sequence of (vid, pid) pairs; pid may be None to accept any pid for given vid
    strict_keywords: case-insensitive keywords to match in port descriptions as fallback
    require_handshake: whether to actually try a handshake to confirm the device
    handshake_command: optional bytes to send for handshake (default b'PING\n')
    handshake_responses: substrings considered a valid handshake response (case-insensitive)
    handshake_timeout: seconds to wait during handshake
    """

    DEFAULT_ACCEPTED_VID_PID: Sequence[Tuple[int, Optional[int]]] = (
        (0x2341, None),     # Official Arduino
        (0x2A03, None),     # Arduino (SAMD)
        (0x1A86, 0x7523),   # CH340
        (0x0403, 0x6001),   # FTDI
        (0x10C4, 0xEA60),   # CP210x
        (0x067B, 0x2303),   # Prolific
    )

    DEFAULT_STRICT_KEYWORDS: Sequence[str] = (
        "arduino", "mega", "uno", "nano", "leonardo", "pro micro", "esp32", "esp8266"
    )

    def __init__(
        self,
        baudrate: int = DEFAULT_BAUD,
        accepted_vid_pid: Optional[Sequence[Tuple[int, Optional[int]]]] = None,
        strict_keywords: Optional[Sequence[str]] = None,
        require_handshake: bool = False,
        handshake_command: Optional[bytes] = b"PING\n",
        handshake_responses: Optional[Sequence[str]] = None,
        handshake_timeout: float = DEFAULT_TIMEOUT,
    ):
        self.baudrate = baudrate
        self.accepted_vid_pid = accepted_vid_pid or self.DEFAULT_ACCEPTED_VID_PID
        self.strict_keywords = [k.lower() for k in (strict_keywords or self.DEFAULT_STRICT_KEYWORDS)]
        self.require_handshake = require_handshake
        self.handshake_command = handshake_command
        self.handshake_responses = [r.lower() for r in (handshake_responses or ("pong", "arduino", "mega"))]
        self.handshake_timeout = handshake_timeout

        self._serial: Optional[serial.Serial] = None
        self._device: Optional[str] = None

    def _matches_vid_pid(self, vid: Optional[int], pid: Optional[int]) -> bool:
        if vid is None:
            return False
        for v, p in self.accepted_vid_pid:
            if vid == v and (p is None or pid == p):
                return True
        return False

    def _gather_ports(self) -> List[serial.tools.list_ports.ListPortInfo]:
        return list(serial.tools.list_ports.comports())

    def _score_port(self, p: serial.tools.list_ports.ListPortInfo):
        # sort so higher-quality candidates come first
        s = 0
        if getattr(p, "serial_number", None):
            s += 100
        if getattr(p, "product", None):
            s += 10
        if getattr(p, "manufacturer", None):
            s += 5
        dev = (p.device or "").lower()
        if dev.startswith("/dev/ttyacm") or dev.startswith("/dev/ttyusb") or "cu.usb" in dev or "usbserial" in dev:
            s += 1
        # returned key sorts ascending; negative score ensures larger s sorts first
        return (-s, len(dev))

    def _default_handshake(self, ser: serial.Serial) -> bool:
        """Non-invasive handshake: write handshake_command and check for one of handshake_responses."""
        if not self.handshake_command:
            return False
        try:
            prev_timeout = ser.timeout
            ser.timeout = self.handshake_timeout
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            time.sleep(0.02)
            ser.write(self.handshake_command)
            ser.flush()
            start = time.time()
            buf = b""
            while time.time() - start < self.handshake_timeout:
                chunk = ser.read(64)
                if chunk:
                    buf += chunk
                    text = buf.decode(errors="ignore").lower()
                    for r in self.handshake_responses:
                        if r in text:
                            return True
                else:
                    time.sleep(0.01)
            # no response matched
            return False
        except Exception as e:
            logger.debug("Handshake failed: %s", e)
            return False
        finally:
            try:
                ser.timeout = prev_timeout
            except Exception:
                pass

    def list_candidates(self) -> List[serial.tools.list_ports.ListPortInfo]:
        ports = self._gather_ports()
        candidates: List[serial.tools.list_ports.ListPortInfo] = []
        for p in ports:
            vid = getattr(p, "vid", None)
            pid = getattr(p, "pid", None)
            if self._matches_vid_pid(vid, pid):
                candidates.append(p)
        if not candidates:
            # fallback: match keywords in description/product/manufacturer/hwid
            for p in ports:
                desc = (p.description or "").lower()
                prod = (getattr(p, "product", "") or "").lower()
                manu = (p.manufacturer or "").lower()
                hwid = (p.hwid or "").lower()
                combined = " ".join([desc, prod, manu, hwid])
                if any(k in combined for k in self.strict_keywords):
                    candidates.append(p)
        candidates.sort(key=self._score_port)
        return candidates

    def find_port(self) -> Optional[str]:
        candidates = self.list_candidates()
        logger.debug("Found candidate ports: %s", [p.device for p in candidates])
        for p in candidates:
            logger.info("Candidate: %s (%s) hwid=%s", p.device, p.description, p.hwid)
            if not self.require_handshake:
                return p.device
            try:
                # try handshake
                with serial.Serial(p.device, self.baudrate, timeout=self.handshake_timeout) as ser:
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                    ok = False
                    try:
                        ok = self._default_handshake(ser)
                    except Exception as e:
                        logger.debug("Handshake raised exception on %s: %s", p.device, e)
                    if ok:
                        logger.info("Handshake OK on %s", p.device)
                        return p.device
                    else:
                        logger.debug("Handshake not confirmed on %s", p.device)
            except Exception as e:
                logger.debug("Could not open %s: %s", p.device, e)
                continue
        logger.info("No serial port matched criteria")
        return None

    def connect(self, port: Optional[str] = None, open_if_needed: bool = True) -> Optional[serial.Serial]:
        if self._serial and self._serial.is_open:
            return self._serial

        port_to_use = port or self.find_port()
        if not port_to_use:
            raise ArduinoConnectorError("No serial port found for Arduino-like device")

        try:
            ser = serial.Serial(port_to_use, self.baudrate, timeout=1)
            time.sleep(0.1)
            self._serial = ser
            self._device = port_to_use
            logger.info("Connected to %s", port_to_use)
            return ser
        except Exception as e:
            raise ArduinoConnectorError(f"Failed to open port {port_to_use}: {e}")

    def disconnect(self) -> None:
        if self._serial:
            try:
                if self._serial.is_open:
                    self._serial.close()
            except Exception:
                pass
        self._serial = None
        self._device = None

    @property
    def connected(self) -> bool:
        return bool(self._serial and self._serial.is_open)

    @property
    def device(self) -> Optional[str]:
        return self._device

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()


class MotorController:
    """High level controller to send moves and manage move duration settings.

    For GUI applications: do not call blocking methods (send_sequence, test_all_motors)
    from the UI thread — run them on a worker thread instead.

    Parameters
    ----------
    available_moves: list of legal move tokens (default: common cube moves)
    port: optional serial port path (if provided, auto-detect is skipped unless None and auto_detect=True)
    baudrate: serial baud
    timeout: general timeout for operations
    auto_detect: try to find port automatically if port is None
    per_move_time: estimated seconds per token (used to compute fallback sequence timeout)
    require_handshake / handshake params are passed to the internal ArduinoConnector if needed.
    """

    def __init__(
        self,
        available_moves: Optional[List[str]] = None,
        port: Optional[str] = None,
        baudrate: int = DEFAULT_BAUD,
        timeout: float = DEFAULT_TIMEOUT,
        auto_detect: bool = True,
        per_move_time: float = 0.06,
        require_handshake: bool = False,
        handshake_command: Optional[bytes] = b"PING\n",
        handshake_responses: Optional[Sequence[str]] = None,
    ):
        if available_moves is None:
            available_moves = [
                "U", "U'", "U2",
                "R", "R'", "R2",
                "F", "F'", "F2",
                "D", "D'", "D2",
                "L", "L'", "L2",
                "B", "B'", "B2",
            ]
        self.available_moves = available_moves
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.auto_detect = auto_detect
        self.per_move_time = per_move_time
        self.ack_per_move = True  # default; can be toggled by user
        self.simulation_mode = False

        self.ser: Optional[serial.Serial] = None
        self._connector = ArduinoConnector(
            baudrate=self.baudrate,
            require_handshake=require_handshake,
            handshake_command=handshake_command,
            handshake_responses=handshake_responses,
        )

        # Move-duration coalescer state
        self._desired_duration: Optional[int] = None
        self._last_sent_duration: Optional[int] = None
        self._duration_lock = threading.Lock()
        self._duration_event = threading.Event()
        self._shutdown_worker = False
        self._worker: Optional[threading.Thread] = None

        # Worker parameters
        self._MIN_IDLE_BEFORE_SEND = 0.15
        self._MIN_INTERVAL_BETWEEN_WRITES = 0.08

        # user callback (optional) to receive decoded lines from Arduino
        # signature: fn(line: str) -> None
        self.on_receive: Optional[Callable[[str], None]] = None

        self._start_worker()

    def _start_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._shutdown_worker = False
            self._worker = threading.Thread(target=self._duration_worker, name="MotorDurationWorker", daemon=True)
            self._worker.start()

    def _find_port(self) -> Optional[str]:
        return self._connector.find_port()

    def connect(self) -> bool:
        if self.port is None and self.auto_detect:
            try:
                self.port = self._find_port()
            except Exception:
                self.port = None

        if self.port is None:
            logger.warning("No serial port found; switching to simulation mode.")
            self.simulation_mode = True
            self.ser = None
            return False

        try:
            logger.info("Opening serial %s @ %d", self.port, self.baudrate)
            ser = self._connector.connect(port=self.port)
            self.ser = ser
            # give device time to boot/reset (often Arduinos reset on open)
            time.sleep(2.0)
            try:
                if self.ser:
                    self.ser.reset_input_buffer()
            except Exception:
                pass
            self.simulation_mode = False
            return True
        except ArduinoConnectorError as e:
            logger.exception("Failed to open serial port via connector: %s", e)
            self.simulation_mode = True
            self.ser = None
            return False
        except Exception as e:
            logger.exception("Failed to open serial port: %s", e)
            self.simulation_mode = True
            self.ser = None
            return False

    def send_sequence(self, sequence: str, timeout_per_token: float = 0.5) -> bool:
        """Send a sequence of space-separated tokens to the Arduino.

        If ack_per_move is True: send each token and wait for an OK per move.
        If False: send full sequence and wait once for final OK.

        Returns True on success (received OK); False on timeout or error.
        """
        moves = [tok for tok in sequence.split() if tok]
        if not moves:
            logger.debug("Empty sequence; nothing to send.")
            return True

        if self.simulation_mode or self.ser is None:
            logger.debug("[SIM] send_sequence: %s", moves)
            # simulate some delay
            time.sleep(min(len(moves) * self.per_move_time, 0.1))
            # optionally notify callback
            if self.on_receive:
                self.on_receive("OK")
            return True

        if self.ack_per_move:
            for mv in moves:
                try:
                    self._write_line(mv)
                except Exception:
                    logger.exception("Failed writing move: %s", mv)
                    return False
                timeout = max(0.2, self.per_move_time + timeout_per_token)
                ok = self._wait_for_ok(timeout)
                if not ok:
                    logger.warning("No OK for move %s", mv)
                    return False
            return True
        else:
            # send full sequence and wait once for OK
            try:
                self._write_line(" ".join(moves))
            except Exception:
                logger.exception("Failed writing full sequence")
                return False
            # compute reasonable timeout = per_move_time * count + user timeout_per_token * count + safety
            total_timeout = max(1.0, len(moves) * (self.per_move_time + timeout_per_token) + 0.5)
            ok = self._wait_for_ok(total_timeout)
            if not ok:
                logger.warning("No final OK for sequence (timeout %.2fs).", total_timeout)
                return False
            return True

    def test_all_motors(self, timeout: float = 15.0) -> bool:
        """Run firmware motor test by sending 'T' and waiting for an 'OK'.
        Returns True if OK received before timeout.
        """
        if self.simulation_mode:
            logger.info("[SIM] test_all_motors")
            if self.on_receive:
                self.on_receive("OK")
            return True
        try:
            self._write_line("T")
        except Exception:
            logger.exception("Failed writing test command")
            return False
        return self._wait_for_ok(timeout)

    def set_move_duration(self, ms: int) -> None:
        """Schedule the desired move duration (in ms). Non-blocking."""
        try:
            ms_int = max(1, int(ms))
        except Exception:
            ms_int = 400
        with self._duration_lock:
            self._desired_duration = ms_int
            self._duration_event.set()
        logger.debug("Requested move-duration set to %d ms (scheduled)", ms_int)

    def _write_line(self, line: str) -> None:
        payload = (line + "\n").encode("utf-8")
        if self.simulation_mode or self.ser is None:
            logger.debug("[SIM WRITE] %s", line)
            return
        try:
            self.ser.write(payload)
            self.ser.flush()
            logger.debug("[WRITE] %s", line)
        except Exception as e:
            logger.exception("Serial write failed for line '%s': %s", line, e)
            raise

    def _wait_for_ok(self, timeout: float) -> bool:
        """Wait for a line containing 'OK' (case-insensitive) until deadline."""
        if self.simulation_mode or self.ser is None:
            time.sleep(min(timeout, 0.05))
            return True
        deadline = time.time() + timeout
        buf = b""
        try:
            prev_timeout = self.ser.timeout
            self.ser.timeout = 0.1
            while time.time() < deadline:
                chunk = self.ser.read(256)
                if chunk:
                    buf += chunk
                    # decode and split into lines; keep any partial final fragment in buf
                    try:
                        text = buf.decode(errors="ignore")
                    except Exception:
                        text = ""
                    lines = text.replace("\r", "").split("\n")
                    for line in lines:
                        if not line:
                            continue
                        stripped = line.strip()
                        logger.debug("Received line: %s", stripped)
                        if self.on_receive:
                            # non-blocking callback
                            try:
                                self.on_receive(stripped)
                            except Exception:
                                logger.exception("on_receive callback raised")
                        if "ok" == stripped.lower() or "ok" in stripped.lower():
                            # found OK
                            self.ser.timeout = prev_timeout
                            return True
                    # keep only partial data after last newline
                    if text.endswith("\n"):
                        buf = b""
                    else:
                        # keep the bytes of last partial line
                        last_newline = text.rfind("\n")
                        if last_newline >= 0:
                            partial = text[last_newline+1:]
                            buf = partial.encode(errors="ignore")
                        else:
                            # no newlines yet; keep all
                            buf = buf
                else:
                    time.sleep(0.02)
            logger.warning("Timeout waiting for OK (%.2fs). Buffer: %s", timeout, buf[:200])
            try:
                self.ser.timeout = prev_timeout
            except Exception:
                pass
            return False
        except Exception as e:
            logger.exception("Error reading serial: %s", e)
            try:
                self.ser.timeout = prev_timeout
            except Exception:
                pass
            return False

    def disconnect(self) -> None:
        """Close serial and stop worker."""
        self.close()
        if self.ser:
            try:
                if self.ser.is_open:
                    self.ser.close()
                    logger.info("Serial closed.")
            except Exception as e:
                logger.warning("Error closing serial: %s", e)
        try:
            self._connector.disconnect()
        except Exception:
            pass
        self.ser = None
        self.simulation_mode = False

    def _duration_worker(self) -> None:
        """Background worker coalescing rapid set_move_duration calls."""
        last_write_ts = 0.0
        while not self._shutdown_worker:
            got = self._duration_event.wait(timeout=1.0)
            if self._shutdown_worker:
                break
            if not got:
                continue
            time.sleep(self._MIN_IDLE_BEFORE_SEND)
            with self._duration_lock:
                ms = self._desired_duration
                self._duration_event.clear()
            if ms is None:
                continue
            if self._last_sent_duration is not None and self._last_sent_duration == ms:
                logger.debug("Duration %d ms already sent, skipping", ms)
                continue
            now = time.time()
            dt = now - last_write_ts
            if dt < self._MIN_INTERVAL_BETWEEN_WRITES:
                time.sleep(self._MIN_INTERVAL_BETWEEN_WRITES - dt)
            line = f"S{int(ms)}"
            try:
                self._write_line(line)
                logger.info("Sent duration to Arduino: %d ms", ms)
            except Exception as e:
                logger.exception("Failed to send duration to Arduino: %s", e)
            self._last_sent_duration = ms
            last_write_ts = time.time()

    def scramble(self, num_moves: int) -> List[str]:
        import random
        moves: List[str] = []
        last_move = ""
        for _ in range(num_moves):
            move = random.choice(self.available_moves)
            while last_move and move[0] == last_move[0]:
                move = random.choice(self.available_moves)
            moves.append(move)
            last_move = move
        return moves

    def close(self) -> None:
        """Stop worker thread and wait for it to finish."""
        self._shutdown_worker = True
        try:
            self._duration_event.set()
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=1.0)
        except Exception:
            pass
        self._worker = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="serial port")
    args = parser.parse_args()

    ctrl = MotorController(port=args.port)
    okc = ctrl.connect()
    print("Connected:", not ctrl.simulation_mode)
    print("Running test... (send 'T' test)")
    ok = ctrl.test_all_motors()
    print("Test OK:", ok)
    print("Sending sequence: R U R' U2")
    ok = ctrl.send_sequence("R U R' U2", timeout_per_token=1)
    print("Sequence OK:", ok)
    ctrl.disconnect()