"""
motor_controller.py

Improved MotorController for communicating with the updated Arduino sketch.

Features:
- Auto-detect Arduino serial port (heuristic).
- Uses new human-readable protocol: SEQ, MOVE, SET, GET, TEST, STATUS.
- Robust response parsing (OK, ERR, BUSY, READY, VALUE, STATUS, PROGRESS).
- Blocking API with configurable timeouts and retries.
- Optional progress callback for long sequences.
- Simulation mode when no hardware connected.
- Legacy compatibility: send single-char opcodes (three-char opcodes) if requested.
- Context manager support.

Usage:
    from motor_controller import MotorController

    with MotorController() as mc:
        mc.connect()
        mc.set_steps('F', 820)
        steps = mc.get_steps('F')
        mc.send_sequence("R U R' U2 D L2", progress_callback=my_cb)
"""

from __future__ import annotations
import serial
import serial.tools.list_ports
import time
import logging
from typing import Optional, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Default constants
DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 1.0          # serial read timeout (s) for readline()
DEFAULT_COMMAND_TIMEOUT = 60.0 # how long to wait for OK after sending a SEQ by default
DEFAULT_RETRIES = 2
DEFAULT_SMALL_DELAY = 0.02     # small delay after writes (s)

# Response types we expect from Arduino
RESP_OK = "OK"
RESP_ERR = "ERR"
RESP_BUSY = "BUSY"
RESP_READY = "READY"
RESP_VALUE = "VALUE"
RESP_STATUS = "STATUS"
RESP_PROGRESS = "PROGRESS"  # optional extension printed by Arduino

# If serial isn't found or connection fails, controller behaves in "simulation" mode
# (useful for testing logic without hardware).
class MotorController:
    """
    MotorController manages serial comms with the Arduino-based motor controller.

    Key methods:
      - connect() / disconnect()
      - send_sequence(sequence_str, timeout=..., progress_callback=...)
      - move(face, angle, dir)
      - set_steps(face, steps_per_90)
      - get_steps(face) -> int | None
      - test_all_motors()
      - execute_legacy_opcode(opcode)  # for backward compat if needed
    """

    def __init__(
        self,
        baud_rate: int = DEFAULT_BAUD,
        timeout: float = DEFAULT_TIMEOUT,
        auto_detect: bool = True,
        port: Optional[str] = None,
        simulation_if_missing: bool = True,
        default_command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
    ):
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.port = port
        self.auto_detect = auto_detect
        self.simulation_if_missing = simulation_if_missing
        self.default_command_timeout = default_command_timeout
        self.retries = retries

        self.serial_connection: Optional[serial.Serial] = None
        self.is_connected: bool = False

        # Backwards compatibility map (keeps your original mapping)
        self.gcode_map: Dict[str, str] = {
            "D": "D01", "D'": "D00", "D2": "D11",
            "F": "F01", "F'": "F00", "F2": "F11",
            "R": "R01", "R'": "R00", "R2": "R11",
            "L": "L01", "L'": "L00", "L2": "L11",
            "B": "B01", "B'": "B00", "B2": "B11",
            "U": "U01", "U'": "U00", "U2": "U11"
        }

    # -------------------------
    # Port discovery
    # -------------------------
    def _find_arduino_port(self) -> Optional[str]:
        """
        Heuristic search for likely Arduino / USB-Serial device.
        Tries several keywords, then falls back to any accessible port.
        """
        logger.debug("Scanning serial ports for Arduino-like device...")
        ports = serial.tools.list_ports.comports()
        arduino_keywords = ["arduino", "ch340", "usb serial", "usb-to-serial", "usbuart", "usb serial device"]

        # First pass: match description/manufacturer vs keywords
        for p in ports:
            desc = (p.description or "").lower()
            manu = (p.manufacturer or "").lower()
            if any(k in desc for k in arduino_keywords) or any(k in manu for k in arduino_keywords):
                logger.info(f"Found Arduino-like port: {p.device} ({p.description})")
                return p.device

        # Fallback: try any port that we can open
        for p in ports:
            try:
                test = serial.Serial(p.device, self.baud_rate, timeout=1)
                test.close()
                logger.info(f"No Arduino heuristics matched; using available port: {p.device}")
                return p.device
            except Exception:
                continue

        logger.debug("No serial ports found matching heuristics.")
        return None

    # -------------------------
    # Connect / Disconnect
    # -------------------------
    def connect(self) -> bool:
        """
        Open serial port. If port is not provided and auto_detect is True,
        tries to discover a port.

        Returns True if connected (hardware) OR simulation mode is active.
        """
        if self.port is None and self.auto_detect:
            port = self._find_arduino_port()
            if port is None:
                if self.simulation_if_missing:
                    logger.warning("Arduino not found; entering simulation mode.")
                    self.is_connected = False
                    self.serial_connection = None
                    return True
                logger.error("Arduino not found and simulation disabled.")
                return False
            self.port = port

        if self.port is None:
            # no port and auto-detect off
            if self.simulation_if_missing:
                logger.warning("No port provided; entering simulation mode.")
                self.is_connected = False
                return True
            logger.error("No port provided and simulation disabled.")
            return False

        try:
            logger.info(f"Opening serial port {self.port} at {self.baud_rate} baud...")
            self.serial_connection = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
            # Give Arduino time to reset if it does
            time.sleep(2.0)
            # flush any leftover
            self.serial_connection.reset_input_buffer()
            self.serial_connection.reset_output_buffer()
            self.is_connected = True
            logger.info("Connected to Arduino.")
            return True
        except Exception as e:
            logger.exception(f"Failed to open serial port {self.port}: {e}")
            if self.simulation_if_missing:
                logger.warning("Falling back to simulation mode.")
                self.is_connected = False
                self.serial_connection = None
                return True
            return False

    def disconnect(self) -> None:
        """Close serial if open and disables connected flag."""
        if self.serial_connection:
            try:
                if self.serial_connection.is_open:
                    self.serial_connection.close()
                    logger.info("Serial connection closed.")
            except Exception as e:
                logger.warning(f"Error closing serial: {e}")
        self.serial_connection = None
        self.is_connected = False

    def __enter__(self) -> "MotorController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    # -------------------------
    # Low-level send and receive
    # -------------------------
    def _write_line(self, line: str) -> None:
        """
        Write a single line to Arduino, appending newline.
        In simulation mode, just log.
        """
        line_to_send = line.strip() + "\n"
        if not self.is_connected or self.serial_connection is None:
            logger.debug(f"[SIM WRITE] {line_to_send.strip()}")
            return
        try:
            self.serial_connection.write(line_to_send.encode("utf-8"))
            self.serial_connection.flush()
            # tiny pause to let Arduino process (tunable)
            time.sleep(DEFAULT_SMALL_DELAY)
            logger.debug(f"[WRITE] {line_to_send.strip()}")
        except Exception as e:
            logger.exception(f"Serial write failed: {e}")
            raise

    def _read_line(self, timeout: float) -> Optional[str]:
        """
        Read a single line (terminated by newline) with a local timeout.
        Returns stripped string or None if timed out / disconnected.
        """
        if not self.is_connected or self.serial_connection is None:
            logger.debug("[SIM READ] nothing to read in simulation mode.")
            return None

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.serial_connection.in_waiting:
                    raw = self.serial_connection.readline()
                    if not raw:
                        continue
                    s = raw.decode("utf-8", errors="ignore").strip()
                    if s == "":
                        continue
                    logger.debug(f"[READ] {s}")
                    return s
            except Exception as e:
                logger.exception(f"Serial read error: {e}")
                return None
            time.sleep(0.005)
        logger.debug("Read timed out")
        return None

    def _drain_until_ok_or_err(
        self,
        timeout: float,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        After sending a command that returns BUSY/OK/ERR, use this helper to wait for final status.

        Returns (success(bool), final_response_text_or_errmsg).
        On success returns (True, 'OK'). On error returns (False, 'ERR msg').
        If timeout, returns (False, None).
        Also calls progress_callback when "PROGRESS ..." messages arrive.
        """
        if not self.is_connected:
            # Simulation mode -> immediate OK
            logger.debug("[SIM] drain -> OK")
            return True, RESP_OK

        deadline = time.time() + timeout
        seen_busy = False
        while time.time() < deadline:
            line = self._read_line(timeout= max(0.1, min(1.0, deadline - time.time())))
            if line is None:
                continue
            # parse
            if line == RESP_BUSY:
                seen_busy = True
                logger.debug("Arduino responded BUSY")
                continue
            if line == RESP_READY:
                # some sketches may send READY -- continue waiting for OK/ERR
                logger.debug("Arduino responded READY")
                continue
            if line == RESP_OK:
                logger.debug("Arduino responded OK")
                return True, RESP_OK
            if line.startswith(RESP_ERR):
                # ERR may be "ERR msg"
                logger.error(f"Arduino reported error: {line}")
                return False, line
            if line.startswith(RESP_VALUE) or line.startswith(RESP_STATUS):
                # caller might want these; return them as well (not OK/ERR)
                logger.info(f"Arduino informational: {line}")
                return True, line
            if line.startswith(RESP_PROGRESS):
                # call progress callback if provided, but keep waiting for OK
                if progress_callback:
                    try:
                        progress_callback(line)
                    except Exception as cb_e:
                        logger.exception("progress_callback raised an exception")
                continue
            # Unknown line: log and continue
            logger.debug(f"Arduino -> {line} (unhandled)")
        logger.error("Timeout waiting for final OK/ERR from Arduino")
        return False, None

    # -------------------------
    # High-level commands
    # -------------------------
    def send_sequence(
        self,
        solution_line: str,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        retries: Optional[int] = None,
    ) -> bool:
        """
        Send a full solution line to Arduino using SEQ <moves...>.
        Moves should be in standard cube notation separated by spaces:
            e.g. "R U R' U2 D L2"

        Blocks until Arduino replies OK or ERR. Returns True on success.
        Optionally accepts progress_callback that receives raw response lines such as "PROGRESS 3/20".
        """
        if timeout is None:
            timeout = self.default_command_timeout
        if retries is None:
            retries = self.retries

        if not self.is_connected:
            # simulation mode
            logger.info(f"[SIM] SEQ {solution_line}")
            return True

        cmd = f"SEQ {solution_line.strip()}"
        for attempt in range(1, retries + 1):
            try:
                self._write_line(cmd)
                success, info = self._drain_until_ok_or_err(timeout, progress_callback=progress_callback)
                if success:
                    return True
                # failure - log and maybe retry
                logger.warning(f"SEQ attempt {attempt} failed: {info}")
            except Exception as e:
                logger.exception(f"Exception during SEQ attempt {attempt}: {e}")
            if attempt < retries:
                logger.info("Retrying SEQ...")
                # small backoff
                time.sleep(0.2 * attempt)
        return False

    def move(self, face: str, angle: int = 90, direction: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Perform a single explicit move:
            face: 'R','L','F','B','U','D' (char)
            angle: 90, 180, 270
            direction: 0 or 1 (direction encoding expected by Arduino)
        Returns True on success.
        """
        face = face.strip().upper()
        if angle not in (90, 180, 270):
            raise ValueError("angle must be 90, 180, or 270")
        if direction not in (0, 1, True, False):
            raise ValueError("direction must be 0 or 1")

        if timeout is None:
            timeout = 10.0

        if not self.is_connected:
            logger.info(f"[SIM] MOVE {face} {angle} {direction}")
            return True

        cmd = f"MOVE {face} {angle} {int(bool(direction))}"
        self._write_line(cmd)
        success, info = self._drain_until_ok_or_err(timeout)
        return success

    def set_steps(self, face: str, steps_per_90: int, timeout: Optional[float] = None) -> bool:
        """
        SET <face> <steps_per_90> - store calibration in EEPROM on Arduino.
        """
        face = face.strip().upper()
        if timeout is None:
            timeout = 5.0
        if not self.is_connected:
            logger.info(f"[SIM] SET {face} {steps_per_90}")
            return True
        cmd = f"SET {face} {int(steps_per_90)}"
        self._write_line(cmd)
        success, info = self._drain_until_ok_or_err(timeout)
        return success

    def get_steps(self, face: str, timeout: Optional[float] = None) -> Optional[int]:
        """
        GET <face> -> Arduino responds: VALUE <face> <steps>
        Returns number of steps or None on failure/timeout.
        """
        face = face.strip().upper()
        if timeout is None:
            timeout = 3.0
        if not self.is_connected:
            logger.info(f"[SIM] GET {face} -> returning None in simulation")
            return None

        # clear input, send GET, and wait for VALUE or ERR
        self.serial_connection.reset_input_buffer()
        self._write_line(f"GET {face}")

        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._read_line(timeout= max(0.05, deadline - time.time()))
            if line is None:
                continue
            if line.startswith(RESP_VALUE):
                # format: VALUE F 800
                try:
                    parts = line.split()
                    if len(parts) >= 3 and parts[1].upper() == face:
                        return int(parts[2])
                except Exception:
                    logger.exception("Failed to parse VALUE line")
                    return None
            if line.startswith(RESP_ERR):
                logger.error(f"Arduino returned error for GET: {line}")
                return None
            # ignore other messages while waiting
        logger.error("Timeout waiting for VALUE reply to GET")
        return None

    def test_all_motors(self, timeout: Optional[float] = None) -> bool:
        """Send TEST command to run the hardware test sequence."""
        if timeout is None:
            timeout = 30.0
        if not self.is_connected:
            logger.info("[SIM] TEST")
            return True
        self._write_line("TEST")
        success, info = self._drain_until_ok_or_err(timeout)
        return success

    def get_status(self, timeout: float = 2.0) -> Optional[str]:
        """
        Ask for STATUS. Returns textual status or None on timeout.
        Example Arduino reply: "STATUS READY"
        """
        if not self.is_connected:
            logger.info("[SIM] STATUS -> simulation returns 'STATUS READY'")
            return "STATUS READY"
        self.serial_connection.reset_input_buffer()
        self._write_line("STATUS")
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._read_line(timeout= max(0.05, deadline - time.time()))
            if line is None:
                continue
            if line.startswith(RESP_STATUS):
                return line
            # Arduino might reply READY or BUSY directly
            if line in (RESP_READY, RESP_BUSY):
                return f"STATUS {line}"
            if line.startswith(RESP_ERR):
                logger.error(f"STATUS returned ERR: {line}")
                return line
        logger.error("Timeout waiting for STATUS")
        return None

    # -------------------------
    # Backwards compatibility: send three-char opcode as before
    # -------------------------
    def execute_legacy_opcode(self, opcode: str, per_char_delay: float = 0.05) -> bool:
        """
        Send the legacy 3-character opcode as in your original script.
        The original scheme sent each char separately followed by newline.
        This helper keeps that behaviour.

        Example:
            opcode = "R01"  (3 characters)
        Returns True (simulation) or True/False based on Arduino responses (if available).
        """
        opcode = opcode.strip()
        if len(opcode) != 3:
            raise ValueError("Legacy opcode must be 3 characters long (e.g. 'R01')")

        if not self.is_connected:
            logger.info(f"[SIM] legacy opcode -> {opcode}")
            return True

        # original Arduino expected each char as a separate line in your old sketch.
        # Our new Arduino also supports MOVE/SEQ — but keep compatibility: send as three separate lines.
        try:
            for ch in opcode:
                self._write_line(ch)
                time.sleep(per_char_delay)
            # The legacy sketch acted on arrival; our new sketch may interpret single characters differently.
            # So we don't wait for OK here. If you want an ACK, modify Arduino to reply.
            return True
        except Exception:
            logger.exception("Failed sending legacy opcode")
            return False

# If run directly, quick demo (only when executed as script)
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="MotorController demo CLI")
    parser.add_argument("--port", help="Serial port (optional)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Baud rate")
    args = parser.parse_args()

    mc = MotorController(port=args.port, baud_rate=args.baud)
    mc.connect()
    print("Status:", mc.get_status())
    print("Testing motors (TEST)")
    ok = mc.test_all_motors()
    print("TEST ok:", ok)
    # Example: set and get steps for F face (only run physically if hardware present)
    if mc.is_connected:
        print("Setting F steps to 800")
        mc.set_steps('F', 800)
        print("Read back:", mc.get_steps('F'))
    mc.disconnect()
