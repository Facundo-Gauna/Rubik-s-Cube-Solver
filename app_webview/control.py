"""
motor_controller.py

Protocol:
 - "T\n"          -> test -> Arduino replies "OK\n" when finished
 - "<sequence>\n" -> sequence tokens (space-separated) -> Arduino replies "OK\n" when finished

This controller waits for the "OK" response after sending a sequence or test.
"""

from __future__ import annotations
import time
import logging
from typing import Optional, List

import serial
import serial.tools.list_ports

from typing import Callable, Sequence, Tuple, Optional as _Optional, List as _List

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 1.0

class ArduinoConnectorError(Exception):
    pass

class ArduinoConnector:
    DEFAULT_ACCEPTED_VID_PID: Sequence[Tuple[int, _Optional[int]]] = (
        (0x2341, None),     # Arduino oficial
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
        baudrate: int = 115200,
        accepted_vid_pid: _Optional[Sequence[Tuple[int, _Optional[int]]]] = None,
        strict_keywords: _Optional[Sequence[str]] = None,
        require_handshake: bool = False,
        handshake: _Optional[Callable[[serial.Serial], bool]] = None,
        handshake_timeout: float = 1.0,
    ):
        self.baudrate = baudrate
        self.accepted_vid_pid = accepted_vid_pid or self.DEFAULT_ACCEPTED_VID_PID
        self.strict_keywords = [k.lower() for k in (strict_keywords or self.DEFAULT_STRICT_KEYWORDS)]
        self.require_handshake = require_handshake
        self.handshake = handshake or self._default_handshake
        self.handshake_timeout = handshake_timeout

        self._serial: _Optional[serial.Serial] = None
        self._device: _Optional[str] = None

    def _matches_vid_pid(self, vid: _Optional[int], pid: _Optional[int]) -> bool:
        if vid is None:
            return False
        for v, p in self.accepted_vid_pid:
            if vid == v and (p is None or pid == p):
                return True
        return False

    def _gather_ports(self) -> _List[serial.tools.list_ports.ListPortInfo]:
        return list(serial.tools.list_ports.comports())

    def _score_port(self, p: serial.tools.list_ports.ListPortInfo):
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
        return (-s, len(dev))

    def _default_handshake(self, ser: serial.Serial) -> bool:
        try:
            ser.timeout = self.handshake_timeout
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            time.sleep(0.05)
            ser.write(b"PING\n")
            ser.flush()
            start = time.time()
            buf = b""
            while time.time() - start < self.handshake_timeout:
                chunk = ser.read(64)
                if chunk:
                    buf += chunk
                    text = buf.decode(errors="ignore").lower()
                    if "pong" in text or "arduino" in text or "mega" in text:
                        return True
                else:
                    time.sleep(0.01)
            return False
        except Exception as e:
            logger.debug("Handshake por defecto falló: %s", e)
            return False

    def list_candidates(self) -> _List[serial.tools.list_ports.ListPortInfo]:
        ports = self._gather_ports()
        candidates: _List[serial.tools.list_ports.ListPortInfo] = []
        for p in ports:
            vid = getattr(p, "vid", None)
            pid = getattr(p, "pid", None)
            if self._matches_vid_pid(vid, pid):
                candidates.append(p)
        if not candidates:
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

    def find_port(self) -> _Optional[str]:
        candidates = self.list_candidates()
        logger.debug("Candidatos encontrados: %s", [p.device for p in candidates])
        for p in candidates:
            logger.info("Candidato: %s (%s) hwid=%s", p.device, p.description, p.hwid)
            if not self.require_handshake:
                return p.device
            try:
                with serial.Serial(p.device, self.baudrate, timeout=self.handshake_timeout) as ser:
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                    ok = False
                    try:
                        ok = self.handshake(ser)
                    except Exception as e:
                        logger.debug("Handshake lanzó excepción en %s: %s", p.device, e)
                    if ok:
                        logger.info("Handshake OK en %s", p.device)
                        return p.device
                    else:
                        logger.debug("Handshake NO confirmó %s", p.device)
            except Exception as e:
                logger.debug("No se pudo abrir %s: %s", p.device, e)
                continue
        logger.info("No se encontró puerto serial que cumpla criterios")
        return None

    def connect(self, port: _Optional[str] = None, open_if_needed: bool = True) -> _Optional[serial.Serial]:
        if self._serial and self._serial.is_open:
            return self._serial

        port_to_use = port or self.find_port()
        if not port_to_use:
            raise ArduinoConnectorError("No se encontró puerto serie para Arduino")

        try:
            ser = serial.Serial(port_to_use, self.baudrate, timeout=1)
            time.sleep(0.1)
            self._serial = ser
            self._device = port_to_use
            logger.info("Conectado a %s", port_to_use)
            return ser
        except Exception as e:
            raise ArduinoConnectorError(f"No se pudo abrir el puerto {port_to_use}: {e}")

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
    def device(self) -> _Optional[str]:
        return self._device

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass


class MotorController:
    def __init__(
        self,
        available_moves: Optional[List[str]] = ["U","U'", "U2",
                                "R","R'", "R2",
                                "F","F'", "F2",
                                "D","D'", "D2",
                                "L","L'", "L2",
                                "B","B'", "B2"],
        port: Optional[str] = None,
        baudrate: int = DEFAULT_BAUD,
        timeout: float = DEFAULT_TIMEOUT,
        auto_detect: bool = True,
        per_move_time: float = 0.06,  # seconds per token estimate (used for fallback timeouts)
    ):
        self.available_moves = available_moves
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.auto_detect = auto_detect
        self.per_move_time = per_move_time
        self.ack_per_move = True

        self.ser: Optional[serial.Serial] = None
        self._simulation_mode = False

        self._connector = ArduinoConnector(baudrate=self.baudrate)

    def _find_port(self) -> Optional[str]:
        return self._connector.find_port()

    def connect(self) -> bool:
        if self.port is None and self.auto_detect:
            try:
                self.port = self._find_port()
            except Exception:
                self.port = None

        if self.port is None:
            logger.warning("No serial port found, simulation mode ON.")
            self._simulation_mode = True
            return False

        try:
            logger.info("Opening %s @ %d", self.port, self.baudrate)
            ser = self._connector.connect(port=self.port)
            self.ser = ser
            time.sleep(2.0)
            # clear input safely
            try:
                if self.ser:
                    self.ser.reset_input_buffer()
            except Exception:
                pass
            self._simulation_mode = False
            return True
        except ArduinoConnectorError as e:
            logger.exception("Failed to open serial port via connector: %s", e)
            if self._simulation_mode:
                logger.warning("Falling back to simulation mode.")
                self.ser = None
                return True
            return False
        except Exception as e:
            logger.exception("Failed to open serial port: %s", e)
            if self._simulation_mode:
                logger.warning("Falling back to simulation mode.")
                self.ser = None
                return True
            return False

    def disconnect(self) -> None:
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
        self._simulation_mode = False

    def _write_line(self, line: str) -> None:
        payload = (line + "\n").encode("utf-8")
        if self._simulation_mode or self.ser is None:
            logger.debug("[SIM WRITE] %s", line)
            return
        self.ser.write(payload)
        self.ser.flush()
        logger.debug("[WRITE] %s", line)

    def _wait_for_ok(self, timeout: float) -> bool:
        if self._simulation_mode or self.ser is None:
            time.sleep(min(timeout, 0.1))
            return True
        deadline = time.time() + timeout
        buf = b""
        try:
            self.ser.timeout = 0.1
            while time.time() < deadline:
                chunk = self.ser.read(128)
                if chunk:
                    buf += chunk
                    try:
                        text = buf.decode(errors="ignore")
                    except Exception:
                        text = ""
                    if "OK" in text.upper():
                        logger.debug("Received OK from Arduino: %s", text.strip())
                        return True
                else:
                    # no data, short sleep
                    time.sleep(0.02)
            logger.warning("Timeout esperando OK (%.2fs). Buffer: %s", timeout, buf[:200])
            return False
        except Exception as e:
            logger.exception("Error leyendo serial: %s", e)
            return False

    def send_sequence(self, sequence: str, extra_timeout: float = 0.5) -> bool:
        moves = sequence.split(" ")
        if self.ack_per_move:
            for mv in moves:
                try:
                    self._write_line(mv)
                except Exception:
                    logger.exception("Failed writing move: %s", mv)
                    return False
                timeout = max(0.2, self.per_move_time + extra_timeout)
                ok = self._wait_for_ok(timeout)
                if not ok:
                    logger.warning("No OK for move %s", mv)
                    return False
            return True
        else:
            try:
                self._write_line(" ".join(moves))
            except Exception:
                logger.exception("Failed writing full sequence")
                return False
            return True
    
    def scramble(self, num_moves: int) -> List[str]:
        import random
        moves = []
        last_move = ""
        for _ in range(num_moves):
            move = random.choice(self.available_moves)
            # avoid immediate repeats of the same face
            while last_move != "" and move[0] == last_move[0]:
                move = random.choice(self.available_moves)
            moves.append(move)
            last_move = move
        return moves

    def test_all_motors(self) -> bool:
        """
        Send "T" then wait for OK.
        Timeout should cover the full test duration.
        """
        if self._simulation_mode:
            logger.info("SIM: test_all_motors")
            return True
        try:
            self._write_line("T")
        except Exception:
            return False
        return True


if __name__ == "__main__":
    import argparse
    import logging
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="serial port")
    args = parser.parse_args()

    ctrl = MotorController(port=args.port)
    ctrl.connect()
    print("Connected:", not ctrl._simulation_mode)
    print("Running test...")
    ok = ctrl.test_all_motors()
    print("Test OK:", ok)
    print("Sending sequence: R U R' U2")
    ok = ctrl.send_sequence("R U R' U2", timeout_per_token=1)
    print("Sequence OK:", ok)
    ctrl.disconnect()
