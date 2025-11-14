import time
import json
import base64
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np
import webview

from config import CALIBRATIONS_PATH, CAMERA_RESOLUTION, IMG1_PATH, IMG2_PATH, OK_TIMEOUT, PICTURES_DIR, POLYGON_POSITIONS_PATH_1, POLYGON_POSITIONS_PATH_2, POSITIONS_DIR
from detector import ColorDetector  
from cube_solver import CubeSolver
from control import MotorController

class CameraController:
    def __init__(self):
        self.cap = None
        self.current_camera_index = 0
        self.is_initialized = False

    def initialize(self) -> bool:
        for i in range(6):
            if i == self.current_camera_index:
                continue
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.current_camera_index = i
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                self.cap = cap
                self._configure_camera()
                self.is_initialized = True
                return True
        # fallback index 0
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self.cap = cap
                self.current_camera_index = 0
                self._configure_camera()
                self.is_initialized = True
                return True
        except Exception:
            pass
        return False

    def _configure_camera(self):
        if self.cap:
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception:
                pass

    def capture_frame(self):
        if not self.is_initialized or not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def switch_camera(self) -> bool:
        try:
            if self.cap:
                self.cap.release()
            self.is_initialized = False
            return self.initialize()
        except Exception:
            return False

    def release(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        self.is_initialized = False


class API:

    def __init__(self):
        self.camera = CameraController()
        self.cube_solver = CubeSolver()
        self.motor_controller = MotorController()
        self.detector = ColorDetector(calib_path=CALIBRATIONS_PATH)

        self.running = True
        self.lock = threading.Lock()
        self.last_frame = None     # numpy BGR (copy)
        self.last_b64 = None       # base64 jpeg (no header)
        self._last_b64_update_ts = 0.0
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)

        self.motor_controller.connect()
        PICTURES_DIR.mkdir(parents=True, exist_ok=True)
        POSITIONS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            self.camera.initialize()
        except Exception:
            pass

        self._capture_thread.start()

        self.captured_images_count = 0

        self._detection_result = None
        self._detection_lock = threading.Lock()
        self._detection_thread = None

    # ----- Utils -----
    def _capture_loop(self):
        while self.running:
            try:
                frame = self.camera.capture_frame()
                if frame is not None:
                    with self.lock:
                        self.last_frame = frame.copy()
                    now = time.time()
                    if now - self._last_b64_update_ts > 0.18:
                        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                        if ok:
                            with self.lock:
                                self.last_b64 = base64.b64encode(buf).decode("utf-8")
                                self._last_b64_update_ts = now
                time.sleep(0.01)
            except Exception:
                time.sleep(0.05)


    def get_frame(self) -> str:
        with self.lock:
            if not self.last_b64:
                return ""
            return f"data:image/jpeg;base64,{self.last_b64}"


    def save_frame(self, idx: Optional[int] = None, b64: Optional[str] = None) -> Dict[str, Any]:
        try:
            if idx is None:
                self.captured_images_count = (self.captured_images_count % 2) + 1
                use_idx = 1 if self.captured_images_count == 1 else 2
            else:
                try:
                    use_idx = 1 if int(idx) == 0 else 2
                except Exception:
                    use_idx = 1
            target = IMG1_PATH if use_idx == 1 else IMG2_PATH
            target.parent.mkdir(parents=True, exist_ok=True)

            ok = False
            if b64:
                if "," in b64:
                    _, payload = b64.split(",", 1)
                else:
                    payload = b64
                im_bytes = base64.b64decode(payload)
                nparr = np.frombuffer(im_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    ok = cv2.imwrite(str(target), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                with self.lock:
                    if self.last_frame is None:
                        return {"ok": False, "error": "No frame available"}
                    ok = cv2.imwrite(str(target), self.last_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            if not ok:
                return {"ok": False, "error": "cv2.imwrite failed"}

            with self.lock:
                data_url = f"data:image/jpeg;base64,{self.last_b64}" if self.last_b64 else ""
            return {"ok": True, "path": str(target), "data_url": data_url}
        except Exception as e:
            print("[API.save_frame] Error:", e)
            return {"ok": False, "error": str(e)}


    def save_json(self, content: str, filename: str) -> bool:
        try:
            POSITIONS_DIR.mkdir(parents=True, exist_ok=True)
            path = POSITIONS_DIR / filename
            data = json.loads(content)
            # debug log
            print(f"[API.save_json] saving {filename} keys_count={len(data.keys())} sample_keys={list(data.keys())[:12]}")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[API.save_json] Error: {e}")
            return False


    def load_points(self, idx: int) -> Dict[str, Any]:
        path = POLYGON_POSITIONS_PATH_1 if idx == 0 else POLYGON_POSITIONS_PATH_2
        try:
            raw = json.load(open(path, "r", encoding="utf-8"))
            return raw.get("positions", raw)
        except Exception:
            return {}


    def color_detector(self) -> Dict[str, Any]:
        try:
            pos1 = self.load_points(0)
            pos2 = self.load_points(1)
            if not pos1 or not pos2:
                return {"ok": False, "error": "Positions files missing or empty (positions1.json/positions2.json)"}

            self.detector.calibrate_from_positions(IMG1_PATH,pos1,IMG2_PATH,pos2)

            det1,labs1= self.detector.detect_single_image(IMG1_PATH, pos1, use_gray_world=True)
            det2,labs2= self.detector.detect_single_image(IMG2_PATH, pos2, use_gray_world=True)

            combined = {}
            combined.update(det1)
            combined.update(det2)

            return {"ok": True, "colors": combined}
        except Exception as e:
            print("Error detecting : "+ e)
            return {"ok":False,"colors":{}}


    def send_sequence(self, move: str) -> bool:
        try:
            self.motor_controller._write_line(move)
            ok = self.motor_controller._wait_for_ok(OK_TIMEOUT)
            if not ok:
                print("No OK for move %s", move)
                return False
            with self.lock:
                try:
                    cs = self.cube_solver.cube_state
                    if getattr(cs, "cubie_cube", None) is None:
                        try:
                            cs._sync_from_face_status()
                        except Exception:
                            pass
                except Exception:
                    pass

                try:
                    self.cube_solver.cube_state.move(move)
                except Exception as e:
                    print("[API.send_sequence] Error applying move in backend:", e)
                    return False
            return True
        except Exception as e:
            print("[API.send_sequence] Exception:", e)
            return False


    # ---- Huge botons -------

    def scramble(self, num_moves: int = 10) -> str:
        try:
            moves = self.motor_controller.scramble(num_moves)
            move = " ".join(moves)
            print("Scramble:"+move)
            return move
        except Exception:
            return []


    def solve(self, face_to_color: dict = None) -> Dict[str, Any]:
        try:
            valid, issue = self.cube_solver.solve(face_to_color)

            if valid:
                return {
                    "ok": True,
                    "solution": self.cube_solver.cube_state.solution,
                    "color_str": self.cube_solver.cube_state.color_status,
                    "facelets":self.cube_solver.cube_state.face_status,
                }
            else:
                self.cube_solver.cube_state.solution = ""
                return {
                    "ok": False,
                    "error": str(issue),
                    "color_str": self.cube_solver.cube_state.color_status,
                    "facelets": self.cube_solver.cube_state.face_status,
                }

        except Exception as e:
            return {"ok": False, "error": str(e), "debug": {"exception": str(e)}}


    # ---- Small botons -------
    def connect_arduino(self) -> bool:
        return self.motor_controller.connect()
            

    def switch_camera(self) -> bool:
        try:
            return bool(self.camera.switch_camera())
        except Exception:
            return False


    def reset_cube_state(self):
        try:
            self.cube_solver.cube_state.reset_state()
        except Exception:
            pass


    def shutdown(self):
        self.running = False
        try:
            self.camera.release()
            webview.windows[0].destroy()
        except Exception:
            pass
