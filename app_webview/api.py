import os
import time
import json
import base64
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np

ROOT = Path.cwd()
PICTURES_DIR = ROOT / "templates/pictures"
POSITIONS_DIR = ROOT / "positions"
PICTURES_DIR.mkdir(parents=True, exist_ok=True)
POSITIONS_DIR.mkdir(parents=True, exist_ok=True)

IMG1_PATH = PICTURES_DIR / "image1.png"
IMG2_PATH = PICTURES_DIR / "image2.png"
COLORS_JSON = POSITIONS_DIR / "colors.json"
POLYGON_POSITIONS_PATH_1 = POSITIONS_DIR / "positions1.json"
POLYGON_POSITIONS_PATH_2 = POSITIONS_DIR / "positions2.json"
CAMERA_RESOLUTION = (1920, 1080)
COLOR_INIT_STATE = {}

from detector import ColorDetector  
from cube_status import CubeStatus
from calibrator import CalibrationManager
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
        self.cube_status = CubeStatus()
        self.motor_controller = MotorController()
        self.calibrator = CalibrationManager()

        self.running = True
        self.lock = threading.Lock()
        self.last_frame = None     # numpy BGR (copy)
        self.last_b64 = None       # base64 jpeg (no header)
        self._last_b64_update_ts = 0.0
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)

        self.motor_controller.connect()

        try:
            self.detector = ColorDetector()
        except Exception:
            self.detector = None

        try:
            self.camera.initialize()
        except Exception:
            pass

        self._capture_thread.start()

        self.captured_images_count = 0

        self._detection_result = None
        self._detection_lock = threading.Lock()
        self._detection_thread = None

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

    def _fast_detect(self, image_path: Path, positions: Dict[str, Tuple[int, int]]) -> Tuple[Dict[str,str], Dict[str, Tuple[int,int,int]]]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")

        h_img, w_img = img.shape[:2]
        roi_size = max(14, min(28, int(min(w_img, h_img) * 0.02)))  # adaptive small ROI
        half = roi_size // 2

        center_hsv = {
            'R': (0, 160, 150),
            'O': (12, 170, 200),
            'Y': (28, 200, 200),
            'G': (60, 160, 140),
            'B': (110, 160, 120),
            'W': (0, 20, 230)
        }

        detected = {}
        labs = {}
        for label, pos in positions.items():
            try:
                x, y = int(pos[0]), int(pos[1])
            except Exception:
                detected[label] = 'W'
                labs[label] = (0,0,0)
                continue
            x = max(0, min(w_img-1, x))
            y = max(0, min(h_img-1, y))
            x0, x1 = max(0, x-half), min(w_img, x+half)
            y0, y1 = max(0, y-half), min(h_img, y+half)
            roi = img[y0:y1, x0:x1]
            if roi is None or roi.size == 0:
                detected[label] = 'W'
                labs[label] = (0,0,0)
                continue

            # median color filter: robust and fast
            med_bgr = np.median(roi.reshape(-1,3), axis=0).astype(np.uint8)
            hsv = cv2.cvtColor(np.uint8([[med_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            lab = cv2.cvtColor(np.uint8([[med_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
            H, S, V = int(hsv[0]), int(hsv[1]), int(hsv[2])
            labs[label] = (int(lab[0]), int(lab[1]), int(lab[2]))

            # match to nearest center with simple weighted distance (circular hue)
            best, bestd = 'W', 999.0
            for k, (ch, cs, cvv) in center_hsv.items():
                dh = min(abs(H - ch), 180 - abs(H - ch)) / 90.0  # normalized
                ds = abs(S - cs) / 255.0
                dv = abs(V - cvv) / 255.0
                d = dh * 0.6 + ds * 0.25 + dv * 0.15
                if d < bestd:
                    bestd = d
                    best = k
            detected[label] = best
        return detected, labs

    def color_detector(self) -> Dict[str, Any]:
        try:
            pos1 = self.load_points(0)
            pos2 = self.load_points(1)
            if not pos1 or not pos2:
                return {"ok": False, "error": "Positions files missing or empty (positions1.json/positions2.json)"}

            det1, labs1 = self._fast_detect(IMG1_PATH, pos1)
            det2, labs2 = self._fast_detect(IMG2_PATH, pos2)

            combined = {}
            combined.update(det1)
            combined.update(det2)

            return {"ok": True, "colors": combined}
        except Exception as e:
            try:
                if self.detector:
                    det1, labs1 = self.detector.detect_single_image(IMG1_PATH, pos1)
                    det2, labs2 = self.detector.detect_single_image(IMG2_PATH, pos2)
                    combined = {}
                    combined.update(det1)
                    combined.update(det2)
                    return {"ok": True, "colors": combined}
            except Exception:
                pass
            print("[color_detector] Exception:", e)
            return {"ok": False, "error": str(e)}

    def scramble(self, moves: int = 20) -> List[str]:
        try:
            return self.motor_controller.scramble(moves)
        except Exception:
            return []

    def send_sequence(self, seq: str) -> bool:
        try:
            if not seq or str(seq).strip() == "":
                return False

            moves = seq.split(" ")

            for mv in moves:
                ok = self.motor_controller.send_sequence(mv)
                if not ok:
                    return False
                try:
                    self.cube_status.change_status([mv])
                except Exception:
                    pass
            return True
        except Exception as e:
            print("[API.send_sequence] Error:", e)
            return False


    def solve(self) -> List[str]:
        try:
            ok, msg = self.cube_status.solve()
            if not ok:
                return []
            return self.cube_status.cube_state.solution
        except Exception:
            return []

    def switch_camera(self) -> bool:
        try:
            return bool(self.camera.switch_camera())
        except Exception:
            return False

    def validate_cube_state(self, face_to_color: dict = None) -> Dict[str, Any]:
        if face_to_color == None:
            ok,mens = self.cube_status.solve()
            if ok:
                return {"ok": True, "solution": self.cube_status.cube_state.solution}
            return {"ok": False, "error": mens}
        try:
            try:
                color_str, facelets, sol = self.cube_status.build_facelets_and_solve(face_to_color)
                self.cube_status.cube_state.color_status = color_str
                self.cube_status.cube_state.face_status = facelets
                self.cube_status.cube_state.solution = sol
            except Exception:
                pass
            valid, issue = self.cube_status.validate_state()
            if valid:
                return {"ok": True, "solution": self.cube_status.cube_state.solution}
            else:
                return {"ok": False, "error": issue}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def reset_cube_state(self):
        try:
            self.cube_status.cube_state.reset_state()
        except Exception:
            pass

    def shutdown(self):
        self.running = False
        try:
            self.camera.release()
        except Exception:
            pass
