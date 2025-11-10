# web_api.py
import os
import time
import json
import base64
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np

# IMPORTS desde tu proyecto (usa los módulos que ya tienes en run.py)
from run import CameraController, CubeStatus, MotorController, CalibrationManager
from detector import ColorDetector, SynchronizedColorCorrectionUI  # y PolygonDetector si quieres

# Rutas usadas en run.py (asegúrate los nombres coincidan con tus constantes)
from config import IMG1_PATH, IMG2_PATH, POLYGON_POSITIONS_PATH_1, POLYGON_POSITIONS_PATH_2, CAMERA_RESOLUTION, COLOR_INIT_STATE

class WebAPI:
    def __init__(self, camera_index: int = 0):
        # componentes (reutilizan lógicas ya existentes)
        self.camera = CameraController()
        self.cube_status = CubeStatus()
        self.motor_controller = MotorController()
        self.calibrator = CalibrationManager()

        # captura en background
        self.running = True
        self.lock = threading.Lock()
        self.last_frame = None     # numpy BGR
        self.last_b64 = None       # base64 string (sin header)
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        # iniciar cámara (si hace falta)
        try:
            self.camera.initialize()
        except Exception:
            pass
        self._capture_thread.start()

        # contador de imágenes (mismo comportamiento que UI: 0/1 => 1/2)
        self.captured_images_count = 0

        # resultados de detección (para polling)
        self._detection_result = None
        self._detection_lock = threading.Lock()
        self._detection_thread = None

        # sccui temporario para corrección vía web
        self.sccui: Optional[SynchronizedColorCorrectionUI] = None

        # carpeta para saves (si necesitas)
        self.pictures_dir = Path(os.getcwd()) / "pictures"
        self.pictures_dir.mkdir(exist_ok=True)

    # ---------------- capture thread ----------------
    def _capture_loop(self):
        while self.running:
            try:
                frame = self.camera.capture_frame()
                if frame is not None:
                    with self.lock:
                        self.last_frame = frame.copy()
                        ok, buf = cv2.imencode(".png", frame)
                        if ok:
                            self.last_b64 = base64.b64encode(buf).decode("utf-8")
                time.sleep(0.03)
            except Exception:
                time.sleep(0.1)

    # ---------------- frame endpoints ----------------
    def get_frame(self) -> str:
        """
        Devuelve `data:image/png;base64,...` con el último frame disponible.
        JS debe llamar periódicamente para preview.
        """
        with self.lock:
            if not self.last_b64:
                return ""
            return f"data:image/png;base64,{self.last_b64}"

    def save_frame(self, idx: int, b64: Optional[str] = None) -> bool:
        """
        Guarda un frame: si `b64` es provisto (data URL o base64) lo decodifica; si no, usa last_frame.
        idx -> 0 o 1 (nombres detector0.png/detector1.png u otros según tu convención).
        """
        try:
            if b64:
                # acepta "data:image/png;base64,..." o sólo base64
                if "," in b64:
                    header, payload = b64.split(",", 1)
                else:
                    payload = b64
                data = base64.b64decode(payload)
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                with self.lock:
                    img = None if self.last_frame is None else self.last_frame.copy()

            if img is None:
                return False
            filename = self.pictures_dir / f"detector{idx}.png"
            ok = cv2.imwrite(str(filename), img)
            return bool(ok)
        except Exception as e:
            print("[WebAPI.save_frame] Error:", e)
            return False

    def capture_image(self) -> Dict[str, Any]:
        """
        Equivalente a presionar Space en la UI original:
        - guarda IMG1_PATH o IMG2_PATH según contador interno
        - retorna {ok:bool, idx:int, path:str}
        """
        with self.lock:
            frame = None if self.last_frame is None else self.last_frame.copy()
        if frame is None:
            return {"ok": False, "error": "No frame available"}

        self.captured_images_count += 1
        try:
            if self.captured_images_count == 1:
                cv2.imwrite(str(IMG1_PATH), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                path = str(IMG1_PATH)
                res = {"ok": True, "idx": 1, "path": path}
            elif self.captured_images_count == 2:
                cv2.imwrite(str(IMG2_PATH), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                path = str(IMG2_PATH)
                res = {"ok": True, "idx": 2, "path": path}
                self.captured_images_count = 0
            else:
                res = {"ok": False, "error": "invalid count"}
        except Exception as e:
            res = {"ok": False, "error": str(e)}
        return res

    def get_thumbnails(self) -> Dict[str, Optional[str]]:
        """
        Devuelve rutas o data URLs de IMG1/IMG2 (para mostrar thumbs).
        Devuelve strings: si no existe, None.
        """
        out = {}
        for k, p in (("img1", IMG1_PATH), ("img2", IMG2_PATH)):
            try:
                if Path(p).exists():
                    with open(p, "rb") as f:
                        b = base64.b64encode(f.read()).decode("utf-8")
                        out[k] = f"data:image/png;base64,{b}"
                else:
                    out[k] = None
            except Exception:
                out[k] = None
        return out

    # ---------------- polygon positions ----------------
    def load_points(self, idx: int) -> Dict[str, Any]:
        """
        Devuelve el dict 'positions' del archivo correspondiente (como tu UI lo esperaba).
        idx: 0 -> POLYGON_POSITIONS_PATH_1, 1 -> POLYGON_POSITIONS_PATH_2
        """
        path = POLYGON_POSITIONS_PATH_1 if idx == 0 else POLYGON_POSITIONS_PATH_2
        try:
            raw = json.load(open(path, "r", encoding="utf-8"))
            return raw.get("positions", raw)  # compatibilidad
        except Exception:
            # si falla, devuelve dict vacío
            return {}

    def save_point(self, idx: int, name: str, x: float, y: float, color: Optional[str] = None) -> bool:
        """
        Actualiza un solo punto en el json (mantiene otras posiciones).
        """
        path = POLYGON_POSITIONS_PATH_1 if idx == 0 else POLYGON_POSITIONS_PATH_2
        try:
            try:
                data = json.load(open(path, "r", encoding="utf-8"))
            except Exception:
                data = {"positions": {}}
            if "positions" not in data:
                data["positions"] = {}
            data["positions"][str(name)] = [x, y, color] if color is not None else [x, y]
            json.dump(data, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print("[WebAPI.save_point] Error:", e)
            return False

    # ---------------- detection / color correction ----------------
    def start_color_detection_and_prepare_correction(self) -> Dict[str, Any]:
        """
        Ejecuta la detección de colores sobre IMG1_PATH y IMG2_PATH y construye un objeto
        SynchronizedColorCorrectionUI que se puede usar para correcciones posteriores desde JS.
        Retorna un payload con detections/corrected/corrected_labs y image_paths.
        """
        # verificar imágenes
        if not IMG1_PATH.exists() or not IMG2_PATH.exists():
            return {"ok": False, "error": "Images missing"}

        # leer posiciones desde los files
        pos1 = self.load_points(0)
        pos2 = self.load_points(1)

        try:
            detector = ColorDetector()
            det1, labs1 = detector.detect_single_image(IMG1_PATH, pos1)
            det2, labs2 = detector.detect_single_image(IMG2_PATH, pos2)
        except Exception as e:
            return {"ok": False, "error": f"Color detection failed: {e}"}

        # construir sccui (para editar en web)
        try:
            sccui = SynchronizedColorCorrectionUI(pos1, pos2, det1, labs1, det2, labs2)
            sccui.detections = [det1 if isinstance(det1, dict) else {}, det2 if isinstance(det2, dict) else {}]
            sccui.corrected = [det1 if isinstance(det1, dict) else {}, det2 if isinstance(det2, dict) else {}]
            sccui.corrected_labs = [labs1 if isinstance(labs1, dict) else {}, labs2 if isinstance(labs2, dict) else {}]
            sccui.image_paths = [str(IMG1_PATH), str(IMG2_PATH)]
            self.sccui = sccui
        except Exception as e:
            return {"ok": False, "error": f"SCCUI build failed: {e}"}

        return {"ok": True,
                "detections": sccui.detections,
                "corrected": sccui.corrected,
                "corrected_labs": sccui.corrected_labs,
                "image_paths": sccui.image_paths}

    def apply_color_correction(self, window_index: int, key: str, color_code: str) -> Dict[str, Any]:
        """
        Aplica una corrección sobre sccui.corrected y devuelve el estado actualizado.
        key: por ejemplo "U1", color_code: 'R'|'O'|'B'|'G'|'W'|'Y'
        """
        if not self.sccui:
            return {"ok": False, "error": "No sccui available; run start_color_detection first."}
        try:
            # lógica similar a _apply_color_to_key del UI original
            if not hasattr(self.sccui, 'corrected') or self.sccui.corrected is None:
                self.sccui.corrected = [{}, {}]
            while len(self.sccui.corrected) <= window_index:
                self.sccui.corrected.append({})
            self.sccui.corrected[window_index][str(key)] = str(color_code)

            # opción: mantener labs si existe mapping
            try:
                if hasattr(self.sccui, 'corrected_labs') and self.sccui.corrected_labs and len(self.sccui.corrected_labs) > window_index:
                    lab_map = self.sccui.corrected_labs[window_index]
                    lab = lab_map.get(str(key))
                    if lab is not None:
                        if not hasattr(self.sccui, 'color_centers'):
                            self.sccui.color_centers = {}
                        self.sccui.color_centers[color_code] = lab
            except Exception:
                pass
            return {"ok": True, "corrected": self.sccui.corrected}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---------------- detection of cube state (long op) ----------------
    def detect_cube_state(self) -> Dict[str, Any]:
        """
        Lanza detect_status() en un thread y guarda el resultado para polling.
        Retorna inmediatamente {"ok": True, "started": True}.
        Después JS/poll podrá llamar get_detection_result().
        """
        def _worker():
            try:
                res = self.cube_status.detect_status()
                with self._detection_lock:
                    self._detection_result = res  # objeto DetectionResult (puedes stringify)
            except Exception as e:
                with self._detection_lock:
                    self._detection_result = {"error": str(e)}

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        self._detection_thread = t
        return {"ok": True, "started": True}

    def get_detection_result(self):
        with self._detection_lock:
            if self._detection_result is None:
                return {"ok": False, "pending": True}
            else:
                r = self._detection_result
                self._detection_result = None
                # convertir el resultado a dict serializable si hace falta
                try:
                    # si es un objeto con atributos, devolver solo las propiedades que te interesan:
                    return {"ok": True,
                            "color_str": getattr(r, "color_str", None),
                            "face_str": getattr(r, "face_str", None),
                            "solution_str": getattr(r, "solution_str", None),
                            "has_errors": getattr(r, "has_errors", None)}
                except Exception:
                    return {"ok": True, "raw": str(r)}

    # ---------------- motor / scramble / solve ----------------
    def scramble(self, moves: int = 20) -> Dict[str, Any]:
        try:
            seq = self.motor_controller.scramble(moves)
            seq_str = " ".join(seq)
            ok = self.motor_controller.send_sequence(seq_str)
            if ok:
                # actualizar estado del cubo según motor controller
                self.cube_status.change_status(seq)
                return {"ok": True, "sequence": seq}
            else:
                return {"ok": False, "error": "hardware failed"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def send_sequence(self, seq: str) -> Dict[str, Any]:
        try:
            ok = self.motor_controller.send_sequence(seq)
            if ok:
                self.cube_status.change_status(seq.split())
                return {"ok": True}
            return {"ok": False, "error": "execution failed"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def solve(self) -> Dict[str, Any]:
        try:
            ok, msg = self.cube_status.solve()
            if not ok:
                return {"ok": False, "error": msg}
            sol = self.cube_status.cube_state.solution
            # intentar ejecutar
            executed = self.motor_controller.send_sequence(sol)
            if executed:
                self.cube_status.change_status(sol.split())
            return {"ok": True, "solution": sol, "executed": bool(executed)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---------------- misc ----------------
    def switch_camera(self) -> Dict[str, Any]:
        try:
            ok = self.camera.switch_camera()
            return {"ok": bool(ok)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def validate_cube_state(self) -> Dict[str, Any]:
        try:
            valid, issue = self.cube_status.validate_state()
            return {"ok": True, "valid": bool(valid), "issue": str(issue) if issue else None}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def reset_cube_state(self) -> Dict[str, Any]:
        try:
            self.cube_status.cube_state.reset_state()
            return {"ok": True, "state": self.cube_status.cube_state.color_status}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def shutdown(self):
        self.running = False
        try:
            self.camera.release()
        except Exception:
            pass
