"""api.py — pywebview JS API + integrated MJPEG preview server
================================================================

This file implements the Python <-> JavaScript API used by the pywebview GUI and
hosts a small Flask-based MJPEG server for live preview. Documentation is
embedded inline as comments (English) to explain design, threading model and
common failure modes. Keep this module as a single point of integration for
camera capture, streaming, color detection and motor control.

High-level responsibilities
- Capture frames from a camera (OpenCV VideoCapture) in a background thread.
- Encode frames to JPEG (TurboJPEG when available, fallback to cv2.imencode)
  in a dedicated encoder thread to keep capture responsive.
- Serve a lightweight MJPEG endpoint using Flask running on a background
  Werkzeug server to allow remote or local preview (the frontend consumes it).
- Expose a pywebview-friendly API (`API` class) with methods used by the
  frontend: snapshot/save, color detection, calibration, motor commands, etc.

Threading model / shared state
- Capture thread: reads frames from camera, pushes copies into a single-slot
  encoder queue and updates `last_frame` for synchronous snapshot access.
- Encoder thread: consumes the latest queued frame, resizes to streaming
  resolution and encodes to JPEG bytes. Notifies waiting MJPEG generators via
  a Condition. Uses TurboJPEG if available for speed.
- Flask server: runs in its own background thread (Werkzeug `make_server`
  wrapped in `_Server`), exposing `/stream`, `/snapshot`, and control routes.
- API methods are intended to be called from the UI thread (pywebview) and are
  responsible for light-weight coordination with the background threads.

Important safety notes
- Camera and Flask lifetime are managed by `API.shutdown()`; ensure this is
  called on application exit to avoid dangling threads and opened devices.
- MJPEG client limits are enforced (`_max_stream_clients`) to avoid using too
  much CPU/network when multiple viewers connect.
------------------------------------------------------------------

Copyright (c) 2025 Facundo Gauna & Ulises Carnevale. MIT License.
"""

import queue
import time
import json
import base64
import threading
from typing import Optional, Dict, Any

import cv2
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from flask_cors import CORS
import numpy as np
import webview

# Optional fast JPEG encoder. We attempt to import TurboJPEG but gracefully
# fall back to cv2.imencode when TurboJPEG is not available in the environment.
try:
    from turbojpeg import TurboJPEG
    _TURBOJPEG = TurboJPEG()
    logger.info("[API] TurboJPEG available, will use for encoding.")
except Exception as e:
    _TURBOJPEG = None
    logger.warning("[API] TurboJPEG not available, falling back to cv2.imencode. : %s", e)

from flask import Flask, Response, request, jsonify, make_response
from werkzeug.serving import make_server

from config import (
    CALIBRATIONS_PATH, CAMERA_RESOLUTION, IMG1_PATH, IMG2_PATH, OK_TIMEOUT,
    PICTURES_DIR, POLYGON_POSITIONS_PATH_1, POLYGON_POSITIONS_PATH_2, POSITIONS_DIR,
    MJPEG_HOST, MJPEG_PORT, MJPEG_FPS, MJPEG_JPEG_QUALITY
)
from detector import ColorDetector
from cube_solver import CubeSolver
from control import MotorController


# ---------- CameraController ----------
# Very small wrapper around OpenCV VideoCapture with simple auto-detection,
# basic configuration and safe release semantics. The API expects a single
# CameraController instance and will not open multiple cameras concurrently.
class CameraController:
    def __init__(self):
        self.cap = None
        self.current_camera_index = -1
        self.is_initialized = False

    def initialize(self) -> bool:
        """Try to open a camera index. The loop attempts several common indices
        then falls back to index 0. Returns True on success.
        """
        for i in range(6):
            if i == self.current_camera_index:
                continue
            try:
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
            except Exception:
                pass
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
        """Apply preferred camera settings (resolution/fps). Settings may be
        ignored by some camera drivers. No error is raised on failure.
        """
        if self.cap:
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                logger.info(f"[CameraController] Camera configured to {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0}")
            except Exception:
                pass

    def capture_frame(self):
        """Read a single frame from the capture device. Returns BGR numpy array
        on success or None on failure / no frame.
        """
        if not self.is_initialized or not self.cap.isOpened():
            return None
        try:
            ret, frame = self.cap.read()
            if ret:
                return frame
            return None
        except Exception:
            return None

    def switch_camera(self) -> bool:
        """Release current capture and re-run initialization (useful to cycle
        cameras). Returns True if a camera becomes available.
        """
        try:
            if self.cap:
                self.cap.release()
            self.is_initialized = False
            return self.initialize()
        except Exception:
            return False

    def release(self):
        """Safely close camera resources.
        """
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        self.is_initialized = False


# ---------- Flask server wrapper ----------
class _Server(threading.Thread):
    """Run Werkzeug/Flask in a background daemon thread using `make_server`.

    This wrapper keeps the Flask server lifecycle separate from pywebview so
    we can stop it gracefully on shutdown.
    """

    def __init__(self, app, host, port):
        super().__init__(daemon=True)
        self._app = app
        self._host = host
        self._port = port
        self._server = None

    def run(self):
        try:
            self._server = make_server(self._host, self._port, self._app, threaded=True)
            self._server.serve_forever()
        except Exception as e:
            logger.exception("[MJPEG] Flask server stopped with error: %s", e)

    def shutdown(self):
        try:
            if self._server:
                self._server.shutdown()
        except Exception:
            pass


# ---------- API class (pywebview JS API + Flask endpoints) ----------
class API:
    """Main JS API object exposed to the frontend via pywebview

    Responsibilities:
    - manage camera capture, encoder and MJPEG server
    - expose methods used by frontend: snapshots, save/load positions, color detection
    - forward motor control commands to the MotorController
    """

    def __init__(self):
        # core components: camera, solver, motor controller and detector
        self.camera = CameraController()
        self.cube_solver = CubeSolver()
        self.motor_controller = MotorController()
        self.move_duration_ms = 400  # default (frontend default)

        self.detector = ColorDetector(calib_path=CALIBRATIONS_PATH)

        # threading / shared state
        self.running = True
        self.lock = threading.Lock()
        self.last_frame = None     # latest captured BGR image (copy)
        self.last_saved_frames = [None,None]  # frames saved via UI (img1/img2)
        self.last_b64 = None       # cached base64 JPEG if used by UI
        self._last_b64_update_ts = 0.0

        # capture loop thread (created but not started yet)
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)

        # try to connect to motors (non-fatal — may run in simulation mode)
        try:
            self.motor_controller.connect()
        except Exception:
            pass

        # ensure directories exist for saving snapshots / positions
        PICTURES_DIR.mkdir(parents=True, exist_ok=True)
        POSITIONS_DIR.mkdir(parents=True, exist_ok=True)

        # initialize camera (non-fatal)
        try:
            self.camera.initialize()
            self.camera._configure_camera()
        except Exception:
            pass
        
        # streaming/encoding state
        self._stream_lock = threading.Lock()
        self._stream_cond = threading.Condition(self._stream_lock)
        self._last_jpeg = None            # latest encoded JPEG bytes
        self._last_jpeg_ts = 0.0
        self._encoder_queue = queue.Queue(maxsize=1)  # single-slot: keep latest only
        # stream defaults; frontend can change via /set_stream_params
        self._stream_resolution = (640, 360)
        self._stream_fps = MJPEG_FPS
        self._stream_interval = max(0.01, 1.0 / max(1, self._stream_fps))
        self._stream_quality = MJPEG_JPEG_QUALITY
        
        # streaming client counters to avoid over-subscription
        self._stream_clients = 0
        self._stream_clients_lock = threading.Lock()
        self._max_stream_clients = 1


        # encoder thread (encode frames to JPEG asynchronously)
        self._encoder_thread = threading.Thread(target=self._encoder_loop, daemon=True)
        self._encoder_thread.start()

        # start background capture loop (reads frames and pushes to encoder)
        self._capture_thread.start()

        # detection bookkeeping
        self.captured_images_count = 0
        self._detection_result = None
        self._detection_lock = threading.Lock()
        self._detection_thread = None

        # start MJPEG server (Flask) on a background thread
        self._flask_app = self._create_flask_app()
        self._flask_thread = _Server(self._flask_app, MJPEG_HOST, MJPEG_PORT)
        try:
            self._flask_thread.start()
            logger.info(f"[MJPEG] server started at http://{MJPEG_HOST}:{MJPEG_PORT}/stream")
        except Exception as e:
            logger.exception("[MJPEG] failed to start server: %s", e)

    # ----- Capture & encoder loops -----
    
    def _capture_loop(self):
        """Capture frames from the camera and feed the encoder queue.

        Responsibilities:
        - keep a thread-safe copy of the latest captured frame in `self.last_frame`
        - push a (single) frame copy to the encoder queue — drop previous if
          encoder lags to always encode the most recent frame
        - sleep a small amount to avoid busy-looping
        """
        while self.running:
            try:
                frame = self.camera.capture_frame()  # BGR numpy
                if frame is None:
                    time.sleep(0.01)
                    continue
                with self.lock:
                    # keep latest frame reference (copy to break camera buffer sharing)
                    try:
                        self.last_frame = frame.copy()
                    except Exception:
                        self.last_frame = frame

                # push to encoder queue (single-slot queue): non-blocking
                try:
                    if self._encoder_queue.full():
                        try:
                            _ = self._encoder_queue.get_nowait()
                        except Exception:
                            pass
                    # prefer to enqueue a copy so the encoder works with a stable image
                    try:
                        self._encoder_queue.put_nowait(frame.copy())
                    except Exception:
                        # fallback to enqueuing original (rare)
                        self._encoder_queue.put_nowait(frame)
                except Exception:
                    # ignore failures; encoder will pick up newer frames later
                    pass

                # throttle capture loop according to streaming interval
                time.sleep(self._stream_interval)
            except Exception as e:
                logger.exception("[API._capture_loop] unexpected error: %s", e)
                time.sleep(0.02)

    def _encoder_loop(self):
        """Consume frames from encoder queue, resize and encode to JPEG.

        Uses TurboJPEG if installed (faster). Updates `_last_jpeg` and notifies
        the streaming condition variable so MJPEG clients can pick up new frames.
        """
        import cv2
        while self.running:
            try:
                # wait for a new frame for a short timeout
                try:
                    frame = self._encoder_queue.get(timeout=1.0 / max(1, self._stream_fps))
                except queue.Empty:
                    continue

                # resize to target stream resolution to reduce encoding work
                target_w, target_h = self._stream_resolution
                h, w = frame.shape[:2]
                if (w, h) != (target_w, target_h):
                    frame_small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                else:
                    frame_small = frame

                # encode to JPEG (TurboJPEG preference)
                if _TURBOJPEG:
                    # TurboJPEG.encode accepts quality parameter
                    jpeg_bytes = _TURBOJPEG.encode(frame_small, quality=self._stream_quality)
                else:
                    ok, buf = cv2.imencode(".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), int(self._stream_quality)])
                    if not ok:
                        continue
                    jpeg_bytes = buf.tobytes()

                # publish encoded bytes to MJPEG generator(s)
                with self._stream_cond:
                    self._last_jpeg = jpeg_bytes
                    self._last_jpeg_ts = time.time()
                    # wake up any MJPEG generator waiting for a frame
                    self._stream_cond.notify_all()

            except Exception as e:
                logger.exception("[API._encoder_loop] error: %s", e)
                time.sleep(0.05)

    def _mjpeg_generator(self):
        """Alternative generator used internally in some code paths. Yields
        multipart JPEG frames using the newest encoded frame. Throttles by
        `self._stream_interval` and avoids busy-looping.
        """
        last_sent_ts = 0.0
        while self.running:
            try:
                with self._stream_cond:
                    self._stream_cond.wait(timeout=self._stream_interval)
                    jpeg = self._last_jpeg
                    jpeg_ts = self._last_jpeg_ts

                if not jpeg:
                    time.sleep(0.01)
                    continue

                if jpeg_ts <= last_sent_ts:
                    time.sleep(self._stream_interval)
                    continue

                frame_length = len(jpeg)
                payload = (b'--frame\r\n' +
                           b'Content-Type: image/jpeg\r\n' +
                           b'Content-Length: ' + f"{frame_length}".encode() + b'\r\n\r\n' +
                           jpeg + b'\r\n')
                yield payload

                last_sent_ts = jpeg_ts
                time.sleep(self._stream_interval)
            except GeneratorExit:
                break
            except Exception as e:
                logger.exception("[MJPEG generator] error: %s", e)
                time.sleep(0.1)
        
    # ----- Flask app / endpoints -----

    def _create_flask_app(self):
        """Create and return a Flask application configured with CORS and
        several endpoints used by the frontend for preview and control.

        Endpoints include:
        - GET /health           -> basic liveness JSON
        - GET /stream           -> multipart MJPEG stream
        - POST /set_stream_params -> set width/height/fps/quality for encoder
        - GET /snapshot         -> latest frame as JPEG
        - GET /snapshot_png     -> latest frame as PNG (lossless)
        """
        app = Flask(__name__)
        
        # allow CORS for local development. If deploying on a network, replace
        # '*' with explicit origins for better security.
        CORS(app, resources={r"/*": {"origins": "*"}})
    
        @app.after_request
        def _add_cors_headers(response):
            # defensive: ensure CORS headers exist on every response
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            return response
    
        @app.route("/health")
        def _health():
            return jsonify({"ok": True})

        @app.route("/stream")
        def _mjpeg_stream():
            """MJPEG endpoint. Limits concurrent clients and yields a generator
            which streams the latest encoded JPEG frames as multipart responses.
            """
            # guard slot allocation
            with self._stream_clients_lock:
                if self._stream_clients >= self._max_stream_clients:
                    return make_response(("Too many stream clients", 429))
                self._stream_clients += 1

            def client_gen():
                boundary = b'--frame\r\n'
                try:
                    while self.running:
                        try:
                            with self._stream_cond:
                                got = self._stream_cond.wait(timeout=max(0.5, 1.0 / max(1, self._stream_fps)))
                                jpeg = self._last_jpeg
                            if not jpeg:
                                time.sleep(0.01)
                                continue
                            frame_length = len(jpeg)
                            yield (boundary +
                                   b'Content-Type: image/jpeg\r\n' +
                                   b'Content-Length: ' + f"{frame_length}".encode() + b'\r\n\r\n' +
                                   jpeg + b'\r\n')
                        except GeneratorExit:
                            break
                        except Exception as e:
                            print("[MJPEG generator] error:", e)
                            time.sleep(0.1)
                finally:
                    # release client slot when generator exits
                    try:
                        with self._stream_clients_lock:
                            self._stream_clients = max(0, self._stream_clients - 1)
                    except Exception:
                        pass

            return Response(client_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


        @app.route("/set_stream_params", methods=["POST"])
        def _set_stream_params():
            """Endpoint to update encoder/stream parameters from the frontend.

            Accepts JSON with `width`, `height`, `fps`, `quality` and applies
            bounds checks before committing them to the API instance state.
            """
            try:
                data = request.get_json() or {}
                w = int(data.get("width", self._stream_resolution[0]))
                h = int(data.get("height", self._stream_resolution[1]))
                fps = int(data.get("fps", self._stream_fps))
                quality = int(data.get("quality", self._stream_quality))
                # bounds checks
                w = max(160, min(1920, w))
                h = max(120, min(1080, h))
                fps = max(1, min(30, fps))
                quality = max(10, min(95, quality))

                self._stream_resolution = (w, h)
                self._stream_fps = fps
                self._stream_interval = max(0.01, 1.0 / max(1, self._stream_fps))
                self._stream_quality = quality
                return jsonify({"ok": True, "width": w, "height": h, "fps": fps, "quality": quality})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)})


        @app.route("/snapshot")
        def _snapshot():
            # Return the latest frame as JPEG bytes. Useful for quick download.
            with self.lock:
                lf = None if self.last_frame is None else self.last_frame.copy()
            if lf is None:
                return make_response(("No image", 503))
            ok, buf = cv2.imencode(".jpg", lf, [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_JPEG_QUALITY])
            if not ok:
                return make_response(("Encode failed", 500))
            return Response(buf.tobytes(), mimetype="image/jpeg")

        @app.route("/snapshot_png")
        def _snapshot_png():
            # Return a lossless PNG (useful for calibration image saving).
            with self.lock:
                lf = None if self.last_frame is None else self.last_frame.copy()
            if lf is None:
                return make_response(("No image", 503))
            ok, buf = cv2.imencode(".png", lf, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            if not ok:
                return make_response(("Encode failed", 500))
            return Response(buf.tobytes(), mimetype="image/png")

        return app

    # ----- pywebview-facing functions (API surface) -----

    def save_frame(self, idx: Optional[int] = None, b64: Optional[str] = None) -> Dict[str, Any]:
        """Save a snapshot to disk and return a data_url for UI preview.

        Parameters
        - idx: optional index selecting image1/image2 slot; when None cycles
          between slots.
        - b64: optional base64-encoded image payload provided by frontend
          (backwards-compatible path).

        Returns a JSON-like dict: {ok: bool, path: str, data_url: str}
        """
        try:
            # decide which target file to use (IMG1_PATH or IMG2_PATH)
            if idx is None:
                self.captured_images_count = (self.captured_images_count % 2) + 1
                use_idx = 1 if self.captured_images_count == 1 else 2
            else:
                try:
                    use_idx = 1 if int(idx) == 0 else 2
                except Exception:
                    use_idx = 1
            target = IMG1_PATH if use_idx == 1 else IMG2_PATH

            # If frontend provided a b64 payload, decode and save it
            if b64:
                if "," in b64:
                    _, payload = b64.split(",", 1)
                else:
                    payload = b64
                im_bytes = base64.b64decode(payload)
                nparr = np.frombuffer(im_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    return {"ok": False, "error": "decode failed"}
                with self.lock:
                    self.last_saved_frames[use_idx - 1] = img
                ok = cv2.imwrite(str(target), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                data_url = f"data:image/png;base64,{payload}"
                if not ok:
                    return {"ok": False, "error": "cv2.imwrite failed"}
                return {"ok": True, "path": str(target), "data_url": data_url}

            # Otherwise, use latest server-side frame
            with self.lock:
                lf = None if self.last_frame is None else self.last_frame.copy()
                self.last_saved_frames[use_idx - 1] = lf
            if lf is None:
                return {"ok": False, "error": "No frame available"}

            # Write a PNG to preserve quality for calibration images
            ok = cv2.imwrite(str(target), lf, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok:
                return {"ok": False, "error": "cv2.imwrite failed"}

            # Also return an in-memory JPEG data URL for immediate UI preview
            ok_enc, buf = cv2.imencode(".jpg", lf, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if ok_enc:
                data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
            else:
                data_url = ""

            return {"ok": True, "path": str(target), "data_url": data_url}
        except Exception as e:
            logger.exception("[API.save_frame] Error: %s", e)
            return {"ok": False, "error": str(e)}

    def save_json(self, content: str, filename: str) -> bool:
        """Save arbitrary JSON (positions) to POSITIONS_DIR.

        Expects `content` to be a JSON string. Returns True on success.
        """
        try:
            POSITIONS_DIR.mkdir(parents=True, exist_ok=True)
            path = POSITIONS_DIR / filename
            data = json.loads(content)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logger.exception("[API.save_json] Error: %s", e)
            return False

    def load_points(self, idx: int) -> Dict[str, Any]:
        """Load position map JSON for image 1 or 2. Returns the 'positions' object
        or an empty dict on failure.
        """
        path = POLYGON_POSITIONS_PATH_1 if idx == 0 else POLYGON_POSITIONS_PATH_2
        try:
            raw = json.load(open(path, "r", encoding="utf-8"))
            return raw.get("positions", raw)
        except Exception:
            logger.exception("[API.load_points] Error loading from json : %s", e)
            return {}

    def color_detector(self) -> Dict[str, Any]:
        """Run color detection using the saved images or latest frames.

        Returns: {ok: True, colors: detected_map, labs: lab_map, centroids: calibration}
        or an error dict when images/positions are missing.
        """
        try:
            pos1 = self.load_points(0)
            pos2 = self.load_points(1)
            if not pos1 or not pos2:
                return {"ok": False, "error": "Positions files missing or empty (positions1.json/positions2.json)"}
            
            # prefer saved frames (images saved by the UI) otherwise fallback to last_frame
            with self.lock:
                img1 = self.last_saved_frames[0] if len(self.last_saved_frames) > 0 else None
                img2 = self.last_saved_frames[1] if len(self.last_saved_frames) > 1 else None
                if img1 is None and self.last_frame is not None:
                    img1 = self.last_frame.copy()
                if img2 is None and self.last_frame is not None:
                    img2 = self.last_frame.copy()

            if img1 is None or img2 is None:
                return {"ok": False, "error": "Could not load images for detection"}

            det, labs = self.detector.detect_images(img1_path=IMG1_PATH, pos1=pos1, img2_path=IMG2_PATH, pos2=pos2,
                                                   just_detections=False, img1=img1, img2=img2, use_gray_world=True)

            return {"ok": True, "colors": det, "labs": labs, "centroids": self.detector.centroids_lab}
        except Exception as e:
            logger.exception("[API.color_detector] Error detecting: %s", e)
            return {"ok": False, "colors": {}}

    def calibrate_colors(self, face_to_color_validate : Dict[str,str]) -> bool:
        """Wrapper to call detector.calibrate_from_detection with a validated map
        from the frontend. Expects a dict mapping label->color_letter.
        Returns True on successful calibration.
        """
        try:
            if face_to_color_validate == None:
                logger.error("Face to color map is null")
                return False
            centroids = self.detector.calibrate_from_detection(face_to_color_validate)
            if centroids == None:
                logger.error("Error calibrating colors")
                return False
            logger.info("Centroids detected: %s", centroids)
            return True
        except Exception as e:
            logger.exception("[API.calibrate_colors] Error calibrating colors: %s", e)
            return False

    def set_camera_resolution(self, w: int, h: int) -> bool:
        """Try to set camera capture resolution at runtime. Returns True on
        success. Also updates CAMERA_RESOLUTION global for consistency.
        """
        try:
            if self.camera and self.camera.cap:
                try:
                    self.camera.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    self.camera.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    self.camera.cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    CAMERA_RESOLUTION[0] = w
                    CAMERA_RESOLUTION[1] = h
                    logger.info(f"[API] set_camera_resolution to {w}x{h}")
                    return True
                except Exception as e:
                    logger.exception("[API.set_camera_resolution] error applying to cap: %s", e)
                    return False
            return False
        except Exception as e:
            logger.exception("[API.set_camera_resolution] exception: %s", e)
            return False

    def set_move_duration(self, ms: int) -> bool:
        """Frontend-facing setter for animation/motor move duration (ms).
        Stores value locally and forwards to MotorController if available.
        """
        try:
            ms_int = max(10, int(ms))
            self.move_duration_ms = ms_int
            logger.info(f"[API] move_duration set to {ms_int} ms")
            if hasattr(self.motor_controller, "set_move_duration"):
                try:
                    self.motor_controller.set_move_duration(ms_int)
                except Exception:
                    pass
            return True
        except Exception as e:
            logger.exception("[API.set_move_duration] error: %s", e)
            return False

    def send_sequence(self, move: str, physical:bool = True) -> bool:
        """
        Apply a move string to the internal cube state and optionally send
        it to the physical motor controller. Blocking on motor ack when
        `physical=True`.
        """
        try:
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
                    if len(move) > 2:
                        self.cube_solver.cube_state.apply_sequence(move)
                    else:
                        self.cube_solver.cube_state.move(move)
                except Exception as e:
                    logger.exception("[API.send_sequence] Error applying move in backend: %s", e)
                    return False
            
            if physical:
                self.motor_controller._write_line(move)
                ok = self.motor_controller._wait_for_ok(OK_TIMEOUT)
                if not ok:
                    logger.warning("No OK for move %s", move)
                    return False

            return True
        except Exception as e:
            logger.exception("[API.send_sequence] Exception: %s", e)
            return False

    def scramble(self, num_moves: int = 15) -> str:
        """ Return a scramble sequence generated by MotorController (string). """
        try:
            moves = self.motor_controller.scramble(num_moves)
            move = " ".join(moves)
            logger.info("Scramble: %s", move)
            return move
        except Exception:
            logger.exception("[API.scramble] Error scrambling the cube : ",e)
            return ""


    def solve(self, face_to_color: dict = None) -> Dict[str, Any]:
        """ High-level solve using detected/calibrated face->color mapping.
        Returns structured result containing solution and internal face/status
        or an error dict on failure. """
        try:
            valid, issue = self.cube_solver.solve(face_to_color)
            logger.info("Solve result valid: %s issue: %s", valid, issue)
            if valid:
                return {
                    "ok": True,
                    "solution": self.cube_solver.cube_state.solution,
                    "color_str": self.cube_solver.cube_state.color_status,
                    "facelets": self.cube_solver.cube_state.face_status,
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

    def connect_arduino(self) -> bool:
        """Proxy to motor_controller.connect()."""
        return self.motor_controller.connect()

    def test_motors(self) -> bool:
        """Proxy to MotorController.test_all_motors(); returns False on error."""
        try:
            return self.motor_controller.test_all_motors()
        except Exception:
            logger.exception("[API.test_motors] Error testing motors: %s",e)
            return False

    def switch_camera(self) -> bool:
        """Switch to next available camera (wraps CameraController.switch_camera)."""
        try:
            return bool(self.camera.switch_camera())
        except Exception:
            logger.exception("[API.switch_camera] Error switching camera : %s",e)
            return False

    def reset_cube_state(self):
        """Reset internal cube state used by solver."""
        try:
            self.cube_solver.cube_state.reset_state()
        except Exception as e:
            logger.exception("[API.reset_cube_state] Cube cannot be reseted : %s",e)

    def shutdown(self):
        """Gracefully stop background threads, release camera and shut down the
        Flask server. Also attempts to close the pywebview window.
        """
        self.running = False
        try:
            self.camera.release()
        except Exception:
            pass
        try:
            if hasattr(self, "_flask_thread") and self._flask_thread:
                self._flask_thread.shutdown()
        except Exception:
            pass

        # close webview window if possible
        try:
            webview.windows[0].destroy()
        except Exception:
            pass

