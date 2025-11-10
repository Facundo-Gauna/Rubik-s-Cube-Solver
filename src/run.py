import sys
import time
import json
import math
import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, List
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox, QInputDialog, QGridLayout,
    QGroupBox, QDialog, QDialogButtonBox, QProgressBar, QLineEdit,
    QSizePolicy, QSplitter, QTextEdit, QAction, QTabWidget,
    QShortcut, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPen, QKeySequence

from calibrator import CalibrationManager
from config import CAMERA_RESOLUTION, COLOR_INIT_STATE, COPYRIGHT_MARK, DARK_THEME, IMG1_PATH, IMG2_PATH, POLYGON_POSITIONS_PATH_1, POLYGON_POSITIONS_PATH_2, THUMBNAIL_RESOLUTION, DetectionResult
from cube_status import CubeStatus
from control import MotorController

from detector import PolygonDetector, ColorDetector, SynchronizedColorCorrectionUI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GLOBAL_CSS = """
/* Global modern dark look */
QWidget { background: #0b0f13; color: #e6eef3; font-family: Inter, Helvetica, Arial, sans-serif; }
QGroupBox { border: 1px solid #1f2a33; margin-top: 8px; border-radius:8px; padding: 8px; }
QGroupBox:title { subcontrol-origin: margin; subcontrol-position: top center; padding: 2px 8px; color:#dbe7ef; font-weight:700; }
QPushButton { background: #1b2530; border: 1px solid #26313b; padding: 8px 10px; border-radius: 8px; color:#e6eef3; }
QPushButton#primary { background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #2bb673, stop:1 #13824c); color: #fff; font-weight: 700; }
QPushButton#danger { background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #e65f6b, stop:1 #b43a3f); color: #fff; font-weight: 700; }
QLabel.imageFrame { background:#07090b; border: 1px solid #1f2a33; border-radius:8px; }
QTextEdit { background:#071014; color:#cfe9ff; border-radius:6px; }
QTabWidget::pane { border: 1px solid #1b2730; }
QLineEdit { background:#081018; border: 1px solid #1a2a33; border-radius:6px; padding:6px; color:#e6eef3; }
QProgressBar { border-radius:6px; text-align:center; color:#fff; }
QPushButton.colorSwatch { border-radius: 18px; width: 36px; height: 36px; border: 2px solid rgba(0,0,0,0.15); }
"""

def cvimg_to_qpixmap(img: np.ndarray, target_size: QSize) -> QPixmap:
    if img is None:
        return QPixmap()
    try:
        if len(img.shape) == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        return pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    except Exception as e:
        logger.error(f"cvimg_to_qpixmap error: {e}")
        return QPixmap()

def overlay_badge(img: np.ndarray, text: str) -> np.ndarray:
    try:
        h, w = img.shape[:2]
        badge_w = min(260, max(140, int(w * 0.28)))
        badge_h = 30
        pad = 8
        overlay = img.copy()
        x0, y0 = pad, pad
        x1, y1 = x0 + badge_w, y0 + badge_h
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (12,16,18), -1, cv2.LINE_AA)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (36,46,52), 1, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        color = (220,220,220)
        cv2.putText(overlay, text, (x0 + 12, y0 + 20), font, font_scale, color, 1, cv2.LINE_AA)
        return overlay
    except Exception as e:
        logger.debug(f"overlay_badge failed: {e}")
        return img

def normalize_mapping_keys(mapping: dict) -> dict:
    if not isinstance(mapping, dict):
        return {}
    out = {}
    for k, v in mapping.items():
        out[str(k)] = v
    return out

def draw_fallback_preview(img: np.ndarray, sccui, window_index: int) -> np.ndarray:
    disp = img.copy()
    positions = {}
    try:
        positions = sccui.positions if hasattr(sccui, 'positions') else {}
        if isinstance(positions, dict) and window_index in positions:
            positions = positions[window_index]
        elif isinstance(positions, list) and len(positions) > window_index:
            positions = positions[window_index]
    except Exception:
        positions = {}

    try:
        corrected = sccui.corrected[window_index] if hasattr(sccui, 'corrected') and sccui.corrected and len(sccui.corrected) > window_index else {}
        corrected = normalize_mapping_keys(corrected)
    except Exception:
        corrected = {}
    try:
        detections = sccui.detections[window_index] if hasattr(sccui, 'detections') and sccui.detections and len(sccui.detections) > window_index else {}
        detections = normalize_mapping_keys(detections)
    except Exception:
        detections = {}

    colmap = {'R': (0,0,255), 'O': (0,140,255), 'B': (255,0,0), 'G': (0,200,0), 'W': (240,240,240), 'Y': (0,230,230)}

    if isinstance(positions, dict):
        items = positions.items()
    elif isinstance(positions, list):
        items = [(str(i), p) for i, p in enumerate(positions)]
    else:
        items = []

    for label, pos in items:
        try:
            x, y = int(pos[0]), int(pos[1])
            color_char = corrected.get(str(label)) if isinstance(corrected, dict) else None
            if color_char is None:
                color_char = detections.get(str(label)) if isinstance(detections, dict) else None
            bgr = colmap.get(color_char, (160,160,160))
            cv2.circle(disp, (x, y), 18, bgr, -1, cv2.LINE_AA)
            cv2.circle(disp, (x, y), 18, (30,30,30), 2, cv2.LINE_AA)
            if str(label) in corrected:
                cv2.circle(disp, (x, y), 24, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(disp, str(label), (x - 10, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0) if color_char in ['W','Y'] else (255,255,255), 1, cv2.LINE_AA)
        except Exception:
            continue
    return disp

# ---------------- PolygonDetectorWidget ----------------
class PolygonDetectorWidget(QWidget):
    def __init__(self, faces_for_image: List[str], save_file: Path, image_path: Path, preferred_size: QSize, parent=None):
        super().__init__(parent)
        self.save_file = Path(save_file)
        self.image_path = Path(image_path)
        self.detector = PolygonDetector(faces_for_image, save_file)
        self.preferred_size = preferred_size
        self._setup_ui()
        self._load_image_and_positions()

    def _setup_ui(self):
        self.setMinimumSize(self.preferred_size)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(6)
        self.layout.setContentsMargins(6,6,6,6)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setObjectName('imageFrame')
        self.layout.addWidget(self.image_label, stretch=1)

    def _load_image_and_positions(self):
        try:
            self.detector.setup_image(self.image_path)
            if self.save_file.exists():
                try:
                    raw = json.load(open(self.save_file,'r'))
                    if isinstance(raw, dict):
                        self.detector.positions = {str(k): (int(v[0]), int(v[1])) for k, v in raw.items()}
                except Exception as e:
                    logger.info(f"No valid polygon json: {e}")
            self.update_display()
        except Exception as e:
            logger.error(f"Polygon load failed: {e}")
            self.image_label.setText('Error loading image')

    def update_display(self):
        try:
            img = self.detector._draw_interface()
            if img is None:
                self.image_label.setText('Preview unavailable')
                return
            img_with_badge = overlay_badge(img, self.image_path.name)
            pix = cvimg_to_qpixmap(img_with_badge, self.image_label.size())
            self.image_label.setPixmap(pix)
        except Exception as e:
            logger.error(f"Polygon display error: {e}")
            self.image_label.setText('Preview unavailable')

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.update_display()

    def _map_pos_to_image(self, pos: QPoint) -> Tuple[int, int]:
        pos_in_label = self.image_label.mapFromParent(pos)
        pix = self.image_label.pixmap()
        if pix is None or self.detector.img is None:
            return (0, 0)
        iw, ih = self.detector.img.shape[1], self.detector.img.shape[0]
        lw, lh = self.image_label.width(), self.image_label.height()
        pw, ph = pix.width(), pix.height()
        x_off = (lw - pw) // 2
        y_off = (lh - ph) // 2
        x = pos_in_label.x() - x_off
        y = pos_in_label.y() - y_off
        x = max(0, min(pw-1, x)); y = max(0, min(ph-1, y))
        img_x = int(x * (iw / pw)); img_y = int(y * (ih / ph))
        return img_x, img_y

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            imgx, imgy = self._map_pos_to_image(event.pos())
            handled = False
            try:
                if hasattr(self.detector, '_handle_mouse_down'):
                    self.detector._handle_mouse_down(imgx, imgy)
                    handled = True
                elif hasattr(self.detector, 'mouse_down'):
                    self.detector.mouse_down(imgx, imgy); handled = True
            except Exception as e:
                logger.debug(f"Detector mouse_down raised: {e}")
            if not handled:
                best_k = None; best_d = 1e9
                for k, (px, py) in self.detector.positions.items():
                    d = math.hypot(px - imgx, py - imgy)
                    if d < best_d:
                        best_d = d; best_k = k
                if best_k and best_d < 28:
                    try:
                        self.detector.positions[best_k] = (imgx, imgy)
                    except Exception:
                        pass
            self.update_display()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            imgx, imgy = self._map_pos_to_image(event.pos())
            try:
                if hasattr(self.detector, '_handle_mouse_move'):
                    self.detector._handle_mouse_move(imgx, imgy)
                else:
                    if hasattr(self.detector, 'last_touched') and self.detector.last_touched:
                        self.detector.positions[self.detector.last_touched] = (imgx, imgy)
            except Exception:
                pass
            self.update_display()

    def save_positions(self):
        try:
            if hasattr(self.detector, '_save_positions_to_disk'):
                self.detector._save_positions_to_disk()
            else:
                with open(self.save_file, 'w') as f:
                    json.dump({k: list(v) for k, v in self.detector.positions.items()}, f)
        except Exception as e:
            logger.error(f"Save positions error: {e}")

    def close_and_release(self):
        try:
            if hasattr(self.detector, 'close'):
                self.detector.close()
        except Exception:
            pass

    def get_positions(self):
        return {str(k): tuple(v) for k, v in self.detector.positions.items()}

# ---------------- ColorCorrectionWidget ----------------
class ColorCorrectionWidget(QWidget):
    def __init__(self, sccui: SynchronizedColorCorrectionUI, window_index: int, preferred_size: QSize, parent=None):
        super().__init__(parent)
        self.sccui = sccui
        self.window_index = window_index
        self.preferred_size = preferred_size
        self.active_color: Optional[str] = None
        self._setup_ui()
        try:
            if not hasattr(self.sccui, 'detections') or not self.sccui.detections:
                self.sccui.detections = [{}, {}]
            else:
                self.sccui.detections = [normalize_mapping_keys(self.sccui.detections[0]), normalize_mapping_keys(self.sccui.detections[1])]
        except Exception:
            self.sccui.detections = [{}, {}]
        try:
            if not hasattr(self.sccui, 'corrected') or self.sccui.corrected is None:
                self.sccui.corrected = [self.sccui.detections[0].copy() if self.sccui.detections and isinstance(self.sccui.detections[0], dict) else {}, self.sccui.detections[1].copy() if self.sccui.detections and isinstance(self.sccui.detections[1], dict) else {}]
            else:
                self.sccui.corrected = [normalize_mapping_keys(self.sccui.corrected[0]) if len(self.sccui.corrected) > 0 else {}, normalize_mapping_keys(self.sccui.corrected[1]) if len(self.sccui.corrected) > 1 else {}]
        except Exception:
            self.sccui.corrected = [{}, {}]
        try:
            if not hasattr(self.sccui, 'corrected_labs') or self.sccui.corrected_labs is None:
                self.sccui.corrected_labs = [{}, {}]
            else:
                self.sccui.corrected_labs = [normalize_mapping_keys(self.sccui.corrected_labs[0]) if len(self.sccui.corrected_labs) > 0 else {}, normalize_mapping_keys(self.sccui.corrected_labs[1]) if len(self.sccui.corrected_labs) > 1 else {}]
        except Exception:
            self.sccui.corrected_labs = [{}, {}]
        if not hasattr(self.sccui, 'image_paths') or self.sccui.image_paths is None:
            self.sccui.image_paths = [str(IMG1_PATH), str(IMG2_PATH)]
        self.update_display()

    def _setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(6,6,6,6); self.layout.setSpacing(6)
        self.image_label = QLabel(self); self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setObjectName('imageFrame')
        self.layout.addWidget(self.image_label, stretch=1)

        pal = QHBoxLayout(); pal.setSpacing(12); pal.setContentsMargins(12,8,12,8)
        self.color_buttons = {}
        palette = [('R','Red',(220,60,60)), ('O','Orange',(255,165,70)), ('B','Blue',(80,130,255)), ('G','Green',(80,200,100)), ('W','White',(240,240,240)), ('Y','Yellow',(245,220,80))]
        for code, name, rgb in palette:
            btn = QPushButton('')
            btn.setObjectName('colorSwatch')
            btn.setToolTip(name + f' ({code})')
            r, g, b = rgb
            btn.setStyleSheet(f"background: rgb({r},{g},{b}); border-radius:18px; border: 2px solid rgba(0,0,0,0.15);")
            btn.setFixedSize(36,36)
            btn.clicked.connect(lambda checked, c=code, b=btn: self.set_active_color(c, b))
            pal.addWidget(btn)
            self.color_buttons[code] = btn
        self.layout.addLayout(pal)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.update_display()

    def update_display(self):
        try:
            img = cv2.imread(str(self.sccui.image_paths[self.window_index]))
            if img is None:
                self.image_label.setText('Could not load image')
                return
            try:
                disp = self.sccui._draw_interface(img, self.window_index)
                if disp is None:
                    raise RuntimeError('sccui._draw_interface returned None')
            except Exception as e:
                logger.debug(f"sccui._draw_interface failed: {e}")
                disp = draw_fallback_preview(img, self.sccui, self.window_index)
            disp = overlay_badge(disp, Path(str(self.sccui.image_paths[self.window_index])).name)
            pix = cvimg_to_qpixmap(disp, self.image_label.size())
            self.image_label.setPixmap(pix)
        except Exception as e:
            logger.error(f"Correction update error: {e}")
            self.image_label.setText('Error rendering preview')

    def _find_nearest_sticker(self, img_x: int, img_y: int, threshold: int = 28) -> Optional[str]:
        try:
            positions = None
            if hasattr(self.sccui, 'positions'):
                positions = self.sccui.positions
                if isinstance(positions, dict) and self.window_index in positions:
                    positions = positions[self.window_index]
                elif isinstance(positions, list) and len(positions) > self.window_index:
                    positions = positions[self.window_index]
            if not positions:
                return None
            best_k = None; best_d = 1e9
            if isinstance(positions, dict):
                items = positions.items()
            else:
                items = [(str(i), p) for i, p in enumerate(positions)]
            for k, v in items:
                try:
                    px, py = int(v[0]), int(v[1])
                except Exception:
                    continue
                d = math.hypot(px - img_x, py - img_y)
                if d < best_d:
                    best_d = d; best_k = k
            if best_d <= threshold:
                return str(best_k)
            return None
        except Exception as e:
            #logger.debug(f"_find_nearest_sticker error: {e}")
            return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pix = self.image_label.pixmap()
            if pix is None:
                return
            pos_in_label = self.image_label.mapFromParent(event.pos())
            pw, ph = pix.width(), pix.height()
            img_cv = cv2.imread(str(self.sccui.image_paths[self.window_index]))
            if img_cv is None:
                return
            iw = img_cv.shape[1]; ih = img_cv.shape[0]
            lw, lh = self.image_label.width(), self.image_label.height()
            x_off = (lw - pw)//2; y_off = (lh - ph)//2
            x = pos_in_label.x() - x_off; y = pos_in_label.y() - y_off
            x = max(0, min(pw-1, x)); y = max(0, min(ph-1, y))
            img_x = int(x * (iw / pw)); img_y = int(y * (ih / ph))

            if self.active_color:
                key = self._find_nearest_sticker(img_x, img_y)
                if not key:
                    #QMessageBox.information(self, 'No sticker nearby', 'No sticker found near the click position. Try clicking closer to the sticker center.')
                    return
                self._apply_color_to_key(str(key), self.active_color)
                return

            try:
                self.sccui._mouse_cb(None, img_x, img_y, None, None, self.window_index)
            except Exception as e:
                logger.debug(f"Correction mouse forward: {e}")
            self.update_display()

    def set_active_color(self, code: str, btn: QPushButton):
        prev = self.active_color
        if prev == code:
            self.active_color = None
        else:
            self.active_color = code

        for c, b in self.color_buttons.items():
            if c == 'R': base = 'background: rgb(220,60,60);'
            elif c == 'O': base = 'background: rgb(255,165,70);'
            elif c == 'B': base = 'background: rgb(80,130,255);'
            elif c == 'G': base = 'background: rgb(80,200,100);'
            elif c == 'W': base = 'background: rgb(240,240,240);'
            elif c == 'Y': base = 'background: rgb(245,220,80);'
            else: base = ''
            style = base + ' border-radius:18px; border: 2px solid rgba(0,0,0,0.15);'
            b.setStyleSheet(style)

            if self.active_color == c:
                effect = QGraphicsDropShadowEffect(b)
                effect.setBlurRadius(18)
                effect.setColor(QColor(80,180,120,160))
                effect.setOffset(0, 0)
                b.setGraphicsEffect(effect)
            else:
                b.setGraphicsEffect(None)

        self.update_display()

    def _apply_color_to_key(self, key: str, color_code: str):
        try:
            if not hasattr(self.sccui, 'corrected') or self.sccui.corrected is None:
                self.sccui.corrected = [{}, {}]
            if isinstance(self.sccui.corrected, list) and len(self.sccui.corrected) <= self.window_index:
                while len(self.sccui.corrected) <= self.window_index:
                    self.sccui.corrected.append({})
            if self.sccui.corrected[self.window_index] is None:
                self.sccui.corrected[self.window_index] = {}
            key_str = str(key)
            self.sccui.corrected[self.window_index][key_str] = color_code
            try:
                if hasattr(self.sccui, 'corrected_labs') and self.sccui.corrected_labs and len(self.sccui.corrected_labs) > self.window_index:
                    lab_map = normalize_mapping_keys(self.sccui.corrected_labs[self.window_index])
                    lab = lab_map.get(key_str)
                    if lab is not None:
                        if not hasattr(self.sccui, 'color_centers'):
                            self.sccui.color_centers = {}
                        self.sccui.color_centers[color_code] = lab
            except Exception:
                pass
            self.update_display()
        except Exception as e:
            logger.error(f"_apply_color_to_key failed: {e}")
            QMessageBox.critical(self, 'Error', f"Failed to apply color: {e}")

# ---------------- Workers ----------------
class ColorDetectWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, positions1, positions2):
        super().__init__()
        self.positions1 = positions1
        self.positions2 = positions2
        self.detector = ColorDetector()

    def run(self):
        try:
            self.progress.emit('Detecting colors image 1...')
            det1, labs1 = self.detector.detect_single_image(IMG1_PATH, self.positions1)
            self.progress.emit('Image 1 done; detecting image 2...')
            det2, labs2 = self.detector.detect_single_image(IMG2_PATH, self.positions2)
            self.progress.emit('Color detection finished')
            self.finished.emit((det1, labs1, det2, labs2))
        except Exception as e:
            self.error.emit(str(e))

# ---------------- Dialog flow ----------------
class CombinedDetectionDialog(QDialog):
    def __init__(self, parent: QWidget, cube_status: CubeStatus):
        super().__init__(parent)
        self.setWindowTitle('Polygon Editors — define sticker positions')
        self.resize(1400, 860)
        self.setStyleSheet(GLOBAL_CSS)
        self.cube_status = cube_status
        self.poly_size = QSize(760, 540)
        self.corr_size = QSize(900, 640)
        self._setup_ui()

    def _setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10,10,10,10); self.layout.setSpacing(8)
        self.top_container = QWidget(); self.top_layout = QHBoxLayout(self.top_container)
        self.top_layout.setSpacing(8); self.top_layout.setContentsMargins(6,6,6,6)
        self.poly1 = PolygonDetectorWidget(['U','F','L'], POLYGON_POSITIONS_PATH_1 , IMG1_PATH, self.poly_size)
        self.poly2 = PolygonDetectorWidget(['D','R','B'], POLYGON_POSITIONS_PATH_2, IMG2_PATH, self.poly_size)
        self.top_layout.addWidget(self.poly1); self.top_layout.addWidget(self.poly2)
        self.layout.addWidget(self.top_container, stretch=1)
        btn_row = QHBoxLayout(); btn_row.addStretch()
        self.cont_btn = QPushButton('Continue → Detect Colors'); self.cont_btn.setObjectName('primary')
        self.cont_btn.setFixedSize(260, 44)
        btn_row.addWidget(self.cont_btn); btn_row.addStretch()
        self.layout.addLayout(btn_row)
        self.cont_btn.clicked.connect(self._on_continue_detect)

    def _clear_top_container(self):
        for i in reversed(range(self.top_layout.count())):
            w = self.top_layout.itemAt(i).widget()
            if w: w.setParent(None)

    def _on_continue_detect(self):
        self.poly1.save_positions(); self.poly2.save_positions()
        self.poly1.close_and_release(); self.poly2.close_and_release()
        pos1 = self.poly1.get_positions(); pos2 = self.poly2.get_positions()
        self.cont_btn.setEnabled(False)
        self.progress = QProgressBar(self); self.progress.setRange(0,0)
        self.layout.insertWidget(1, self.progress)
        self.worker = ColorDetectWorker(pos1, pos2)
        self.worker.progress.connect(lambda m: self.setWindowTitle(f"Polygon Editors — {m}"))
        self.worker.error.connect(self._on_detect_error)
        self.worker.finished.connect(self._on_color_detected)
        self.worker.start()

    def _on_detect_error(self, e: str):
        QMessageBox.critical(self, 'Color detection error', e)
        self.cont_btn.setEnabled(True)
        if hasattr(self,'progress'):
            self.progress.hide(); self.progress.deleteLater()

    def _on_color_detected(self, payload):
        if hasattr(self,'progress'):
            self.progress.hide(); self.progress.deleteLater()
        det1, labs1, det2, labs2 = payload
        try:
            sccui = SynchronizedColorCorrectionUI(self.poly1.get_positions(), self.poly2.get_positions(), det1, labs1, det2, labs2)
        except Exception as e:
            QMessageBox.critical(self, 'SCCUI error', str(e))
            self.cont_btn.setEnabled(True); return
        sccui.detections = [normalize_mapping_keys(det1) if isinstance(det1, dict) else {}, normalize_mapping_keys(det2) if isinstance(det2, dict) else {}]
        sccui.corrected = [normalize_mapping_keys(det1) if isinstance(det1, dict) else {}, normalize_mapping_keys(det2) if isinstance(det2, dict) else {}]
        sccui.corrected_labs = [normalize_mapping_keys(labs1) if isinstance(labs1, dict) else {}, normalize_mapping_keys(labs2) if isinstance(labs2, dict) else {}]
        if not hasattr(sccui, 'image_paths') or sccui.image_paths is None:
            sccui.image_paths = [str(IMG1_PATH), str(IMG2_PATH)]

        corr = CorrectionDialog(self, self.cube_status, sccui, self.corr_size)
        res = corr.exec_()
        if res == QDialog.Accepted:
            self.accept()
        else:
            self._clear_top_container()
            self.poly1 = PolygonDetectorWidget(['U','F','L'], Path('polygon1.json'), IMG1_PATH, self.poly_size)
            self.poly2 = PolygonDetectorWidget(['D','R','B'], Path('polygon2.json'), IMG2_PATH, self.poly_size)
            self.top_layout.addWidget(self.poly1); self.top_layout.addWidget(self.poly2)
            self.cont_btn.setEnabled(True)

class CorrectionDialog(QDialog):
    def __init__(self, parent, cube_status: CubeStatus, sccui: SynchronizedColorCorrectionUI, preferred_size: QSize):
        super().__init__(parent)
        self.setWindowTitle('Color Correction — fix detected colors')
        self.resize(1500, 900)
        self.setStyleSheet(GLOBAL_CSS)
        self.cube_status = cube_status
        self.sccui = sccui
        self.pref = preferred_size
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8,8,8,8); layout.setSpacing(8)
        top = QHBoxLayout()
        self.cw1 = ColorCorrectionWidget(self.sccui, 0, self.pref); self.cw2 = ColorCorrectionWidget(self.sccui, 1, self.pref)
        top.addWidget(self.cw1); top.addWidget(self.cw2)
        layout.addLayout(top)
        btn_row = QHBoxLayout(); btn_row.addStretch()
        self.cancel_btn = QPushButton('Cancel'); self.cancel_btn.setObjectName('danger'); self.cancel_btn.setFixedSize(160,44)
        self.finalize_btn = QPushButton('Finalize & Validate'); self.finalize_btn.setObjectName('primary'); self.finalize_btn.setFixedSize(260,52)
        btn_row.addWidget(self.cancel_btn); btn_row.addSpacing(16); btn_row.addWidget(self.finalize_btn); btn_row.addStretch()
        layout.addLayout(btn_row)
        self.cancel_btn.clicked.connect(self.reject)
        self.finalize_btn.clicked.connect(self._on_finalize)

    def _on_finalize(self):
        try:
            self.cube_status.corrected1 = self.sccui.corrected[0]
            self.cube_status.corrected2 = self.sccui.corrected[1]
            color_str, facelet, sol = self.cube_status.build_facelets_and_solve()
            QMessageBox.information(self, 'Validation OK', 'Corrected mapping validated successfully.')
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, 'Validation failed', f"Validation failed: {e}")

# ---------------- Main app UI ----------------

class CameraController:
    
    def __init__(self):
        self.cap = None
        self.current_camera_index = 0
        self.is_initialized = False
        self.last_frame = None

    def initialize(self) -> bool:
        for i in range(6):
            if i == self.current_camera_index:
                continue
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.current_camera_index = i
                if self.cap is not None:
                    self.cap.release()
                self.cap = cap
                self._configure_camera()
                self.is_initialized = True
                logger.info(f"Camera initialized on index {i}")
                return True
        if self.current_camera_index == -1:
            logger.error("No camera found")
            return False
        self.cap = cv2.VideoCapture(self.current_camera_index)
        self._configure_camera()
        self.is_initialized = True
        logger.info("Using last known camera index")
        return True

    def _configure_camera(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
            self.cap.set(cv2.CAP_PROP_FPS, 30)

    def capture_frame(self):
        if not self.is_initialized or not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = frame
            return frame
        return None

    def switch_camera(self) -> bool:
        if self.cap:
            self.cap.release()
        self.is_initialized = False
        return self.initialize()

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.is_initialized = False
        logger.info("Camera released")

class DetectionWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, cube_status: CubeStatus):
        super().__init__()
        self.cube_status = cube_status
        self._is_running = True

    def run(self):
        try:
            self.progress.emit("Starting detection...")
            result = self.cube_status.detect_status()
            if self._is_running:
                self.finished.emit(result)
        except Exception as e:
            if self._is_running:
                self.error.emit(str(e))

    def stop(self):
        self._is_running = False

class CameraWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Camera feed will appear here")
        self.setStyleSheet("border-radius: 8px; background-color: #111; color: #bbb; padding: 6px;")
        self.pos1 = None
        self.pos2 = None
        self.load_positions()

    def set_frame(self, frame: np.ndarray, idx: int = 1, new_pos: bool = False):
        """
        Draw camera frame and overlay saved polygon points for the given image index.
        idx: 1 or 2
        new_pos: if True will reload positions from disk before drawing.
        """
        if frame is None or frame.size == 0:
            return
        if new_pos:
            self.load_positions()

        try:
            # original frame dims
            ih, iw = frame.shape[:2]

            # convert to RGB QImage
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * iw
            qimage = QImage(rgb_image.data, iw, ih, bytes_per_line, QImage.Format_RGB888)

            # scaled pixmap keeping aspect ratio
            scaled_pixmap = QPixmap.fromImage(qimage).scaled(
                self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            pw, ph = scaled_pixmap.width(), scaled_pixmap.height()

            # final pixmap (background + scaled image centered)
            final_pixmap = QPixmap(self.width(), self.height())
            final_pixmap.fill(QColor("#111"))

            painter = QPainter(final_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)

            x_off = (self.width() - pw) // 2
            y_off = (self.height() - ph) // 2
            painter.drawPixmap(x_off, y_off, scaled_pixmap)

            # draw rounded border
            pen = QPen(QColor(80, 80, 80))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRoundedRect(1, 1, self.width() - 2, self.height() - 2, 6, 6)

            # overlay saved positions (map saved-image coords -> pixmap coords)
            positions = self.pos1 if idx == 1 else self.pos2
            img_width, img_height = CAMERA_RESOLUTION
            if positions:
                fill_brush = QColor(255, 220, 80, 180)  # semi-transparent yellow
                outline_pen = QPen(QColor(200, 180, 60, 220))
                outline_pen.setWidth(2)
                painter.setPen(outline_pen)
                painter.setBrush(fill_brush)

                for k, v in positions.items():
                    try:
                        vx, vy = v
                        # detect normalized coords: small floats <= 1.0
                        if 0.0 <= vx <= 1.0 and 0.0 <= vy <= 1.0:
                            img_x = int(vx * img_width)
                            img_y = int(vy * img_height)
                        else:
                            # assume pixel coords already
                            img_x = int(vx)
                            img_y = int(vy)

                        # map image (base_w,base_h) coords to pixmap coords (pw,ph)
                        px = x_off + int(img_x * (pw / img_width))
                        py = y_off + int(img_y * (ph / img_height))

                        # clamp into pixmap area (avoid drawing outside)
                        px = max(x_off, min(x_off + pw - 1, px))
                        py = max(y_off, min(y_off + ph - 1, py))

                        # draw marker (+ halo)
                        painter.setBrush(fill_brush)
                        painter.setPen(outline_pen)
                        painter.drawEllipse(px - 6, py - 6, 12, 12)

                        halo_pen = QPen(QColor(255, 255, 255, 80))
                        halo_pen.setWidth(1)
                        painter.setPen(halo_pen)
                        painter.setBrush(Qt.NoBrush)
                        painter.drawEllipse(px - 10, py - 10, 20, 20)

                        # small numeric label
                        label_pen = QPen(QColor(20, 20, 20))
                        painter.setPen(label_pen)
                        font = painter.font()
                        font.setPointSize(8)
                        font.setBold(True)
                        painter.setFont(font)
                        painter.drawText(px - 6, py + 4, str(k))

                    except Exception as ee:
                        logger.debug(f"Failed drawing position {k}:{v} -> {ee}")
                        continue

                # small top-left helper label on the image
                helper_pen = QPen(QColor(220, 220, 220, 200))
                painter.setPen(helper_pen)
                font = painter.font()
                font.setPointSize(9)
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(x_off + 8, y_off + 20, "Polygon guide loaded")

            painter.end()

            # set result to label
            self.setPixmap(final_pixmap)
        except Exception as e:
            logger.error(f"Error setting camera frame with overlay: {e}")

    def load_positions(self):
        try:
            raw1 = json.load(open(POLYGON_POSITIONS_PATH_1,'r'))["positions"]
            raw2 = json.load(open(POLYGON_POSITIONS_PATH_2,'r'))["positions"]
            if isinstance(raw1, dict):
                self.pos1 = {str(k): (int(v[0]), int(v[1])) for k, v in raw1.items()}
            if isinstance(raw2, dict):
                self.pos2 = {str(k): (int(v[0]), int(v[1])) for k, v in raw2.items()}
        except Exception as e:
            logger.info(f"No valid polygon json: {e}")

class ThumbnailWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(THUMBNAIL_RESOLUTION[0], THUMBNAIL_RESOLUTION[1])
        self.setStyleSheet("border-radius:6px; background-color: #111; padding:2px;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("No image")

    def set_image(self, image_path: Path):
        if not image_path.exists():
            self.clear()
            self.setText("No image")
            return
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                self.clear(); self.setText("Load error"); return
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            final = QPixmap(self.width(), self.height())
            final.fill(QColor("#111"))
            painter = QPainter(final)
            x = (self.width()-pixmap.width())//2
            y = (self.height()-pixmap.height())//2
            painter.drawPixmap(x,y,pixmap)
            painter.end()
            self.setPixmap(final)
        except Exception as e:
            logger.error(f"Error loading thumbnail {image_path}: {e}")
            self.clear(); self.setText("Error")

class MoveButton(QPushButton):
    def __init__(self, move: str, parent=None):
        super().__init__(move, parent)
        self.move = move
        self.setFixedSize(24, 20)
        self.setCursor(Qt.PointingHandCursor)
        move_colors = {
            'U': '#FF6B6B', 'F': '#6BFF6B', 'L': '#6B6BFF',
            'D': '#FFD26B', 'R': '#FF6BFF', 'B': '#6BFFFF'
        }
        color = move_colors.get(move, '#777')
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: black;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:pressed {{
                background-color: #333;
                color: white;
            }}
        """)

class CubeWidget(QWidget):
    stickerClicked = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.STICKER_SIZE = 28
        self.STICKER_GAP = 4
        self.SIDE_GAP = 12
        self.SIDE_SIZE = self.STICKER_SIZE * 3 + self.STICKER_GAP * 4
        self.SIDE_POSITIONS = {'U': (1, 0), 'L': (0, 1), 'F': (1, 1), 'R': (2, 1), 'B': (3, 1), 'D': (1, 2)}
        self.color_palette = {
            'R': QColor(200, 30, 30), 'O': QColor(255, 140, 0), 'B': QColor(30, 60, 200),
            'G': QColor(40, 180, 40), 'W': QColor(240, 240, 240), 'Y': QColor(230, 230, 50)
        }
        self.setMinimumSize(self.SIDE_SIZE * 4 + self.SIDE_GAP * 5, self.SIDE_SIZE * 3 + self.SIDE_GAP * 4)
        self.setStyleSheet("background:transparent;")
        self.status = COLOR_INIT_STATE
        self.selected_sticker = None

    def set_state(self, status: str):
        self.status = status
        self.update()

    def get_sticker_at_pos(self, pos) -> Tuple[Optional[str], Optional[str]]:
        x, y = pos.x(), pos.y()
        total_width = self.SIDE_SIZE * 4 + self.SIDE_GAP * 5
        total_height = self.SIDE_SIZE * 3 + self.SIDE_GAP * 4
        x_offset = (self.width() - total_width) // 2
        y_offset = (self.height() - total_height) // 2
        for side, (x_pos, y_pos) in self.SIDE_POSITIONS.items():
            offset_x = x_offset + x_pos * self.SIDE_SIZE + (x_pos + 1) * self.SIDE_GAP
            offset_y = y_offset + y_pos * self.SIDE_SIZE + (y_pos + 1) * self.SIDE_GAP
            if (offset_x <= x <= offset_x + self.SIDE_SIZE and offset_y <= y <= offset_y + self.SIDE_SIZE):
                rel_x = x - offset_x
                rel_y = y - offset_y
                sticker_col = int(rel_x // (self.STICKER_SIZE + self.STICKER_GAP))
                sticker_row = int(rel_y // (self.STICKER_SIZE + self.STICKER_GAP))
                if 0 <= sticker_row < 3 and 0 <= sticker_col < 3:
                    idx = sticker_row * 3 + sticker_col
                    return side, f"{side}{idx+1}"
        return None, None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            side, sticker_pos = self.get_sticker_at_pos(event.pos())
            if side and sticker_pos:
                self.selected_sticker = (side, sticker_pos)
                self.stickerClicked.emit(side, sticker_pos)
                self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(22, 22, 22))
        total_width = self.SIDE_SIZE * 4 + self.SIDE_GAP * 5
        total_height = self.SIDE_SIZE * 3 + self.SIDE_GAP * 4
        x_offset = (self.width() - total_width) // 2
        y_offset = (self.height() - total_height) // 2
        face_order = ['U', 'R', 'F', 'D', 'L', 'B']
        for side_idx, side in enumerate(face_order):
            x_pos, y_pos = self.SIDE_POSITIONS[side]
            offset_x = x_offset + x_pos * self.SIDE_SIZE + (x_pos + 1) * self.SIDE_GAP
            offset_y = y_offset + y_pos * self.SIDE_SIZE + (y_pos + 1) * self.SIDE_GAP
            for i in range(9):
                row, col = i//3, i%3
                sticker_x = offset_x + col * (self.STICKER_SIZE + self.STICKER_GAP) + self.STICKER_GAP
                sticker_y = offset_y + row * (self.STICKER_SIZE + self.STICKER_GAP) + self.STICKER_GAP
                try:
                    color_char = self.status[side_idx * 9 + i]
                except Exception:
                    color_char = '?'
                color = self.color_palette.get(color_char, QColor(120,120,120))
                if (self.selected_sticker and self.selected_sticker[0] == side and self.selected_sticker[1] == f"{side}{i+1}"):
                    painter.setPen(QPen(QColor(255, 255, 0), 3))
                else:
                    painter.setPen(QPen(QColor(60,60,60), 1))
                painter.fillRect(sticker_x, sticker_y, self.STICKER_SIZE, self.STICKER_SIZE, color)
                painter.drawRect(sticker_x, sticker_y, self.STICKER_SIZE, self.STICKER_SIZE)

class ColorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_color = None
        self.setWindowTitle("Select Sticker Color")
        self.setStyleSheet(GLOBAL_CSS)
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        label = QLabel("Choose color for the sticker:")
        layout.addWidget(label)
        grid = QGridLayout()
        colors = [
            ('R', 'Red', QColor(200,30,30)),
            ('O', 'Orange', QColor(255,140,0)),
            ('B', 'Blue', QColor(30,60,200)),
            ('G', 'Green', QColor(40,180,40)),
            ('W', 'White', QColor(240,240,240)),
            ('Y', 'Yellow', QColor(230,230,50))
        ]
        for i, (char, name, qcol) in enumerate(colors):
            btn = QPushButton(name)
            btn.setFixedSize(90, 36)
            text_color = 'black' if name in ['White', 'Yellow'] else 'white'
            btn.setStyleSheet(f"background:{qcol.name()}; color:{text_color}; font-weight:bold; border-radius:6px;")
            btn.clicked.connect(lambda checked, c=char: self._choose(c))
            grid.addWidget(btn, i//3, i%3)
        layout.addLayout(grid)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose(self, c: str):
        self.selected_color = c
        self.accept()

# ---------------- MainWindow ----------------
class MainWindow(QMainWindow):
    update_status_signal = pyqtSignal(str)
    detection_finished_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.cube_status = CubeStatus()
        self.camera_controller = CameraController()
        self.motor_controller = MotorController()
        self.calibrator = CalibrationManager()

        self.captured_images_count = 0
        self.detection_worker: Optional[DetectionWorker] = None
        self.is_processing = False
        self.prime_mode = False
        self.new_positions = False

        self._setup_ui()
        self._connect_signals()
        self._initialize_camera()
        self._initialize_hardware()

    def _setup_ui(self):
        self.setWindowTitle("Rubik's Cube Solver — Unified UI")
        self.setGeometry(80, 80, 1400, 900)
        self.setStyleSheet(GLOBAL_CSS)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("View")
        toggle_theme = QAction("Toggle Theme", self)
        toggle_theme.triggered.connect(self._toggle_theme)
        view_menu.addAction(toggle_theme)

        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        root = QSplitter(Qt.Horizontal)
        root.setHandleWidth(8)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        vis_row = QHBoxLayout()
        vis_row.setSpacing(12)

        cube_box = QGroupBox("Cube State")
        cube_layout = QVBoxLayout(cube_box)
        self.cube_widget = CubeWidget()
        self.cube_widget.stickerClicked.connect(self._on_sticker_clicked)
        cube_layout.addWidget(self.cube_widget)
        cube_box.setMinimumWidth(520)
        vis_row.addWidget(cube_box, stretch=3)

        cam_box = QGroupBox("Camera")
        cam_layout = QVBoxLayout(cam_box)
        self.camera_widget = CameraWidget()
        cam_layout.addWidget(self.camera_widget)
        cam_box.setMinimumWidth(520)
        vis_row.addWidget(cam_box, stretch=3)

        left_layout.addLayout(vis_row, stretch=4)

        lower_row = QHBoxLayout()
        lower_row.setSpacing(10)

        thumb_group = QGroupBox("Captured Images")
        t_layout = QHBoxLayout(thumb_group)
        self.thumb1 = ThumbnailWidget()
        self.thumb2 = ThumbnailWidget()
        t_layout.addWidget(self._make_thumb_column("Image 1", self.thumb1, "Centers: Blue, Red, Yellow"))
        t_layout.addWidget(self._make_thumb_column("Image 2", self.thumb2, "Centers: Green, Orange, White"))
        lower_row.addWidget(thumb_group, stretch=1)

        tabs = QTabWidget()
        tabs.setMinimumWidth(380)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("background-color:#0f0f0f; color:#ddd; font-family:monospace; font-size:11px;")
        tabs.addTab(self.log_view, "Logs")

        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        status_layout.addWidget(self.progress_bar)
        self.status_message = QLabel("Ready")
        self.status_message.setStyleSheet("padding:6px; background:#151515; border-radius:4px; color:#ddd;")
        status_layout.addWidget(self.status_message)
        status_layout.addStretch()
        tabs.addTab(status_tab, "Status")

        lower_row.addWidget(tabs, stretch=2)
        left_layout.addLayout(lower_row, stretch=2)
        root.addWidget(left_container)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(12)
        right_container.setFixedWidth(420)

        right_layout.addWidget(self._create_detection_group())
        right_layout.addWidget(self._create_solving_group())
        right_layout.addWidget(self._create_custom_moves_group())
        right_layout.addWidget(self._create_hardware_group())
        right_layout.addWidget(self._create_validation_group())
        right_layout.addWidget(self._create_system_group())

        copyright_label = QLabel(COPYRIGHT_MARK)
        copyright_label.setStyleSheet("color:#888; font-size:10px;")
        copyright_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(copyright_label)
        right_layout.addStretch()

        root.addWidget(right_container)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Ready — Press Space to capture images")

    # ... (UI helper methods: _make_thumb_column, _create_* groups, signal connections, camera init)
    # For brevity those methods are identical to the previous runUi.py implementation; we include them below.

    def _make_thumb_column(self, title: str, widget: ThumbnailWidget, info: str) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight:bold; color:#ddd;")
        layout.addWidget(lbl)
        layout.addWidget(widget)
        info_lbl = QLabel(info)
        info_lbl.setStyleSheet("background:#111; padding:6px; border-radius:6px; color:#ddd;")
        layout.addWidget(info_lbl)
        return box

    def _create_detection_group(self):
        g = QGroupBox("Cube Detection")
        l = QVBoxLayout(g)

        # New: Open interactive detection panel (polygon editors + color corrector)
        self.open_det_panel_btn = QPushButton("Open Detections Panel")
        self.open_det_panel_btn.clicked.connect(self._open_detections_panel)
        self.open_det_panel_btn.setToolTip("Open interactive polygon editors and color corrector")
        #self.open_det_panel_btn.setObjectName('primary')
        l.addWidget(self.open_det_panel_btn)

        self.calibrator_btn = QPushButton("Calibrate Colors")
        self.calibrator_btn.clicked.connect(self._calibrate_colors)
        l.addWidget(self.calibrator_btn)
        return g

    def _create_solving_group(self):
        g = QGroupBox("Cube Solving")
        l = QVBoxLayout(g)
        self.scramble_btn = QPushButton("Scramble Cube")
        self.scramble_btn.clicked.connect(self._scramble_cube)
        l.addWidget(self.scramble_btn)

        self.solve_btn = QPushButton("Solve Cube")
        self.solve_btn.clicked.connect(self._solve_cube)
        l.addWidget(self.solve_btn)
        return g

    def _create_custom_moves_group(self):
        g = QGroupBox("Custom Moves")
        l = QVBoxLayout(g)

        self.sequence_input = QLineEdit()
        self.sequence_input.setPlaceholderText("Enter moves: R U R' F B2 ...")
        l.addWidget(self.sequence_input)

        seq_row = QHBoxLayout()
        self.send_sequence_btn = QPushButton("Send")
        self.send_sequence_btn.clicked.connect(self._send_custom_sequence)
        seq_row.addWidget(self.send_sequence_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_sequence)
        seq_row.addWidget(self.clear_btn)
        l.addLayout(seq_row)

        grid = QGridLayout()
        grid.setSpacing(8)
        moves = ['U', 'F', 'L', 'D', 'R', 'B']
        self.move_buttons = {}
        for i, m in enumerate(moves):
            btn = MoveButton(m)
            btn.clicked.connect(lambda checked, mv=m: self._move_button_click(mv))
            btn.setContextMenuPolicy(Qt.CustomContextMenu)
            btn.customContextMenuRequested.connect(lambda pos, mv=m: self._move_button_right_click(mv))
            grid.addWidget(btn, i//3, i%3)
            self.move_buttons[m] = btn
        l.addLayout(grid)

        quick = QHBoxLayout()
        self.prime_toggle_btn = QPushButton("Prime: OFF")
        self.prime_toggle_btn.setCheckable(True)
        self.prime_toggle_btn.toggled.connect(self._toggle_prime_mode)
        quick.addWidget(self.prime_toggle_btn)

        self.add_space_btn = QPushButton("Space")
        self.add_space_btn.clicked.connect(self._add_space)
        quick.addWidget(self.add_space_btn)
        l.addLayout(quick)
        return g

    def _create_hardware_group(self):
        g = QGroupBox("Hardware")
        l = QVBoxLayout(g)
        self.motor_test_btn = QPushButton("Test Motors")
        self.motor_test_btn.clicked.connect(self._test_motors)
        l.addWidget(self.motor_test_btn)

        self.camera_switch_btn = QPushButton("Switch Camera")
        self.camera_switch_btn.clicked.connect(self._switch_camera)
        l.addWidget(self.camera_switch_btn)

        self.arduino_connection = QPushButton("Toggle Arduino Connection")
        self.arduino_connection.clicked.connect(self._arduino_connection)
        l.addWidget(self.arduino_connection)

        self.hardware_status = QLabel("Hardware: checking...")
        self.hardware_status.setStyleSheet("padding:6px; background:#0b0b0b; color:#ddd;")
        l.addWidget(self.hardware_status)
        return g

    def _create_validation_group(self):
        g = QGroupBox("Validation")
        l = QVBoxLayout(g)
        self.validate_btn = QPushButton("Validate Cube State")
        self.validate_btn.clicked.connect(lambda: self._validate_cube_state(show_warning=True))
        l.addWidget(self.validate_btn)

        self.validation_result = QLabel("Not validated yet")
        self.validation_result.setWordWrap(True)
        self.validation_result.setStyleSheet("padding:8px; background:#0b0b0b; color:#ddd;")
        l.addWidget(self.validation_result)
        return g

    def _create_system_group(self):
        g = QGroupBox("System")
        l = QVBoxLayout(g)
        self.reset_btn = QPushButton("Reset Cube State")
        self.reset_btn.clicked.connect(self._reset_cube_state)
        l.addWidget(self.reset_btn)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)
        self.exit_btn.setStyleSheet("background:#800; color:white; font-weight:bold;")
        l.addWidget(self.exit_btn)
        return g

    def _connect_signals(self):
        self.space_shortcut = QShortcut(QKeySequence("Space"), self)
        self.space_shortcut.activated.connect(self._capture_image)

        self.sequence_input.returnPressed.connect(self._send_custom_sequence)

        for key, func in {
            'F': self._motor_f, 'Ctrl+F': self._motor_f_reverse,
            'B': self._motor_b, 'Ctrl+B': self._motor_b_reverse,
            'R': self._motor_r, 'Ctrl+R': self._motor_r_reverse,
            'L': self._motor_l, 'Ctrl+L': self._motor_l_reverse,
            'D': self._motor_d, 'Ctrl+D': self._motor_d_reverse
        }.items():
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(func)

        self.update_status_signal.connect(self._update_status)
        self.detection_finished_signal.connect(self._on_detection_finished)

    def _initialize_camera(self):
        if self.camera_controller.initialize():
            self.camera_timer = QTimer()
            self.camera_timer.timeout.connect(self._update_camera_display)
            self.camera_timer.start(33)
        else:
            self._show_warning("No camera", "Camera not available. Camera functions disabled.")

    def _initialize_hardware(self):
        if not self.motor_controller._simulation_mode and self.motor_controller.connect():
            self.hardware_status.setText("Hardware: Connected ✓")
            self.hardware_status.setStyleSheet("padding:6px; background:#003300; color:#bfffbf;")
        else:
            self.hardware_status.setText("Hardware: Disconnected (Simulation)")
            self.hardware_status.setStyleSheet("padding:6px; background:#333300; color:#fffb8f;")

    # ---------------- camera & capture ----------------
    def _update_camera_display(self):
        frame = self.camera_controller.capture_frame()
        if frame is not None:
            self.camera_widget.set_frame(frame,self.captured_images_count+1,self.new_positions)
        if self.new_positions:
            self.new_positions = False

    def _capture_image(self):
        if self.is_processing:
            return
        frame = self.camera_controller.last_frame
        if frame is None:
            self._show_warning("No Frame", "No camera frame available to capture")
            return
        self.captured_images_count += 1
        if self.captured_images_count == 1:
            cv2.imwrite(str(IMG1_PATH), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            self.thumb1.set_image(IMG1_PATH)
            self._log("Captured Image 1")
            self.update_status_signal.emit("Image 1 captured. Press Space for image 2.")
        elif self.captured_images_count == 2:
            cv2.imwrite(str(IMG2_PATH), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            self.thumb2.set_image(IMG2_PATH)
            self._log("Captured Image 2")
            self.update_status_signal.emit("Image 2 captured. Click 'Detect Cube State' to process.")
            self.captured_images_count = 0

    # ---------------- detection & solving ----------------
    def _detect_cube_state(self):
        if not IMG1_PATH.exists() or not IMG2_PATH.exists():
            self._show_warning("Images Required", "Capture both images (press Space twice) before detecting.")
            return
        self._set_processing_state(True, "Starting detection...")
        self.detection_worker = DetectionWorker(self.cube_status)
        self.detection_worker.finished.connect(self.detection_finished_signal)
        self.detection_worker.error.connect(self._on_detection_error)
        self.detection_worker.progress.connect(self.update_status_signal)
        self.detection_worker.start()

    def _open_detections_panel(self):
        dlg = CombinedDetectionDialog(self, self.cube_status)
        res = dlg.exec_()
        if res == QDialog.Accepted:
            # After user finished interactive flow, update thumbnails/cube state
            try:
                # If build_facelets_and_solve already ran inside correction, update cube_state
                if hasattr(self.cube_status, 'corrected1') and self.cube_status.corrected1:
                    # Rebuild facelets (cube_status.build_facelets_and_solve did validation already)
                    try:
                        color_str, facelet, sol = self.cube_status.build_facelets_and_solve()
                        self.cube_status.cube_state.color_status = color_str or self.cube_status.cube_state.color_status
                        self.cube_status.cube_state.face_status = facelet or self.cube_status.cube_state.face_status
                        self.cube_status.cube_state.solution = sol or self.cube_status.cube_state.solution
                    except Exception:
                        pass
                self.thumb1.set_image(IMG1_PATH); self.thumb2.set_image(IMG2_PATH)
                self._update_cube_display()
                self._log("Interactive detection completed")
            except Exception as e:
                self._show_error("Update Error", str(e))

    def _on_detection_finished(self, result: DetectionResult):
        self._set_processing_state(False, "Detection completed")
        self._log("Detection finished")
        try:
            self.new_positions = True
            self.cube_status.cube_state.color_status = result.color_str
            self.cube_status.cube_state.face_status = result.face_str
            self.cube_status.cube_state.solution = result.solution_str
            self._update_cube_display()
            if result.has_errors:
                self.validation_result.setText("Detection completed with warnings - verify colors")
                self.validation_result.setStyleSheet("color:orange; background:#332200; padding:8px;")
            else:
                self.validation_result.setText("Detection completed successfully!")
                self.validation_result.setStyleSheet("color:green; background:#002200; padding:8px;")
            self._show_info("Detection", "Detection completed. Please verify the colors.")
        except Exception as e:
            self._show_error("Detection Error", str(e))

    def _on_detection_error(self, msg: str):
        self._set_processing_state(False)
        self._show_error("Detection Error", msg)

    def _calibrate_colors(self):
        try:
            self.calibrator.run_calibration()
            self._log("Calibration finished")
        except Exception as e:
            self._show_error("Calibration Error", str(e))

    def _scramble_cube(self):
        num_moves, ok = QInputDialog.getInt(self, "Scramble", "Number of moves:", value=20, min=1, max=200)
        if not ok: return
        self._set_processing_state(True, f"Scrambling {num_moves} moves")
        try:
            moves = self.motor_controller.scramble(num_moves)
            seq = " ".join(moves)
            if self.motor_controller.send_sequence(seq):
                self.cube_status.change_status(moves)
                self._update_cube_display()
                self._log(f"Scrambled: {len(moves)} moves")
                self.update_status_signal.emit("Scramble executed")
            else:
                self._show_error("Scramble Error", "Hardware failed to execute scramble")
        except Exception as e:
            self._show_error("Scramble Error", str(e))
        finally:
            self._set_processing_state(False)

    def _solve_cube(self):
        if self.cube_status.cube_state.color_status == COLOR_INIT_STATE:
            self._show_info("Solve", "Cube already solved")
            return
        if not self._validate_cube_state(show_warning=True):
            return
        self._set_processing_state(True, "Solving...")
        try:
            ok, mens = self.cube_status.solve()
            if not ok:
                self._show_error("Solve Error", mens)
                return
            sol = self.cube_status.cube_state.solution
            self.update_status_signal.emit(f"Solution: {sol}")
            if self.motor_controller.send_sequence(sol):
                self.cube_status.change_status(sol.split())
                self._update_cube_display()
                self._log("Solution executed on hardware")
                self._show_info("Solved", f"Solution: {sol}")
            else:
                self._show_error("Execution Error", "Failed to run solution on hardware")
        except Exception as e:
            self._show_error("Solve Error", str(e))
        finally:
            self._set_processing_state(False)

    def _send_custom_sequence(self):
        moves = self.sequence_input.text()
        if not moves:
            self._show_warning("No moves", "Enter a move sequence first")
            return
        self._set_processing_state(True, "Executing custom sequence...")
        try:
            if self.motor_controller.send_sequence(moves):
                self.cube_status.change_status(moves.split())
                self._update_cube_display()
                self._log(f"Executed custom sequence: {moves}")
                self.sequence_input.clear()
            else:
                self._show_error("Execution Error", "Hardware failed to execute sequence")
        except Exception as e:
            self._show_error("Execution Error", str(e))
        finally:
            self._set_processing_state(False)

    def _move_button_click(self, move: str):
        cur = self.sequence_input.text()
        addition = move + ("'" if self.prime_mode else "")
        self.sequence_input.setText((cur + " " + addition).strip())

    def _move_button_right_click(self, move: str):
        cur = self.sequence_input.text()
        self.sequence_input.setText((cur + " " + move + "'").strip())

    def _toggle_prime_mode(self, checked: bool):
        self.prime_mode = checked
        self.prime_toggle_btn.setText("Prime: ON" if checked else "Prime: OFF")

    def _clear_sequence(self):
        self.sequence_input.clear()

    def _add_space(self):
        txt = self.sequence_input.text()
        if txt and not txt.endswith(' '):
            self.sequence_input.setText(txt + ' ')

    def _test_motors(self):
        self._set_processing_state(True, "Testing motors...")
        try:
            if self.motor_controller.test_all_motors():
                self._show_info("Motor Test", "All motors OK")
                self._log("Motor test OK")
            else:
                self._show_error("Motor Test", "Some motors failed")
        except Exception as e:
            self._show_error("Motor Test", str(e))
        finally:
            self._set_processing_state(False)

    def _arduino_connection(self):
        if not self.motor_controller._simulation_mode:
            self.motor_controller._simulation_mode = True
        else:
            if self.motor_controller.ser is not None:
                self.motor_controller._simulation_mode = False
            else:
                self.motor_controller._simulation_mode = self.motor_controller.connect()
        self._initialize_hardware()

    def _motor_f(self): self.motor_controller.send_sequence("F")
    def _motor_d(self): self.motor_controller.send_sequence("D")
    def _motor_b(self): self.motor_controller.send_sequence("B")
    def _motor_r(self): self.motor_controller.send_sequence("R")
    def _motor_l(self): self.motor_controller.send_sequence("L")
    def _motor_f_reverse(self): self.motor_controller.send_sequence("F'")
    def _motor_d_reverse(self): self.motor_controller.send_sequence("D'")
    def _motor_b_reverse(self): self.motor_controller.send_sequence("B'")
    def _motor_r_reverse(self): self.motor_controller.send_sequence("R'")
    def _motor_l_reverse(self): self.motor_controller.send_sequence("L'")

    def _switch_camera(self):
        self.update_status_signal.emit("Switching camera...")
        if self.camera_controller.switch_camera():
            self.update_status_signal.emit("Camera switched")
        else:
            self._show_warning("Camera", "No other cameras found")

    def _validate_cube_state(self, show_warning: bool = False) -> bool:
        is_valid, issue = self.cube_status.validate_state()
        if is_valid:
            self.validation_result.setText("✓ Cube state valid")
            self.validation_result.setStyleSheet("color:green; background:#002200; padding:8px;")
            return True
        else:
            message = "Cube state invalid:\n" + str(issue)
            self.validation_result.setText(message)
            self.validation_result.setStyleSheet("color:red; background:#220000; padding:8px;")
            if show_warning:
                reply = QMessageBox.question(self, "Invalid", f"{message}\n\nContinue anyway?", QMessageBox.Yes | QMessageBox.No)
                return reply == QMessageBox.Yes
            return False

    def _reset_cube_state(self):
        reply = QMessageBox.question(self, "Reset", "Reset cube state?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.cube_status.cube_state.reset_state()
            self._update_cube_display()
            self.validation_result.setText("Cube state reset")
            self._log("Cube reset to default")

    def _update_cube_display(self):
        self.cube_widget.set_state(self.cube_status.cube_state.color_status)

    def _update_status(self, message: str):
        self.statusBar().showMessage(message)
        self.status_message.setText(message)
        self._log(message)

    def _set_processing_state(self, processing: bool, message: str = ""):
        self.is_processing = processing
        controls = [
            self.solve_btn, self.scramble_btn, self.solve_btn, self.motor_test_btn,
            self.camera_switch_btn, self.validate_btn, self.reset_btn, self.send_sequence_btn,
            self.prime_toggle_btn, self.clear_btn, self.add_space_btn
        ]
        for c in controls:
            c.setEnabled(not processing)
        for b in self.move_buttons.values():
            b.setEnabled(not processing)
        if processing:
            self.progress_bar.show()
        else:
            self.progress_bar.hide()
        if message:
            self.update_status_signal.emit(message)

    def _log(self, message: str):
        ts = time.strftime('%H:%M:%S')
        self.log_view.append(f"[{ts}] {message}")

    def _show_info(self, title: str, message: str):
        QMessageBox.information(self, title, message)

    def _show_warning(self, title: str, message: str):
        QMessageBox.warning(self, title, message)

    def _show_error(self, title: str, message: str):
        QMessageBox.critical(self, title, message)

    def _show_about(self):
        QMessageBox.information(self, "About", "Rubik's Cube Solver — Unified UI\nVersion: 3.0")

    def _toggle_theme(self):
        self._show_info("Theme", "Theme toggling placeholder.")

    def _on_sticker_clicked(self, side: str, sticker_pos: str):
        dialog = ColorDialog(self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_color:
            try:
                self.cube_status.update_sticker(side, sticker_pos, dialog.selected_color)
                self._update_cube_display()
                self._log(f"Sticker {sticker_pos} set to {dialog.selected_color}")
            except Exception as e:
                self._show_error("Color Update", str(e))

    def closeEvent(self, event):
        if self.detection_worker and self.detection_worker.isRunning():
            self.detection_worker.stop()
            self.detection_worker.wait(2000)
        self.camera_controller.release()
        try:
            self.motor_controller.disconnect()
        except Exception:
            pass
        event.accept()

# ---------------- main ----------------
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("Rubik's Cube Solver")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
