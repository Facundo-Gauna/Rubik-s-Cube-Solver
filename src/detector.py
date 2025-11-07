#!/usr/bin/env python3
"""
manual_detector2.py
Single-file replacement preserving your parallel pipeline but fixing logic errors,
using calibration medians to map clusters -> color, and adding diagnostics + solver flow.

Expect to be used inside your project where:
 - config.* constants exist (IMG1_PATH, IMG2_PATH, POLYGON_POSITIONS_PATH_1, POLYGON_POSITIONS_PATH_2, ...)
 - calibrator.load_calibration() and calibrator.get_hsv_ranges() exist and return structures
 - ThreadSyncManager is available in config and provides events: windows_ready, ready_to_finish,
   and attributes: lock, finish_requested, exception
"""

import json
import math
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    import kociemba
except Exception:
    kociemba = None

# optional local robust face-cube verification (if you already have it)
try:
    import kociemba_module
except Exception:
    kociemba_modul = None

try:
    from scipy.spatial import distance
except Exception:
    distance = None

# Project imports (your code provided these names earlier)
from calibrator import load_calibration
from config import (
    CENTER_INDICES, CENTER_LABELS, COLOR_ORDER, FACE_ORDER, FACE_TO_COLOR, COLOR_TO_FACE,
    IMG1_PATH, IMG2_PATH, KEY_WAIT_TIME, POLYGON_POSITIONS_PATH_1,
    POLYGON_POSITIONS_PATH_2, CALIBRATIONS_PATH, THREAD_TIMEOUT, ThreadSyncManager
)

# ----- Constants (kept from your file) -----
WINDOW_TITLE_PREFIX = "Cube Detector"
CIRCLE_RADIUS = 20
SELECTED_CIRCLE_RADIUS = 28
CIRCLE_COLOR = (0, 200, 0)
SELECTED_CIRCLE_COLOR = (0, 255, 255)
BORDER_COLOR = (0, 0, 0)
TEXT_COLOR = (0, 0, 0)

ROI_SIZE_MIN = 5
ROI_SIZE_MAX = 10
ROI_DISTANCE_FACTOR = 0.45
CLUSTER_COUNT = 6
KMEANS_RANDOM_STATE = 0
KMEANS_N_INIT = 10

COLOR_MAP_BGR = {
    'R': (0, 0, 255),
    'O': (0, 165, 255),
    'Y': (0, 255, 255),
    'G': (0, 255, 0),
    'B': (255, 0, 0),
    'W': (255, 255, 255),
    'unknown': (128, 128, 128)
}

DEFAULT_HSV_RANGES = {
    'R': [((0, 80, 50), (10, 255, 255)), ((170, 80, 50), (179, 255, 255))],
    'O': [((10, 80, 50), (25, 255, 255))],
    'Y': [((25, 80, 50), (35, 255, 255))],
    'G': [((36, 40, 40), (85, 255, 255))],
    'B': [((90, 40, 40), (140, 255, 255))],
    'W': [((0, 0, 180), (179, 80, 255))]
}

COLOR_NAMES = ['R', 'O', 'Y', 'G', 'B', 'W']
EXPECTED_COLORS = ['R','O','Y','G','B','W']

class PolygonDetector:
    """
    Define and save positions for an image. Each image holds exactly 27 positions (3 faces x 9 labels).
    faces_for_image is a list like ['U','F','L'] or ['D','R','B'].
    """

    def __init__(self, faces_for_image: List[str], save_file: Path):
        self.faces_for_image = faces_for_image
        self.save_file = save_file
        self.img = None
        self.labels = [f + str(i) for f in self.faces_for_image for i in range(1, 10)]
        self.positions: Dict[str, Tuple[int, int]] = {}
        self.selected_point: Optional[str] = None
        self.dragging = False
        self.drag_offset = (0, 0)

    def _create_default_positions_for(self, img_shape) -> Dict[str, Tuple[int, int]]:
        h, w = img_shape[:2]
        rows, cols = 3, 9
        positions = {}
        for i, label in enumerate(self.labels):
            row = i // cols
            col = i % cols
            x = int((col + 0.5) * (w / cols))
            y = int((row + 0.5) * (h / rows))
            positions[label] = (x, y)
        return positions

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._handle_mouse_down(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.selected_point:
            self._handle_mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def _handle_mouse_down(self, x: int, y: int):
        closest = None
        min_d = float('inf')
        for label, (px, py) in self.positions.items():
            d = math.hypot(x - px, y - py)
            if d < min_d and d < 40:
                min_d = d
                closest = label
        if closest:
            self.selected_point = closest
            self.dragging = True
            px, py = self.positions[closest]
            self.drag_offset = (x - px, y - py)

    def _handle_mouse_move(self, x: int, y: int):
        nx = max(0, min(self.img.shape[1] - 1, x - self.drag_offset[0]))
        ny = max(0, min(self.img.shape[0] - 1, y - self.drag_offset[1]))
        self.positions[self.selected_point] = (int(nx), int(ny))

    def _draw_interface(self) -> np.ndarray:
        disp = self.img.copy()
        for label in self.labels:
            x, y = self.positions[label]
            is_selected = label == self.selected_point
            color = SELECTED_CIRCLE_COLOR if is_selected else CIRCLE_COLOR
            radius = SELECTED_CIRCLE_RADIUS if is_selected else CIRCLE_RADIUS
            cv2.circle(disp, (int(x), int(y)), radius, color, -1)
            cv2.circle(disp, (int(x), int(y)), radius, BORDER_COLOR, 2)
            cv2.putText(disp, label, (int(x) + 36, int(y) + 6), cv2.FONT_HERSHEY_DUPLEX, 1, TEXT_COLOR, 1)
        return disp

    def _load_saved_positions(self) -> Dict:
        if not self.save_file.exists():
            return {}
        try:
            with open(self.save_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load saved positions: {e}")
            return {}

    def _save_positions_to_disk(self):
        data = self._load_saved_positions()
        data['positions'] = {k: [int(v[0]), int(v[1])] for k, v in self.positions.items()}
        try:
            with open(self.save_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Template saved to {self.save_file}")
        except Exception as e:
            print(f"Error saving positions: {e}")

    def setup_image(self, image_path: Path) -> Dict[str, Tuple[int, int]]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")
        self.img = img
        saved = self._load_saved_positions()
        default_positions = self._create_default_positions_for(img.shape)
        self.positions = {k: tuple(v) for k, v in saved.get('positions', {}).items() if k in self.labels}
        for k, v in default_positions.items():
            if k not in self.positions:
                self.positions[k] = v
        return {k: tuple(v) for k, v in self.positions.items()}


class SynchronizedPolygonDetector:
    """
    Manage two PolygonDetectors in parallel and coordinate finish via ThreadSyncManager.
    First image faces: ['U','F','L']
    Second image faces: ['D','R','B']
    """

    def __init__(self):
        self.image1_path = IMG1_PATH
        self.image2_path = IMG2_PATH
        self.faces_image1 = ['U', 'F', 'L']   # enforced
        self.faces_image2 = ['D', 'R', 'B']   # enforced
        self.save_file1 = POLYGON_POSITIONS_PATH_1
        self.save_file2 = POLYGON_POSITIONS_PATH_2

        self.positions1: Optional[Dict[str, Tuple[int, int]]] = None
        self.positions2: Optional[Dict[str, Tuple[int, int]]] = None
        self.sync_manager = ThreadSyncManager()
        # counting local readiness for both windows
        self._window_ready_count = 0
        self._window_ready_lock = threading.Lock()

    def define_positions_both(self) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]]:
        thread1 = threading.Thread(target=self._define_image, args=(1,))
        thread2 = threading.Thread(target=self._define_image, args=(2,))
        thread1.start()
        thread2.start()

        # Wait for both windows to be ready
        # The worker threads will increment _window_ready_count and set sync_manager.windows_ready
        start = time.time()
        while True:
            with self._window_ready_lock:
                if self._window_ready_count >= 2:
                    # mark windows ready in ThreadSyncManager
                    try:
                        self.sync_manager.windows_ready.set()
                    except Exception:
                        pass
                    break
            if time.time() - start > THREAD_TIMEOUT:
                break
            time.sleep(0.05)

        # Wait until a finish request (ENTER pressed in either window)
        self.sync_manager.ready_to_finish.wait()
        with self.sync_manager.lock:
            self.sync_manager.finish_requested = True

        thread1.join()
        thread2.join()

        if self.sync_manager.exception:
            raise self.sync_manager.exception

        return self.positions1, self.positions2

    def _define_image(self, image_num: int):
        try:
            if image_num == 1:
                polygon_detector = PolygonDetector(self.faces_image1, self.save_file1)
                image_path = self.image1_path
                window_suffix = "Image 1"
                positions_attr = "positions1"
            else:
                polygon_detector = PolygonDetector(self.faces_image2, self.save_file2)
                image_path = self.image2_path
                window_suffix = "Image 2"
                positions_attr = "positions2"

            polygon_detector.setup_image(image_path)
            default_positions = polygon_detector._create_default_positions_for(polygon_detector.img.shape)

            window_name = f"{WINDOW_TITLE_PREFIX} - {window_suffix} - {Path(image_path).name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, polygon_detector._mouse_callback)

            # increment ready count
            with self._window_ready_lock:
                self._window_ready_count += 1

            print(f"{window_suffix} window ready: {window_name}")

            # main loop
            while True:
                display = polygon_detector._draw_interface()
                cv2.imshow(window_name, display)

                # if finish requested by other window, exit too
                with self.sync_manager.lock:
                    if self.sync_manager.finish_requested:
                        break

                key = cv2.waitKey(KEY_WAIT_TIME) & 0xFF

                if key == 13:  # ENTER
                    print(f"ENTER pressed in {window_suffix} - signaling ready_to_finish")
                    with self.sync_manager.lock:
                        if not self.sync_manager.finish_requested:
                            self.sync_manager.finish_requested = True
                        try:
                            self.sync_manager.ready_to_finish.set()
                        except Exception:
                            pass
                    break
                elif key == 27:  # ESC
                    print(f"ESC pressed in {window_suffix} - canceling")
                    self.sync_manager.exception = Exception("User cancelled polygon definition")
                    try:
                        self.sync_manager.ready_to_finish.set()
                    except Exception:
                        pass
                    break
                elif key == ord('r'):
                    polygon_detector.positions = default_positions.copy()

            # if finished normally, save positions and attach to parent
            if not self.sync_manager.exception:
                polygon_detector._save_positions_to_disk()
                setattr(self, positions_attr, {k: tuple(v) for k, v in polygon_detector.positions.items()})

            cv2.destroyWindow(window_name)
        except Exception as e:
            self.sync_manager.exception = e
            try:
                self.sync_manager.ready_to_finish.set()
            except Exception:
                pass


class ColorDetector:
    """Detects colors from image positions using clustering + calibration medians."""

    def __init__(self):
        # Load calibration once
        self.calibration = load_calibration() or {}
        # If calibration provides hsv_ranges, normalize them; otherwise use defaults
        cfg_ranges = None
        try:
            cfg_ranges = self.calibration.get('hsv_ranges')
        except Exception:
            cfg_ranges = None
        if cfg_ranges:
            self.hsv_ranges = self.normalize_ranges(cfg_ranges)
        else:
            self.hsv_ranges = DEFAULT_HSV_RANGES
        # hold calibration stats for mapping clusters -> color
        self.calib_stats = self.calibration.get('color_statistics', {}) if self.calibration else {}

    def detect_single_image(self, image_path: Path, positions: Dict[str, Tuple[int, int]]):
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")

        roi_size = self._compute_optimal_roi_size(positions, img.shape)
        labels, features, hsv_values, lab_values = self._extract_features(img, positions, roi_size)

        if len(features) < 6:
            print(f"Warning: only {len(features)} samples found in {Path(image_path).name}")
            return self._create_unknown_results(positions)

        cluster_labels = self._cluster_features(features)

        # compute cluster center HSV medians
        cl_points = defaultdict(list)
        for i, cid in enumerate(cluster_labels):
            cl_points[cid].append(hsv_values[i])

        centers = {cid: np.median(np.array(pts), axis=0).tolist() for cid, pts in cl_points.items()}

        # map clusters -> colors with calibration-first approach
        cluster_colors = self._assign_colors_to_clusters(centers, cluster_labels)

        detected, detected_lab = self._create_detection_results(labels, cluster_labels, cluster_colors, lab_values, positions)

        detected_count = sum(1 for v in detected.values() if v != 'unknown')
        print(f"Detected {detected_count}/{len(detected)} on {Path(image_path).name}")
        return detected, detected_lab

    def _extract_features(self, img: np.ndarray, positions: Dict[str, Tuple[int, int]], roi_size: int):
        labels, features, hsv_values, lab_values = [], [], [], []
        for label, (x, y) in positions.items():
            x, y = int(x), int(y)
            roi = self._extract_roi(img, x, y, roi_size)
            if roi is None or roi.size == 0:
                continue
            dominant_bgr = self._get_robust_dominant_color(roi)
            if dominant_bgr is None:
                continue
            hsv = cv2.cvtColor(np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2HSV)[0][0].tolist()
            lab = cv2.cvtColor(np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2LAB)[0][0].tolist()
            features.append([float(lab[1]), float(lab[2])])  # a,b
            hsv_values.append(hsv)
            lab_values.append(tuple(int(v) for v in lab))
            labels.append(label)
        return labels, features, hsv_values, lab_values

    def _extract_roi(self, img: np.ndarray, x: int, y: int, size: int):
        r = size // 2
        h, w = img.shape[:2]
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        roi = img[y0:y1, x0:x1]
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        return roi

    def _get_robust_dominant_color(self, roi: np.ndarray):
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:,:,2]
        mask_non_highlight = v_channel < 245  # Remove very bright pixels (highlights)

        # Apply the same mask to both BGR and LAB spaces
        roi_bgr = roi.reshape(-1, 3)
        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).reshape(-1, 3)

        mask_flat = mask_non_highlight.reshape(-1)

        # Filter out highlighted pixels
        roi_bgr_filtered = roi_bgr[mask_flat]
        roi_lab_filtered = roi_lab[mask_flat]

        # If all pixels were highlights, use the original
        if len(roi_bgr_filtered) == 0:
            roi_bgr_filtered = roi_bgr
            roi_lab_filtered = roi_lab

        # Now use LAB for outlier detection on the non-highlight pixels
        pixels_lab = roi_lab_filtered.astype(float)

        if len(pixels_lab) == 0:
            # Fallback: median of original ROI in BGR
            med = np.median(roi_bgr, axis=0).astype(np.uint8)
            return tuple(med)

        # IQR outlier removal in LAB space
        Q1 = np.percentile(pixels_lab, 25, axis=0)
        Q3 = np.percentile(pixels_lab, 75, axis=0)
        IQR = Q3 - Q1

        # Avoid division by zero for constant color regions
        IQR = np.where(IQR == 0, 1, IQR)

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        mask = np.all((pixels_lab >= lower) & (pixels_lab <= upper), axis=1)
        filt_lab = pixels_lab[mask]

        if len(filt_lab) > 0:
            # Use median of inliers in LAB space, then convert to BGR
            med_lab = np.median(filt_lab, axis=0).astype(np.uint8)
            bgr = cv2.cvtColor(np.uint8([[med_lab]]), cv2.COLOR_LAB2BGR)[0][0]
            return tuple(bgr)
        else:
            # Fallback: median of non-highlight pixels in BGR
            med_bgr = np.median(roi_bgr_filtered, axis=0).astype(np.uint8)
            return tuple(med_bgr)

    def _cluster_features(self, features: List[List[float]]):
        arr = np.array(features)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(arr)
        kmeans = KMeans(n_clusters=CLUSTER_COUNT, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
        return kmeans.fit_predict(scaled)

    def normalize_ranges(self, hsv_ranges):
        out = {}
        for color, ranges in hsv_ranges.items():
            out[color] = []
            for low, high in ranges:
                low = list(low); high = list(high)
                if low[1] > high[1]:
                    low[1], high[1] = high[1], low[1]
                if low[2] > high[2]:
                    low[2], high[2] = high[2], low[2]
                if low[0] <= high[0]:
                    out[color].append((tuple(low), tuple(high)))
                else:
                    out[color].append((tuple(low), (179, high[1], high[2])))
                    out[color].append(((0, low[1], low[2]), tuple(high)))
        return out

    # ----- improved mapping function -----
    def _assign_colors_to_clusters(self, hsv_centers_of_clusters: Dict[int, List[float]], cluster_labels=None) -> Dict[int, str]:
        """
        Map each cluster center (HSV) to a color letter.
        Strategy:
          1. If calibration medians exist -> greedily match clusters to nearest calib medians (circular hue distance).
          2. Then attempt to use hsv_ranges to verify/complete assignments.
        """
        cluster_colors = {}

        # Build calib medians
        calib_centers = {}
        for color, info in (self.calib_stats or {}).items():
            hm = info.get('hsv_median')
            if hm:
                calib_centers[color] = hm

        # If calibration exists, perform greedy matching by distance
        if calib_centers:
            # create sorted list of (dist, cluster, color)
            trip = []
            for cid, center in hsv_centers_of_clusters.items():
                for color, cm in calib_centers.items():
                    d = self.hsv_circular_distance(center, cm)
                    trip.append((d, cid, color))
            trip.sort(key=lambda x: x[0])
            used_colors = set()
            used_clusters = set()
            for d, cid, color in trip:
                if cid in used_clusters or color in used_colors:
                    continue
                cluster_colors[cid] = color
                used_clusters.add(cid)
                used_colors.add(color)

        # For any unassigned cluster, try to match using hsv_ranges (fallback)
        # Build center map -> color using _get_color_from_hsv
        for cid, center in hsv_centers_of_clusters.items():
            if cid in cluster_colors:
                continue
            color_guess = self._get_color_from_hsv(center)
            cluster_colors[cid] = color_guess

        # Final check: ensure each cluster has a color (if missing, assign arbitrary)
        all_colors = list(self.hsv_ranges.keys())
        for i, cid in enumerate(sorted(hsv_centers_of_clusters.keys())):
            if cid not in cluster_colors:
                cluster_colors[cid] = all_colors[i % len(all_colors)]

        return cluster_colors

    def _get_color_from_hsv(self, hsv):
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        # white check
        wl, wh = self.hsv_ranges['W'][0]
        if wl[0] <= h <= wh[0] and wl[1] <= s <= wh[1] and wl[2] <= v <= wh[2]:
            return 'W'
        # other colors
        for cname, ranges in self.hsv_ranges.items():
            if cname == 'W':
                continue
            for low, high in ranges:
                if low[0] <= h <= high[0] and low[1] <= s <= high[1] and low[2] <= v <= high[2]:
                    return cname
        return 'unknown'

    def hsv_circular_distance(self, hsv1, hsv2, wh=(1.0, 0.8, 0.4)):
        dh = min(abs(int(hsv1[0]) - int(hsv2[0])), 180 - abs(int(hsv1[0]) - int(hsv2[0])))
        ds = abs(int(hsv1[1]) - int(hsv2[1]))
        dv = abs(int(hsv1[2]) - int(hsv2[2]))
        h_score = (dh / 90.0)  # normalized
        s_score = (ds / 255.0)
        v_score = (dv / 255.0)
        w_h, w_s, w_v = wh
        return math.sqrt(w_h * h_score * h_score + w_s * s_score * s_score + w_v * v_score * v_score)

    def _compute_optimal_roi_size(self, positions: Dict[str, Tuple[int, int]], img_shape) -> int:
        pts = np.array(list(positions.values()))
        if pts.shape[0] < 2 or distance is None:
            return 20
        dist = distance.cdist(pts, pts)
        np.fill_diagonal(dist, np.inf)
        min_d = float(np.min(dist))
        size = max(ROI_SIZE_MIN, min(int(min_d * ROI_DISTANCE_FACTOR), ROI_SIZE_MAX))
        return size

    def _create_unknown_results(self, positions: Dict[str, Tuple[int, int]]):
        return {k: 'unknown' for k in positions.keys()}, {k: (0, 0, 0) for k in positions.keys()}

    def _create_detection_results(self, labels, cluster_labels, cluster_colors, lab_values, positions):
        detected = {lab: cluster_colors.get(int(cid), 'unknown') for lab, cid in zip(labels, cluster_labels)}
        detected_lab = {lab: labv for lab, labv in zip(labels, lab_values)}
        for k in positions.keys():
            if k not in detected:
                detected[k] = 'unknown'
            if k not in detected_lab:
                detected_lab[k] = (0, 0, 0)
        return detected, detected_lab

def combine_corrected_maps(map1, map2):
    """Devuelve dict combinado label->color y listado ordenado de labels"""
    combined = {}
    if map1:
        combined.update(map1)
    if map2:
        combined.update(map2)
    return combined

def count_colors(combined_map):
    """Devuelve Counter de colores y mapping color->list(labels)"""
    cnt = Counter()
    buckets = {c: [] for c in EXPECTED_COLORS}
    for lbl, col in combined_map.items():
        if col in EXPECTED_COLORS:
            cnt[col] += 1
            buckets[col].append(lbl)
        else:
            # consider 'unknown' or other tokens as not counted
            pass
    return cnt, buckets

def find_center_issues(combined_map):
    """Comprueba presencia/duplicados en labels de centros U5..B5"""
    issues = []
    centers_assigned = {}
    for cen in CENTER_LABELS:
        col = combined_map.get(cen)
        if col is None or col == 'unknown':
            issues.append(f"Center {cen} no asignado")
        else:
            if col in centers_assigned:
                issues.append(f"Centro duplicado: color {col} en {centers_assigned[col]} y {cen}")
            else:
                centers_assigned[col] = cen
    return issues

def build_diag_message(cnt, buckets):
    lines = []
    for c in EXPECTED_COLORS:
        lines.append(f"{c}: {cnt.get(c,0)}")
    return ' | '.join(lines)

class SynchronizedColorCorrectionUI:
    """Run two color correction windows in parallel; assignments are returned."""

    def __init__(self,positions1: Dict[str, Tuple[int, int]], positions2: Dict[str, Tuple[int, int]],
                 detections1: Dict[str, str], labs1: Dict[str, Tuple[int, int, int]],
                 detections2: Dict[str, str], labs2: Dict[str, Tuple[int, int, int]]):
        self.image_paths = [IMG1_PATH, IMG2_PATH]
        self.positions = [positions1, positions2]
        self.detections = [detections1, detections2]
        self.labs = [labs1, labs2]
        self.corrected = [None, None]
        self.corrected_labs = [None, None]
        self.selected = [None, None]
        self.sync_manager = ThreadSyncManager()
        self._window_ready_count = 0
        self._window_ready_lock = threading.Lock()
        self.color_centers = {}

    def run_both(self):
        while True:
            thread1 = threading.Thread(target=self._run_single_ui, args=(0,))
            thread2 = threading.Thread(target=self._run_single_ui, args=(1,))
            thread1.start()
            thread2.start()

            # wait until both windows ready
            start = time.time()
            while True:
                with self._window_ready_lock:
                    if self._window_ready_count >= 2:
                        try:
                            self.sync_manager.windows_ready.set()
                        except Exception:
                            pass
                        break
                if time.time() - start > THREAD_TIMEOUT:
                    break
                time.sleep(0.05)

            print("\n=== BOTH COLOR CORRECTION WINDOWS ARE READY ===")
            print("Instructions:")
            print("- Click stickers to select, then press 1-6 to assign colors")
            print("- 1=Red, 2=Orange, 3=Yellow, 4=Green, 5=Blue, 6=White")
            print("- Press ENTER in EITHER window when BOTH are ready")
            print("- Both windows close when ENTER is pressed in either one\n")

            # wait finish
            self.sync_manager.ready_to_finish.wait()
            with self.sync_manager.lock:
                self.sync_manager.finish_requested = True

            thread1.join()
            thread2.join()

            ok, diag = self._valid_detections()
            self._last_diag = diag  # para pintar en UI
            if ok:
                break  # accept and return
            elif self.sync_manager.exception: 
                raise self.sync_manager.exception
            else:
                # notify user via console and keep loop: reopen windows for correction
                print("\nValidation failed: ", diag.get('message_long'))
                print("Please correct the highlighted stickers and press ENTER again.")
                # reset sync_manager for next iteration:
                self.sync_manager = ThreadSyncManager()
                # reset windows ready counter so windows will be recreated
                with self._window_ready_lock:
                    self._window_ready_count = 0

        return (self.corrected[0], self.corrected_labs[0], self.corrected[1], self.corrected_labs[1])

    def _valid_detections(self):
        if self.corrected[0] is None or self.corrected[1] is None:
            return False, {"reason":"corrections missing"}

        combined = combine_corrected_maps(self.corrected[0], self.corrected[1])
        cnt, buckets = count_colors(combined)

        diag = {}
        diag['counts'] = cnt
        diag['buckets'] = buckets

        center_issues = find_center_issues(combined)
        diag['center_issues'] = center_issues

        missing = []
        extras = []
        for c in EXPECTED_COLORS:
            n = cnt.get(c,0)
            if n < 9:
                missing.append((c, 9-n))
            elif n > 9:
                extras.append((c, n-9))
        diag['missing'] = missing
        diag['extras'] = extras

        highlight_missing = []   # labels that are unassigned or 'unknown'
        highlight_extra = []     # labels assigned to a color that is in excess (candidates to change)

        for lbl, col in combined.items():
            if col not in EXPECTED_COLORS:
                highlight_missing.append(lbl)

        for color, overcount in extras:
            assigned = list(buckets.get(color, []))
            non_centers = [l for l in assigned if l not in CENTER_LABELS]
            pick_from = non_centers if len(non_centers) >= overcount else assigned
            highlight_extra.extend(pick_from[:overcount])

        diag['highlight_missing'] = highlight_missing
        diag['highlight_extra'] = highlight_extra

        # ok if no missing/extras and centers ok
        ok = (len(missing) == 0 and len(extras) == 0 and len(center_issues) == 0)
        if not ok:
            diag['message_short'] = "Faltan/Exceden stickers o centros duplicados"
            diag['message_long'] = (
                f"Conteo: {build_diag_message(cnt, buckets)}; "
                f"Missing: {missing}; Extras: {extras}; Centers: {center_issues}"
            )
        else:
            diag['message_short'] = "OK"
            diag['message_long'] = "Todos los colores aparecen 9 veces y centros válidos."

        return ok, diag

    def _run_single_ui(self, window_index: int):
        try:
            img = cv2.imread(str(self.image_paths[window_index]))
            if img is None:
                raise FileNotFoundError(f"Cannot open image: {self.image_paths[window_index]}")
            window_name = f"Color Correction - Image {window_index + 1} - {Path(self.image_paths[window_index]).name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, lambda e, x, y, f, p: self._mouse_cb(e, x, y, f, p, window_index))
            with self._window_ready_lock:
                self._window_ready_count += 1
            # initialize corrected storage
            self.corrected[window_index] = self.detections[window_index].copy()
            self.corrected_labs[window_index] = self.labs[window_index].copy()
            print(f"Color Correction Window {window_index + 1} ready: {window_name}")

            while True:
                display = self._draw_interface(img, window_index)
                cv2.imshow(window_name, display)
                with self.sync_manager.lock:
                    if self.sync_manager.finish_requested:
                        break
                key = cv2.waitKey(KEY_WAIT_TIME) & 0xFF
                if key == 13:
                    print(f"ENTER in Color Correction window {window_index+1}")
                    with self.sync_manager.lock:
                        if not self.sync_manager.finish_requested:
                            self.sync_manager.finish_requested = True
                            try:
                                self.sync_manager.ready_to_finish.set()
                            except Exception:
                                pass
                    break
                elif key == 27:
                    print(f"ESC in Color Correction window {window_index+1}")
                    self.sync_manager.exception = Exception("User cancelled color correction")
                    try:
                        self.sync_manager.ready_to_finish.set()
                    except Exception:
                        pass
                    break
                elif ord('1') <= key <= ord('6') and self.selected[window_index]:
                    idx = key - ord('1')
                    color = COLOR_NAMES[idx]
                    selected_label = self.selected[window_index]
                    self.corrected[window_index][selected_label] = color
                    # if there is a lab sample, store center
                    if self.corrected_labs[window_index].get(selected_label):
                        self.color_centers[color] = self.corrected_labs[window_index][selected_label]
                    print(f"Set {selected_label} -> {color} (Image {window_index+1})")
            cv2.destroyWindow(window_name)
        except Exception as e:
            self.sync_manager.exception = e
            try:
                self.sync_manager.ready_to_finish.set()
            except Exception:
                pass

    def _mouse_cb(self, event, x, y, flags, param, window_index: int):
        if event == cv2.EVENT_LBUTTONDOWN:
            min_d = float('inf'); closest = None
            for label, (px, py) in self.positions[window_index].items():
                d = math.hypot(x - px, y - py)
                if d < min_d and d < 40:
                    min_d = d; closest = label
            if closest:
                self.selected[window_index] = closest
                print(f"Selected {closest} on Image {window_index + 1}")

    def _draw_interface(self, img: np.ndarray, window_index: int) -> np.ndarray:
        disp = img.copy()
        positions = self.positions[window_index]
        detections = self.corrected[window_index]
        selected = self.selected[window_index]
        for label, (x, y) in positions.items():
            color_name = detections.get(label, 'unknown')
            color_bgr = COLOR_MAP_BGR.get(color_name, (128, 128, 128))
            cv2.circle(disp, (int(x), int(y)), 28, color_bgr, -1)
            cv2.circle(disp, (int(x), int(y)), 28, BORDER_COLOR, 2)
            if label == selected:
                cv2.circle(disp, (int(x), int(y)), 36, (255, 255, 0), 3)
            cv2.putText(disp, f"{label}:{color_name}", (int(x) + 16, int(y) + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)
        # instructions
        lines = [
            f"Image {window_index + 1} - Click sticker, press 1-6 to assign color",
            "1=Red, 2=Orange, 3=Yellow, 4=Green, 5=Blue, 6=White",
            "Press ENTER when BOTH images are ready"
        ]
        for i, ln in enumerate(lines):
            cv2.putText(disp, ln, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
                # show diagnostics if available

        if hasattr(self, '_last_diag') and self._last_diag:
            diag = self._last_diag
            # print short message
            cv2.putText(disp, diag.get('message_short',''), (10, disp.shape[0]-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            # highlight missing in red border
            for lbl in diag.get('highlight_missing', []):
                if lbl in positions:
                    x,y = positions[lbl]
                    cv2.circle(disp, (int(x),int(y)), 20, (0,0,255), 3)
            # highlight extras in orange
            for lbl in diag.get('highlight_extra', []):
                if lbl in positions:
                    x,y = positions[lbl]
                    cv2.circle(disp, (int(x),int(y)), 20, (0,140,255), 3)
        return disp


ROT90 = [6, 3, 0, 7, 4, 1, 8, 5, 2]


def rotate_block_list(vals9: List, times: int) -> List:
    out = list(vals9)
    for _ in range(times % 4):
        out = [out[i] for i in ROT90]
    return out


def build_color_net_text(color_str: str) -> str:
    faces = ['U', 'R', 'F', 'D', 'L', 'B']
    out = []
    for fi, face in enumerate(faces):
        out.append(f"\n{face}:")
        block = color_str[fi * 9:(fi + 1) * 9]
        for r in range(3):
            out.append(' '.join(block[r * 3:(r + 1) * 3]))
    return '\n'.join(out)


def brute_force_face_orientations(face_color_string: str):
    if kociemba is None:
        print("kociemba not installed; cannot brute force.")
        return None, None
    blocks = [list(face_color_string[i * 9:(i + 1) * 9]) for i in range(6)]
    import itertools
    t0 = time.time()
    print("Brute-force: trying all face rotations (4^6)...")
    for comb in itertools.product(range(4), repeat=6):
        candidate_blocks = [''.join(rotate_block_list(b, comb[i])) for i, b in enumerate(blocks)]
        candidate_color_str = ''.join(candidate_blocks)
        color_to_face = {}
        ok = True
        for face_letter, idx in CENTER_INDICES.items():
            c = candidate_color_str[idx]
            if c in color_to_face:
                ok = False
                break
            color_to_face[c] = face_letter
        if not ok:
            continue
        candidate_facelets = ''.join(color_to_face[c] for c in candidate_color_str)
        try:
            sol = kociemba.solve(candidate_facelets)
            print(f"Found valid orientation (brute-force) after {time.time() - t0:.2f}s.")
            return candidate_facelets, sol
        except Exception:
            continue
    print("Brute-force exhausted; no valid orientation found.")
    return None, None


class ParallelCubeDetector:
    def __init__(self):
        self.detections1 = None
        self.labs1 = None
        self.detections2 = None
        self.labs2 = None

        self.corrected1 = None
        self.corrected2 = None

    def detect_both_images(self):
        print("=== STEP 1: Defining positions for both images ===")
        sync_detector = SynchronizedPolygonDetector()
        positions1, positions2 = sync_detector.define_positions_both()
        if not positions1 or not positions2:
            raise Exception("Failed to get positions for both images")

        print("\n=== STEP 2: Detecting colors in parallel ===")
        detector = ColorDetector()
        def job1():
            self.detections1, self.labs1 = detector.detect_single_image(IMG1_PATH, positions1)
            print("✓ Image 1 colors detected")

        def job2():
            self.detections2, self.labs2 = detector.detect_single_image(IMG2_PATH, positions2)
            print("✓ Image 2 colors detected")

        t1 = threading.Thread(target=job1)
        t2 = threading.Thread(target=job2)
        t1.start(); t2.start()
        t1.join(); t2.join()

        print("✓ Both image colors detected successfully!")
        
        print("\n=== STEP 3: Color Correction for both images ===")
        if not (positions1 and self.detections1 and self.labs1 and positions2 and self.detections2 and self.labs2):
            raise Exception("Must run detection before color correction")

        sync_correction = SynchronizedColorCorrectionUI(
            IMG1_PATH, IMG2_PATH,
            positions1, positions2,
            self.detections1, self.labs1,
            self.detections2, self.labs2
        )

        self.corrected1, corrected_labs1, self.corrected2, corrected_labs2 = sync_correction.run_both()
        print("✓ Both image colors corrected successfully!")



def main():
    print("=== CUBE DETECTION PIPELINE STARTED ===")
    try:
        parallel_detector = ParallelCubeDetector()
        parallel_detector.detect_both_images()

        color_str, facelet_str, solution = parallel_detector.build_facelets_and_solve()
        print("\n=== PROCESS COMPLETE ===")
        print("Color string:", color_str)
        print("Facelet string:", facelet_str)
        if solution:
            print("Solution:", solution)
        else:
            print("No solution (kociemba not installed) - inspect facelet string above")

        return True
    except Exception as e:
        print(f"\n=== ERROR: {e} ===")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    ok = main()
    exit(0 if ok else 1)
