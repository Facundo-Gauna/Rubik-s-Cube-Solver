from __future__ import annotations

import json
import math
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np

from config import IMG1_PATH,IMG2_PATH,POLYGON_POSITIONS_PATH_1,POLYGON_POSITIONS_PATH_2


def bgr2lab(inputColor: Tuple[int, int, int]) -> List[float]:
    r = inputColor[2] / 255.0
    g = inputColor[1] / 255.0
    b = inputColor[0] / 255.0

    def to_linear(v: float) -> float:
        if v > 0.04045:
            return ((v + 0.055) / 1.055) ** 2.4
        return v / 12.92

    R = to_linear(r) * 100.0
    G = to_linear(g) * 100.0
    B = to_linear(b) * 100.0

    X = R * 0.4124 + G * 0.3576 + B * 0.1805
    Y = R * 0.2126 + G * 0.7152 + B * 0.0722
    Z = R * 0.0193 + G * 0.1192 + B * 0.9505

    X /= 95.047
    Y /= 100.0
    Z /= 108.883

    def f(t: float) -> float:
        if t > 0.008856:
            return t ** (1.0 / 3.0)
        return (7.787 * t) + (16.0 / 116.0)

    fx = f(X)
    fy = f(Y)
    fz = f(Z)

    L = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return [float(L), float(a), float(b)]


def ciede2000(Lab_1: List[float], Lab_2: List[float]) -> float:
    C_25_7 = 6103515625  # 25**7

    L1, a1, b1 = Lab_1[0], Lab_1[1], Lab_1[2]
    L2, a2, b2 = Lab_2[0], Lab_2[1], Lab_2[2]
    C1 = math.sqrt(a1 * a1 + b1 * b1)
    C2 = math.sqrt(a2 * a2 + b2 * b2)
    C_ave = (C1 + C2) / 2.0
    G = 0.5 * (1.0 - math.sqrt((C_ave ** 7) / (C_ave ** 7 + C_25_7)))

    L1_, L2_ = L1, L2
    a1_, a2_ = (1.0 + G) * a1, (1.0 + G) * a2
    b1_, b2_ = b1, b2

    C1_ = math.sqrt(a1_ * a1_ + b1_ * b1_)
    C2_ = math.sqrt(a2_ * a2_ + b2_ * b2_)

    def atan2_safe(y, x):
        return math.atan2(y, x)

    if b1_ == 0 and a1_ == 0:
        h1_ = 0.0
    else:
        h1_ = atan2_safe(b1_, a1_)
        if h1_ < 0:
            h1_ += 2.0 * math.pi

    if b2_ == 0 and a2_ == 0:
        h2_ = 0.0
    else:
        h2_ = atan2_safe(b2_, a2_)
        if h2_ < 0:
            h2_ += 2.0 * math.pi

    dL_ = L2_ - L1_
    dC_ = C2_ - C1_
    dh_ = h2_ - h1_
    if C1_ * C2_ == 0:
        dh_ = 0.0
    elif dh_ > math.pi:
        dh_ -= 2.0 * math.pi
    elif dh_ < -math.pi:
        dh_ += 2.0 * math.pi

    dH_ = 2.0 * math.sqrt(max(0.0, C1_ * C2_)) * math.sin(dh_ / 2.0)

    L_ave = (L1_ + L2_) / 2.0
    C_ave = (C1_ + C2_) / 2.0

    _dh = abs(h1_ - h2_)
    _sh = h1_ + h2_
    C1C2 = C1_ * C2_

    if _dh <= math.pi and C1C2 != 0:
        h_ave = (h1_ + h2_) / 2.0
    elif _dh > math.pi and _sh < 2.0 * math.pi and C1C2 != 0:
        h_ave = (h1_ + h2_) / 2.0 + math.pi
    elif _dh > math.pi and _sh >= 2.0 * math.pi and C1C2 != 0:
        h_ave = (h1_ + h2_) / 2.0 - math.pi
    else:
        h_ave = h1_ + h2_

    T = 1.0 - 0.17 * math.cos(h_ave - math.pi / 6.0) + 0.24 * math.cos(2.0 * h_ave) + 0.32 * math.cos(3.0 * h_ave + math.pi / 30.0) - 0.20 * math.cos(4.0 * h_ave - 63.0 * math.pi / 180.0)

    h_ave_deg = h_ave * 180.0 / math.pi
    if h_ave_deg < 0:
        h_ave_deg += 360.0
    elif h_ave_deg > 360.0:
        h_ave_deg -= 360.0

    dTheta = 30.0 * math.exp(-(((h_ave_deg - 275.0) / 25.0) ** 2.0))

    R_C = 2.0 * math.sqrt((C_ave ** 7.0) / (C_ave ** 7.0 + C_25_7))
    S_C = 1.0 + 0.045 * C_ave
    S_H = 1.0 + 0.015 * C_ave * T

    Lm50s = (L_ave - 50.0) ** 2.0
    S_L = 1.0 + 0.015 * Lm50s / math.sqrt(20.0 + Lm50s)
    R_T = -math.sin(dTheta * math.pi / 90.0) * R_C

    k_L, k_C, k_H = 1.0, 1.0, 1.0

    f_L = dL_ / (k_L * S_L)
    f_C = dC_ / (k_C * S_C)
    f_H = dH_ / (k_H * S_H)

    dE_00 = math.sqrt(max(0.0, f_L * f_L + f_C * f_C + f_H * f_H + R_T * f_C * f_H))
    return float(dE_00)


DEFAULT_FACE_TO_COLOR = {
    'U': 'B',
    'R': 'O',
    'F': 'Y',
    'D': 'G',
    'L': 'R',
    'B': 'W'
}

# canonical BGR approximations (used if no calibration present)
CANONICAL_BGR = {
    'R': (30, 30, 200),
    'O': (10, 120, 200),
    'Y': (0, 200, 200),
    'G': (0, 150, 40),
    'B': (180, 50, 0),
    'W': (200, 200, 200)
}


def _extract_roi(img: np.ndarray, x: int, y: int, size: int) -> Optional[np.ndarray]:
    r = max(1, int(size // 2))
    h, w = img.shape[:2]
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = img[y0:y1, x0:x1].copy()
    if roi.size == 0:
        return None
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    return roi


def _compute_optimal_roi_size(pts: List[Tuple[int, int]], img_shape) -> int:
    if len(pts) < 2:
        return 20
    arr = np.array(pts, dtype=float)
    dists = []
    for i in range(len(arr)):
        others = np.delete(arr, i, axis=0)
        if others.size == 0:
            continue
        dif = others - arr[i]
        dd = np.hypot(dif[:, 0], dif[:, 1])
        dists.append(dd.min())
    if len(dists) == 0:
        return 20
    median_nn = float(np.median(dists))
    # roi_factor tunable; default later in class is 0.28
    size = int(max(8, min(64, median_nn * 0.28)))
    return size


def _median_lab_from_roi(roi: np.ndarray) -> Optional[List[float]]:
    if roi is None or roi.size == 0:
        return None
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return None

    # HSV mask to remove bright specular pixels
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].reshape(-1)
    V = hsv[:, :, 2].reshape(-1)

    # center circular mask to avoid edges
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = int(min(h, w) * 0.4)
    circle_mask = ((x - cx) ** 2 + (y - cy) ** 2) <= (r * r)
    circle_mask_flat = circle_mask.reshape(-1)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)

    mask_non_spec = (S > 30) | (V < 240)
    mask = mask_non_spec & circle_mask_flat
    if mask.sum() < max(1, 0.05 * lab.shape[0]):
        # relax: use center 25%
        r2 = int(min(h, w) * 0.25)
        circle2 = ((x - cx) ** 2 + (y - cy) ** 2) <= (r2 * r2)
        mask = circle2.reshape(-1)

    if mask.sum() == 0:
        med = np.median(lab, axis=0)
        return [float(med[0]), float(med[1]), float(med[2])]

    lab_f = lab[mask]
    # IQR trimming
    Q1 = np.percentile(lab_f, 25, axis=0)
    Q3 = np.percentile(lab_f, 75, axis=0)
    IQR = Q3 - Q1
    IQR[IQR == 0] = 1.0
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    good_mask = np.all((lab_f >= lower) & (lab_f <= upper), axis=1)
    if good_mask.sum() > max(1, 0.2 * lab_f.shape[0]):
        med_lab = np.median(lab_f[good_mask], axis=0)
    else:
        med_lab = np.median(lab_f, axis=0)
    return [float(med_lab[0]), float(med_lab[1]), float(med_lab[2])]


def apply_gray_world(img: np.ndarray) -> np.ndarray:
    imgf = img.astype(np.float32)
    avgB = imgf[:, :, 0].mean()
    avgG = imgf[:, :, 1].mean()
    avgR = imgf[:, :, 2].mean()
    avg = (avgB + avgG + avgR) / 3.0
    # avoid division by zero
    if avgB == 0 or avgG == 0 or avgR == 0:
        return img
    imgf[:, :, 0] = np.clip(imgf[:, :, 0] * (avg / avgB), 0, 255)
    imgf[:, :, 1] = np.clip(imgf[:, :, 1] * (avg / avgG), 0, 255)
    imgf[:, :, 2] = np.clip(imgf[:, :, 2] * (avg / avgR), 0, 255)
    return imgf.astype(np.uint8)


def get_face_from_label(label: str) -> str:
    return label[0].upper() if label and isinstance(label, str) else ''


class ColorDetector:
    def __init__(self,
                 calib_path: Optional[str] = None,
                 use_gray_world: bool = False,
                 roi_factor: float = 0.28,
                 center_threshold: float = 14.0,
                 global_threshold: float = 22.0,
                 persist_calib_if_centers_ok: bool = False):
        """
        calib_path: optional json path to save centroids (color_letter -> [L,a,b])
        roi_factor: multiplier for neighbor distance when computing ROI size
        center_threshold: dE threshold to accept sticker ~= face center
        global_threshold: dE threshold for fallback marking white/uncertain
        persist_calib_if_centers_ok: if True and all centers measured OK, overwrite stored calib
        """
        self.calib_path = Path(calib_path) if calib_path else None
        self.use_gray_world = use_gray_world
        self.roi_factor = roi_factor
        self.center_threshold = float(center_threshold)
        self.global_threshold = float(global_threshold)
        self.persist_calib_if_centers_ok = persist_calib_if_centers_ok

        # load calibration if present
        self.centroids_lab: Dict[str, List[float]] = {}
        if self.calib_path and self.calib_path.exists():
            try:
                data = json.loads(self.calib_path.read_text(encoding='utf-8'))
                self.centroids_lab = {k: list(map(float, v)) for k, v in data.items()}
            except Exception:
                self.centroids_lab = {}

        # ensure all colors present
        for c, bgr in CANONICAL_BGR.items():
            if c not in self.centroids_lab:
                self.centroids_lab[c] = bgr2lab(bgr)

    def save_calibration(self) -> bool:
        if not self.calib_path:
            return False
        try:
            self.calib_path.parent.mkdir(parents=True, exist_ok=True)
            self.calib_path.write_text(json.dumps(self.centroids_lab, indent=2), encoding='utf-8')
            return True
        except Exception:
            return False

    def calibrate_from_positions(self,
                                 img1_path: Path,
                                 pos1: Dict[str, Tuple[int, int]],
                                 img2_path: Path,
                                 pos2: Dict[str, Tuple[int, int]],
                                 face_to_color_expected: Optional[Dict[str, str]] = None) -> Dict[str, List[float]]:
 
        face_to_color_expected = face_to_color_expected or DEFAULT_FACE_TO_COLOR

        # map face -> candidate labels (central row)
        candidates = {
            'U': ['U4', 'U5', 'U6'],
            'F': ['F4', 'F5', 'F6'],
            'L': ['L4', 'L5', 'L6'],
            'D': ['D4', 'D5', 'D6'],
            'B': ['B4', 'B5', 'B6'],
            'R': ['R4', 'R5', 'R6']
        }

        def med_lab_for_label(img_path: Path, positions: Dict[str, Tuple[int, int]], label: str) -> Optional[List[float]]:
            if label not in positions:
                return None
            x, y = positions[label]
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            # compute ROI size with provided roi_factor
            pts = list(positions.values())
            arr = np.array(pts, dtype=float)
            # compute neighbor median
            if arr.size == 0:
                size = 20
            else:
                # compute nearest neighbor distances
                dists = []
                for i in range(len(arr)):
                    others = np.delete(arr, i, axis=0)
                    if others.size == 0: continue
                    dd = np.hypot((others - arr[i])[:, 0], (others - arr[i])[:, 1])
                    dists.append(dd.min())
                median_nn = float(np.median(dists)) if dists else 40.0
                size = int(max(8, min(64, median_nn * self.roi_factor)))
            roi = _extract_roi(img, int(x), int(y), size)
            return _median_lab_from_roi(roi)

        centroids = {}
        # for each face collect candidate medians (from appropriate image)
        for face, labels in candidates.items():
            vals = []
            for lab in labels:
                which_img = img1_path if lab[0] in ['U', 'F', 'L'] else img2_path
                which_pos = pos1 if lab[0] in ['U', 'F', 'L'] else pos2
                med = med_lab_for_label(which_img, which_pos, lab)
                if med is not None:
                    vals.append(med)
            if vals:
                # stack and IQR trim
                arr = np.array(vals, dtype=float)
                Q1 = np.percentile(arr, 25, axis=0)
                Q3 = np.percentile(arr, 75, axis=0)
                IQR = Q3 - Q1
                IQR[IQR == 0] = 1.0
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                mask = np.all((arr >= lower) & (arr <= upper), axis=1)
                if mask.sum() > 0:
                    med = np.median(arr[mask], axis=0)
                else:
                    med = np.median(arr, axis=0)
                color_letter = face_to_color_expected.get(face)
                if color_letter:
                    centroids[color_letter] = [float(med[0]), float(med[1]), float(med[2])]

        # fill missing with existing or canonical
        for c in CANONICAL_BGR.keys():
            if c not in centroids:
                if c in self.centroids_lab:
                    centroids[c] = list(self.centroids_lab[c])
                else:
                    centroids[c] = bgr2lab(CANONICAL_BGR[c])

        # commit
        self.centroids_lab = centroids
        if self.calib_path:
            self.save_calibration()
        return self.centroids_lab

    def detect_single_image(self,
                            image_path: Path,
                            positions: Dict[str, Tuple[int, int]],
                            face_to_color_expected: Optional[Dict[str, str]] = None,
                            use_gray_world: Optional[bool] = None) -> Tuple[Dict[str, str], Dict[str, Tuple[float, float, float]]]:
        """
        Detect colors for a single image with given positions mapping label->(x,y).
        Forzamos los centros segun face_to_color_expected (por convención).
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")
        if use_gray_world is None:
            use_gray_world = self.use_gray_world
        if use_gray_world:
            img = apply_gray_world(img)

        h_img, w_img = img.shape[:2]
        pts = list(positions.values())
        # compute ROI using dynamic factor
        # recompute with self.roi_factor
        if len(pts) >= 2:
            arr = np.array(pts, dtype=float)
            dists = []
            for i in range(len(arr)):
                others = np.delete(arr, i, axis=0)
                if others.size == 0:
                    continue
                dd = np.hypot((others - arr[i])[:, 0], (others - arr[i])[:, 1])
                dists.append(dd.min())
            median_nn = float(np.median(dists)) if dists else 40.0
            roi_size = int(max(8, min(64, median_nn * self.roi_factor)))
        else:
            roi_size = 20

        detected: Dict[str, str] = {}
        labs: Dict[str, Tuple[float, float, float]] = {}
        face_to_color_expected = face_to_color_expected or DEFAULT_FACE_TO_COLOR

        # collect medians
        medlabs_per_label: Dict[str, Optional[List[float]]] = {}
        for label, pos in positions.items():
            try:
                x, y = int(pos[0]), int(pos[1])
            except Exception:
                medlabs_per_label[label] = None
                continue
            x = max(0, min(w_img - 1, x))
            y = max(0, min(h_img - 1, y))
            roi = _extract_roi(img, x, y, roi_size)
            med = _median_lab_from_roi(roi)
            medlabs_per_label[label] = med
            if med is not None:
                labs[label] = (float(med[0]), float(med[1]), float(med[2]))
            else:
                labs[label] = (0.0, 0.0, 0.0)

        # Build centroids_temp starting from stored centroids
        centroids_temp = dict(self.centroids_lab)

        # centers_by_face mapping (labels)
        centers_by_face = {
            'U': 'U5', 'R': 'R5', 'F': 'F5',
            'D': 'D5', 'L': 'L5', 'B': 'B5'
        }

        measured_centers = {}
        # measure centers and decide whether to accept them as centroids_temp anchors
        for face, center_label in centers_by_face.items():
            expected_color = face_to_color_expected.get(face)
            med = medlabs_per_label.get(center_label)
            if med is not None and expected_color:
                # check distance to stored centroid; if very different, treat with caution
                stored = self.centroids_lab.get(expected_color)
                if stored is not None:
                    try:
                        d = ciede2000(med, stored)
                    except Exception:
                        d = float(np.linalg.norm(np.array(med) - np.array(stored)))
                else:
                    d = 0.0
                # if d is reasonable (not huge), accept measured center; else fallback to stored
                if d < 35.0:
                    centroids_temp[expected_color] = med
                    measured_centers[expected_color] = med
                else:
                    # reject measured center (likely specular/outlier), keep stored or canonical
                    if expected_color not in centroids_temp:
                        centroids_temp[expected_color] = bgr2lab(CANONICAL_BGR.get(expected_color, (200, 200, 200)))
                # enforce detected center color regardless (we must not contradict convention)
                detected[center_label] = expected_color
            else:
                # center measurement failed -> ensure centroids_temp has fallback
                if expected_color not in centroids_temp:
                    centroids_temp[expected_color] = bgr2lab(CANONICAL_BGR.get(expected_color, (200, 200, 200)))
                # still assign detected center label to expected color (force convention)
                if expected_color:
                    detected[center_label] = expected_color

        # optional: if persist_calib_if_centers_ok and we have all 6 measured_centers -> persist
        if self.persist_calib_if_centers_ok and len(measured_centers) == 6:
            # update persistent centroids with these measured values (robust median could be applied)
            for k, v in measured_centers.items():
                self.centroids_lab[k] = v
            if self.calib_path:
                self.save_calibration()

        # classification for non-center stickers
        for label, med_lab in medlabs_per_label.items():
            # skip centers: already assigned
            if label in detected:
                continue
            if med_lab is None:
                detected[label] = 'W'
                continue

            face = get_face_from_label(label)

            # attempt 1: if face center exists, compare to its centroid and favor it
            assigned = None
            if face in centers_by_face:
                expected_center_color = face_to_color_expected.get(face)
                if expected_center_color and expected_center_color in centroids_temp:
                    try:
                        d_center = ciede2000(med_lab, centroids_temp[expected_center_color])
                    except Exception:
                        d_center = float(np.linalg.norm(np.array(med_lab) - np.array(centroids_temp[expected_center_color])))
                    if d_center <= self.center_threshold:
                        assigned = expected_center_color
                        detected[label] = assigned
                        continue

            # attempt 2: compare to all centroids_temp
            best = None
            best_d = float('inf')
            for color_letter, ref_lab in centroids_temp.items():
                try:
                    d = ciede2000(med_lab, ref_lab)
                except Exception:
                    d = float(np.linalg.norm(np.array(med_lab) - np.array(ref_lab)))
                if d < best_d:
                    best_d = d
                    best = color_letter

            if best is None:
                detected[label] = 'W'
                continue

            if best_d > self.global_threshold:
                # fallback using HSV heuristic: if very desaturated & bright => white
                lab_arr = np.uint8([[[int(round(med_lab[0])), int(round(med_lab[1])), int(round(med_lab[2]))]]])
                try:
                    bgr = cv2.cvtColor(lab_arr, cv2.COLOR_LAB2BGR)[0][0]
                    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                    H, S, V = int(hsv[0]), int(hsv[1]), int(hsv[2])
                except Exception:
                    H, S, V = 0, 0, 255
                if S < 55 and V > 230:
                    detected[label] = 'W'
                else:
                    detected[label] = best
            else:
                detected[label] = best

        # conservative per-face correction: force a few outliers to center color if center assigned
        for face, center_label in centers_by_face.items():
            center_color = face_to_color_expected.get(face)
            if not center_color:
                continue
            face_keys = [face + str(i) for i in range(1, 10)]
            vals = [detected[k] for k in face_keys if k in detected]
            if not vals:
                continue
            ctr = Counter(vals)
            most, cnt = ctr.most_common(1)[0]
            # if center appears less than 6 times, try to correct up to 3 outliers
            center_count = vals.count(center_color)
            if center_count < 6:
                outliers = [k for k in face_keys if k in detected and detected[k] != center_color]
                for k in outliers[:3]:
                    detected[k] = center_color

        return detected, labs

    def debug_detection_output(self, image_path: Path, positions: Dict[str, Tuple[int, int]]):
        """
        Debug visualization: dibuja ROIs, centros y muestra dE / clasificación
        """
        try:
            detected, labs = self.detect_single_image(image_path, positions)
        except Exception as e:
            print("DEBUG: detect_single_image falló:", e)
            return

        img = cv2.imread(str(image_path))
        if img is None:
            print("DEBUG: no se pudo abrir imagen:", image_path)
            return

        h_img, w_img = img.shape[:2]
        pts = list(positions.values())
        # recompute roi_size local
        if len(pts) >= 2:
            arr = np.array(pts, dtype=float)
            dists = []
            for i in range(len(arr)):
                others = np.delete(arr, i, axis=0)
                if others.size == 0:
                    continue
                dd = np.hypot((others - arr[i])[:, 0], (others - arr[i])[:, 1])
                dists.append(dd.min())
            median_nn = float(np.median(dists)) if dists else 40.0
            roi_size = int(max(8, min(64, median_nn * self.roi_factor)))
        else:
            roi_size = 20

        print("\n--- INICIO DE DEBUGGING DE DETECCIÓN ---")
        print(f"Tamaño de ROI calculado: {roi_size} píxeles de radio.")
        print(f"{'Color':<6} | {'Etiqueta':<7} | {'Coordenadas':<12} | {'Lab Mediana':<22} | {'Mejor dE (CIEDE2000)':<20} | {'Clasificación'}")
        print("-" * 110)

        # compute dE to chosen centroid for display
        for label, pos in positions.items():
            x, y = int(pos[0]), int(pos[1])
            med_lab = labs.get(label)
            if med_lab is None or med_lab == (0.0, 0.0, 0.0):
                print(f"{'N/A':<6} | {label:<7} | ({x},{y:<3}) | [ROI failed]")
                # mark red
                cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
                continue

            assigned = detected.get(label, '?')
            # compute best dE vs assigned centroid
            ref = self.centroids_lab.get(assigned) or bgr2lab(CANONICAL_BGR.get(assigned, (200, 200, 200)))
            try:
                d = ciede2000(list(med_lab), ref)
            except Exception:
                d = float(np.linalg.norm(np.array(list(med_lab)) - np.array(ref)))
            lab_str = f"L:{med_lab[0]:.1f}, a:{med_lab[1]:.1f}, b:{med_lab[2]:.1f}"
            print(f"{assigned:<6} | {label:<7} | ({x},{y:<3}) | {lab_str:<30} | {d:<20.2f} | {assigned}")
            # draw ROI/center/label
            r = roi_size // 2
            cv2.rectangle(img, (max(0, x - r), max(0, y - r)), (min(w_img - 1, x + r), min(h_img - 1, y + r)), (255, 200, 0), 1)
            # center circle color by assigned
            color_bgr = (255, 255, 255)
            if assigned in CANONICAL_BGR:
                color_bgr = CANONICAL_BGR[assigned]
            cv2.circle(img, (x, y), 5, color_bgr, -1)
            cv2.putText(img, f"{assigned}:{d:.1f}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        print("-" * 110)
        cv2.imshow('DEBUG DETECCION - Presiona una tecla', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("--- FIN DE DEBUGGING DE DETECCIÓN ---\n")


if __name__ == '__main__':
    print('detector.py: self-check debug mode')
    try:
        det = ColorDetector(calib_path="positions/calibrations.json")
        try:
            pos1 = json.load(open(POLYGON_POSITIONS_PATH_1, 'r', encoding='utf-8')).get('positions', json.load(open(POLYGON_POSITIONS_PATH_1, 'r', encoding='utf-8')))
        except Exception:
            pos1 = {}
        try:
            pos2 = json.load(open(POLYGON_POSITIONS_PATH_2, 'r', encoding='utf-8')).get('positions', json.load(open(POLYGON_POSITIONS_PATH_2, 'r', encoding='utf-8')))
        except Exception:
            pos2 = {}
        if pos1:
            print("Lanzando debug_detection_output para IMG1...")
            det.debug_detection_output(IMG1_PATH, pos1)
            det.debug_detection_output(IMG2_PATH, pos2)

        else:
            print("No pude cargar positions1.json; revisá la ruta.")
    except Exception as e:
        print("Error inicial:", e)
