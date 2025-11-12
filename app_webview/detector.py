import itertools
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import kociemba
import numpy as np

from config import CENTER_INDICES

DEFAULT_HSV_RANGES = {
    'R': [((0, 80, 50), (10, 255, 255)), ((170, 80, 50), (179, 255, 255))],
    'O': [((10, 80, 50), (25, 255, 255))],
    'Y': [((25, 80, 50), (35, 255, 255))],
    'G': [((36, 40, 40), (85, 255, 255))],
    'B': [((90, 40, 40), (140, 255, 255))],
    'W': [((0, 0, 180), (179, 80, 255))]
}

def _median_bgr_from_roi(roi: np.ndarray):
    """Robust median BGR color using LAB outlier removal (fallbacks included)."""
    if roi is None or roi.size == 0:
        return None
    roi_bgr = roi.reshape(-1, 3)
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    mask_non_highlight = hsv[:, 2] < 245
    if mask_non_highlight.sum() == 0:
        # use median of all
        med = np.median(roi_bgr, axis=0).astype(np.uint8)
        return tuple(med.tolist())
    roi_bgr_f = roi_bgr[mask_non_highlight]
    roi_lab_f = roi_lab[mask_non_highlight]
    if roi_lab_f.shape[0] == 0:
        med = np.median(roi_bgr, axis=0).astype(np.uint8)
        return tuple(med.tolist())

    # IQR outlier removal in LAB
    Q1 = np.percentile(roi_lab_f, 25, axis=0)
    Q3 = np.percentile(roi_lab_f, 75, axis=0)
    IQR = Q3 - Q1
    IQR[IQR == 0] = 1.0
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = np.all((roi_lab_f >= lower) & (roi_lab_f <= upper), axis=1)
    if mask.sum() > 0:
        med_lab = np.median(roi_lab_f[mask], axis=0).astype(np.uint8)
        bgr = cv2.cvtColor(np.uint8([[med_lab]]), cv2.COLOR_LAB2BGR)[0][0]
        return tuple(bgr.tolist())
    else:
        med = np.median(roi_bgr_f, axis=0).astype(np.uint8)
        return tuple(med.tolist())


class ColorDetector:
    """
    Detección por sticker: lee posiciones (label->[x,y]) y la imagen,
    extrae un ROI por marcador, determina el BGR/LAB/HSV promedio y mapea
    a una letra entre R,O,Y,G,B,W.
    """

    def __init__(self, hsv_ranges: Optional[Dict[str, List[Tuple[Tuple[int,int,int],Tuple[int,int,int]]]]] = None):
        self.hsv_ranges = hsv_ranges or DEFAULT_HSV_RANGES

    def detect_single_image(self, image_path: Path, positions: Dict[str, Tuple[int,int]]):
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(...)
        h_img, w_img = img.shape[:2]

        roi_size = 20

        labs = {}
        detected = {}

        center_hsv = {
            'R': (0, 180, 150),
            'O': (14, 170, 160),
            'Y': (28, 200, 200),
            'G': (60, 170, 150),
            'B': (110, 160, 120),
            'W': (0, 30, 230)
        }
        
        for label, (x, y) in positions.items():
            x = int(max(0, min(w_img-1, x)))
            y = int(max(0, min(h_img-1, y)))
            r = roi_size//2
            x0, x1 = max(0, x-r), min(w_img, x+r)
            y0, y1 = max(0, y-r), min(h_img, y+r)
            roi = img[y0:y1, x0:x1]
            if roi.size == 0:
                detected[label] = 'W'
                labs[label] = (0,0,0)
                continue
            # compute median color quickly
            med_bgr = np.median(roi.reshape(-1,3), axis=0).astype(np.uint8)
            hsv = cv2.cvtColor(np.uint8([[med_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            labs[label] = tuple(int(v) for v in cv2.cvtColor(np.uint8([[med_bgr]]), cv2.COLOR_BGR2LAB)[0][0])

            # map to nearest center using circular hue distance
            best = None; bestd = 1e9
            H,S,V = int(hsv[0]), int(hsv[1]), int(hsv[2])
            for k,(ch,cs,cvv) in center_hsv.items():
                dh = min(abs(H-ch), 180-abs(H-ch))/180.0
                ds = abs(S-cs)/255.0
                dv = abs(V-cvv)/255.0
                d = dh*0.6 + ds*0.25 + dv*0.15
                if d < bestd:
                    bestd = d; best = k
            detected[label] = best or 'W'
        return detected, labs

    def _extract_roi(self, img: np.ndarray, x: int, y: int, size: int):
        r = max(1, size // 2)
        h, w = img.shape[:2]
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        roi = img[y0:y1, x0:x1]
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        return roi

    def _compute_optimal_roi_size(self, pts: List[Tuple[int, int]], img_shape):
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
        size = int(max(8, min(40, median_nn * 0.45)))
        return size

    def _map_hsv_to_letter(self, h: int, s: int, v: int) -> str:
        """
        Map HSV to one of letters using hsv_ranges + robust fallbacks.
        Returns one of 'R','O','Y','G','B','W' (or 'W' as very safe fallback).
        """
        # quick white check: low saturation and high value
        if s < 60 and v > 200:
            return 'W'
    
        # helper to test a single (low,high) pair with wrap-around for H
        def in_hsv_range(h, s, v, low, high):
            low_h, low_s, low_v = low
            high_h, high_s, high_v = high
            # Hue wrap-around
            if low_h <= high_h:
                h_ok = (low_h <= h <= high_h)
            else:
                h_ok = (h >= low_h or h <= high_h)
            s_ok = (low_s <= s <= high_s)
            v_ok = (low_v <= v <= high_v)
            return h_ok and s_ok and v_ok
    
        # check configured HSV ranges first
        for cname, ranges in (self.hsv_ranges or DEFAULT_HSV_RANGES).items():
            for low, high in ranges:
                if in_hsv_range(h, s, v, low, high):
                    return cname
    
        # fallback centers if none matched: coarse distance
        centers = {
            'R': (0, 190, 130),
            'O': (14, 170, 150),
            'Y': (28, 200, 200),
            'G': (60, 170, 120),
            'B': (110, 180, 110),
            'W': (0, 30, 240)
        }
        best = None
        bestd = 1e9
        for k, c in centers.items():
            ch, cs, cv = c
            dh = min(abs(h - ch), 180 - abs(h - ch)) / 180.0
            ds = abs(s - cs) / 255.0
            dv = abs(v - cv) / 255.0
            # weights: hue more important
            d = dh * 0.6 + ds * 0.3 + dv * 0.1
            if d < bestd:
                bestd = d
                best = k
    
        return best or 'W'
    
def save_colors_file(out_path: Path , combined : dict) -> bool:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[ColorCorrector.save_colors_file] Error: {e}")
        return False


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