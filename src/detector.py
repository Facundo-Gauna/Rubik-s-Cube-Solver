"""
detector.py — color detector for Rubik's-cube stickers
======================================================

This module implements a heuristic, image-based color detector tailored to
identify the 27 visible stickers from three faces of a Rubik's cube in two
photographs. It is intentionally lightweight, dependency-minimal (OpenCV + numpy),
and designed for integration with a larger solver/GUI. That said, the approach is
heuristic and **frequently error-prone** under varying illumination, specular
reflections, heavy compression, or strong perspective — see "Limitations" below.

## Primary classes / functions

* ColorDetector

  * High-level class exposing the main workflow:

    * `detect_on_image(img, positions, use_clahe=True, top_k_candidates=2)`:
      classify a set of labeled positions (e.g. "U1","U2",..., "B9") inside a
      single image. Returns `(detected_map, labs_map)`.
    * `detect_images(img1_path, pos1, img2_path, pos2, ...)`:
      convenience wrapper that reads images (or accepts numpy arrays), stores
      them as `last_img1/2`, runs detection for both images and optionally
      returns the LAB medians per sticker for later calibration.
    * `calibrate_from_detection(validated=None)`:
      build/update canonical centroids (CIE-L*a*b* lists) using measured
      sticker medians. If `validated` (label->color_letter) is provided, groups
      by color; otherwise computes robust per-face medians from stored positions.
    * `save_calibration()` / calibration loading on init:
      persist/load centroids as JSON at `calib_path`.
    * `debug_detection_output(image_path, positions)`:
      visual debug overlay + console table with medians/assignments.
  * Constructor options of note:

    * `calib_path`: Path to JSON calibration file (color_letter -> [L,a,b]).
    * `use_gray_world`: apply simple gray-world white-balance pre-step.
    * `roi_factor`: fraction used to compute sticker ROI from neighbor distances.
    * `center_threshold` / `global_threshold`: decision thresholds (CIEDE2000).
    * `persist_calib_if_centers_ok`: if true, update stored centroids from
      reliable center measurements.

* Utility functions (module-level)

  * `bgr2lab(bgr_tuple)` — fallback BGR→CIE-Lab conversion.
  * `ciede2000(Lab1, Lab2)` — CIEDE2000 color-difference implementation.
  * `_median_lab_from_labhsv(lab_roi_cie, hsv_roi)` — robust median extraction
    from a ROI: circular mask, specularity filtering (S/V), IQR outlier rejection.
  * `apply_gray_world(img)` — simple global channel scaling white-balance.
  * Several small helpers for vectorized conversions and ROI sizing.

## Data formats / assumptions

* `positions` argument: `Dict[str, Tuple[int,int]]` mapping labels like `"U1"` to
  image pixel coordinates `(x,y)`. Two position maps are expected (img1 covers
  faces `U,F,L`; img2 covers `D,R,B`) in the provided workflows.
* Calibration JSON format (loaded/saved at `calib_path`):

  ```json
  {
    "R": [L, a, b],
    "O": [L, a, b],
    "Y": [L, a, b],
    "G": [L, a, b],
    "B": [L, a, b],
    "W": [L, a, b]
  }
  ```

  Values are floats representing CIE-L*a*b* (L ~ 0..100).
* Canonical fallback BGR colors are provided in `CANONICAL_BGR` and converted
  to Lab on startup if no calibration exists.

## Algorithm / implementation notes

* Input image processing:

  * optional gray-world white-balance,
  * optional CLAHE applied to the L channel for local contrast normalization.
* Color representation: OpenCV LAB -> converted to **CIE-L*a*b*** floats for
  perceptual distances (vectorized).
* Sticker measurement: for each labeled position a circular-ish ROI is formed
  (scaled by neighbor distances), HSV used to reject specular pixels, then a
  robust median computed after IQR-based outlier removal.
* Assignment:

  * quick shortlist by Euclidean distance in Lab against stored centroids,
  * exact ranking by CIEDE2000 for top candidates (expensive but accurate),
  * special handling for face centers (U5, R5, ...): if the measured center
    is close enough to its expected color centroid, that color is used as
    a strong prior for other stickers on the same face.
* Post-correction: per-face conservative correction tries to reduce obvious
  outliers (e.g., force stickers to center color when center dominates).

## Limitations & common failure modes (important)

Be explicit: this system is brittle under many real-world conditions. Typical
issues you'll encounter:

* **Global illumination / white-balance shifts** — changes in scene color cast
  make fixed centroids mismatch; detection degrades unless centroids are
  re-centered or the image is normalized.
* **Specular highlights** (bright shiny spots) on stickers produce extreme V
  values that can bias medians if not well rejected.
* **Shadows & mixed lighting** cause overlapping Lab clusters (stickers drift).
* **Perspective distortion & non-uniform ROIs** (sticker appears skewed or
  partially occluded) reduce the ROI representativity.
* **JPEG artifacts / compression** and sensor noise add small-scale variation.
* **Color ambiguity** (orange vs. red, or white vs. very pale yellow) — clusters
  may overlap in a/b plane and CIEDE2000 distances can be close.
* **Threshold sensitivity** — many heuristics (`S>30`, `V<240`, `center_threshold`, `global_threshold`)
  need per-environment tuning; defaults may be far from optimal for some photos.

## Suggested improvements (concise list)

These are grouped from low-effort/high-impact to more involved changes.

Quick / high-impact

* **Per-image Lab shift**: compute a median offset between measured centers
  and stored centroids, apply it as a global shift to all centroids before
  final assignment — compensates global color cast (recommended first step).
* **Better specular masking**: more aggressive detection/removal of pixels with
  very high V and low S, or morphological erosion of specular areas within ROI.
* **Scale-aware ROI sizing**: incorporate image resolution / cube size estimate
  so ROI is not too large/small.

Algorithmic / medium effort

* **Mahalanobis or learned covariances** per-color instead of plain Euclidean:
  penalize low-variance directions more — improves separation when centroids
  have anisotropic spread.
* **Small constrained assignment solver**: treat each face globally and solve
  assignment (minimize total cost) instead of independent per-sticker greedy
  choices — reduces inconsistent mixes on the same face.
* **Local clusterization per-face**: cluster the 9 stickers on a face to find
  local centroids and map them to canonical colors (works well when calibration
  is poor but stickers are internally consistent).
* **Perspective rectification**: detect sticker quadrilaterals, compute homography
  per-face and sample canonical, fronto-parallel patch for more uniform color.
* **Save per-color variance from calibration** so future runs can use Mahalanobis
  distance or set adaptive thresholds.

Advanced / longer-term

* **Train a small classifier** (k-NN / RF / light NN) on labeled patch dataset
  (Lab, chromaticity, HSV histograms) — often more robust than hand-tuned rules.
* **Retinex / multi-scale color constancy** (MSRCR) as pre-processing to handle
  mixed lighting; beware artifacts and tune carefully.
* **Active capture**: require a controlled white reference or color card in a
  calibration shot when possible — yields best long-term accuracy.

## Debugging & observability tips

* Dump `label, x, y, L,a,b, assigned_color, d_best, d_center, is_probably_white`
  for each image to CSV — inspect numeric patterns quickly.
* Plot `(a,b)` scatter of all measured medians colored by assigned label to
  visually detect overlapping clusters.
* Use `debug_detection_output()` to generate overlays (rectangles, assigned
  canonical colors) and a printed table of medians — very helpful to spot
  wrongly sized ROIs or failed median computations.

## Performance notes

* The implementation already shortlists candidates by Euclidean distance and
  computes CIEDE2000 only for the top-K (`top_k_candidates`) — this is a good
  trade-off between speed and perceptual accuracy.
* Avoid lowering vectorized full-image operations (Lab/HSV conversion) — they
  are fast and preferable to repeated small conversions per-ROI.
* If runtime matters, reduce `top_k_candidates` or cache conversions when
  processing multiple similar images.

## Design & integration notes

* `ColorDetector` is intentionally independent of any GUI / I/O — pass numpy
  images and position dicts extracted by your own detection pipeline.
* Persisted calibration (`calib_path`) stores color centroids only — no
  per-environment metadata is kept (e.g. exposure or camera model), so
  re-calibration is recommended when capture conditions change.
* The module emphasizes robustness via medians + IQR outlier rejection rather
  than per-pixel voting; this makes it resilient to small noise but not to
  systematic lighting/color casts.

-------------------------------------------------------------------------------

Copyright (c) 2025 Facundo Gauna & Ulises Carnevale. Licensed under MIT License.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np

from config import CALIBRATIONS_PATH,FACE_TO_COLOR,CANONICAL_BGR

# ---------- Color math helpers ----------

def bgr2lab(inputColor: Tuple[int, int, int]) -> List[float]:
    """
    Convert a BGR tuple (0..255) to CIE L*a*b* (L ~ 0..100, a/b centered around 0).
    Kept for canonical fallbacks.
    """
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
    """
    CIEDE2000 implementation (unchanged numeric algorithm).
    Expects Lab_1 and Lab_2 as CIE-L*a*b* (L ~ 0..100).
    """
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

    T = 1.0 - 0.17 * math.cos(h_ave - math.pi / 6.0) + 0.24 * math.cos(2.0 * h_ave) + \
        0.32 * math.cos(3.0 * h_ave + math.pi / 30.0) - 0.20 * math.cos(4.0 * h_ave - 63.0 * math.pi / 180.0)

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

# ---------- Image / ROI helpers ----------
def _median_lab_from_labhsv(lab_roi_cie: np.ndarray, hsv_roi: np.ndarray) -> Optional[List[float]]:
    """
    Optimized: receive ROI already converted to LAB (CIE floats) and HSV (uint8).
    lab_roi_cie: HxWx3 floats in CIE scale.
    hsv_roi: HxWx3 uint8
    """
    if lab_roi_cie is None or lab_roi_cie.size == 0:
        return None
    h, w = lab_roi_cie.shape[:2]
    if h == 0 or w == 0:
        return None

    lab = lab_roi_cie.reshape(-1, 3).astype(float)
    S = hsv_roi[:, :, 1].reshape(-1)
    V = hsv_roi[:, :, 2].reshape(-1)

    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = int(min(h, w) * 0.4)
    circle_mask = ((x - cx) ** 2 + (y - cy) ** 2) <= (r * r)
    circle_mask_flat = circle_mask.reshape(-1)

    mask_non_spec = (S > 30) | (V < 240)
    mask = mask_non_spec & circle_mask_flat
    if mask.sum() < max(1, 0.05 * lab.shape[0]):
        r2 = int(min(h, w) * 0.25)
        circle2 = ((x - cx) ** 2 + (y - cy) ** 2) <= (r2 * r2)
        mask = circle2.reshape(-1)

    if mask.sum() == 0:
        med = np.median(lab, axis=0)
        return [float(med[0]), float(med[1]), float(med[2])]

    lab_f = lab[mask]
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
    if avgB == 0 or avgG == 0 or avgR == 0:
        return img
    avg = (avgB + avgG + avgR) / 3.0
    imgf[:, :, 0] = np.clip(imgf[:, :, 0] * (avg / avgB), 0, 255)
    imgf[:, :, 1] = np.clip(imgf[:, :, 1] * (avg / avgG), 0, 255)
    imgf[:, :, 2] = np.clip(imgf[:, :, 2] * (avg / avgR), 0, 255)
    return imgf.astype(np.uint8)


def _compute_median_nn(positions: Dict[str, Tuple[int, int]]) -> float:
    pts = np.array(list(positions.values()), dtype=float)
    if pts.shape[0] < 2:
        return 40.0
    dists = []
    for i in range(len(pts)):
        others = np.delete(pts, i, axis=0)
        if others.size == 0:
            continue
        dd = np.hypot((others - pts[i])[:, 0], (others - pts[i])[:, 1])
        if dd.size > 0:
            dists.append(dd.min())
    return float(np.median(dists)) if dists else 40.0


def _lab_array_from_bgr_dict(d: Dict[str, Tuple[int, int, int]]) -> Dict[str, List[float]]:
    """
    Convert a dict of BGR tuples -> dict of CIE-Lab lists.
    Vectorized: convert all BGRs -> OpenCV LAB -> to CIE.
    """
    letters = list(d.keys())
    bgrs = np.array([d[k] for k in letters], dtype=np.uint8).reshape(-1, 1, 3)
    lab_op = cv2.cvtColor(bgrs, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    # convert to CIE
    lab_cie = np.empty_like(lab_op)
    lab_cie[:, 0] = lab_op[:, 0] * (100.0 / 255.0)
    lab_cie[:, 1] = lab_op[:, 1] - 128.0
    lab_cie[:, 2] = lab_op[:, 2] - 128.0
    return {letters[i]: lab_cie[i].tolist() for i in range(len(letters))}


def _is_probably_white(med_lab: List[float], img: np.ndarray, x: int, y: int, roi_size: int) -> bool:
    try:
        h, w = img.shape[:2]
        x0 = max(0, int(x - roi_size))
        y0 = max(0, int(y - roi_size))
        x1 = min(w, int(x + roi_size + 1))
        y1 = min(h, int(y + roi_size + 1))
        patch = img[y0:y1, x0:x1]
        if patch.size == 0:
            return False
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1].reshape(-1)
        V = hsv[:, :, 2].reshape(-1)
        s20 = np.percentile(S, 20)
        v95 = np.percentile(V, 95)
        return (s20 < 60) and (v95 > 220)
    except Exception:
        return False



class ColorDetector:
    def __init__(self,
                 calib_path: Optional[str] = None,
                 use_gray_world: bool = False,
                 roi_factor: float = 0.28,
                 center_threshold: float = 14.0,
                 global_threshold: float = 22.0,
                 persist_calib_if_centers_ok: bool = False):

        self.calib_path = Path(calib_path) if calib_path else CALIBRATIONS_PATH
        self.use_gray_world = use_gray_world
        self.roi_factor = roi_factor
        self.center_threshold = float(center_threshold)
        self.global_threshold = float(global_threshold)
        self.persist_calib_if_centers_ok = persist_calib_if_centers_ok

        self.centroids_lab: Dict[str, List[float]] = {}
        if self.calib_path and self.calib_path.exists():
            try:
                data = json.loads(self.calib_path.read_text(encoding='utf-8'))
                # assume stored centroids are CIE-Lab lists already
                self.centroids_lab = {k: list(map(float, v)) for k, v in data.items()}
            except Exception:
                self.centroids_lab = {}

        canonical_lab = _lab_array_from_bgr_dict(CANONICAL_BGR)
        for c in CANONICAL_BGR.keys():
            if c not in self.centroids_lab:
                self.centroids_lab[c] = list(canonical_lab[c])

        self._update_centroid_np()

        # memory of last images used in calibration/detection
        self.last_img1: Optional[np.ndarray] = None
        self.last_img2: Optional[np.ndarray] = None
        self.last_pos1: Optional[Dict[str, Tuple[int, int]]] = None
        self.last_pos2: Optional[Dict[str, Tuple[int, int]]] = None
        self.face_to_color_expected: Dict[str, str] = FACE_TO_COLOR

        # reuse CLAHE object
        self._clahe = None

    def _update_centroid_np(self):
        self._centroid_letters = list(self.centroids_lab.keys())
        if not self._centroid_letters:
            self._centroid_np = np.empty((0, 3), dtype=float)
        else:
            self._centroid_np = np.array([self.centroids_lab[k] for k in self._centroid_letters], dtype=float)

    def save_calibration(self) -> bool:
        if not self.calib_path:
            return False
        try:
            self.calib_path.parent.mkdir(parents=True, exist_ok=True)
            self.calib_path.write_text(json.dumps(self.centroids_lab, indent=2), encoding='utf-8')
            self._update_centroid_np()
            return True
        except Exception:
            return False

    def calibrate_from_detection(self, validated: Optional[Dict[str, str]] = None) -> Dict[str, List[float]]:
        """
        Improved calibration:
        - If `validated` provided: expects dict label->color_letter (e.g. {'U1':'B', 'U2':'B', ...})
          and computes centroids grouped by color using only those validated positions.
        - If `validated` is None: fallback to robust per-face median using all available positions
          stored in self.last_pos1/self.last_pos2 (legacy behavior).
        Requires self.last_img1/2 and self.last_pos1/2 to be set (they are set by detect_images).
        Returns updated centroids (CIE-Lab).
        """
        # Ensure we have images and positions in memory
        if self.last_img1 is None or self.last_img2 is None or self.last_pos1 is None or self.last_pos2 is None:
            print("Some image or position is null, calibration won't be possible")
            return self.centroids_lab

        # prepare images (apply gray world if configured)
        i1 = apply_gray_world(self.last_img1) if self.use_gray_world else self.last_img1.copy()
        i2 = apply_gray_world(self.last_img2) if self.use_gray_world else self.last_img2.copy()

        def med_lab_for_label_in_images(label: str) -> Optional[List[float]]:
            # choose correct image/positions for label and compute median LAB for that ROI.
            label_str = label if isinstance(label, str) else str(label)
            prefix = label_str[0] if label_str else ''
            if prefix in ['U', 'F', 'L']:
                pos = self.last_pos1
                img = i1
                pts = list(pos.values()) if pos else []
            else:
                pos = self.last_pos2
                img = i2
                pts = list(pos.values()) if pos else []

            if not pos or label_str not in pos:
                return None

            x, y = pos[label_str]
            arr = np.array(pts, dtype=float) if len(pts) >= 2 else np.zeros((0, 2))
            if arr.size == 0:
                size = 20
            else:
                dists = []
                for i in range(len(arr)):
                    others = np.delete(arr, i, axis=0)
                    if others.size == 0:
                        continue
                    dd = np.hypot((others - arr[i])[:, 0], (others - arr[i])[:, 1])
                    dists.append(dd.min())
                median_nn = float(np.median(dists)) if dists else 40.0
                size = int(max(8, min(64, median_nn * self.roi_factor)))

            # Prepare lab_cie and hsv once for the chosen image (small cost)
            lab_op_full = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab_cie_full = np.empty_like(lab_op_full, dtype=float)
            lab_cie_full[:, :, 0] = lab_op_full[:, :, 0] * (100.0 / 255.0)
            lab_cie_full[:, :, 1] = lab_op_full[:, :, 1] - 128.0
            lab_cie_full[:, :, 2] = lab_op_full[:, :, 2] - 128.0
            hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # compute ROI slice
            r = max(1, int(size // 2))
            h, w = img.shape[:2]
            x0, x1 = max(0, int(x - r)), min(w, int(x + r + 1))
            y0, y1 = max(0, int(y - r)), min(h, int(y + r + 1))
            if x1 <= x0 or y1 <= y0:
                return None
            lab_roi = lab_cie_full[y0:y1, x0:x1]
            hsv_roi = hsv_full[y0:y1, x0:x1]
            return _median_lab_from_labhsv(lab_roi, hsv_roi)

        # If validated provided, group by color letter
        centroids: Dict[str, List[float]] = {}
        if validated:
            color_groups: Dict[str, List[List[float]]] = {k: [] for k in CANONICAL_BGR.keys()}
            for label, color_letter in validated.items():
                if not label or not color_letter:
                    continue
                color_letter = str(color_letter).upper()
                if color_letter not in color_groups:
                    continue
                try:
                    med = med_lab_for_label_in_images(label)
                except Exception:
                    med = None
                if med is not None:
                    color_groups[color_letter].append(med)

            for color_letter, vals in color_groups.items():
                if vals:
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
                    centroids[color_letter] = [float(med[0]), float(med[1]), float(med[2])]
        else:
            # fallback: compute per-face group medians using ALL available stickers
            # group by face letter using self.last_pos1/self.last_pos2
            face_groups = {'U': {}, 'R': {}, 'F': {}, 'D': {}, 'L': {}, 'B': {}}

            # prep lab/hsv for both images once
            lab1_op = cv2.cvtColor(i1, cv2.COLOR_BGR2LAB)
            lab1_cie = np.empty_like(lab1_op, dtype=float)
            lab1_cie[:, :, 0] = lab1_op[:, :, 0] * (100.0 / 255.0)
            lab1_cie[:, :, 1] = lab1_op[:, :, 1] - 128.0
            lab1_cie[:, :, 2] = lab1_op[:, :, 2] - 128.0
            hsv1 = cv2.cvtColor(i1, cv2.COLOR_BGR2HSV)

            lab2_op = cv2.cvtColor(i2, cv2.COLOR_BGR2LAB)
            lab2_cie = np.empty_like(lab2_op, dtype=float)
            lab2_cie[:, :, 0] = lab2_op[:, :, 0] * (100.0 / 255.0)
            lab2_cie[:, :, 1] = lab2_op[:, :, 1] - 128.0
            lab2_cie[:, :, 2] = lab2_op[:, :, 2] - 128.0
            hsv2 = cv2.cvtColor(i2, cv2.COLOR_BGR2HSV)

            def med_for_label_from_mem(img_lab_cie, img_hsv, pos_map, label):
                if not pos_map or label not in pos_map:
                    return None
                x, y = pos_map[label]
                pts = list(pos_map.values())
                arr = np.array(pts, dtype=float) if len(pts) >= 2 else np.zeros((0, 2))
                if arr.size == 0:
                    size = 20
                else:
                    dists = []
                    for i in range(len(arr)):
                        others = np.delete(arr, i, axis=0)
                        if others.size == 0:
                            continue
                        dd = np.hypot((others - arr[i])[:, 0], (others - arr[i])[:, 1])
                        dists.append(dd.min())
                    median_nn = float(np.median(dists)) if dists else 40.0
                    size = int(max(8, min(64, median_nn * self.roi_factor)))

                r = max(1, int(size // 2))
                h, w = img_lab_cie.shape[:2]
                x0, x1 = max(0, int(x - r)), min(w, int(x + r + 1))
                y0, y1 = max(0, int(y - r)), min(h, int(y + r + 1))
                if x1 <= x0 or y1 <= y0:
                    return None
                lab_roi = img_lab_cie[y0:y1, x0:x1]
                hsv_roi = img_hsv[y0:y1, x0:x1]
                return _median_lab_from_labhsv(lab_roi, hsv_roi)

            # pos1 (U,F,L)
            for label in (self.last_pos1 or {}):
                face = label[0].upper()
                med = med_for_label_from_mem(lab1_cie, hsv1, self.last_pos1, label)
                if med is not None:
                    face_groups.setdefault(face, {})[label] = med

            # pos2 (D,R,B)
            for label in (self.last_pos2 or {}):
                face = label[0].upper()
                med = med_for_label_from_mem(lab2_cie, hsv2, self.last_pos2, label)
                if med is not None:
                    face_groups.setdefault(face, {})[label] = med

            for face, label_map in face_groups.items():
                vals = list(label_map.values())
                if vals:
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
                    color_letter = self.face_to_color_expected.get(face)
                    if color_letter:
                        centroids[color_letter] = [float(med[0]), float(med[1]), float(med[2])]

        # fill missing with previous calibration or canonical
        for c in CANONICAL_BGR.keys():
            if c not in centroids:
                if c in self.centroids_lab:
                    centroids[c] = list(self.centroids_lab[c])
                else:
                    centroids[c] = bgr2lab(CANONICAL_BGR[c])

        self.centroids_lab = centroids
        self._update_centroid_np()
        if self.calib_path:
            self.save_calibration()
        return self.centroids_lab


    def detect_on_image(self,
                        img: np.ndarray,
                        positions: Dict[str, Tuple[int, int]],
                        use_clahe: bool = True,
                        top_k_candidates: int = 2) -> Tuple[Dict[str, str], Dict[str, Tuple[float, float, float]]]:
        """
        Detect colors for a single image.
        Vectorized: convert full image to LAB (OpenCV) once and to CIE-Lab float once.
        Also compute HSV once.
        Returns (detected_map, labs_map)
        """
        if img is None:
            raise ValueError("img is None")
        face_to_color_expected = self.face_to_color_expected

        img_proc = apply_gray_world(img) if self.use_gray_world else img.copy()

        # Apply CLAHE once on L channel if requested
        lab_op_full = cv2.cvtColor(img_proc, cv2.COLOR_BGR2LAB)
        if use_clahe:
            if self._clahe is None:
                self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            L, A, B = cv2.split(lab_op_full)
            L2 = self._clahe.apply(L)
            lab_op_full = cv2.merge([L2, A, B])
            img_proc = cv2.cvtColor(lab_op_full, cv2.COLOR_LAB2BGR)

        # Convert full opencv-LAB to CIE-Lab floats (vectorized)
        lab_cie_full = np.empty_like(lab_op_full, dtype=float)
        lab_cie_full[:, :, 0] = lab_op_full[:, :, 0] * (100.0 / 255.0)
        lab_cie_full[:, :, 1] = lab_op_full[:, :, 1] - 128.0
        lab_cie_full[:, :, 2] = lab_op_full[:, :, 2] - 128.0

        # compute HSV full
        hsv_full = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV)

        h_img, w_img = img_proc.shape[:2]

        pts = list(positions.values())
        if len(pts) >= 2:
            median_nn = _compute_median_nn(positions)
            roi_size = int(max(8, min(64, median_nn * self.roi_factor)))
        else:
            roi_size = 20

        detected: Dict[str, str] = {}
        labs: Dict[str, Tuple[float, float, float]] = {}

        medlabs_per_label: Dict[str, Optional[List[float]]] = {}
        for label, pos in positions.items():
            try:
                x, y = int(pos[0]), int(pos[1])
            except Exception:
                medlabs_per_label[label] = None
                labs[label] = (0.0, 0.0, 0.0)
                continue
            x = max(0, min(w_img - 1, x))
            y = max(0, min(h_img - 1, y))
            r = max(1, int(roi_size // 2))
            x0, x1 = max(0, x - r), min(w_img, x + r + 1)
            y0, y1 = max(0, y - r), min(h_img, y + r + 1)
            if x1 <= x0 or y1 <= y0:
                med = None
            else:
                lab_roi = lab_cie_full[y0:y1, x0:x1]
                hsv_roi = hsv_full[y0:y1, x0:x1]
                med = _median_lab_from_labhsv(lab_roi, hsv_roi)
            medlabs_per_label[label] = med
            if med is not None:
                labs[label] = (float(med[0]), float(med[1]), float(med[2]))
            else:
                labs[label] = (0.0, 0.0, 0.0)

        cent_np = self._centroid_np
        letters = self._centroid_letters

        centers_by_face = {
            'U': 'U5', 'R': 'R5', 'F': 'F5',
            'D': 'D5', 'L': 'L5', 'B': 'B5'
        }

        measured_centers: Dict[str, List[float]] = {}
        for face, center_label in centers_by_face.items():
            expected_color = face_to_color_expected.get(face)
            med = medlabs_per_label.get(center_label)
            if med is not None and expected_color:
                stored = self.centroids_lab.get(expected_color)
                if stored is not None:
                    try:
                        d = ciede2000(med, stored)
                    except Exception:
                        d = float(np.linalg.norm(np.array(med) - np.array(stored)))
                else:
                    d = 0.0
                if d < 35.0:
                    measured_centers[expected_color] = med
                detected[center_label] = expected_color
            else:
                if expected_color and expected_color not in self.centroids_lab:
                    self.centroids_lab[expected_color] = bgr2lab(CANONICAL_BGR.get(expected_color, (200, 200, 200)))
                if expected_color:
                    detected[center_label] = expected_color

        if self.persist_calib_if_centers_ok and len(measured_centers) == 6:
            for k, v in measured_centers.items():
                self.centroids_lab[k] = v
            if self.calib_path:
                self.save_calibration()
            self._update_centroid_np()
            cent_np = self._centroid_np

        for label, med_lab in medlabs_per_label.items():
            if label in detected:
                continue
            if med_lab is None:
                detected[label] = 'W'
                continue

            med_np = np.array(med_lab, dtype=float)
            if cent_np.size == 0:
                best = None
                best_d = float('inf')
                for c, ref in self.centroids_lab.items():
                    d = float(np.linalg.norm(med_np - np.array(ref)))
                    if d < best_d:
                        best_d = d
                        best = c
                assigned = best or 'W'
                detected[label] = assigned
                continue

            diff = cent_np - med_np
            d2 = np.sum(diff * diff, axis=1)
            idxs = np.argsort(d2)[:max(1, min(top_k_candidates, len(d2)))]
            best_exact = None
            best_exact_d = float('inf')
            for ii in idxs:
                cand_letter = letters[ii]
                ref_lab = self.centroids_lab.get(cand_letter)
                if ref_lab is None:
                    continue
                try:
                    d_exact = ciede2000(med_lab, ref_lab)
                except Exception:
                    d_exact = float(np.linalg.norm(med_np - np.array(ref_lab)))
                if d_exact < best_exact_d:
                    best_exact_d = d_exact
                    best_exact = cand_letter

            if best_exact is None:
                detected[label] = 'W'
                continue

            face = label[0].upper() if label and isinstance(label, str) else ''
            if face in centers_by_face:
                expected_center_color = face_to_color_expected.get(face)
                if expected_center_color and expected_center_color in self.centroids_lab:
                    try:
                        d_center = ciede2000(med_lab, self.centroids_lab[expected_center_color])
                    except Exception:
                        d_center = float(np.linalg.norm(med_np - np.array(self.centroids_lab[expected_center_color])))
                    if d_center <= self.center_threshold:
                        detected[label] = expected_center_color
                        continue

            if best_exact_d > self.global_threshold:
                pos = positions.get(label)
                if pos:
                    x, y = int(pos[0]), int(pos[1])
                    if _is_probably_white(med_lab, img_proc, x, y, roi_size):
                        detected[label] = 'W'
                    else:
                        detected[label] = best_exact
                else:
                    detected[label] = best_exact
            else:
                detected[label] = best_exact

        # conservative per-face correction: try to reduce obvious outliers
        for face, center_label in centers_by_face.items():
            center_color = face_to_color_expected.get(face)
            if not center_color:
                continue
            face_keys = [face + str(i) for i in range(1, 10)]
            vals = [detected[k] for k in face_keys if k in detected]
            if not vals:
                continue
            center_count = vals.count(center_color)
            if center_count < 6:
                outliers = [k for k in face_keys if k in detected and detected[k] != center_color]
                for k in outliers[:3]:
                    detected[k] = center_color

        return detected, labs

    def detect_images(self,
                      img1_path: Path,
                      pos1: Dict[str, Tuple[int, int]],
                      img2_path: Path,
                      pos2: Dict[str, Tuple[int, int]],
                      just_detections: bool = True,
                      face_to_color_expected: Optional[Dict[str, str]] = None,
                      use_gray_world: Optional[bool] = None,
                      img1: Optional[np.ndarray] = None,
                      img2: Optional[np.ndarray] = None) -> Tuple[Dict[str, str], Optional[Dict[str, Tuple[float, float, float]]]]:
        """
        High-level API used by api.py.
        Accepts either img1/img2 numpy arrays or img1_path/img2_path file paths.
        Stores last_img1/2 and last_pos1/2 for later calibration.
        """
        try:
            self.last_img1 = cv2.imread(str(img1_path)) if img1 is None else img1
            self.last_img2 = cv2.imread(str(img2_path)) if img2 is None else img2
            self.last_pos1 = pos1
            self.last_pos2 = pos2
            if use_gray_world is not None:
                self.use_gray_world = use_gray_world

            if face_to_color_expected is not None:
                self.face_to_color_expected = face_to_color_expected

            detected1, labs1 = self.detect_on_image(self.last_img1, pos1)
            detected2, labs2 = self.detect_on_image(self.last_img2, pos2)

            det = {}
            det.update(detected1)
            det.update(detected2)
            if just_detections:
                return det, None

            labs = {}
            labs.update(labs1)
            labs.update(labs2)
            return det, labs
        except Exception as e:
            print("Error detecting colors in img : ", e)
            return {}, None

    def debug_detection_output(self, image_path: Path, positions: Dict[str, Tuple[int, int]]):
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print("DEBUG: no se pudo abrir imagen:", image_path)
                return
            detected, labs = self.detect_on_image(img, positions, use_clahe=True)
        except Exception as e:
            print("DEBUG: detect_on_image falló:", e)
            return

        disp = img.copy()
        h_img, w_img = disp.shape[:2]
        pts = list(positions.values())
        median_nn = _compute_median_nn(positions) if len(pts) >= 2 else 40.0
        roi_size = int(max(8, min(64, median_nn * self.roi_factor)))

        print("\n--- DEBUG DETECCIÓN (MEJORADO) ---")
        print(f"Tamaño ROI (estimado): {roi_size}")
        print(f"{'Label':<6} | {'Assigned':<8} | {'Coords':<12} | {'Lab Mediana':<24}")
        print("-" * 100)

        for label, pos in positions.items():
            x, y = int(pos[0]), int(pos[1])
            med_lab = labs.get(label)
            assigned = detected.get(label, '?')
            if med_lab is None or (isinstance(med_lab, (list, tuple)) and tuple(map(float, med_lab)) == (0.0, 0.0, 0.0)):
                print(f"{label:<6} | {'N/A':<8} | ({x},{y})    | [ROI failed]")
                cv2.circle(disp, (x, y), 4, (0, 0, 255), -1)
                continue
            lab_str = f"L:{med_lab[0]:.1f},a:{med_lab[1]:.1f},b:{med_lab[2]:.1f}"
            print(f"{label:<6} | {assigned:<8} | ({x},{y})    | {lab_str:<24}")

            r = roi_size // 2
            cv2.rectangle(disp, (max(0, x - r), max(0, y - r)), (min(w_img - 1, x + r), min(h_img - 1, y + r)), (255, 200, 0), 1)
            color_bgr = (255, 255, 255)
            if assigned in CANONICAL_BGR:
                color_bgr = CANONICAL_BGR[assigned]
            cv2.circle(disp, (x, y), 6, color_bgr, -1)
            cv2.putText(disp, f"{assigned}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        print("-" * 100)
        cv2.imshow('DEBUG DETECCION - Presiona una tecla', disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("--- FIN DEBUG ---\n")


if __name__ == '__main__':
    print('detector.py: self-check debug mode (refactor)')
    try:
        try:
            js1 = json.load(open("positions/positions1.json", 'r', encoding='utf-8'))
            pos1 = js1.get("positions", js1)
        except Exception as e:
            print("Error loading pos 1 : ", e)
            pos1 = {}
        try:
            js2 = json.load(open("positions/positions2.json", 'r', encoding='utf-8'))
            pos2 = js2.get("positions", js2)
        except Exception:
            print("Error loading pos 2 : ", e)
            pos2 = {}
        
        det = ColorDetector(calib_path="positions/calibrations.json")
        
        if not pos1 or not pos2: 
            exit(1)

        while True:
            resp = input("Detect(d) or debug(db) or exit(x): ")
            if resp == "d":
                print("Lanzando debug_detection_output para IMG1...")
                det.debug_detection_output("templates/pictures/image1.png", pos1)
                det.debug_detection_output("templates/pictures/image2.png", pos2)
                print("No pude cargar positions1.json; revisá la ruta.")
            elif resp == "x":
                break
            elif resp == "db":
                print("Detecting images...")
                det.detect_images("templates/pictures/image1.png", pos1, "templates/pictures/image2.png", pos2, use_gray_world=True)
                det.calibrate_from_detection()
                print("Detection and calibration done.")
    except Exception as e:
        print("Error:", e)
