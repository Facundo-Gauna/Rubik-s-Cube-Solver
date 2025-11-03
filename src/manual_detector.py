# manual_detector2.py
# Single-file replacement for the manual detector two-photo workflow (photo1: U,F,L ; photo2: D,R,B)
# Features:
# - Polygon-based fixed positions (drag/save)
# - KMeans-based detection (colors + LAB centers stored)
# - ColorCorrection UI that edits both color and LAB
# - Per-face geometric rotation estimation (from P1->P3 vector)
# - Greedy rotation selection to ensure unique centers; brute-force (4^6) fallback
# - FaceCube / CubieCube verification (if kociemba_mod available)
# - Diagnostic net printing

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    import kociemba
except Exception:
    kociemba = None

try:
    import kociemba_module as kociemba_mod
except Exception:
    kociemba_mod = None

try:
    from scipy.spatial import distance
except Exception:
    distance = None

from config import config
from app_types import CubeState, CENTER_INDICES

class PolygonDetector:
    """
    Define and save positions for an image. Each image holds exactly 27 positions (3 faces x 9 labels).
    For image1 we expect faces ['U','F','L'] (in that order, each 9 labels).
    For image2 we expect faces ['D','R','B'] (in that order).
    """
    def __init__(self, faces_for_image: List[str], save_file: Path):
        # faces_for_image: e.g. ['U','F','L'] for image1
        self.faces_for_image = faces_for_image
        self.save_file = save_file
        self.img = None
        self.current_faces = []
        # create ordered label list like ['U1'..'U9','F1'..'F9','L1'..'L9']
        self.labels = [f + str(i) for f in self.faces_for_image for i in range(1,10)]
        self.positions: Dict[str, Tuple[int,int]] = {}
        self.selected_point: Optional[str] = None
        self.dragging = False
        self.drag_offset = (0,0)

    def define_positions_for(self, image_path: Path) -> Dict[str, Tuple[int,int]]:
        print(f"Defining positions on {image_path}")
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

        window_name = f"Define positions for {image_path.name} - Drag points, ENTER to save"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        print("Instructions: drag circles to move, ENTER save, ESC cancel, r reset to defaults")
        try:
            while True:
                display = self._draw_interface()
                cv2.imshow(window_name, display)
                key = cv2.waitKey(20) & 0xFF
                if key == 13:
                    self._save_positions_to_disk()
                    break
                elif key == 27:
                    print("Cancelled by user")
                    break
                elif key == ord('r'):
                    self.positions = default_positions.copy()
        finally:
            cv2.destroyAllWindows()
        return {k: tuple(v) for k, v in self.positions.items()}

    def _create_default_positions_for(self, img_shape):
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
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.selected_point:
            nx = max(0, min(self.img.shape[1]-1, x - self.drag_offset[0]))
            ny = max(0, min(self.img.shape[0]-1, y - self.drag_offset[1]))
            self.positions[self.selected_point] = (int(nx), int(ny))
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def _draw_interface(self):
        disp = self.img.copy()
        for label in self.labels:
            x, y = self.positions[label]
            color = (0, 200, 0) if label != self.selected_point else (0,255,255)
            r = 10 if label != self.selected_point else 14
            cv2.circle(disp, (int(x), int(y)), r, color, -1)
            cv2.circle(disp, (int(x), int(y)), r, (0,0,0), 2)
            cv2.putText(disp, label, (int(x)+18, int(y)+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return disp

    def _load_saved_positions(self):
        if not self.save_file.exists():
            return {}
        try:
            with open(self.save_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_positions_to_disk(self):
        data = {}
        if self.save_file.exists():
            try:
                with open(self.save_file, 'r') as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data['positions'] = {k: [int(v[0]), int(v[1])] for k, v in self.positions.items()}
        with open(self.save_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Template saved to {self.save_file}")

class ColorDetector:
    def __init__(self):
        # HSV heuristics used when mapping clusters -> letters
        self.hsv_ranges = {
            'R': [((0, 80, 50), (10, 255, 255)), ((170, 80, 50), (179, 255, 255))],
            'O': [((10, 80, 50), (25, 255, 255))],
            'Y': [((25, 80, 50), (35, 255, 255))],
            'G': [((36, 40, 40), (85, 255, 255))],
            'B': [((90, 40, 40), (140, 255, 255))],
            'W': [((0, 0, 180), (179, 80, 255))]
        }

    def detect_single_image(self, image_path: Path, positions: Dict[str, Tuple[int,int]]) -> Tuple[Dict[str,str], Dict[str,Tuple[int,int]]]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(image_path)
        roi_size = self._compute_optimal_roi_size(positions, img.shape)
        labels = []
        features = []
        hsv_values = []
        lab_values = []
        for label, (x, y) in positions.items():
            x, y = int(x), int(y)
            roi = self._extract_roi(img, x, y, roi_size)
            if roi is None or roi.size == 0:
                continue
            dominant_bgr = self._get_robust_dominant_color(roi)
            if dominant_bgr is None:
                continue
            hsv = cv2.cvtColor(np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            lab = cv2.cvtColor(np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
            features.append([float(lab[1]), float(lab[2])])
            hsv_values.append(hsv)
            lab_values.append(tuple(int(v) for v in lab))
            labels.append(label)
        if len(features) < 6:
            print(f"Warning: only {len(features)} samples found in {image_path.name}")
            return {k:'unknown' for k in positions.keys()}, {k:(0,0,0) for k in positions.keys()}
        cluster_labels = self._cluster_features(features)
        cluster_colors = self._assign_colors_to_clusters(hsv_values, cluster_labels)
        detected = {lab: cluster_colors.get(cid, 'unknown') for lab, cid in zip(labels, cluster_labels)}
        detected_lab = {lab: labv for lab, labv in zip(labels, lab_values)}
        # fill missing
        for k in positions.keys():
            if k not in detected:
                detected[k] = 'unknown'
            if k not in detected_lab:
                detected_lab[k] = (0,0,0)
        print(f"Detected {sum(1 for v in detected.values() if v!='unknown')}/{len(detected)} on {image_path.name}")
        return detected, detected_lab

    def _extract_roi(self, img, x, y, size):
        r = size // 2
        h, w = img.shape[:2]
        x0 = max(0, x - r); x1 = min(w, x + r + 1)
        y0 = max(0, y - r); y1 = min(h, y + r + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        roi = img[y0:y1, x0:x1]
        roi = cv2.GaussianBlur(roi, (3,3), 0)
        return roi

    def _get_robust_dominant_color(self, roi):
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape(-1, 3).astype(float)
        Q1 = np.percentile(pixels, 25, axis=0)
        Q3 = np.percentile(pixels, 75, axis=0)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = np.all((pixels >= lower) & (pixels <= upper), axis=1)
        filt = pixels[mask]
        if len(filt) > 0:
            med = np.median(filt, axis=0).astype(np.uint8)
            bgr = cv2.cvtColor(np.uint8([[med]]), cv2.COLOR_LAB2BGR)[0][0]
            return bgr
        med = np.median(roi.reshape(-1,3), axis=0).astype(np.uint8)
        return med

    def _cluster_features(self, features):
        arr = np.array(features)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(arr)
        kmeans = KMeans(n_clusters=6, random_state=0, n_init=10)
        labels = kmeans.fit_predict(scaled)
        return labels

    def _assign_colors_to_clusters(self, hsv_values, cluster_labels):
        cl_points = defaultdict(list)
        for i, cid in enumerate(cluster_labels):
            cl_points[cid].append(hsv_values[i])
        centers = {cid: np.median(np.array(pts), axis=0) for cid, pts in cl_points.items()}
        cluster_colors = {}
        used = set()
        # try to match each center to HSV ranges
        for cid, center in centers.items():
            col = self._get_color_from_hsv(center)
            if col != 'unknown' and col not in used:
                cluster_colors[cid] = col
                used.add(col)
        # remaining clusters -> remaining colors (deterministic order)
        all_colors = list(self.hsv_ranges.keys())
        rem_colors = [c for c in all_colors if c not in used]
        rem_clusters = [c for c in centers.keys() if c not in cluster_colors]
        for cid, col in zip(rem_clusters, rem_colors):
            cluster_colors[cid] = col
        return cluster_colors

    def _get_color_from_hsv(self, hsv):
        h, s, v = hsv
        wl, wh = self.hsv_ranges['W'][0]
        if wl[0] <= h <= wh[0] and wl[1] <= s <= wh[1] and wl[2] <= v <= wh[2]:
            return 'W'
        for cname, ranges in self.hsv_ranges.items():
            if cname == 'W':
                continue
            for low, high in ranges:
                if low[0] <= h <= high[0] and low[1] <= s <= high[1] and low[2] <= v <= high[2]:
                    return cname
        return 'unknown'

    def _compute_optimal_roi_size(self, positions, img_shape):
        pts = np.array(list(positions.values()))
        if pts.shape[0] < 2 or distance is None:
            return 20
        dist = distance.cdist(pts, pts)
        np.fill_diagonal(dist, np.inf)
        min_d = float(np.min(dist))
        size = max(8, min(int(min_d * 0.35), 40))
        return size

class ColorCorrectionUI:
    """
    Display each image with circles on sticker positions.
    Clicking selects a sticker; keys 1..6 assign colors (R,O,Y,G,B,W).
    When a sticker color is set manually, its LAB center is stored/updated.
    TAB switches between images. ENTER accepts; ESC cancels (revert).
    """
    def __init__(self):
        # BGR colors for visualization only
        self.color_map_bgr = {'R':(0,0,255),'O':(0,165,255),'Y':(0,255,255),'G':(0,255,0),'B':(255,0,0),'W':(255,255,255)}
        self.color_names = ['R','O','Y','G','B','W']
        self.selected = None
        self.current_image_idx = 0
        self.image_paths = []
        self.images = [None, None]
        self.positions = [None, None]
        self.colors = [None, None]   # dict label->color char
        self.labs = [None, None]     # dict label->lab tuple
        # color center mapping (LAB) updated when user sets a sticker => color_centers[color] = lab
        self.color_centers = {}

    def run(self, img1_path: Path, img2_path: Path,
            positions1: Dict[str,Tuple[int,int]], positions2: Dict[str,Tuple[int,int]],
            detections1: Dict[str,str], labs1: Dict[str,Tuple[int,int,int]],
            detections2: Dict[str,str], labs2: Dict[str,Tuple[int,int,int]]) -> Tuple[Dict[str,str], Dict[str,Tuple[int,int,int]], Dict[str,str], Dict[str,Tuple[int,int,int]]]:
        self.image_paths = [img1_path, img2_path]
        self.images[0] = cv2.imread(str(img1_path))
        self.images[1] = cv2.imread(str(img2_path))
        if self.images[0] is None or self.images[1] is None:
            raise FileNotFoundError("One of the images could not be opened")
        self.positions[0] = positions1
        self.positions[1] = positions2
        self.colors[0] = detections1.copy()
        self.colors[1] = detections2.copy()
        self.labs[0] = labs1.copy()
        self.labs[1] = labs2.copy()

        window = "Color Correction UI: click sticker to select; 1..6 assign color; TAB switch; ENTER accept"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, self._mouse_cb)
        print("Color Correction UI: click sticker to select; 1..6 assign color; TAB switch; r rotate face; ENTER accept")
        try:
            while True:
                display = self._draw()
                cv2.imshow(window, display)
                key = cv2.waitKey(20) & 0xFF
                if key == 13:  # ENTER
                    break
                elif key == 27:  # ESC
                    # cancel -> revert (just return original detections passed in)
                    print("Canceled by user; reverting")
                    return detections1, labs1, detections2, labs2
                elif key == 9:  # TAB
                    self.current_image_idx = 1 - self.current_image_idx
                elif ord('1') <= key <= ord('6') and self.selected:
                    idx = key - ord('1')
                    color = self.color_names[idx]
                    # set color and update lab center mapping
                    self.colors[self.current_image_idx][self.selected] = color
                    labval = self.labs[self.current_image_idx].get(self.selected)
                    if labval:
                        self.color_centers[color] = labval
                    print(f"Set {self.selected} -> {color} (image {self.current_image_idx+1}) and updated LAB")
                elif key == ord('r'):
                    # rotate face visualization only (useful if user wants to check orientation)
                    # rotation done by dragging in polygon UI, not here
                    pass
        finally:
            cv2.destroyAllWindows()
        return self.colors[0], self.labs[0], self.colors[1], self.labs[1]

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            posmap = self.positions[self.current_image_idx]
            min_d = float('inf'); closest = None
            for lab, (px, py) in posmap.items():
                d = math.hypot(x-px, y-py)
                if d < min_d and d < 40:
                    min_d = d; closest = lab
            if closest:
                self.selected = closest
                print(f"Selected {closest} on image {self.current_image_idx+1}")

    def _draw(self):
        disp = self.images[self.current_image_idx].copy()
        pos = self.positions[self.current_image_idx]
        det = self.colors[self.current_image_idx]
        for label, (x,y) in pos.items():
            color_name = det.get(label, 'unknown')
            color_bgr = self.color_map_bgr.get(color_name, (128,128,128))
            cv2.circle(disp, (int(x),int(y)), 14, color_bgr, -1)
            cv2.circle(disp, (int(x),int(y)), 14, (0,0,0), 2)
            if label == self.selected:
                cv2.circle(disp, (int(x),int(y)), 18, (255,255,0), 3)
            cv2.putText(disp, f"{label}:{color_name}", (int(x)+16,int(y)+6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255),1)
        info = f"Image {self.current_image_idx+1} - ENTER accept - TAB switch"
        cv2.putText(disp, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
        return disp

ROT90 = [6,3,0,7,4,1,8,5,2]

def rotate_block_list(vals9: List, times: int) -> List:
    out = list(vals9)
    for _ in range(times % 4):
        out = [out[i] for i in ROT90]
    return out

def build_color_net_text(color_str: str) -> str:
    faces = ['U','R','F','D','L','B']
    out = []
    for fi, face in enumerate(faces):
        out.append(f"\n{face}:")
        block = color_str[fi*9:(fi+1)*9]
        for r in range(3):
            out.append(' '.join(block[r*3:(r+1)*3]))
    return '\n'.join(out)

def coords_toprow_angle(labels9: List[str], positions: Dict[str,Tuple[int,int]]) -> float:
    p1 = positions[labels9[0]]    # top-left
    p3 = positions[labels9[2]]    # top-right
    vx = p3[0] - p1[0]
    vy = p3[1] - p1[1]
    ang = math.degrees(math.atan2(vy, vx))
    # normalize
    if ang < -180:
        ang += 360
    if ang >= 180:
        ang -= 360
    return ang

def guess_two_candidates(labels9: List[str], positions: Dict[str,Tuple[int,int]]):
    ang = coords_toprow_angle(labels9, positions)
    quant = int(round(ang / 90.0))
    candA = (-quant) % 4
    candB = (quant) % 4
    return candA, candB, ang

def brute_force_face_orientations(face_color_string: str):
    """
    Tries all 4^6 = 4096 face rotation combos. Returns (valid_facelets, solution) or (None,None).
    """
    if kociemba is None:
        print("kociemba not installed; cannot brute force.")
        return None, None
    blocks = [face_color_string[i*9:(i+1)*9] for i in range(6)]
    import itertools, time
    print("Brute-force: trying all face rotations (4^6)...")
    t0 = time.time()
    for comb in itertools.product(range(4), repeat=6):
        candidate_blocks = [rotate_block_list(blocks[i], comb[i]) for i in range(6)]
        candidate_color_str = ''.join(''.join(b) for b in candidate_blocks)
        # map centers -> faces
        color_to_face = {}
        ok = True
        for face_letter, idx in CENTER_INDICES.items():
            c = candidate_color_str[idx]
            if c in color_to_face:
                ok = False; break
            color_to_face[c] = face_letter
        if not ok:
            continue
        candidate_facelets = ''.join(color_to_face[c] for c in candidate_color_str)
        try:
            sol = kociemba.solve(candidate_facelets)
            t1 = time.time()
            print(f"Found valid orientation after {t1-t0:.2f}s. combo: {comb}")
            return candidate_facelets, sol
        except Exception:
            continue
    print("Brute-force exhausted; no valid orientation found.")
    return None, None

class ManualDetectionWorkflow:
    def __init__(self):
        # face groupings per image: user specified "photo1 = U,F,L ; photo2 = D,R,B"
        self.faces_image1 = ['U','F','L']
        self.faces_image2 = ['D','R','B']
        # create polygon detectors for both images (save to positions1.json and positions2.json)
        self.poly1 = PolygonDetector(self.faces_image1, config.SAVE_FILE1)
        self.poly2 = PolygonDetector(self.faces_image2, config.SAVE_FILE2)
        self.detector = ColorDetector()
        self.ui = ColorCorrectionUI()
        self.cube_state = CubeState()

    def build_facelets(self,
                       det1: Dict[str,str],
                       labs1: Dict[str,Tuple[int,int,int]],
                       det2: Dict[str,str],
                       labs2: Dict[str,Tuple[int,int,int]],
                       pos1: Dict[str,Tuple[int,int]],
                       pos2: Dict[str,Tuple[int,int]]) -> Tuple[str,str]:
        """
        - Merge detections & labs & positions
        - For each face, guess rotation (two candidates) and pick good candidate greedily to favor unique centers.
        - If final centers not unique or kociemba rejects, try greedy per-face rotation improvement, then brute force (4^6).
        Returns (color_str, facelet_str) where facelet_str is URFDLB letters for each facelet.
        """

        # merge maps for convenience
        merged_det = {}
        merged_det.update(det1 or {})
        merged_det.update(det2 or {})
        merged_lab = {}
        merged_lab.update(labs1 or {})
        merged_lab.update(labs2 or {})
        merged_pos = {}
        merged_pos.update(pos1 or {})
        merged_pos.update(pos2 or {})

        # define expected ordered face labels for each of the 6 canonical faces in the polygon layout
        # each face's labels come in the polygon saved order (top-left -> ... -> bottom-right)
        face_labels_ordered = []
        for f in self.faces_image1 + self.faces_image2:  # results order: U,F,L,D,R,B
            face_labels_ordered.append([f + str(i) for i in range(1,10)])

        # canonical result list in U,R,F,D,L,B order required by solver
        # BUT note: our detection faces order is [U,F,L,D,R,B]; we must reorder blocks to canonical
        # canonical faces order for building color string:
        canonical_faces_req = ['U','R','F','D','L','B']

        # Step A: per-face guess rotations (two candidates each)
        per_face_candidates = {}  # face -> (candA,candB, angle)
        face_to_labels = {}
        for labels in face_labels_ordered:
            face = labels[0][0]  # first char e.g. 'U' or 'F'
            face_to_labels[face] = labels
            if any(lbl not in merged_pos for lbl in labels):
                raise ValueError(f"Missing positions for face {face}: expected labels {labels}")
            candA, candB, ang = guess_two_candidates(labels, merged_pos)
            per_face_candidates[face] = (candA, candB, ang)

        # Step B: choose one candidate rotation per face greedily to try to keep centers unique
        chosen_rot = {}
        chosen_blocks = {}
        chosen_centers = set()
        for face in face_to_labels.keys():
            labels = face_to_labels[face]
            candA, candB, ang = per_face_candidates[face]
            blockA = [merged_det[l] for l in labels]
            blockA_rot = rotate_block_list(blockA, candA)
            blockB = [merged_det[l] for l in labels]
            blockB_rot = rotate_block_list(blockB, candB)
            cA = blockA_rot[4]
            cB = blockB_rot[4]
            # prefer candidate not colliding with already chosen centers
            if cA not in chosen_centers:
                chosen_rot[face] = candA
                chosen_blocks[face] = blockA_rot
                chosen_centers.add(cA)
            elif cB not in chosen_centers:
                chosen_rot[face] = candB
                chosen_blocks[face] = blockB_rot
                chosen_centers.add(cB)
            else:
                # both collide - choose candA by default
                chosen_rot[face] = candA
                chosen_blocks[face] = blockA_rot
                chosen_centers.add(cA)

        # Now we have chosen_blocks in the detection order (faces_image1 + faces_image2 order)
        # We need to build the 6 blocks in canonical solver order U,R,F,D,L,B
        # map: detection order -> canonical index
        # detection order currently: e.g. ['U','F','L','D','R','B']
        detection_order = list(face_to_labels.keys())
        # reorder chosen_blocks into canonical ordering
        blocks_by_face = {face: chosen_blocks[face] for face in chosen_blocks}

        # Build color blocks in canonical order
        canonical_blocks = []
        for face in canonical_faces_req:
            if face not in blocks_by_face:
                # possibly our detection order had other mapping (defensive)
                raise ValueError(f"Face {face} not present in chosen blocks (check mapping)")
            canonical_blocks.append(blocks_by_face[face])

        # Build color_str
        color_str = ''.join(''.join(block) for block in canonical_blocks)
        print("Built color string:", color_str)
        print(build_color_net_text(color_str))
        print("Center colors (by face):", {face: color_str[idx] for face, idx in CENTER_INDICES.items()})
        # quick counts
        ctr = Counter(color_str)
        print("Color counts:", ctr)

        # Quick centers uniqueness check
        centers = [color_str[idx] for idx in CENTER_INDICES.values()]
        if len(set(centers)) != 6:
            print("Centers not unique after greedy rotation selection; attempting per-face improvement.")
            # Try per-face local improvement: for each face try rotations 0..3 pick rotation avoiding duplicates
            faces_list = canonical_faces_req.copy()
            improved = False
            # do greedy improvement loop that tries to reduce duplicate centers
            for face in faces_list:
                labels = [face + str(i) for i in range(1,10)]
                best_rot = chosen_rot[face]
                best_center = rotate_block_list([merged_det[l] for l in labels], best_rot)[4]
                # try all rotations and pick center that reduces duplicates globally
                best_choice = best_rot
                best_score = None
                for r in range(4):
                    block_r = rotate_block_list([merged_det[l] for l in labels], r)
                    # compute centers if this rotation used
                    trial_centers = []
                    for f in canonical_faces_req:
                        if f == face:
                            trial_centers.append(block_r[4])
                        else:
                            trial_centers.append(rotate_block_list([merged_det[f+str(i)] for i in range(1,10)], chosen_rot[f])[4])
                    # score number distinct
                    score = len(set(trial_centers))
                    if best_score is None or score > best_score:
                        best_score = score
                        best_choice = r
                # apply best_choice
                chosen_rot[face] = best_choice
                chosen_blocks[face] = rotate_block_list([merged_det[face+str(i)] for i in range(1,10)], best_choice)
            # rebuild canonical blocks & color_str after improvement
            canonical_blocks = [chosen_blocks[face] for face in canonical_faces_req]
            color_str = ''.join(''.join(b) for b in canonical_blocks)
            centers = [color_str[idx] for idx in CENTER_INDICES.values()]
            if len(set(centers)) == 6:
                print("Centers became unique after local improvement.")
                improved = True
            else:
                print("Local improvement couldn't ensure unique centers.")
                improved = False

            if not improved:
                # fallback to brute force rotations of faces (4^6)
                print("Falling back to brute-force rotations across faces (4^6 combinations). This can take a moment.")
                cand_facelets, sol = brute_force_face_orientations(color_str)
                if cand_facelets:
                    print("Brute-force produced a valid facelet string.")
                    return color_str, cand_facelets
                else:
                    raise ValueError("Failed to produce valid facelet mapping after brute-force.")
        else:
            print("Centers appear unique; proceeding to FaceCube verification (if available)")

        # Map centers -> face letters and convert color_str -> facelet_str
        color_to_face = {}
        for face, idx in CENTER_INDICES.items():
            c = color_str[idx]
            if c in color_to_face:
                print("Duplicate center color found during mapping:", c)
            color_to_face[c] = face

        facelet_str = ''.join(color_to_face.get(c, '?') for c in color_str)
        print("Converted to facelet string:", facelet_str)

        # Final verification attempt: use kociemba_mod FaceCube if available to get good diagnostics
        if kociemba_mod is not None:
            try:
                fc = kociemba_mod.FaceCube(facelet_str)
                cub = fc.toCubieCube()
                if cub is None:
                    print("FaceCube -> CubieCube produced None (invalid).")
                    # try brute force fallback
                    cand_facelets, sol = brute_force_face_orientations(color_str)
                    if cand_facelets:
                        return color_str, cand_facelets
                    raise ValueError("FaceCube->CubieCube verification failed")
                else:
                    # convert back to facelet to normalize orientation and check
                    back = cub.toFaceCube().to_String()
                    print("FaceCube verification OK.")
            except Exception as e:
                print("FaceCube diagnostic error:", e)
                # try brute force fallback
                cand_facelets, sol = brute_force_face_orientations(color_str)
                if cand_facelets:
                    return color_str, cand_facelets
                raise ValueError("FaceCube verification failed and brute-force failed.")
        else:
            # if no kociemba_mod, we still attempt to call kociemba.solve to check
            if kociemba is not None:
                try:
                    kociemba.solve(facelet_str)
                    print("kociemba accepted the facelet string (initial orientation).")
                except Exception as e:
                    print("kociemba rejected initial facelets:", e)
                    cand_facelets, sol = brute_force_face_orientations(color_str)
                    if cand_facelets:
                        return color_str, cand_facelets
                    raise ValueError("kociemba rejected and brute force failed.")
            else:
                print("No kociemba library available to validate. Returning built facelets for inspection.")

        return color_str, facelet_str

    def run(self) -> Tuple[bool,str,str]:
        # define positions for images (drag/save)
        pos1 = self.poly1.define_positions_for(config.IMG1_PATH)
        pos2 = self.poly2.define_positions_for(config.IMG2_PATH)

        # detect colors + labs
        det1, labs1 = self.detector.detect_single_image(config.IMG1_PATH, pos1)
        det2, labs2 = self.detector.detect_single_image(config.IMG2_PATH, pos2)

        # color correction UI (manually refine)
        det1_corr, labs1_corr, det2_corr, labs2_corr = self.ui.run(config.IMG1_PATH, config.IMG2_PATH,
                                                                     pos1, pos2, det1, labs1, det2, labs2)

        print("Final corrected detections (image1) colors:", det1_corr)
        sample_labs1 = {k: labs1_corr[k] for k in list(labs1_corr.keys())[:6]}
        print("Final corrected detections (image1) labs sample:", sample_labs1)
        print("Final corrected detections (image2) colors:", det2_corr)
        sample_labs2 = {k: labs2_corr[k] for k in list(labs2_corr.keys())[:6]}
        print("Final corrected detections (image2) labs sample:", sample_labs2)

        # build facelets and attempt solve
        try:
            color_str, facelet = self.build_facelets(det1_corr, labs1_corr, det2_corr, labs2_corr, pos1, pos2)
        except Exception as e:
            print("Failed to build/verify facelets:", e)
            return False, '', ''

        # call solver
        if kociemba is not None:
            try:
                sol = kociemba.solve(facelet)
                print("Solution:", sol)
                return True, sol, color_str
            except Exception as e:
                print("kociemba failed:", e)
                # final fallback: try brute-force orientation as last try
                cand_facelets, sol = brute_force_face_orientations(color_str)
                if cand_facelets:
                    try:
                        sol = kociemba.solve(cand_facelets)
                        print("Solved after brute-force orientation. Solution:", sol)
                        return True, sol, color_str
                    except Exception as e2:
                        print("Even after brute-force orientation kociemba failed:", e2)
                        return False, '', color_str
                return False, '', color_str
        else:
            print("kociemba not installed; returning facelet string for inspection.")
            return True, '', color_str

if __name__ == "__main__":
    wf = ManualDetectionWorkflow()
    ok, solution, color_str = wf.run()
    if ok and solution:
        print("Solved:", solution)
    elif ok:
        print("Facelets built (no solver run):", color_str)
    else:
        print("Failed to generate valid solution.")
