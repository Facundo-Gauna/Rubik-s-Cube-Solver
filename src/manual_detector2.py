#!/usr/bin/env python3
"""
qbr_with_polygon_template.py

QBR-style 6-face capture and solver, but uses a fixed polygon template (3x3)
for sampling sticker positions instead of QBR's dynamic contour detection.

Place next to your project files: config.py, app_types.py, vision_params.py, etc.
Run: python qbr_with_polygon_template.py

Author: adapted for user's project
"""

import json
import math
import itertools
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import numpy as np

# kociemba (solver) and optional local kociemba module with FaceCube diagnostics
try:
    import kociemba
except Exception:
    kociemba = None

try:
    import kociemba_module as kmod  # your local copy of the library that exposes FaceCube
except Exception:
    kmod = None

# project imports (these must exist in your repo)
from config import config
from app_types import CubeState, CENTER_INDICES, clean_color_string

# vision params (QBR thresholds)
try:
    import vision_params
except Exception:
    class DummyVP:
        rgb_L = 50
        sat_W = 60
        val_W = 150
        sigma_W = 300
        sigma_C = 5
        delta_C = 5
        orange_L = 6
        orange_H = 23
        yellow_H = 50
        green_H = 100
        blue_H = 160
        face_col = []
        face_hsv = []
        cube_col = {}
        cube_hsv = {}
    vision_params = DummyVP()

# ---------------------------
# Configuration: if you want to force capture order (QBR-style), set FIXED_CAPTURE_ORDER.
# Example (colors as single letters): FIXED_CAPTURE_ORDER = ['G','R','B','O','W','Y']
# If None, the code will map images to cube faces by reading the center sample.
FIXED_CAPTURE_ORDER = None
# If you want to force a particular file naming for six images, edit these:
DEFAULT_FILENAMES = [
    "detector_1.png",
    "detector_2.png",
    "detector_3.png",
    "detector_4.png",
    "detector_5.png",
    "detector_6.png",
]

TEMPLATE_FILE = Path(__file__).parent / "positions_template.json"

# ---------------------------
# PolygonTemplate: save/load 3x3 positions, interactive editor
# ---------------------------
class PolygonTemplate:
    labels = [f"P{i}" for i in range(1, 10)]

    def __init__(self):
        self.positions: Dict[str, Tuple[int,int]] = {}
        self._img = None
        self._sel = None
        self._dragging = False
        self._offset = (0,0)

    def _default_positions(self, w, h):
        pos = {}
        rows, cols = 3, 3
        for i, label in enumerate(self.labels):
            r = i // cols
            c = i % cols
            x = int((c + 0.5) * (w / cols))
            y = int((r + 0.5) * (h / rows))
            pos[label] = (x, y)
        return pos

    def load(self):
        if not TEMPLATE_FILE.exists():
            return {}
        try:
            with open(TEMPLATE_FILE, 'r') as f:
                obj = json.load(f)
            if 'positions' in obj:
                self.positions = {k: tuple(v) for k, v in obj['positions'].items() if k in self.labels}
            return self.positions
        except Exception:
            return {}

    def save(self):
        with open(TEMPLATE_FILE, 'w') as f:
            json.dump({'positions': {k: [int(v[0]), int(v[1])] for k, v in self.positions.items()}}, f, indent=2)
        print("Template saved to", TEMPLATE_FILE)

    def define_on_image(self, image_path: Path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(image_path)
        self._img = img.copy()
        h, w = img.shape[:2]
        saved = self.load()
        default = self._default_positions(w, h)
        self.positions = default.copy()
        self.positions.update({k: tuple(v) for k, v in saved.get('positions', {}).items()}) if saved else None

        window = f"Define template - drag dots - ENTER to save - r reset - ESC cancel ({image_path.name})"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, self._mouse_cb)
        print("Instructions: drag circles to move; ENTER save; ESC cancel; r reset to defaults")
        try:
            while True:
                disp = self._draw_display()
                cv2.imshow(window, disp)
                key = cv2.waitKey(20) & 0xFF
                if key == 13:  # Enter
                    self.save()
                    break
                elif key == 27:  # ESC
                    print("Cancelled.")
                    break
                elif key == ord('r'):
                    self.positions = default.copy()
        finally:
            cv2.destroyAllWindows()
        return self.positions

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            closest = None
            md = float('inf')
            for k, (px, py) in self.positions.items():
                d = math.hypot(px - x, py - y)
                if d < md and d < 40:
                    md = d
                    closest = k
            if closest:
                self._sel = closest
                self._dragging = True
                px, py = self.positions[closest]
                self._offset = (x - px, y - py)
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging and self._sel:
            nx = max(0, min(self._img.shape[1]-1, x - self._offset[0]))
            ny = max(0, min(self._img.shape[0]-1, y - self._offset[1]))
            self.positions[self._sel] = (int(nx), int(ny))
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False

    def _draw_display(self):
        d = self._img.copy()
        for k, (x, y) in self.positions.items():
            color = (0, 255, 0) if k != self._sel else (0, 255, 255)
            r = 10 if k != self._sel else 14
            cv2.circle(d, (int(x), int(y)), r, color, -1)
            cv2.circle(d, (int(x), int(y)), r, (0, 0, 0), 2)
            cv2.putText(d, k, (int(x) + 12, int(y) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return d

# ---------------------------
# QBR color decision (simplified, faithful to qbr getcolor)
# ---------------------------
def getcolor_from_hsv(hsv_pixel):
    mh, ms, mv = hsv_pixel
    # white check
    if ms <= vision_params.sat_W and mv >= vision_params.val_W:
        return 'white'
    if vision_params.orange_L <= mh < vision_params.orange_H:
        return 'orange'
    if vision_params.orange_H <= mh < vision_params.yellow_H:
        return 'yellow'
    if vision_params.yellow_H <= mh < vision_params.green_H:
        if ms < 150:
            return 'white'
        else:
            return 'green'
    if vision_params.green_H <= mh < vision_params.blue_H:
        if ms < 150:
            return 'white'
        else:
            return 'blue'
    # else red
    return 'red'

COLOR_NAME_TO_LETTER = {
    'red': 'R',
    'orange': 'O',
    'yellow': 'Y',
    'green': 'G',
    'blue': 'B',
    'white': 'W'
}

# ---------------------------
# Sampling: take ROI around template positions and compute median HSV & LAB
# ---------------------------
def sample_face_from_image(img, positions: Dict[str, Tuple[int,int]], roi_radius=12):
    colors_map = {}
    labs_map = {}
    hsv_map = {}
    for label, (x, y) in positions.items():
        x = int(x); y = int(y)
        x0 = max(0, x - roi_radius); x1 = min(img.shape[1], x + roi_radius + 1)
        y0 = max(0, y - roi_radius); y1 = min(img.shape[0], y + roi_radius + 1)
        roi = img[y0:y1, x0:x1]
        if roi is None or roi.size == 0:
            colors_map[label] = 'X'
            labs_map[label] = (0,0,0)
            hsv_map[label] = (0,0,0)
            continue
        roi_blur = cv2.GaussianBlur(roi, (5,5), 0)
        hsv_roi = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)
        lab_roi = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2LAB)
        med_h = int(np.median(hsv_roi[:,:,0]))
        med_s = int(np.median(hsv_roi[:,:,1]))
        med_v = int(np.median(hsv_roi[:,:,2]))
        med_lab = tuple(int(x) for x in np.median(lab_roi.reshape(-1,3), axis=0))
        cname = getcolor_from_hsv((med_h, med_s, med_v))
        color_letter = COLOR_NAME_TO_LETTER.get(cname, 'X')
        colors_map[label] = color_letter
        labs_map[label] = med_lab
        hsv_map[label] = (med_h, med_s, med_v)
    return colors_map, labs_map, hsv_map

# ---------------------------
# Color Correction UI: edits color letters and updates LAB when user changes the color.
# Also supports rotating the face (r) and switching (TAB).
# ---------------------------
class ColorCorrectionUI:
    def __init__(self):
        # mapping of color letter -> BGR for display
        self.color_order = ['R','O','Y','G','B','W']
        self.color_bgr = {
            'R': (0,0,255),
            'O': (0,165,255),
            'Y': (0,255,255),
            'G': (0,255,0),
            'B': (255,0,0),
            'W': (255,255,255)
        }
        self.selected = None
        self.idx = 0
        self.window = "Color Correction - TAB switch - 1..6 set color - r rotate - ENTER accept"

    def run(self, image_paths: List[Path], positions: Dict[str, Tuple[int,int]], colors_list: List[Dict[str,str]], labs_list: List[Dict[str,Tuple[int,int,int]]]):
        frames = [cv2.imread(str(p)) for p in image_paths]
        # keep original copies to allow cancel
        orig_colors = [dict(c) for c in colors_list]
        orig_labs = [dict(l) for l in labs_list]
        self.frames = frames
        self.positions = positions
        self.colors = [dict(c) for c in colors_list]
        self.labs = [dict(l) for l in labs_list]

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._mouse_cb)
        print("Color Correction UI: click sticker to select; 1..6 to set color; r rotate face; TAB switch image; ENTER accept; ESC cancel")
        try:
            while True:
                img = self.frames[self.idx].copy()
                for label, (x, y) in self.positions.items():
                    color_letter = self.colors[self.idx].get(label, 'X')
                    bgr = self.color_bgr.get(color_letter, (128,128,128))
                    cv2.circle(img, (int(x), int(y)), 16, bgr, -1)
                    cv2.circle(img, (int(x), int(y)), 16, (0,0,0), 2)
                    if label == self.selected:
                        cv2.circle(img, (int(x), int(y)), 22, (255,255,0), 3)
                    lab = self.labs[self.idx].get(label, (0,0,0))
                    cv2.putText(img, f"{label}:{color_letter} L{lab[0]} a{lab[1]} b{lab[2]}", (int(x)+16, int(y)+6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
                info = f"Image {self.idx+1}/{len(self.frames)} - TAB switch - ENTER accept"
                cv2.putText(img, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.imshow(self.window, img)
                key = cv2.waitKey(20) & 0xFF
                if key == 13:  # ENTER
                    break
                elif key == 27:  # ESC -> revert
                    self.colors = orig_colors
                    self.labs = orig_labs
                    break
                elif key == 9:  # TAB
                    self.idx = (self.idx + 1) % len(self.frames)
                elif ord('1') <= key <= ord('6'):
                    if self.selected is not None:
                        idx = key - ord('1')
                        letter = self.color_order[idx]
                        self.colors[self.idx][self.selected] = letter
                        # update LAB to canonical value derived from BGR
                        bgr = np.uint8([[self.color_bgr[letter]]])
                        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0][0]
                        self.labs[self.idx][self.selected] = (int(lab[0]), int(lab[1]), int(lab[2]))
                        print(f"Set {self.selected} -> {letter} (image {self.idx+1}) and updated LAB")
                elif key == ord('r'):
                    self._rotate_current_face()
        finally:
            cv2.destroyAllWindows()
        return self.colors, self.labs

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mind = float('inf'); closest = None
            for label, (px, py) in self.positions.items():
                d = math.hypot(px - x, py - y)
                if d < mind and d < 40:
                    mind = d; closest = label
            if closest:
                self.selected = closest
                print("Selected", closest, "on image", self.idx+1)

    def _rotate_current_face(self):
        ROT90 = [6,3,0,7,4,1,8,5,2]
        labels = [f"P{i}" for i in range(1, 10)]
        curr_colors = [self.colors[self.idx][l] for l in labels]
        curr_labs = [self.labs[self.idx][l] for l in labels]
        new_colors = [None]*9
        new_labs = [None]*9
        for out_idx, src_idx in enumerate(ROT90):
            new_colors[out_idx] = curr_colors[src_idx]
            new_labs[out_idx] = curr_labs[src_idx]
        for i, l in enumerate(labels):
            self.colors[self.idx][l] = new_colors[i]
            self.labs[self.idx][l] = new_labs[i]
        print(f"Rotated face {self.idx+1} clockwise")

# ---------------------------
# Build color string and facelet string (URFDLB) from six images of P1..P9
# - supports either fixed order list or mapping by centers
# ---------------------------
def build_from_six_images_by_centers(colors_per_img: List[Dict[str,str]], color_to_side_map: Dict[str,str]):
    """
    colors_per_img : list of 6 dicts mapping P1..P9 -> color letter
    color_to_side_map : dict mapping color letter -> face letter (U,R,F,D,L,B)
    Returns (color_str, facelet_str)
    """
    # first map each image to a cube face using center P5
    face_map = {}
    used_faces = set()
    for idx, cmap in enumerate(colors_per_img):
        center = cmap.get('P5')
        if center is None:
            raise ValueError(f"Image {idx+1} missing center")
        face_letter = color_to_side_map.get(center)
        if face_letter is None:
            raise ValueError(f"Center color {center} unknown in provided mapping")
        if face_letter in used_faces:
            raise ValueError(f"Duplicate mapping for face {face_letter} from image {idx+1}")
        used_faces.add(face_letter)
        face_map[face_letter] = [cmap[f"P{i}"] for i in range(1,10)]
    if len(face_map) != 6:
        raise ValueError("Not all 6 faces found from centers")
    order = ['U','R','F','D','L','B']
    color_str = ''.join(''.join(face_map[f]) for f in order)
    facelet_str = ''.join(color_to_side_map[ch] for ch in color_str)
    return clean_color_string(color_str), facelet_str

def build_from_six_images_fixed_order(colors_per_img: List[Dict[str,str]], fixed_colors_order: List[str]):
    """
    colors_per_img: list of 6 dicts (capture order)
    fixed_colors_order: list of 6 color letters that correspond to images 1..6 centers (e.g. ['G','R','B','O','W','Y'])
    Returns color_str and facelet_str using CubeState mapping.
    """
    if len(colors_per_img) != 6 or len(fixed_colors_order) != 6:
        raise ValueError("Expect 6 images and 6 fixed order color letters")
    cs = CubeState()
    # map each provided color letter to cube face via cs.color_to_side
    # fixed_colors_order[i] corresponds to colors_per_img[i]
    face_map = {}
    used_faces = set()
    for i, color_letter in enumerate(fixed_colors_order):
        face_letter = cs.color_to_side.get(color_letter)
        if face_letter is None:
            raise ValueError(f"Fixed order color {color_letter} not recognized by CubeState")
        if face_letter in used_faces:
            raise ValueError("Duplicate face in fixed order")
        used_faces.add(face_letter)
        cmap = colors_per_img[i]
        face_map[face_letter] = [cmap[f"P{j}"] for j in range(1, 10)]
    if len(face_map) != 6:
        raise ValueError("Fixed order produce incomplete face map")
    order = ['U','R','F','D','L','B']
    color_str = ''.join(''.join(face_map[f]) for f in order)
    cs_map = cs.color_to_side  # map side->color? we need color->side
    color_to_side = {v:k for k,v in cs_map.items()}
    facelet_str = ''.join(color_to_side[ch] for ch in color_str)
    return clean_color_string(color_str), facelet_str

# ---------------------------
# ASCII net for human checking
# ---------------------------
def print_ascii_net(color_str, facelet_str):
    blocks = [color_str[i*9:(i+1)*9] for i in range(6)]
    fblocks = [facelet_str[i*9:(i+1)*9] for i in range(6)]
    print("\n=== NET (color | face) ===")
    # U (index 0)
    u, uf = blocks[0], fblocks[0]
    print("      U")
    print(f"      {u[0:3]} | {uf[0:3]}")
    print(f"      {u[3:6]} | {uf[3:6]}")
    print(f"      {u[6:9]} | {uf[6:9]}")
    # Middle L(4),F(2),R(1),B(5)
    l,f,r,b = blocks[4], blocks[2], blocks[1], blocks[5]
    lf, ff, rf, bf = fblocks[4], fblocks[2], fblocks[1], fblocks[5]
    print("\nL | F | R | B")
    for row in range(3):
        print(f"{l[row*3:(row+1)*3]} | {f[row*3:(row+1)*3]} | {r[row*3:(row+1)*3]} | {b[row*3:(row+1)*3]}   |   "
              f"{lf[row*3:(row+1)*3]} | {ff[row*3:(row+1)*3]} | {rf[row*3:(row+1)*3]} | {bf[row*3:(row+1)*3]}")
    d, df = blocks[3], fblocks[3]
    print("\n      D")
    print(f"      {d[0:3]} | {df[0:3]}")
    print(f"      {d[3:6]} | {df[3:6]}")
    print(f"      {d[6:9]} | {df[6:9]}")
    print()

# ---------------------------
# Verification helper using kmod (if available) else try kociemba.solve and interpret exceptions
# ---------------------------
def verify_facelet_string(facelet_str):
    # returns 0 if ok or error code int otherwise
    if kmod is not None:
        try:
            fc = kmod.FaceCube(facelet_str)
            cc = fc.toCubieCube()
            return cc.verify()
        except Exception as e:
            print("kmod verify failed:", e)
            return -99
    else:
        # try calling kociemba.solve to see if it throws; can't get verify code
        try:
            if kociemba is None:
                print("kociemba not installed; cannot verify reliably")
                return -98
            kociemba.solve(facelet_str)  # if no exception, it's valid
            return 0
        except Exception as e:
            print("kociemba.solve error:", e)
            return -2

# ---------------------------
# Rotation-only brute force (4^6 combos)
# ---------------------------
def try_face_rotation_bruteforce(color_str_initial, color_to_face_map):
    ROT90 = [6,3,0,7,4,1,8,5,2]
    blocks = [color_str_initial[i*9:(i+1)*9] for i in range(6)]
    def rot_block(block, t):
        b = block
        for _ in range(t % 4):
            b = ''.join(b[i] for i in ROT90)
        return b
    print("Trying face-rotation brute-force (4096 combos)...")
    for comb in itertools.product(range(4), repeat=6):
        candidate_blocks = [rot_block(blocks[i], comb[i]) for i in range(6)]
        candidate_color_str = ''.join(candidate_blocks)
        try:
            facelet = ''.join(color_to_face_map[ch] for ch in candidate_color_str)
        except KeyError:
            continue
        verr = verify_facelet_string(facelet)
        if verr == 0:
            try:
                sol = kociemba.solve(facelet) if kociemba else None
            except Exception:
                sol = None
            print("Found valid rotation combo:", comb)
            return candidate_color_str, facelet, comb, sol
    return None

# ---------------------------
# Center mapping prompt (user can override default CubeState mapping)
# ---------------------------
def prompt_center_mapping(detected_centers: List[Tuple[int,str]]):
    cs = CubeState()
    print("\nDetected centers (image index -> color):")
    for i,c in detected_centers:
        print(f"  [{i}] center: {c}")
    print("\nDefault mapping (color -> face) from CubeState.color_to_side:")
    print({v:k for k,v in cs.side_to_color.items()} if hasattr(cs,'side_to_color') else cs.color_to_side)
    print("If default is correct press ENTER. Otherwise supply overrides like 'W:U R:R Y:F G:D O:L B:B'")
    s = input("Mapping override (ENTER to accept default): ").strip()
    if s == "":
        # derive color->face mapping from cs.color_to_side or from cs.side_to_color
        if hasattr(cs, 'color_to_side') and cs.color_to_side:
            return cs.color_to_side.copy()
        # fallback: invert side_to_color if present
        if hasattr(cs, 'side_to_color') and cs.side_to_color:
            inv = {v:k for k,v in cs.side_to_color.items()}
            return inv
        return cs.color_to_side.copy()
    mapping = {}
    for token in s.split():
        if ':' not in token:
            print("Bad token:", token)
            continue
        col, face = token.split(':',1)
        col = col.strip().upper()
        face = face.strip().upper()
        if col == '' or face not in ('U','R','F','D','L','B'):
            print("Invalid mapping token:", token)
            continue
        mapping[col] = face
    # fill missing using cs mapping
    if len(mapping) != 6:
        base = prompt_center_mapping([]) if False else (cs.color_to_side.copy() if hasattr(cs,'color_to_side') and cs.color_to_side else {v:k for k,v in cs.side_to_color.items()})
        base.update(mapping)
        return base
    return mapping

# ---------------------------
# Main workflow
# ---------------------------
def main():
    imgdir = config.PICTURES_DIR
    image_paths = [imgdir / n for n in DEFAULT_FILENAMES]

    template = PolygonTemplate()
    # if we have at least detector_1 file, use that to edit template (load existing if present)
    if image_paths[0].exists():
        loaded = template.load()
        if not loaded:
            template.define_on_image(image_paths[0])
            template.load()
    else:
        # capture a quick frame from webcam to edit template
        print("No images found. Opening webcam to capture one frame for template definition. Press SPACE to capture.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam for template capture")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            disp = frame.copy()
            cv2.putText(disp, "Press SPACE to capture template frame (ESC to quit)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
            cv2.imshow("Template capture", disp)
            k = cv2.waitKey(20) & 0xFF
            if k == 32:
                cap.release()
                cv2.destroyAllWindows()
                tmp = imgdir / "template_capture.png"
                cv2.imwrite(str(tmp), frame)
                template.define_on_image(tmp)
                break
            elif k == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

    # Now gather the six images: if files exist use them, otherwise capture via webcam in sequence
    images_to_use = []
    if all(p.exists() for p in image_paths):
        images_to_use = image_paths
    else:
        print("Images not found; capturing 6 faces sequentially from webcam.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        for i in range(6):
            print(f"Place face #{i+1} in view and press SPACE to capture (you can capture in the QBR fixed order if you prefer).")
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                disp = frame.copy()
                # draw template positions
                for label, (x, y) in template.positions.items():
                    cv2.circle(disp, (int(x), int(y)), 10, (0,255,0), -1)
                    cv2.circle(disp, (int(x), int(y)), 10, (0,0,0), 2)
                cv2.putText(disp, f"Capture face {i+1} - SPACE to save - ESC to quit", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
                cv2.imshow("Capture 6 faces", disp)
                k = cv2.waitKey(20) & 0xFF
                if k == 32:
                    fn = imgdir / f"detector_{i+1}.png"
                    cv2.imwrite(str(fn), frame)
                    images_to_use.append(fn)
                    print("Saved", fn)
                    break
                elif k == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return
        cap.release()
        cv2.destroyAllWindows()

    # Sample each image using the template positions
    roi_radius =  max(8, int(min(cv2.imread(str(images_to_use[0])).shape[:2]) * 0.03))
    all_colors = []
    all_labs = []
    all_hsvs = []
    for p in images_to_use:
        img = cv2.imread(str(p))
        colors_map, labs_map, hsv_map = sample_face_from_image(img, template.positions, roi_radius=roi_radius)
        all_colors.append(colors_map)
        all_labs.append(labs_map)
        all_hsvs.append(hsv_map)
        print(f"Detected (sample) on {p.name}: center {colors_map.get('P5')}")

    # Offer user color-correction UI (updates LAB when color changed)
    ui = ColorCorrectionUI()
    corrected_colors, corrected_labs = ui.run(images_to_use, template.positions, all_colors, all_labs)

    # Determine center mapping: either fixed order or prompt mapping by centers
    cs = CubeState()
    if FIXED_CAPTURE_ORDER is not None:
        # Use fixed capture order (list of 6 color letters corresponding to images 1..6)
        fixed_order = FIXED_CAPTURE_ORDER
        try:
            color_str, facelet_str = build_from_six_images_fixed_order(corrected_colors, fixed_order)
        except Exception as e:
            print("Failed to build strings with fixed order:", e)
            return
        color_to_face_map = {v:k for k,v in cs.side_to_color.items()} if hasattr(cs,'side_to_color') else cs.color_to_side.copy()
    else:
        # auto detect centers, ask user to confirm mapping
        detected_centers = [(i+1, cmap.get('P5')) for i,cmap in enumerate(corrected_colors)]
        color_to_face_map = prompt_center_mapping(detected_centers)
        try:
            color_str, facelet_str = build_from_six_images_by_centers(corrected_colors, color_to_face_map)
        except Exception as e:
            print("Failed to build/verify facelets:", e)
            return

    print("Built color string:", color_str)
    print("Facelet string:", facelet_str)

    # Verify
    verr = verify_facelet_string(facelet_str)
    if verr != 0:
        print("VERIFY error:", verr)
        print_ascii_net(color_str, facelet_str)
        # Try rotation brute-force (only orientation)
        res = try_face_rotation_bruteforce(color_str, color_to_face_map)
        if res is not None:
            new_color_str, new_facelet, combo, sol = res
            print("Solved after face-rotation brute force. Rotation combo:", combo)
            if sol:
                print("Solution:", sol)
            else:
                print("No moves returned (kociemba missing) but cube verified OK.")
            return
        else:
            print("Rotation brute-force failed; please inspect and correct with UI or re-capture faces.")
            return
    else:
        # verified ok
        print("Facelets verified OK.")
        try:
            if kociemba is None:
                print("kociemba not installed; cannot compute solution.")
            else:
                sol = kociemba.solve(facelet_str)
                print("Solution:", sol)
        except Exception as e:
            print("kociemba.solve failed:", e)
            print_ascii_net(color_str, facelet_str)
            # optionally attempt rotation brute-force
            res = try_face_rotation_bruteforce(color_str, color_to_face_map)
            if res is not None:
                _, _, combo, sol = res
                print("Solved after rotation brute-force:", combo)
                if sol:
                    print("Solution:", sol)
            else:
                print("Rotation brute-force couldn't find a valid orientation. Manual correction required.")

if __name__ == "__main__":
    main()
