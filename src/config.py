"""config.py — project configuration
------------------------------------

This file centralizes default runtime constants for the Rubik's-cube solver
project. Keep in mind these are *defaults* for development;
prefer overriding them at runtime or via a small settings loader for deployments.

Notes / warnings
- Paths are constructed using `Path.cwd()` which is evaluated at import time.
  For packaged/distributed usage prefer `Path(__file__).parent` or provide a
  user-configurable override.
- Many values (resolution, MJPEG quality, roi heuristics in the detector) are
  trade-offs between accuracy and CPU usage. See tuning notes near each group.

-------------------------------------------------------------------------------

Copyright (c) 2025 Facundo Gauna & Ulises Carnevale. Licensed under MIT License.
"""

from pathlib import Path
from typing import Dict, List

# ---------------- Rubik cube configurations ----------------

# 54-character flattened strings describing initial sticker layout.
# Each character is a color-letter (R,O,Y,G,B,W) or face-letter (U,R,F,D,L,B).
# These defaults represent the kociemba order (sort cube).
COLOR_INIT_STATE: str = "BBBBBBBBBOOOOOOOOOYYYYYYYYYGGGGGGGGGRRRRRRRRRWWWWWWWWW"
FACES_INIT_STATE: str = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

# Map face letter -> small integer index used by kociemba classes to rotate the cube str representation.
MOVE_INDEX = {'U':0, 'R':1, 'F':2, 'D':3, 'L':4, 'B':5}

# These constants assume a flattened 54-element sticker ordering (0..53).
# CENTER_INDICES maps face letter -> integer index of that face's center
# in the flattened array. Useful for quick access to measured center stickers.
CENTER_INDICES: Dict[str,int] ={ 'U': 4, 'R': 13, 'F': 22, 'D': 31, 'L': 40, 'B': 49}

# Canonical color and face orderings
COLOR_ORDER: List[str] = ['B','O','Y','G','R','W']
FACE_ORDER: List[str] = ['U','R','F','D','L','B']
FACE_TO_COLOR: Dict[str,str] = {'U':'B', 'R':'O', 'F':'Y', 'D':'G', 'L':'R', 'B':'W'}

# Initial map bgr map to determinate the color detection if the calibration hasnt been done.
CANONICAL_BGR = {
    'R': (30, 30, 200),
    'O': (10, 120, 200),
    'Y': (0, 200, 200),
    'G': (0, 150, 40),
    'B': (180, 50, 0),
    'W': (200, 200, 200)
}

# ---------------- Face rotation fixes ----------------
# Per-face rotation fix: integer 0..3 meaning number of 90° clockwise
# rotations to apply to the source numbering of each face so that the
# sticker ordering matches the solver's expected orientation.
# Adjust these values if a captured face appears rotated relative to the
# solver convention (common when the camera/cube orientation differs).
FACE_ROTATIONS: Dict[str, int] = {
    'U': 0,
    'R': 2,
    'F': 0,
    'D': 3,
    'L': 0,
    'B': 2,
}


# ---------------- Capture ----------------
# Resolution choice is a trade-off: higher resolution -> better detections
# per sticker but more CPU. For constrained platforms try [1280,720].
CAMERA_RESOLUTION: list[int, int] = [1920, 1080]   # could be (1280,720) for better performance
CAMERA_FPS : int = 30


# Generic timeout used by arduino that wait for an "OK"/ack reply.
OK_TIMEOUT : float = 0.6


# ---------------- Filesystem paths ----------------
# ROOT is built from the current working directory at import time. If you run
# the program from a different working directory adjust this or compute paths
# relative to __file__ instead.
ROOT = Path.cwd() / "src"
PICTURES_DIR  = ROOT / "templates/pictures"
POSITIONS_DIR = ROOT / "positions"

IMG1_PATH = PICTURES_DIR / "image1.png"
IMG2_PATH = PICTURES_DIR / "image2.png"

# Where calibrated color centroids (Lab arrays) are persisted.
CALIBRATIONS_PATH : Path = ROOT / "positions/calibrations.json"
# Default JSON files that contain sticker position maps (label -> (x,y)).
POLYGON_POSITIONS_PATH_1 = POSITIONS_DIR / "positions1.json"
POLYGON_POSITIONS_PATH_2 = POSITIONS_DIR / "positions2.json"


# ---------------- MJPEG / preview settings ----------------

# Local MJPEG preview server used by the UI for live previews. These values
# control network binding, target FPS and JPEG compression quality. Lower
# JPEG_QUALITY reduces CPU and bandwidth at the cost of stronger compression
# artifacts which may affect color fidelity during debugging/calibration.
MJPEG_HOST = "127.0.0.1"
MJPEG_PORT = 5001
MJPEG_FPS = 6                # target preview frames per second (tune for performance)
MJPEG_JPEG_QUALITY = 30      # 0..100; lower -> less CPU/bandwidth, more compression

