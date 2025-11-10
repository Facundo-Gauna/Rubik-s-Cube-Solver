from pathlib import Path
import threading
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field

@dataclass
class DetectionResult:
    color_str: str
    solution_str: str
    face_str: str
    has_errors: bool

COLOR_INIT_STATE: str = "BBBBBBBBBOOOOOOOOOYYYYYYYYYGGGGGGGGGRRRRRRRRRWWWWWWWWW"
FACES_INIT_STATE: str = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

@dataclass
class CubeState:
    color_status: str = COLOR_INIT_STATE
    face_status: str =  FACES_INIT_STATE
    solution: str = ''

    def face_status_set(self, face_status_new : str):
        self.face_status = face_status_new
        colors = ''
        for face in face_status_new:
            colors += FACE_TO_COLOR[face]
        self.color_status = colors

    def colors_status_set(self, colors_status_new : str):
        self.color_status = colors_status_new
        faces= ''
        for colors in colors_status_new:
            faces += COLOR_TO_FACE[colors]
        self.face_status = faces

    def reset_state(self):
        self.color_status = COLOR_INIT_STATE
        self.face_status =  FACES_INIT_STATE
        self.solution = ''

def init_const():
    PICTURES_DIR.mkdir(exist_ok=True)

CAMERA_RESOLUTION: Tuple[int, int] = (1980, 1080)
THUMBNAIL_RESOLUTION: Tuple[int, int] = (160, 90)

BASE_DIR: Path = Path(__file__).parent
PICTURES_DIR: Path = BASE_DIR / "pictures"

IMG1_PATH : Path = PICTURES_DIR / "detector1.png"
IMG2_PATH : Path = PICTURES_DIR / "detector2.png"
POLYGON_POSITIONS_PATH_1 : Path = BASE_DIR / "positions1.json"
POLYGON_POSITIONS_PATH_2 : Path = BASE_DIR / "positions2.json"
CALIBRATIONS_PATH : Path = BASE_DIR / "calibrations.json"

# Cube Consts

CENTER_INDICES: Dict[str,int] ={ 'U': 4, 'R': 13, 'F': 22, 'D': 31, 'L': 40, 'B': 49}

COLOR_ORDER: List[str] = ['B','O','Y','G','R','W']
FACE_ORDER: List[str] = ['U','R','F','D','L','B']

FACE_TO_COLOR: Dict[str,str] = {'U':'B', 'R':'O', 'F':'Y', 'D':'G', 'L':'R', 'B':'W'}

COLOR_TO_FACE: Dict[str,str] ={'B':'U', 'O':'R', 'Y':'F', 'G':'D', 'R':'L', 'W':'B'}

CENTER_LABELS : List[str] = ['U5','R5','F5','D5','L5','B5']

# face groupings per image: user specified "photo1 = U,F,L ; photo2 = D,R,B"
FACE_ORDER_IMG_1 : List[str ]=  ['U','F','L']
FACE_ORDER_IMG_2 : List[str] =  ['D','R','B']


# Threading settings
THREAD_TIMEOUT : float = 10.0
KEY_WAIT_TIME : int = 10

# Theme
DARK_THEME: str = """
    QMainWindow, QWidget, QDialog {
        background-color: #2b2b2b;
        color: #ffffff;
    }
    QGroupBox {
        color: #ffffff;
        font-weight: bold;
        border: 2px solid #555555;
        border-radius: 5px;
        margin-top: 1ex;
        padding-top: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px 0 5px;
        color: #ffffff;
    }
    QPushButton {
        background-color: #404040;
        color: #ffffff;
        border: 1px solid #555555;
        border-radius: 3px;
        padding: 5px 10px;
        font-weight: bold;
        min-height: 20px;
    }
    QPushButton:hover {
        background-color: #505050;
        border: 1px solid #666666;
    }
    QPushButton:pressed {
        background-color: #606060;
    }
    QPushButton:disabled {
        background-color: #333333;
        color: #777777;
    }
    QLabel {
        color: #ffffff;
    }
    QStatusBar {
        background-color: #2b2b2b;
        color: #ffffff;
    }
"""

COPYRIGHT_MARK: str = "@Creators: Facundo and Ulises"


class ThreadSyncManager:
    """Manages threading synchronization for parallel operations"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.ready_to_finish = threading.Event()
        self.finish_requested = False
        self.windows_ready = threading.Event()
        self.exception = None
    
    def reset(self):
        """Reset all synchronization states"""
        self.ready_to_finish.clear()
        self.finish_requested = False
        self.windows_ready.clear()
        self.exception = None