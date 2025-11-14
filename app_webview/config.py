from pathlib import Path
from typing import Dict, Tuple, List

COLOR_INIT_STATE: str = "BBBBBBBBBOOOOOOOOOYYYYYYYYYGGGGGGGGGRRRRRRRRRWWWWWWWWW"
FACES_INIT_STATE: str = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
MOVE_INDEX = {'U':0, 'R':1, 'F':2, 'D':3, 'L':4, 'B':5}

CAMERA_RESOLUTION: Tuple[int, int] = (1980, 1080)
OK_TIMEOUT : float = 0.6

ROOT = Path.cwd() / "app_webview"
PICTURES_DIR  = ROOT / "templates/pictures"
POSITIONS_DIR = ROOT / "positions"

IMG1_PATH = PICTURES_DIR / "image1.png"
IMG2_PATH = PICTURES_DIR / "image2.png"
CALIBRATIONS_PATH : Path = ROOT / "positions/calibrations.json"
POLYGON_POSITIONS_PATH_1 = POSITIONS_DIR / "positions1.json"
POLYGON_POSITIONS_PATH_2 = POSITIONS_DIR / "positions2.json"

CENTER_INDICES: Dict[str,int] ={ 'U': 4, 'R': 13, 'F': 22, 'D': 31, 'L': 40, 'B': 49}

COLOR_ORDER: List[str] = ['B','O','Y','G','R','W']
FACE_ORDER: List[str] = ['U','R','F','D','L','B']

FACE_TO_COLOR: Dict[str,str] = {'U':'B', 'R':'O', 'F':'Y', 'D':'G', 'L':'R', 'B':'W'}

COLOR_TO_FACE: Dict[str,str] ={'B':'U', 'O':'R', 'Y':'F', 'G':'D', 'R':'L', 'W':'B'}

CENTER_LABELS : List[str] = ['U5','R5','F5','D5','L5','B5']

FACE_ORDER_IMG_1 : List[str ]=  ['U','F','L']
FACE_ORDER_IMG_2 : List[str] =  ['D','R','B']
