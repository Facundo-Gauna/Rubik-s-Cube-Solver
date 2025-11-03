from pathlib import Path
import re
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass

@dataclass
class DetectionResult:
    color_str: str
    solution_str: str
    has_errors: bool

@dataclass  
class StickerPosition:
    label: str
    x: int
    y: int
    color: Optional[str] = None

@dataclass
class CubeState:
    # Standard solved state: U:R, R:G, F:W, D:O, L:B, B:Y
    color_status: str = 'RRRRRRRRRGGGGGGGGGWWWWWWWWWOOOOOOOOOBBBBBBBBBYYYYYYYYY'
    face_status: str = 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'
    side_to_color: Dict[str, str] = None
    color_to_side : Dict[str,str] = None
    face_color_order = ['B','O','Y','G','R','W']
    face_side_order = ['U','R','F','D','L','B']
    def __post_init__(self):
        if self.side_to_color is None:
            # Standard mapping for display and kociemba
            self.side_to_color = {
                'U':'B',
                'R':'O',
                'F':'Y',
                'D':'G',
                'L':'R',
                'B':'W'
            }
        if self.color_to_side is None:
            self.color_to_side = {
                v:k for k,v in self.side_to_color.items()
            }

CENTER_INDICES = {'U': 4, 'R': 13, 'F': 22, 'D': 31, 'L': 40, 'B': 49}

def clean_color_string(s: str) -> str:
    return re.sub(r'[^A-Za-z]', '', s).upper()