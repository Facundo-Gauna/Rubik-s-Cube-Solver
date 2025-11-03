from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Dict, Tuple

@dataclass
class AppConfig:
    # Base paths
    BASE_DIR: Path = Path(__file__).parent
    PICTURES_DIR: Path = BASE_DIR / "pictures"
    DETECTIONS_DIR: Path = BASE_DIR / "detections" 
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Create directories
    def __post_init__(self):
        self.PICTURES_DIR.mkdir(exist_ok=True)
        self.DETECTIONS_DIR.mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(exist_ok=True)
        if self.CUBE_PALETTE is None:
            self.CUBE_PALETTE = {
            'R': [((0, 80, 50), (10, 255, 255)), ((170, 80, 50), (179, 255, 255))],
            'O': [((10, 80, 50), (25, 255, 255))],
            'Y': [((25, 80, 50), (35, 255, 255))],
            'G': [((36, 40, 40), (85, 255, 255))],
            'B': [((90, 40, 40), (140, 255, 255))],
            'W': [((0, 0, 180), (179, 80, 255))]
        }
    
    # File paths
    @property
    def IMG1_PATH(self) -> Path:
        return self.PICTURES_DIR / "detector1.png"
    
    @property
    def IMG2_PATH(self) -> Path:
        return self.PICTURES_DIR / "detector2.png"
    
    @property
    def SAVE_FILE1(self) -> Path:
        return self.BASE_DIR / "positions1.json"
    @property
    def SAVE_FILE2(self) -> Path:
        return self.BASE_DIR / "positions2.json"
    
    # Camera settings
    CAMERA_RESOLUTION: Tuple[int, int] = (1980, 1080)
    THUMBNAIL_RESOLUTION: Tuple[int, int] = (160, 90)
    

    # Detection settings
    MIN_ROI: int = 8
    ROI_FRACTION: float = 0.45
    WHITE_CHROMA_THRESHOLD: float = 14.0
    WHITE_LIGHTNESS_THRESHOLD: float = 200.0
    
    # Cube configuration
    SIDE_ORDER: Tuple[str, ...] = ('U', 'R', 'F', 'D', 'L', 'B')
    COLOR_LETTERS: Tuple[str, ...] = ('B', 'G', 'O', 'R', 'W', 'Y')
    CUBE_PALETTE: Dict[str, Tuple[int, int, int]] = None

    # UI settings
    POLYGON_WIDTH: int = 10
    POLYGON_HEIGHT: int = 10
    
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
    
    COPYRIGHT_MARK: str = "@Creator: Facundo and Ulises"

config = AppConfig()