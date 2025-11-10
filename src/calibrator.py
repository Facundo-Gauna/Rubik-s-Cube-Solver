import json
import math
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from config import CALIBRATIONS_PATH, IMG1_PATH, IMG2_PATH, ThreadSyncManager

# =============================================================================
# CALIBRATION CONFIGURATION
# =============================================================================

# Window settings
WINDOW_TITLE_PREFIX = "Cube Calibration"
CIRCLE_RADIUS = 15
SELECTED_CIRCLE_RADIUS = 20
CIRCLE_COLOR = (0, 200, 0)
SELECTED_CIRCLE_COLOR = (0, 255, 255)
BORDER_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
HIGHLIGHT_COLOR = (255, 255, 0)
    
# Color settings
COLOR_MAP_BGR = {
    'R': (0, 0, 255),
    'O': (0, 165, 255), 
    'Y': (0, 255, 255),
    'G': (0, 255, 0),
    'B': (255, 0, 0),
    'W': (255, 255, 255)
}
    
# Detection settings
ROI_SIZE = 30  # Size of region to sample around each point
MIN_SAMPLES = 5  # Minimum samples for robust color calculation
    
# Threading settings
THREAD_TIMEOUT = 10.0
KEY_WAIT_TIME = 10
    
# Default HSV ranges (will be updated during calibration)

COLOR_ORDER_TO_CALIBRATE: List[str] = ['B','R','Y','G','O','W']

# =============================================================================
# CALIBRATION DETECTOR
# =============================================================================

class CalibrationDetector:
    """
    Handles calibration point definition and color sampling for a single image.
    Each image calibrates 3 colors from the COLOR_ORDER.
    """
    
    def __init__(self, colors_for_image: List[str], save_file: Path, image_path: Path):
        self.colors_for_image = colors_for_image  # e.g., ['B','O','Y'] for image1
        self.save_file = save_file
        self.image_path = image_path
        self.img = None
        self.positions: Dict[str, Tuple[int, int]] = {}
        self.selected_color: Optional[str] = None
        self.dragging = False
        self.drag_offset = (0, 0)
        self.calibration_data: Dict[str, Dict] = {}

    def _create_default_positions(self, img_shape: Tuple[int, int]) -> Dict[str, Tuple[int, int]]:
        """Create default positions in a vertical line on the left side"""
        h, w = img_shape[:2]
        positions = {}
        num_colors = len(self.colors_for_image)
        
        for i, color in enumerate(self.colors_for_image):
            # Position points vertically centered on the left side
            x = int(w * 0.2)  # 20% from left
            y = int((i + 1) * (h / (num_colors + 1)))  # Evenly spaced vertically
            positions[color] = (x, y)
            
        return positions

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for point dragging"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._handle_mouse_down(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.selected_color:
            self._handle_mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def _handle_mouse_down(self, x: int, y: int):
        """Handle mouse button down event"""
        closest = None
        min_d = float('inf')
        
        for color, (px, py) in self.positions.items():
            d = math.hypot(x - px, y - py)
            if d < min_d and d < 50:  # 50 pixel threshold for selection
                min_d = d
                closest = color
                
        if closest:
            self.selected_color = closest
            self.dragging = True
            px, py = self.positions[closest]
            self.drag_offset = (x - px, y - py)

    def _handle_mouse_move(self, x: int, y: int):
        """Handle mouse movement while dragging"""
        nx = max(0, min(self.img.shape[1] - 1, x - self.drag_offset[0]))
        ny = max(0, min(self.img.shape[0] - 1, y - self.drag_offset[1]))
        self.positions[self.selected_color] = (int(nx), int(ny))

    def _draw_interface(self) -> np.ndarray:
        """Draw the calibration interface with color points and instructions"""
        disp = self.img.copy()
        
        for color in self.colors_for_image:
            x, y = self.positions[color]
            is_selected = color == self.selected_color
            
            color_bgr = COLOR_MAP_BGR[color]
            radius = (SELECTED_CIRCLE_RADIUS if is_selected 
                     else CIRCLE_RADIUS)
            
            # Draw circle with color fill
            cv2.circle(disp, (int(x), int(y)), radius, color_bgr, -1)
            cv2.circle(disp, (int(x), int(y)), radius, BORDER_COLOR, 2)
            
            # Highlight selected circle
            if is_selected:
                cv2.circle(disp, (int(x), int(y)), radius + 4, 
                          HIGHLIGHT_COLOR, 3)
            
            # Draw color label
            label = f"{color} ({x},{y})"
            cv2.putText(disp, label, (int(x) + 25, int(y) + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
        
        # Add instructions
        instructions = [
            f"Calibration: {', '.join(self.colors_for_image)}",
            "Drag colored circles to center of corresponding color stickers",
            "Press ENTER when all points are correctly positioned",
            "Press ESC to cancel, R to reset positions"
        ]
        
        for i, line in enumerate(instructions):
            cv2.putText(disp, line, (10, 30 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
        
        return disp

    def _load_saved_positions(self) -> Dict:
        """Load previously saved calibration positions from file"""
        if not self.save_file.exists():
            return {}
            
        try:
            with open(self.save_file, 'r') as f:
                data = json.load(f)
                return data.get('calibration_positions', {})
        except Exception as e:
            print(f"Warning: Could not load saved calibration positions: {e}")
            return {}

    def _save_positions_to_disk(self):
        """Save current calibration positions to disk"""
        try:
            # Load existing data to preserve other calibration information
            existing_data = {}
            if self.save_file.exists():
                with open(self.save_file, 'r') as f:
                    existing_data = json.load(f)
            
            # Update only the positions
            existing_data['calibration_positions'] = {
                color: [int(x), int(y)] 
                for color, (x, y) in self.positions.items()
            }
            
            with open(self.save_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
            print(f"Calibration positions saved to {self.save_file}")
            
        except Exception as e:
            print(f"Error saving calibration positions: {e}")

    def setup_image(self):
        """Setup image and load/default positions"""
        self.img = cv2.imread(str(self.image_path))
        if self.img is None:
            raise FileNotFoundError(f"Cannot open image: {self.image_path}")
            
        saved_positions = self._load_saved_positions()
        default_positions = self._create_default_positions(self.img.shape)

        # Load saved positions or use defaults
        for color in self.colors_for_image:
            if color in saved_positions:
                self.positions[color] = tuple(saved_positions[color])
            else:
                self.positions[color] = default_positions[color]

    def sample_colors(self) -> Dict[str, Dict]:
        """
        Sample colors from the current positions and return HSV statistics
        Returns: {color: {'hsv_median': (h,s,v), 'hsv_std': (h,s,v), 'samples': n}}
        """
        calibration_results = {}
        
        for color, (x, y) in self.positions.items():
            hsv_values = self._sample_region_hsv(x, y)
            
            if len(hsv_values) < MIN_SAMPLES:
                print(f"Warning: Not enough samples for color {color}")
                continue
                
            hsv_median = np.median(hsv_values, axis=0).astype(int)
            hsv_std = np.std(hsv_values, axis=0).astype(int)
            
            calibration_results[color] = {
                'hsv_median': tuple(hsv_median.tolist()),
                'hsv_std': tuple(hsv_std.tolist()),
                'samples': len(hsv_values),
                'position': [int(x), int(y)]
            }
            
            print(f"Color {color}: Median HSV={tuple(hsv_median)}, Samples={len(hsv_values)}")
            
        return calibration_results

    def _sample_region_hsv(self, x: int, y: int) -> np.ndarray:
        """Sample HSV values from a region around the point"""
        r = ROI_SIZE // 2
        h, w = self.img.shape[:2]
        
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        
        if x1 <= x0 or y1 <= y0:
            return np.array([])
            
        roi = self.img[y0:y1, x0:x1]
        
        # Apply Gaussian blur to reduce noise
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Reshape to list of pixels and remove black/dark pixels
        pixels = hsv_roi.reshape(-1, 3)
        brightness_mask = pixels[:, 2] > 30  # Filter out very dark pixels
        filtered_pixels = pixels[brightness_mask]
        
        return filtered_pixels

# =============================================================================
# PARALLEL CALIBRATION SYSTEM
# =============================================================================

class SynchronizedCalibration:
    """Manages two CalibrationDetectors with synchronized completion"""
    
    def __init__(self):
        self.image1_path = IMG1_PATH
        self.image2_path = IMG2_PATH
        self.colors_image1 = COLOR_ORDER_TO_CALIBRATE[:3]  # First 3 colors from COLOR_ORDER
        self.colors_image2 = COLOR_ORDER_TO_CALIBRATE[3:]  # Last 3 colors from COLOR_ORDER
        self.calibration_file = CALIBRATIONS_PATH
        
        self.calibration_data1: Optional[Dict[str, Dict]] = None
        self.calibration_data2: Optional[Dict[str, Dict]] = None
        self.positions1: Optional[Dict[str, Tuple[int, int]]] = None
        self.positions2: Optional[Dict[str, Tuple[int, int]]] = None
        self.sync_manager = ThreadSyncManager()

    def calibrate_both(self) -> Dict[str, Dict]:
        """Run both calibration UIs with synchronized completion and return combined data"""
        thread1 = threading.Thread(target=self._calibrate_image, args=(1,))
        thread2 = threading.Thread(target=self._calibrate_image, args=(2,))
        
        thread1.start()
        thread2.start()
        
        # Wait for both windows to be created
        self.sync_manager.windows_ready.wait(timeout=THREAD_TIMEOUT)
        
        print("\n=== BOTH CALIBRATION WINDOWS ARE READY ===")
        print("Instructions:")
        print("- Drag colored circles to center of corresponding color stickers")
        print("- Ensure each circle is centered on a clean, well-lit sticker")
        print("- Press ENTER in EITHER window when BOTH are ready")
        print("- Both windows close when ENTER is pressed in either one\n")
        
        # Wait for finish request
        self.sync_manager.ready_to_finish.wait()
        
        # Set finish flag so both detectors know to exit
        with self.sync_manager.lock:
            self.sync_manager.finish_requested = True
        
        # Wait for threads to complete
        thread1.join()
        thread2.join()
        
        if self.sync_manager.exception:
            raise self.sync_manager.exception
        
        # Combine and save calibration data
        combined_data = self._combine_and_save_calibration()
        
        print("✓ Calibration completed successfully!")
        return combined_data

    def _calibrate_image(self, image_num: int):
        """Run calibration for a single image with synchronization"""
        try:
            if image_num == 1:
                colors = self.colors_image1
                image_path = self.image1_path
                calibration_attr = "calibration_data1"
                positions_attr = "positions1"
                window_suffix = "Image 1"
            else:
                colors = self.colors_image2
                image_path = self.image2_path
                calibration_attr = "calibration_data2"
                positions_attr = "positions2"
                window_suffix = "Image 2"

            # Create calibration detector
            detector = CalibrationDetector(colors, self.calibration_file, image_path)
            detector.setup_image()
            
            window_name = f"{WINDOW_TITLE_PREFIX} - {window_suffix}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, detector._mouse_callback)
            
            # Signal that window is ready
            with self.sync_manager.lock:
                if hasattr(self, '_window_ready_count'):
                    self.sync_manager.windows_ready.set()
                else:
                    self._window_ready_count = 1
            
            print(f"Calibration Window {image_num} ready: {window_name}")
            
            # Main event loop
            while True:
                display = detector._draw_interface()
                cv2.imshow(window_name, display)
                
                # Check if finish was requested
                with self.sync_manager.lock:
                    if self.sync_manager.finish_requested:
                        break
                
                key = cv2.waitKey(KEY_WAIT_TIME) & 0xFF
                
                if key == 13:  # ENTER
                    self._handle_enter_key(window_suffix)
                    break
                elif key == 27:  # ESC
                    self._handle_escape_key(window_suffix)
                    break
                elif key == ord('r'):
                    # Reset to default positions
                    default_positions = detector._create_default_positions(detector.img.shape)
                    detector.positions = default_positions.copy()
            
            # Process calibration if finished normally
            if not self.sync_manager.exception and self.sync_manager.finish_requested:
                detector._save_positions_to_disk()
                setattr(self, calibration_attr, detector.sample_colors())
                setattr(self, positions_attr, detector.positions.copy())
            
            cv2.destroyWindow(window_name)
            
        except Exception as e:
            self.sync_manager.exception = e
            self.sync_manager.ready_to_finish.set()

    def _handle_enter_key(self, window_suffix: str):
        """Handle ENTER key press"""
        print(f"ENTER pressed in {window_suffix} Calibration - finishing both...")
        with self.sync_manager.lock:
            if not self.sync_manager.finish_requested:
                self.sync_manager.finish_requested = True
                self.sync_manager.ready_to_finish.set()

    def _handle_escape_key(self, window_suffix: str):
        """Handle ESC key press"""
        print(f"ESC pressed in {window_suffix} Calibration - cancelling both...")
        self.sync_manager.exception = Exception("User cancelled calibration")
        self.sync_manager.ready_to_finish.set()

    def _combine_and_save_calibration(self) -> Dict[str, Dict]:
        """Combine calibration data from both images and save to file"""
        if not self.calibration_data1 or not self.calibration_data2:
            raise Exception("Calibration data incomplete")
        
        # Combine data from both images
        combined_data = {}
        combined_data.update(self.calibration_data1)
        combined_data.update(self.calibration_data2)
        
        # Generate HSV ranges from calibration data
        hsv_ranges = self._generate_hsv_ranges(combined_data)
        
        # Prepare complete calibration data
        calibration_output = {
            'timestamp': self._get_timestamp(),
            'hsv_ranges': hsv_ranges,
            'calibration_points': {
                'image1': {color: self.positions1[color] for color in self.colors_image1},
                'image2': {color: self.positions2[color] for color in self.colors_image2}
            },
            'color_statistics': combined_data
        }
        
        # Save to file
        self._save_calibration_data(calibration_output)
        
        return calibration_output

    def _generate_hsv_ranges(self, color_data: Dict[str, Dict]) -> Dict[str, List[Tuple[Tuple, Tuple]]]:
        """Generate HSV ranges from calibrated color statistics"""
        hsv_ranges = {}
        
        for color, stats in color_data.items():
            h_median, s_median, v_median = stats['hsv_median']
            h_std, s_std, v_std = stats['hsv_std']
            
            # Calculate ranges based on median ± 2*std, clamped to valid HSV ranges
            if color == 'R':
                # Red needs special handling due to hue wrap-around
                range1 = (
                    (max(0, h_median - 2*h_std), max(50, s_median - 2*s_std), max(50, v_median - 2*v_std)),
                    (min(10, h_median + 2*h_std), min(255, s_median + 2*s_std), min(255, v_median + 2*v_std))
                )
                range2 = (
                    (max(170, h_median - 2*h_std), max(50, s_median - 2*s_std), max(50, v_median - 2*v_std)),
                    (min(179, h_median + 2*h_std), min(255, s_median + 2*s_std), min(255, v_median + 2*v_std))
                )
                hsv_ranges[color] = [range1, range2]
            else:
                # For other colors, single range
                lower = (
                    max(0, h_median - 2*h_std),
                    max(40, s_median - 2*s_std), 
                    max(40, v_median - 2*v_std)
                )
                upper = (
                    min(179, h_median + 2*h_std),
                    min(255, s_median + 2*s_std),
                    min(255, v_median + 2*v_std)
                )
                hsv_ranges[color] = [(lower, upper)]
        
        return hsv_ranges

    def _save_calibration_data(self, calibration_data: Dict):
        """Save calibration data to JSON file"""
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            print(f"Calibration data saved to {self.calibration_file}")
        except Exception as e:
            print(f"Error saving calibration data: {e}")
            raise

    def _get_timestamp(self) -> str:
        """Get current timestamp for calibration metadata"""
        from datetime import datetime
        return datetime.now().isoformat()

# =============================================================================
# CALIBRATION MANAGER
# =============================================================================

class CalibrationManager:
    """High-level manager for the calibration process"""
    
    def __init__(self):
        self.calibration_file = CALIBRATIONS_PATH
        self.calibration_data: Optional[Dict] = None

    def run_calibration(self) -> bool:
        print("=== CUBE COLOR CALIBRATION ===")
        print("This process will calibrate the color detection system.")
        print("You will need to position markers on 6 color stickers (3 per image).")
        print("Make sure each sticker is clean and well-lit.\n")
        
        try:
            parallel_calibrator = SynchronizedCalibration()
            self.calibration_data = parallel_calibrator.calibrate_both()
            
            self._print_calibration_summary()
            
            print("\n=== CALIBRATION COMPLETE ===")
            return True
            
        except Exception as e:
            print(f"\n=== CALIBRATION FAILED: {e} ===")
            return False

    def _print_calibration_summary(self):
        if not self.calibration_data:
            return
            
        print("\n--- CALIBRATION SUMMARY ---")
        print(f"Timestamp: {self.calibration_data['timestamp']}")
        
        print("\nCalibrated Colors:")
        for color in ['B', 'O', 'Y', 'G', 'R', 'W']:
            if color in self.calibration_data['color_statistics']:
                stats = self.calibration_data['color_statistics'][color]
                print(f"  {color}: HSV {stats['hsv_median']} (±{stats['hsv_std']})")
        
        print(f"\nCalibration data saved to: {self.calibration_file}")

def load_calibration() -> Optional[Dict]:
    """Load existing calibration data from file"""
    try:
        if CALIBRATIONS_PATH.exists():
            with open(CALIBRATIONS_PATH, 'r') as f:
                calibration_data = json.load(f)
            print("Calibration data loaded successfully")
            return calibration_data
        else:
            print("No calibration file found")
            return None
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return None

if __name__ == "__main__":
    c = CalibrationManager()
    c.run_calibration()