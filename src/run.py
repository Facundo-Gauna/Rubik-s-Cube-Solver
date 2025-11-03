import sys
import time
import cv2
import kociemba
import numpy as np
import concurrent.futures
import logging
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QMessageBox, QInputDialog, QGridLayout, 
    QGroupBox, QDialog, QDialogButtonBox, QProgressDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPen, QKeySequence
from PyQt5.QtWidgets import QShortcut

from config import config
from app_types import DetectionResult
from cube_status import CubeStatus
from control import MotorController
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraController:
    """Handles camera operations and frame capture"""
    
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_camera_index: int = -1
        self.is_initialized: bool = False
        self.last_frame: Optional[np.ndarray] = None
        
    def initialize(self) -> bool:
        """Initialize camera with auto-detection"""
        for i in range(10):
            if i == self.current_camera_index:
                continue
                
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.current_camera_index = i
                if self.cap is not None:
                    self.cap.release()
                self.cap = cap
                self._configure_camera()
                self.is_initialized = True
                logger.info(f"Camera initialized on index {i}")
                return True
                
        logger.error("No camera found")
        return False
        
    def _configure_camera(self):
        """Configure camera settings"""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame"""
        if not self.is_initialized or not self.cap or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = frame
            return frame
        return None
        
    def switch_camera(self) -> bool:
        """Switch to next available camera"""
        if self.cap:
            self.cap.release()
        self.is_initialized = False
        return self.initialize()
        
    def release(self):
        """Release camera resources"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.is_initialized = False
        logger.info("Camera released")


class DetectionWorker(QThread):
    """Worker thread for cube detection to prevent UI freezing"""
    
    finished = pyqtSignal(object)  # Emits DetectionResult
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, cube_status: CubeStatus, method: str = "manual"):
        super().__init__()
        self.cube_status = cube_status
        self.method = method
        self._is_running = True
        
    def run(self):
        """Run detection in background thread"""
        try:
            self.progress.emit(f"Starting {self.method} detection...")
            
            if self.method == "manual":
                result = self.cube_status.detect_status_manual()
            elif self.method == "cnn":
                # Fallback to manual if CNN not available
                result = self.cube_status.detect_status_manual()
            else:
                raise ValueError(f"Unknown detection method: {self.method}")
                
            if self._is_running:
                self.finished.emit(result)
                
        except Exception as e:
            if self._is_running:
                self.error.emit(str(e))
                
    def stop(self):
        """Stop the worker"""
        self._is_running = False


class CameraWidget(QLabel):
    """Widget to display camera feed"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Camera feed will appear here")
        self.setStyleSheet("border: 1px solid #555555; background-color: #1a1a1a;")
        
    def set_frame(self, frame: np.ndarray):
        """Update the widget with a new frame from the camera"""
        if frame is None or frame.size == 0:
            return
            
        try:
            # Convert BGR to RGB for display
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            # Create QImage and scale to fit widget
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.width(), self.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
            
        except Exception as e:
            logger.error(f"Error setting camera frame: {e}")


class ThumbnailWidget(QLabel):
    """Widget for displaying captured image thumbnails"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(config.THUMBNAIL_RESOLUTION[0], config.THUMBNAIL_RESOLUTION[1])
        self.setStyleSheet("border: 1px solid #555555; background-color: #1a1a1a;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("No image")
        
    def set_image(self, image_path: Path):
        """Load and display thumbnail from image file"""
        if not image_path.exists():
            self.clear()
            self.setText("No image")
            return
            
        try:
            # Load image with OpenCV for processing
            image = cv2.imread(str(image_path))
            if image is None:
                self.clear()
                self.setText("Load error")
                return
                
            # Convert to RGB and resize
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_image, config.THUMBNAIL_RESOLUTION, interpolation=cv2.INTER_AREA)
            
            # Convert to QPixmap
            h, w, ch = resized.shape
            bytes_per_line = ch * w
            qt_image = QImage(resized.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(qt_image))
            
        except Exception as e:
            logger.error(f"Error loading thumbnail {image_path}: {e}")
            self.clear()
            self.setText("Error")


class MainWindow(QMainWindow):
    """Main application window"""
    
    # Signals for thread-safe UI updates
    update_status_signal = pyqtSignal(str)
    update_cube_display_signal = pyqtSignal()
    detection_finished_signal = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        self.cube_status = CubeStatus()
        self.camera_controller = CameraController()
        self.motor_controller = MotorController()  # Auto-detect simulation
        
        self.captured_images_count: int = 0
        self.detection_worker: Optional[DetectionWorker] = None
        self.is_processing: bool = False
        
        self._setup_ui()
        self._connect_signals()
        self._initialize_camera()
        self._initialize_hardware()
        
    def _setup_ui(self):
        """Initialize the UI components"""
        self.setStyleSheet(config.DARK_THEME)
        self.setWindowTitle("Rubik's Cube Solver - Professional Edition")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Left panel - cube display and camera
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 2)
        
        # Right panel - controls
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready - Capture two images with Space bar")
        
    def _create_left_panel(self) -> QWidget:
        """Create the left panel with cube display and camera"""
        panel = QWidget()
        panel.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555555; border-radius: 8px;")
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Cube display
        cube_group = QGroupBox("Cube State")
        cube_layout = QVBoxLayout(cube_group)
        self.cube_widget = CubeWidget()
        self.cube_widget.stickerClicked.connect(self._on_sticker_clicked)
        cube_layout.addWidget(self.cube_widget)
        layout.addWidget(cube_group)
        
        # Camera feed
        camera_group = QGroupBox("Camera Feed")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_widget = CameraWidget()
        camera_layout.addWidget(self.camera_widget)
        layout.addWidget(camera_group, 1)
        
        # Thumbnails
        thumb_group = QGroupBox("Captured Images")
        thumb_layout = QHBoxLayout(thumb_group)
        
        self.thumb1 = ThumbnailWidget()
        self.thumb2 = ThumbnailWidget()
        
        thumb_layout.addWidget(QLabel("Image 1:"))
        thumb_layout.addWidget(self.thumb1)
        thumb_layout.addWidget(QLabel("Image 2:"))
        thumb_layout.addWidget(self.thumb2)
        thumb_layout.addStretch()
        
        layout.addWidget(thumb_group)
        
        return panel
        
    def _create_right_panel(self) -> QWidget:
        """Create the right panel with controls"""
        panel = QWidget()
        panel.setMaximumWidth(450)
        panel.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555555; border-radius: 8px;")
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Detection group
        detect_group = self._create_detection_group()
        layout.addWidget(detect_group)
        
        # Solving group
        solve_group = self._create_solving_group()
        layout.addWidget(solve_group)
        
        # Hardware control group
        hardware_group = self._create_hardware_group()
        layout.addWidget(hardware_group)
        
        # Validation group
        validation_group = self._create_validation_group()
        layout.addWidget(validation_group)
        
        # System group
        system_group = self._create_system_group()
        layout.addWidget(system_group)
        
        # Copyright
        copyright_label = QLabel(config.COPYRIGHT_MARK)
        copyright_label.setStyleSheet("color: #888888; font-size: 10px; padding: 10px;")
        copyright_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(copyright_label)
        
        layout.addStretch()
        
        return panel
        
    def _create_detection_group(self) -> QGroupBox:
        """Create detection controls group"""
        group = QGroupBox("Cube Detection")
        layout = QVBoxLayout(group)
        
        self.detect_cnn_btn = QPushButton("🧠 Detect with AI (CNN)")
        self.detect_cnn_btn.clicked.connect(self._detect_cube_state_cnn)
        self.detect_cnn_btn.setToolTip("Use trained neural network for detection")
        layout.addWidget(self.detect_cnn_btn)
        
        self.detect_manual_btn = QPushButton("✋ Manual Detection")
        self.detect_manual_btn.clicked.connect(self._detect_cube_state_manual)
        self.detect_manual_btn.setToolTip("Manually adjust positions and detect colors")
        layout.addWidget(self.detect_manual_btn)
        
        return group
        
    def _create_solving_group(self) -> QGroupBox:
        """Create solving controls group"""
        group = QGroupBox("Cube Solving")
        layout = QVBoxLayout(group)
        
        self.scramble_btn = QPushButton("🌀 Scramble Cube")
        self.scramble_btn.clicked.connect(self._scramble_cube)
        layout.addWidget(self.scramble_btn)
        
        self.solve_btn = QPushButton("✅ Solve Cube")
        self.solve_btn.clicked.connect(self._solve_cube)
        layout.addWidget(self.solve_btn)
        
        self.custom_moves_btn = QPushButton("🔧 Custom Moves")
        self.custom_moves_btn.clicked.connect(self._custom_moves)
        layout.addWidget(self.custom_moves_btn)
        
        return group
        
    def _create_hardware_group(self) -> QGroupBox:
        """Create hardware controls group"""
        group = QGroupBox("Hardware Control")
        layout = QVBoxLayout(group)
        
        self.motor_test_btn = QPushButton("⚙️ Test Motors")
        self.motor_test_btn.clicked.connect(self._test_motors)
        layout.addWidget(self.motor_test_btn)
        
        self.camera_switch_btn = QPushButton("📷 Switch Camera")
        self.camera_switch_btn.clicked.connect(self._switch_camera)
        layout.addWidget(self.camera_switch_btn)
        
        self.arduino_connection = QPushButton("♾️ Arduino Connection")
        self.arduino_connection.clicked.connect(self._arduino_connection)
        layout.addWidget(self.arduino_connection)

        # Hardware status
        self.hardware_status = QLabel("Hardware: Checking...")
        self.hardware_status.setStyleSheet("padding: 5px; border-radius: 3px; background-color: #333333;")
        layout.addWidget(self.hardware_status)
        
        return group
        
    def _create_validation_group(self) -> QGroupBox:
        """Create validation controls group"""
        group = QGroupBox("State Validation")
        layout = QVBoxLayout(group)
        
        self.validate_btn = QPushButton("✓ Validate Cube State")
        self.validate_btn.clicked.connect(self._validate_cube_state)
        layout.addWidget(self.validate_btn)
        
        self.validation_result = QLabel("Not yet validated")
        self.validation_result.setStyleSheet("padding: 8px; border-radius: 4px; background-color: #333333; min-height: 40px;")
        self.validation_result.setWordWrap(True)
        layout.addWidget(self.validation_result)
        
        return group
        
    def _create_system_group(self) -> QGroupBox:
        """Create system controls group"""
        group = QGroupBox("System")
        layout = QVBoxLayout(group)
        
        self.reset_btn = QPushButton("🔄 Reset Cube State")
        self.reset_btn.clicked.connect(self._reset_cube_state)
        layout.addWidget(self.reset_btn)
        
        self.exit_btn = QPushButton("🚪 Exit")
        self.exit_btn.setStyleSheet("background-color: #8B0000; color: white; font-weight: bold;")
        self.exit_btn.clicked.connect(self.close)
        layout.addWidget(self.exit_btn)
        
        return group
        
    def _connect_signals(self):
        # Keyboard shortcuts
        self.space_shortcut = QShortcut(QKeySequence("Space"), self)
        self.space_shortcut.activated.connect(self._capture_image)
        
        #Manually testers
        self.f_motor = QShortcut(QKeySequence("F"), self)
        self.f_motor_reverse = QShortcut(QKeySequence("Ctrl+F"), self)
        self.f_motor.activated.connect(self._motor_f)
        self.f_motor_reverse.activated.connect(self._motor_f_reverse)

        self.b_motor = QShortcut(QKeySequence("B"), self)
        self.b_motor_reverse = QShortcut(QKeySequence("Ctrl+B"), self)
        self.b_motor.activated.connect(self._motor_b)
        self.b_motor_reverse.activated.connect(self._motor_b_reverse)

        self.r_motor = QShortcut(QKeySequence("R"), self)
        self.r_motor_reverse = QShortcut(QKeySequence("Ctrl+R"), self)
        self.r_motor.activated.connect(self._motor_r)
        self.r_motor_reverse.activated.connect(self._motor_r_reverse)

        self.l_motor = QShortcut(QKeySequence("L"), self)
        self.l_motor_reverse = QShortcut(QKeySequence("Ctrl+L"), self)
        self.l_motor.activated.connect(self._motor_l)
        self.l_motor_reverse.activated.connect(self._motor_l_reverse)

        self.d_motor = QShortcut(QKeySequence("D"), self)
        self.d_motor_reverse = QShortcut(QKeySequence("Ctrl+D"), self)
        self.d_motor.activated.connect(self._motor_d)
        self.d_motor_reverse.activated.connect(self._motor_d_reverse)

        # Thread-safe update signals
        self.update_status_signal.connect(self._update_status_bar)
        self.update_cube_display_signal.connect(self._update_cube_display)
        self.detection_finished_signal.connect(self._on_detection_finished)
        
    def _initialize_camera(self):
        """Initialize camera and start timer"""
        if self.camera_controller.initialize():
            # Start camera update timer
            self.camera_timer = QTimer()
            self.camera_timer.timeout.connect(self._update_camera_display)
            self.camera_timer.start(33)  # ~30 FPS
        else:
            self._show_warning("Camera Warning", 
                             "Could not initialize camera. Camera functions will be disabled.")
            
    def _initialize_hardware(self):
        """Initialize hardware controller"""
        if self.motor_controller.connect():
            self.hardware_status.setText("Hardware: Connected ✓")
            self.hardware_status.setStyleSheet("padding: 5px; border-radius: 3px; background-color: #002200; color: #00FF00;")
        else:
            self.hardware_status.setText("Hardware: Disconnected (Simulation Mode)")
            self.hardware_status.setStyleSheet("padding: 5px; border-radius: 3px; background-color: #222200; color: #FFFF00;")
            
    def _update_camera_display(self):
        """Update camera display with latest frame"""
        frame = self.camera_controller.capture_frame()
        if frame is not None:
            self.camera_widget.set_frame(frame)
            
    def _capture_image(self):
        """Capture image when space bar is pressed"""
        if self.is_processing:
            return
            
        frame = self.camera_controller.last_frame
        if frame is None:
            self._show_warning("No Frame", "No camera frame available to capture")
            return
            
        self.captured_images_count += 1
        
        if self.captured_images_count == 1:
            # Save first image
            cv2.imwrite(str(config.IMG1_PATH), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            self.thumb1.set_image(config.IMG1_PATH)
            self.update_status_signal.emit("Image 1 captured. Press Space for image 2.")
            
        elif self.captured_images_count == 2:
            # Save second image
            cv2.imwrite(str(config.IMG2_PATH), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            self.thumb2.set_image(config.IMG2_PATH)
            self.update_status_signal.emit("Image 2 captured. Click 'Detect Cube State' to process.")
            self.captured_images_count = 0  # Reset for next capture cycle
            
    def _detect_cube_state_manual(self):
        """Start manual detection process"""
        if not self._check_images_captured():
            return
            
        self._set_processing_state(True, "Starting manual detection...")
        
        # Run detection in background thread
        self.detection_worker = DetectionWorker(self.cube_status, "manual")
        self.detection_worker.finished.connect(self.detection_finished_signal)
        self.detection_worker.error.connect(self._on_detection_error)
        self.detection_worker.progress.connect(self.update_status_signal)
        self.detection_worker.start()
        
    def _detect_cube_state_cnn(self):
        """Start CNN detection process"""
        if not self._check_images_captured():
            return
            
        self._set_processing_state(True, "Starting AI detection...")
        self.update_status_signal.emit("CNN detection not yet implemented - using manual detection")
        
        # Fallback to manual detection
        self._detect_cube_state_manual()
        
    def _on_detection_finished(self, result: DetectionResult):
        """Handle completed detection"""
        self._set_processing_state(False, "Detection completed")

        # Update cube state
        print("Prev status : "+self.cube_status.current_status)

        self.cube_status.current_status = result.status_string
        self.cube_status.side_to_color = result.side_to_color

        print("Curr status : "+self.cube_status.current_status)
        
        self._update_cube_display()

        # Update validation
        if result.has_errors:
            self.validation_result.setText("Detection completed with warnings - please verify colors")
            self.validation_result.setStyleSheet("color: orange; background-color: #332200; padding: 8px; border-radius: 4px;")
        else:
            self.validation_result.setText("Detection completed successfully!")
            self.validation_result.setStyleSheet("color: green; background-color: #002200; padding: 8px; border-radius: 4px;")

        self._show_info("Detection Complete", 
                       f"Detection completed using {result.method} method. Please verify the colors and correct if needed.") 
    
    def _on_detection_error(self, error_message: str):
        """Handle detection errors"""
        self._set_processing_state(False, "Detection failed")
        self._show_error("Detection Error", error_message)
        
    def _scramble_cube(self):
        """Scramble the cube with random moves"""
        num_moves, ok = QInputDialog.getInt(
            self, "Scramble Cube", 
            "Number of scramble moves:", 
            value=20, min=1, max=100, step=1
        )
        
        if not ok:
            return
            
        self._set_processing_state(True, f"Scrambling with {num_moves} moves...")
        
        try:
            moves = self.motor_controller.generate_scramble(num_moves)
            
            # Execute moves on hardware
            if self.motor_controller.execute_sequence(moves):
                # Update cube state
                self.cube_status.current_status = self.cube_status.change_status(
                    self.cube_status.current_status, moves
                )
                self._update_cube_display()
                self.update_status_signal.emit(f"Scrambled with {num_moves} moves")
            else:
                self._show_error("Scramble Error", "Failed to execute scramble moves on hardware")
                
        except Exception as e:
            self._show_error("Scramble Error", f"Scrambling failed: {str(e)}")
        finally:
            self._set_processing_state(False)
            
    def _solve_cube(self):
        """Solve the cube using Kociemba algorithm"""
        if not self._validate_cube_state(show_warning=True):
            return
            
        self._set_processing_state(True, "Solving cube...")
        try:
            # Convert to face notation for Kociemba
            face_status = self.cube_status.convert_to_face_status()
            solution = kociemba.solve(face_status).split()
            
            self.update_status_signal.emit(f"Solution found: {' '.join(solution)}")
            
            # Execute solution
            if self.motor_controller.execute_sequence(solution):
                # Update cube state to solved
                self.cube_status.current_status = self.cube_status.change_status(
                    self.cube_status.current_status, solution
                )
                self._update_cube_display()
                self.update_status_signal.emit("Cube solved successfully!")
                self._show_info("Solution", f"Solution: {' '.join(solution)}")
            else:
                self._show_error("Solve Error", "Failed to execute solution on hardware")
                
        except Exception as e:
            self._show_error("Solve Error", f"Solving failed: {str(e)}")
        finally:
            self._set_processing_state(False)
            
    def _custom_moves(self):
        moves, ok = QInputDialog.getText(
            self, "Custom Moves", 
            "Enter moves (e.g., R U R' F B2 D'):"
        )
        
        if not ok or not moves.strip():
            return
            
        move_list = moves.split()
        self._set_processing_state(True, f"Executing {len(move_list)} custom moves...")
        
        try:
            if self.motor_controller.execute_sequence(move_list):
                # Update cube state
                self.cube_status.current_status = self.cube_status.change_status(
                    self.cube_status.current_status, move_list
                )
                self._update_cube_display()
                self.update_status_signal.emit("Custom moves executed successfully")
            else:
                self._show_error("Execution Error", "Failed to execute moves on hardware")
                
        except Exception as e:
            self._show_error("Execution Error", f"Custom moves failed: {str(e)}")
        finally:
            self._set_processing_state(False)
            
    def _test_motors(self):
        """Test all motors"""
        self._set_processing_state(True, "Testing motors...")
        
        try:
            if self.motor_controller.test_all_motors():
                self.update_status_signal.emit("Motor test completed successfully")
                self._show_info("Motor Test", "All motors tested successfully!")
            else:
                self._show_error("Motor Test", "Motor test completed with errors")
        except Exception as e:
            self._show_error("Motor Test", f"Motor test failed: {str(e)}")
        finally:
            self._set_processing_state(False)
        
    def _arduino_connection(self):
        if self.motor_controller.is_connected:
            self.motor_controller.is_connected = False
        else:
            if self.motor_controller.serial_connection != None:
                self.motor_controller.is_connected = True
            else:
                self.motor_controller.is_connected = self.motor_controller.connect()
        self._initialize_hardware()

    def _motor_f(self):
        self.motor_controller.execute_move("F")
    def _motor_d(self):
        self.motor_controller.execute_move("D")
    def _motor_b(self):
        self.motor_controller.execute_move("B")
    def _motor_r(self):
        self.motor_controller.execute_move("R")
    def _motor_l(self):
        self.motor_controller.execute_move("L")
    def _motor_f_reverse(self):
        self.motor_controller.execute_move("F'")
    def _motor_d_reverse(self):
        self.motor_controller.execute_move("D'")
    def _motor_b_reverse(self):
        self.motor_controller.execute_move("B'")
    def _motor_r_reverse(self):
        self.motor_controller.execute_move("R'")
    def _motor_l_reverse(self):
        self.motor_controller.execute_move("L'")

    def _switch_camera(self):
        """Switch to next available camera"""
        self.update_status_signal.emit("Switching camera...")
        if self.camera_controller.switch_camera():
            self.update_status_signal.emit("Camera switched successfully")
        else:
            self._show_warning("Camera Switch", "No other cameras found")
            
    def _validate_cube_state(self, show_warning: bool = False) -> bool:
        """Validate the current cube state"""
        is_valid, issues = self.cube_status.validate_state()
        
        if is_valid:
            self.validation_result.setText("✓ Cube state is valid!")
            self.validation_result.setStyleSheet("color: green; background-color: #002200; padding: 8px; border-radius: 4px;")
            return True
        else:
            message = "Cube state is invalid:\n" + "\n".join(issues)
            self.validation_result.setText(message)
            self.validation_result.setStyleSheet("color: red; background-color: #220000; padding: 8px; border-radius: 4px;")
            
            if show_warning:
                reply = QMessageBox.question(
                    self, "Invalid Cube State", 
                    f"{message}\n\nContinue anyway?",
                    QMessageBox.Yes | QMessageBox.No
                )
                return reply == QMessageBox.Yes
                
            return False
            
    def _reset_cube_state(self):
        """Reset cube to default state"""
        reply = QMessageBox.question(
            self, "Reset Cube State",
            "Are you sure you want to reset the cube state?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.cube_status = CubeStatus()  # Create fresh instance
            self._update_cube_display()
            self.validation_result.setText("Cube state reset")
            self.validation_result.setStyleSheet("padding: 8px; border-radius: 4px; background-color: #333333;")
            self.update_status_signal.emit("Cube state reset to default")
            
    def _on_sticker_clicked(self, side: str, sticker_pos: str):
        """Handle sticker click for manual color correction"""
        
        dialog = ColorDialog(self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_color:
            try:
                self.cube_status.update_sticker(side, sticker_pos, dialog.selected_color)
                self._update_cube_display()
                self.update_status_signal.emit(f"Changed {sticker_pos} to {dialog.selected_color}")
            except Exception as e:
                self._show_error("Color Update Error", f"Failed to update color: {str(e)}")
                
    def _check_images_captured(self) -> bool:
        """Check if required images are captured"""
        if not config.IMG1_PATH.exists():
            self._show_warning("Images Required", "Please capture at least one image using Space bar")
            return False
        return True
        
    def _set_processing_state(self, processing: bool, message: str = ""):
        """Update UI for processing state"""
        self.is_processing = processing
        
        # Enable/disable buttons
        buttons = [
            self.detect_cnn_btn, self.detect_manual_btn, self.scramble_btn,
            self.solve_btn, self.custom_moves_btn, self.motor_test_btn,
            self.camera_switch_btn, self.validate_btn, self.reset_btn
        ]
        
        for button in buttons:
            button.setEnabled(not processing)
            
        if message:
            self.update_status_signal.emit(message)
            
    def _update_cube_display(self):
        """Update the cube display widget"""
        self.cube_widget.set_state(
            self.cube_status.current_status, 
            self.cube_status.side_to_color
        )
        
    def _update_status_bar(self, message: str):
        """Update status bar (thread-safe)"""
        self.statusBar().showMessage(message)
        
    def _show_info(self, title: str, message: str):
        """Show information message"""
        QMessageBox.information(self, title, message)
        
    def _show_warning(self, title: str, message: str):
        """Show warning message"""
        QMessageBox.warning(self, title, message)
        
    def _show_error(self, title: str, message: str):
        """Show error message"""
        QMessageBox.critical(self, title, message)
        
    def closeEvent(self, event):
        """Handle application close"""
        # Stop any running processes
        if self.detection_worker and self.detection_worker.isRunning():
            self.detection_worker.stop()
            self.detection_worker.wait(2000)  # Wait up to 2 seconds
            
        # Release resources
        self.camera_controller.release()
        self.motor_controller.disconnect()
        
        event.accept()


class CubeWidget(QWidget):
    stickerClicked = pyqtSignal(str, str)  # Signal: (face, position)
    
    def __init__(self):
        super().__init__()
        self.STICKER_SIZE = 28
        self.STICKER_GAP = 3
        self.SIDE_GAP = 10
        self.SIDE_SIZE = self.STICKER_SIZE * 3 + self.STICKER_GAP * 4
        self.SIDE_POSITIONS = {
            'U': (1, 0), 'L': (0, 1), 'F': (1, 1), 
            'R': (2, 1), 'B': (3, 1), 'D': (1, 2)
        }
        
        self.color_palette = {
            'R': QColor(255, 0, 0),      # Red
            'O': QColor(255, 128, 0),    # Orange
            'B': QColor(0, 0, 255),      # Blue
            'G': QColor(0, 255, 0),      # Green
            'W': QColor(255, 255, 255),  # White
            'Y': QColor(255, 255, 0)     # Yellow
        }
        
        self.setMinimumSize(
            self.SIDE_SIZE * 4 + self.SIDE_GAP * 5, 
            self.SIDE_SIZE * 3 + self.SIDE_GAP * 4
        )
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #555555;")
        
        self.status = 'RRRRRRRRRGGGGGGGGGWWWWWWWWWOOOOOOOOOBBBBBBBBBYYYYYYYYY'  # Solved state
        self.selected_sticker = None
        
    def set_state(self, status: str, side_to_color: Dict[str, str]):
        self.status = status
        self.update()
        
    def get_sticker_at_pos(self, pos) -> Tuple[Optional[str], Optional[str]]:
        x, y = pos.x(), pos.y()
        
        for side, (x_pos, y_pos) in self.SIDE_POSITIONS.items():
            offset_x = x_pos * self.SIDE_SIZE + (x_pos + 1) * self.SIDE_GAP
            offset_y = y_pos * self.SIDE_SIZE + (y_pos + 1) * self.SIDE_GAP
            
            # Check if click is within this side
            if (offset_x <= x <= offset_x + self.SIDE_SIZE and 
                offset_y <= y <= offset_y + self.SIDE_SIZE):
                
                rel_x = x - offset_x
                rel_y = y - offset_y
                
                # Check if click is in the sticker area
                sticker_col = int(rel_x // (self.STICKER_SIZE + self.STICKER_GAP))
                sticker_row = int(rel_y // (self.STICKER_SIZE + self.STICKER_GAP))
                
                if 0 <= sticker_row < 3 and 0 <= sticker_col < 3:
                    sticker_idx = sticker_row * 3 + sticker_col
                    sticker_pos = f"{side}{sticker_idx + 1}"  # U1, U2, ..., U9
                    return side, sticker_pos
        return None, None
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            side, sticker_pos = self.get_sticker_at_pos(event.pos())
            if side and sticker_pos:
                self.selected_sticker = (side, sticker_pos)
                self.stickerClicked.emit(side, sticker_pos)
                self.update()
                
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(26, 26, 26))
        
        # Draw all cube sides - FIXED: Use direct color mapping
        face_order = ['U', 'R', 'F', 'D', 'L', 'B']
        
        for side_idx, side in enumerate(face_order):
            x_pos, y_pos = self.SIDE_POSITIONS[side]
            offset_x = x_pos * self.SIDE_SIZE + (x_pos + 1) * self.SIDE_GAP
            offset_y = y_pos * self.SIDE_SIZE + (y_pos + 1) * self.SIDE_GAP
            
            for i in range(9):
                row, col = i // 3, i % 3
                sticker_x = offset_x + col * (self.STICKER_SIZE + self.STICKER_GAP) + self.STICKER_GAP
                sticker_y = offset_y + row * (self.STICKER_SIZE + self.STICKER_GAP) + self.STICKER_GAP
                
                # Get color letter directly from status string
                color_char = self.status[side_idx * 9 + i]
                color = self.color_palette.get(color_char, QColor(128, 128, 128))  # Gray for unknown
                
                # Highlight selected sticker
                if (self.selected_sticker and 
                    self.selected_sticker[0] == side and 
                    self.selected_sticker[1] == f"{side}{i+1}"):
                    painter.setPen(QPen(QColor(255, 255, 0), 3))  # Yellow border for selected
                else:
                    painter.setPen(QPen(QColor(50, 50, 50), 1))   # Dark border for others
                
                painter.fillRect(sticker_x, sticker_y, self.STICKER_SIZE, self.STICKER_SIZE, color)
                painter.drawRect(sticker_x, sticker_y, self.STICKER_SIZE, self.STICKER_SIZE)

class ColorDialog(QDialog):
    """Dialog for selecting sticker colors - FIXED"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_color = None
        self.setWindowTitle("Select Sticker Color")
        self.setStyleSheet(config.DARK_THEME)
        self.setModal(True)
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        label = QLabel("Choose a color for the selected sticker:")
        label.setWordWrap(True)
        layout.addWidget(label)
        
        # Color buttons - FIXED: Use color letters directly
        color_grid = self._create_color_grid()
        layout.addLayout(color_grid)
        
        # Cancel button
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def _create_color_grid(self) -> QGridLayout:
        """Create the color selection grid - FIXED"""
        color_grid = QGridLayout()
        
        # Use the same color mapping as everywhere else
        colors = [
            ('R', 'Red', QColor(255, 0, 0)),
            ('O', 'Orange', QColor(255, 128, 0)),
            ('B', 'Blue', QColor(0, 0, 255)),
            ('G', 'Green', QColor(0, 255, 0)),
            ('W', 'White', QColor(255, 255, 255)),
            ('Y', 'Yellow', QColor(255, 255, 0))
        ]
        
        for i, (color_char, color_name, color) in enumerate(colors):
            btn = QPushButton(color_name)
            btn.setFixedSize(80, 40)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color.name()};
                    border: 2px solid #aaaaaa;
                    font-weight: bold;
                    color: {'black' if color_name in ['White', 'Yellow'] else 'white'};
                }}
                QPushButton:hover {{
                    border: 3px solid yellow;
                }}
            """)
            # Pass the color LETTER, not the name
            btn.clicked.connect(lambda checked, c=color_char: self.color_selected(c))
            color_grid.addWidget(btn, i // 3, i % 3)
            
        return color_grid
            
    def color_selected(self, color_char: str):
        """Handle color selection - FIXED: Use color letters"""
        self.selected_color = color_char
        self.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application properties
    app.setApplicationName("Rubik's Cube Solver")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("CubeSolver")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Execute application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()