import sys
import time
import cv2
import kociemba
import numpy as np
import concurrent.futures
from typing import Dict, Tuple, Optional, List

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QMessageBox, QInputDialog, QGridLayout, 
    QGroupBox, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QEvent
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPen, QKeySequence
from PyQt5.QtWidgets import QShortcut

from cube_status import CubeStatus
from control import MotorController
from themes import DARK_THEME


class ThreadSafeEvent(QEvent):
    """Custom event for thread-safe UI updates"""
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
    
    def __init__(self, callback):
        super().__init__(ThreadSafeEvent.EVENT_TYPE)
        self.callback = callback
        
    def execute(self):
        self.callback()


class CameraWidget(QLabel):
    """Widget to display camera feed with overlay for cube detection regions"""
    
    def __init__(self, polygons: Dict, parent=None):
        super().__init__(parent)
        self.polygons = polygons
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Camera feed will appear here")
        self.setStyleSheet("border: 1px solid #555555; background-color: #1a1a1a;")
        
    def set_frame(self, frame: np.ndarray):
        """Update the widget with a new frame from the camera"""
        if frame is None or frame.size == 0:
            return
            
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            print(f"Error setting frame: {e}")
        
    def paintEvent(self, event):
        """Override paint event to draw detection regions"""
        super().paintEvent(event)

        if self.pixmap() is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(255, 100, 100), 2)
        painter.setPen(pen)

        # Dynamic font size based on widget size
        font_size = max(10, int(min(self.width(), self.height()) * 0.02))
        font = QFont('Sans', font_size)
        painter.setFont(font)

        pixmap_size = self.pixmap().size()
        pixmap_size.scale(self.size(), Qt.KeepAspectRatio)

        # Calculate scaling and offset
        scale_x = pixmap_size.width() / self.pixmap().width()
        scale_y = pixmap_size.height() / self.pixmap().height()
        offset_x = (self.width() - pixmap_size.width()) / 2
        offset_y = (self.height() - pixmap_size.height()) / 2

        font_metrics = painter.fontMetrics()

        for label, (x, y, w, h) in self.polygons.items():
            # Scale coordinates
            scaled_x = offset_x + x * scale_x
            scaled_y = offset_y + y * scale_y + 20
            scaled_w = w * scale_x
            scaled_h = h * scale_y

            # Convert to integers
            rect_x, rect_y = int(scaled_x), int(scaled_y)
            rect_w, rect_h = int(scaled_w), int(scaled_h)

            # Draw rectangle
            painter.drawRect(rect_x, rect_y, rect_w, rect_h)

            # Calculate text position
            try:
                text_width = font_metrics.horizontalAdvance(label)
            except AttributeError:
                text_width = font_metrics.boundingRect(label).width()
                
            text_x = rect_x + (rect_w - text_width) // 2
            text_y = rect_y - 5

            # Draw text
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(int(text_x), int(text_y), label)


class CubeWidget(QWidget):
    """Widget to display and interact with the Rubik's cube state"""
    stickerClicked = pyqtSignal(str, str)  # Signal: (face, position)
    
    def __init__(self):
        super().__init__()
        # Configuration
        self.STICKER_SIZE = 28
        self.STICKER_GAP = 3
        self.SIDE_GAP = 10
        self.SIDE_SIZE = self.STICKER_SIZE * 3 + self.STICKER_GAP * 4
        self.SIDE_POSITIONS = {
            'U': (1, 0), 'L': (0, 1), 'F': (1, 1), 
            'R': (2, 1), 'B': (3, 1), 'D': (1, 2)
        }
        
        # Color palette
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
        
        # Initialize state
        self.status = 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'
        self.side_to_color = {'U': 'O', 'R': 'B', 'F': 'W', 'D': 'R', 'L': 'G', 'B': 'Y'}
        self.selected_sticker = None
        
    def set_state(self, status: str, side_to_color: Dict[str, str]):
        """Set the cube state"""
        self.status = status
        self.side_to_color = side_to_color
        self.update()
        
    def get_sticker_at_pos(self, pos) -> Tuple[Optional[str], Optional[str]]:
        """Get the sticker at the given position"""
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
                if (rel_x % (self.STICKER_SIZE + self.STICKER_GAP) < self.STICKER_SIZE and
                    rel_y % (self.STICKER_SIZE + self.STICKER_GAP) < self.STICKER_SIZE):
                    
                    col = int(rel_x // (self.STICKER_SIZE + self.STICKER_GAP))
                    row = int(rel_y // (self.STICKER_SIZE + self.STICKER_GAP))
                    
                    if 0 <= row < 3 and 0 <= col < 3:
                        sticker_idx = row * 3 + col
                        sticker_pos = f"{side}{sticker_idx + 1}"
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
        
        # Draw all cube sides
        for side_idx, side in enumerate(['U', 'R', 'F', 'D', 'L', 'B']):
            x_pos, y_pos = self.SIDE_POSITIONS[side]
            offset_x = x_pos * self.SIDE_SIZE + (x_pos + 1) * self.SIDE_GAP
            offset_y = y_pos * self.SIDE_SIZE + (y_pos + 1) * self.SIDE_GAP
            
            for i in range(9):
                row, col = i // 3, i % 3
                sticker_x = offset_x + col * self.STICKER_SIZE + (col + 1) * self.STICKER_GAP
                sticker_y = offset_y + row * self.STICKER_SIZE + (row + 1) * self.STICKER_GAP
                
                color_char = self.status[side_idx * 9 + i]
                color_name = self.side_to_color.get(color_char, 'W')
                color = self.color_palette.get(color_name, QColor(128, 128, 128))
                
                # Highlight selected sticker
                if (self.selected_sticker and 
                    self.selected_sticker[0] == side and 
                    self.selected_sticker[1] == f"{side}{i+1}"):
                    painter.setPen(QPen(QColor(255, 255, 0), 3))
                else:
                    painter.setPen(QPen(QColor(50, 50, 50), 1))
                
                painter.fillRect(sticker_x, sticker_y, self.STICKER_SIZE, self.STICKER_SIZE, color)
                painter.drawRect(sticker_x, sticker_y, self.STICKER_SIZE, self.STICKER_SIZE)


class ColorDialog(QDialog):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_color = None
        self.setWindowTitle("Select Sticker Color")
        self.setStyleSheet(DARK_THEME)
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        label = QLabel("Choose a color for this sticker:")
        label.setWordWrap(True)
        layout.addWidget(label)
        
        # Color buttons
        color_grid = self._create_color_grid()
        layout.addLayout(color_grid)
        
        # Cancel button
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def _create_color_grid(self) -> QGridLayout:
        """Create the color selection grid"""
        color_grid = QGridLayout()
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
                    color: {'black' if color_name in ['W', 'Y'] else 'white'};
                }}
                QPushButton:hover {{
                    border: 3px solid yellow;
                }}
            """)
            btn.clicked.connect(lambda checked, c=color_char: self.color_selected(c))
            color_grid.addWidget(btn, i // 3, i % 3)
            
        return color_grid
            
    def color_selected(self, color_char: str):
        """Handle color selection"""
        self.selected_color = color_char
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cube = CubeStatus()
        self.controller = MotorController()
        
        self._setup_camera()
        self._setup_ui()
        self._connect_signals()
        
        if not self.controller.connect():
            QMessageBox.warning(self, "Controller Warning", 
                              f"Hardware controller initialization failed\n"
                              "Software functions will still work.")
    
    ### ----INITIALIZERS----    
    
    def _setup_camera(self):
        self.cap = None
        self.captured_frames = []
        self.last_frame = None
        
        # Try to find available camera
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.cap_idx = i
                self.cap = cap
                break

        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "Camera Warning", 
                              "Could not open camera. Camera functions will be disabled.")
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Camera timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_camera)
        self.timer.start(30)
        
    def _setup_ui(self):
        """Setup the main UI"""
        self.setStyleSheet(DARK_THEME)
        self.setWindowTitle("Rubik's Cube Solver")
        self.setGeometry(100, 100, 1200, 800)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        layout = QHBoxLayout(self.central_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Left side - cube and camera
        left_widget = self._create_left_widget()
        layout.addWidget(left_widget, 2)
        
        # Right side - controls
        right_widget = self._create_right_widget()
        layout.addWidget(right_widget, 1)
        
    def _create_left_widget(self) -> QWidget:
        """Create left side widget with cube and camera"""
        widget = QWidget()
        widget.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555555; border-radius: 5px;")
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Cube widget
        self.cube_widget = CubeWidget()
        self.cube_widget.stickerClicked.connect(self._on_sticker_clicked)
        layout.addWidget(self.cube_widget)

        # Camera widget
        self.camera_widget = CameraWidget(self.cube.polygons)
        layout.addWidget(self.camera_widget, 1)
        
        # Thumbnails
        thumb_layout = self._create_thumbnail_layout()
        layout.addLayout(thumb_layout)
        
        # Status label
        self.status_label = QLabel("Current status: Ready")
        self.status_label.setStyleSheet("font-weight: bold; color: #ffffff; background-color: #404040; padding: 5px; border-radius: 3px;")
        layout.addWidget(self.status_label)
        
        # Initialize cube view
        self.cube_widget.set_state(self.cube.current_status, self.cube.side_to_color)
        
        return widget
        
    def _create_thumbnail_layout(self) -> QHBoxLayout:
        """Create thumbnail display layout"""
        layout = QHBoxLayout()
        
        self.thumb1 = QLabel()
        self.thumb1.setFixedSize(160, 90)
        self.thumb1.setStyleSheet("border: 1px solid #555555; background-color: #1a1a1a;")
        
        self.thumb2 = QLabel()
        self.thumb2.setFixedSize(160, 90)
        self.thumb2.setStyleSheet("border: 1px solid #555555; background-color: #1a1a1a;")
        
        layout.addWidget(QLabel("Image 1:"))
        layout.addWidget(self.thumb1)
        layout.addWidget(QLabel("Image 2:"))
        layout.addWidget(self.thumb2)
        
        return layout
        
    def _create_right_widget(self) -> QWidget:
        """Create right side widget with controls"""
        widget = QWidget()
        widget.setMaximumWidth(400)
        widget.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555555; border-radius: 5px;")
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Detection group
        detect_group = self._create_detection_group()
        layout.addWidget(detect_group)
        
        # Actions group
        actions_group = self._create_actions_group()
        layout.addWidget(actions_group)
        
        # Validation group
        validation_group = self._create_validation_group()
        layout.addWidget(validation_group)
        
        # Copyright
        copyright_label = QLabel("@Creator: Facundo and Ulises")
        copyright_label.setStyleSheet("color: #888888; font-size: 10px; padding: 10px;")
        copyright_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(copyright_label)
        
        layout.addStretch(1)
        
        return widget
        
    def _create_detection_group(self) -> QGroupBox:
        """Create detection control group"""
        group = QGroupBox("Detection")
        layout = QVBoxLayout(group)
        
        self.detect_btn = QPushButton("🔍 Detect Cube State")
        self.detect_btn.clicked.connect(self._detect_cube_state)
        layout.addWidget(self.detect_btn)
        
        return group
        
    def _create_actions_group(self) -> QGroupBox:
        """Create actions control group"""
        group = QGroupBox("Actions")
        layout = QVBoxLayout(group)
        
        buttons = [
            ("🌀 Scramble Cube", self._scramble_cube),
            ("✅ Solve Cube", self._solve_cube),
            ("🔧 Custom Moves", self._custom_moves),
            ("⚙️ Test Motors", self._test_motors),
            ("📷 Change Camera", self._change_camera),
            ("🔄 Reset cube state (UI)", self._reset_cube_status),
            ("🚪 Exit", self.close)
        ]
        
        for text, slot in buttons:
            btn = QPushButton(text)
            if text == "🚪 Exit":
                btn.setStyleSheet("background-color: #8B0000; color: white; font-weight: bold;")
            btn.clicked.connect(slot)
            layout.addWidget(btn)
        
        return group
        
    def _create_validation_group(self) -> QGroupBox:
        group = QGroupBox("Validation")
        layout = QVBoxLayout(group)
        
        self.validate_btn = QPushButton("✓ Validate Cube State")
        self.validate_btn.clicked.connect(self._validate_cube_state)
        layout.addWidget(self.validate_btn)
        
        self.validation_result = QLabel("Not yet validated")
        self.validation_result.setStyleSheet("padding: 5px; border-radius: 3px;")
        layout.addWidget(self.validation_result)
        
        return group
        
    def _connect_signals(self):
        """Connect keyboard shortcuts"""
        self.space_shortcut = QShortcut(QKeySequence("Space"), self)
        self.space_shortcut.activated.connect(self._on_space_pressed)
        
    def _update_camera(self):
        """Update camera feed"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.last_frame = frame.copy()
                self.camera_widget.set_frame(frame)
    
    ### ----------------    

    def _on_space_pressed(self):
        """Handle space bar for capturing images"""
        if self.last_frame is None:
            return
            
        self.captured_frames.append(self.last_frame.copy())
        idx = len(self.captured_frames)
        
        # Show thumbnail
        thumb_rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        thumb_small = cv2.resize(thumb_rgb, (160, 90))
        h, w, ch = thumb_small.shape
        qimg = QImage(thumb_small.data, w, h, ch * w, QImage.Format_RGB888)
        
        if idx == 1:
            self.thumb1.setPixmap(QPixmap.fromImage(qimg))
            qimg.save(".//pictures//detector1.png")
            self._update_status("Image 1 captured. Press Space for image 2.")
        elif idx == 2:
            self.thumb2.setPixmap(QPixmap.fromImage(qimg))
            qimg.save(".//pictures//detector2.png")
            self._update_status("Image 2 captured. Click 'Detect Cube State' to process.")
        else:
            self.captured_frames = [self.last_frame.copy()]
            self.thumb1.clear()
            self.thumb2.clear()
            self.thumb1.setPixmap(QPixmap.fromImage(qimg))
            self._update_status("Reset: image 1 captured.")

    def _on_sticker_clicked(self, side: str, sticker_pos: str):
        dialog = ColorDialog(self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_color:
            self._update_sticker_color(side, sticker_pos, dialog.selected_color)
    
    def _update_sticker_color(self, side: str, sticker_pos: str, color: str):
        try:
            self.cube.update_sticker(side, sticker_pos, color)
            self._update_status(f"Changed {sticker_pos} to {color}")
            self.cube_widget.set_state(self.cube.current_status, self.cube.side_to_color)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update color: {str(e)}")
    
    def _validate_cube_state(self):
        is_valid, issues = self.cube.validate_state()
        
        if is_valid:
            self.validation_result.setText("Cube state is valid!")
            self.validation_result.setStyleSheet(
                "color: green; background-color: #002200; padding: 5px; border-radius: 3px;"
            )
        else:
            self.validation_result.setText("Cube state is invalid:\n" + "\n".join(issues))
            self.validation_result.setStyleSheet(
                "color: red; background-color: #220000; padding: 5px; border-radius: 3px;"
            )
        
        return is_valid
            
    def _detect_cube_state(self):
        if len(self.captured_frames) < 2:
            QMessageBox.warning(self, "Warning", "Please capture 2 images first by pressing Space")
            return
            
        self._update_status("Detecting cube state...")
        QApplication.processEvents()
        
        frame1, frame2 = self.captured_frames[:2]
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.cube.detect_status_from_frames, frame1, frame2)
        future.add_done_callback(self._on_detection_done)
                
    def _on_detection_done(self, future):
        try:
            status_string, side_to_color, has_error = future.result()
            
            color_status = ''.join(side_to_color.get(face_char, 'W') for face_char in status_string)
            
            self.cube.current_status = color_status
            self.cube_widget.set_state(self.cube.current_status, self.cube.side_to_color)
            
            if has_error:
                self.validation_result.setText("Detection has issues")
                self.validation_result.setStyleSheet(
                    "color: orange; background-color: #332200; padding: 5px; border-radius: 3px;"
                )
                self._update_status("Detection completed with warnings")
            else:
                self.validation_result.setText("Cube state is valid!")
                self.validation_result.setStyleSheet(
                    "color: green; background-color: #002200; padding: 5px; border-radius: 3px;"
                )
                self._update_status("Detection completed successfully")
                
        except Exception as e:
            self._update_status(f"Detection error: {e}")
    
    def _scramble_cube(self):
        num_moves, ok = QInputDialog.getInt(self, "Scramble", "How many random moves?", 10, 1, 100, 1)
        if not ok:
            return
            
        self._update_status("Scrambling...")
        QApplication.processEvents()
        
        try:
            moves = self.controller.generate_scramble(num_moves)
            self.controller.execute_sequence(moves)
            self.cube.current_status = self.cube.change_status( self.cube.current_status, moves)
            self.cube_widget.set_state(self.cube.current_status, self.cube.side_to_color)
            self._update_status(f"Scrambled with {num_moves} moves: {' '.join(moves)}")
        except Exception as e:
            self._update_status(f"Scramble error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Scrambling failed: {str(e)}")
            
    def _solve_cube(self):
        if not self._validate_cube_state():
            reply = QMessageBox.question(
                self, "Invalid Cube", 
                "The cube state is invalid. Solve anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        self._update_status("Solving...")
        QApplication.processEvents()
        
        try:
            face_status = self.cube.convert_to_face_status(self.cube.current_status)
            moves = kociemba.solve(face_status).split()
            self._update_status(f"Executing solution: {' '.join(moves)}")
            
            for move in moves:
                self.controller.execute_move(move)
                QApplication.processEvents()
                
            self.cube.current_status = self.cube.change_status(
                self.cube.current_status, moves
            )
            self.cube_widget.set_state(self.cube.current_status, self.cube.side_to_color)
            self._update_status("Cube solved!")
            QMessageBox.information(self, "Solution", f"Solution: {' '.join(moves)}")
        except Exception as e:
            self._update_status(f"Solve error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Solving failed: {str(e)}")
            
    def _custom_moves(self):
        moves, ok = QInputDialog.getText( self, "Custom Moves", "Enter moves (e.g., R U R'):" )
        if not ok or not moves.strip():
            return
            
        moves = moves.split()
        self._update_status(f"Executing custom moves: {' '.join(moves)}")
        QApplication.processEvents()
        
        self.cube._apply_move_to_list(moves)

        try:
            for move in moves:
                self.controller.execute_move(move)
                QApplication.processEvents()
                
            self.cube.current_status = self.cube.change_status(
                self.cube.current_status, moves
            )
            self.cube_widget.set_state(self.cube.current_status, self.cube.side_to_color)
            self._update_status("Custom moves executed")
        except Exception as e:
            self._update_status(f"Custom moves error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Custom moves failed: {str(e)}")
            
    def _test_motors(self):
        self._update_status("Testing motors...")
        QApplication.processEvents()

        try:        
            if self.controller.test_all_motors():
                self._update_status("Motor test completed")
                QMessageBox.information(self, "Test", "All motors tested successfully!")
            else:
                self._update_status(f"Motor test error")
        except Exception as e:
            self._update_status(f"Motor test error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Motor test failed: {str(e)}")
    
    def _reset_cube_status(self):
        self.cube.current_status = 'OOOOOOOOOBBBBBBBBBWWWWWWWWWRRRRRRRRRGGGGGGGGGYYYYYYYYY'
        self._update_status("Cube UI has been reseted")
        self.cube_widget.set_state(self.cube.current_status, self.cube.side_to_color)

    def _change_camera(self):
        if not self.cap:
            return
            
        for i in range(10):
            if i == self.cap_idx:
                continue
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.cap_idx = i
                self.cap.release()
                self.cap = cap
                break
        
    def _update_status(self, message: str):
        self.status_label.setText(f"Status: {message}")
        
    def closeEvent(self, event):
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()