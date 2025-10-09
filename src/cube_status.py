import colorsys
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
from joblib import load
from tensorflow.keras.models import load_model


class CubeStatus:
    def __init__(self):
        # Polygons for camera 1.0 Mpx HD (1280x720 or 1366x768 resolution)
        # scaled from original 1280x720 -> 160x90 (scale = 0.125)
        self.polygons = {
            "U1": (888, 248, 60, 33), "U2": (1025, 274, 46, 37), "U3": (1169, 313, 53, 37),
            "U4": (765, 283, 62, 35), "U5": (907, 327, 58, 37), "U6": (1053, 371, 65, 42),
            "U7": (607, 339, 56, 42), "U8": (744, 382, 60, 42), "U9": (904, 434, 70, 49),
            "L1": (547, 436, 49, 47), "L2": (670, 501, 51, 60), "L3": (800, 564, 56, 60),
            "L4": (594, 607, 46, 51), "L5": (700, 679, 46, 58), "L6": (826, 765, 49, 60),
            "L7": (619, 744, 39, 46), "L8": (719, 821, 42, 51), "L9": (832, 913, 47, 53),
            "R1": (995, 587, 49, 60), "R2": (1146, 501, 51, 62), "R3": (1269, 420, 44, 58),
            "R4": (986, 782, 51, 58), "R5": (1141, 686, 42, 53), "R6": (1243, 589, 42, 51),
            "R7": (992, 939, 44, 53), "R8": (1124, 823, 42, 47), "R9": (1222, 735, 40, 46)
        }


        # Map labels in the first picture to final label in cube status
        self.pic2_label_mapping = {
            'U1': 'U1', 'U2': 'U2', 'U3': 'U3', 'U4': 'U4', 'U5': 'U5', 'U6': 'U6', 
            'U7': 'U7', 'U8': 'U8', 'U9': 'U9',
            'L1': 'F1', 'L2': 'F2', 'L3': 'F3', 'L4': 'F4', 'L5': 'F5', 'L6': 'F6', 
            'L7': 'F7', 'L8': 'F8', 'L9': 'F9',
            'R1': 'R1', 'R2': 'R2', 'R3': 'R3', 'R4': 'R4', 'R5': 'R5', 'R6': 'R6', 
            'R7': 'R7', 'R8': 'R8', 'R9': 'R9',
        }
        
        # Map labels in the second picture to final label in cube status      
        self.pic1_label_mapping = {
            'U1': 'D3', 'U2': 'D6', 'U3': 'D9', 'U4': 'D2', 'U5': 'D5', 'U6': 'D8', 
            'U7': 'D1', 'U8': 'D4', 'U9': 'D7',
            'L1': 'L9', 'L2': 'L8', 'L3': 'L7', 'L4': 'L6', 'L5': 'L5', 'L6': 'L4', 
            'L7': 'L3', 'L8': 'L2', 'L9': 'L1',
            'R1': 'B9', 'R2': 'B8', 'R3': 'B7', 'R4': 'B6', 'R5': 'B5', 'R6': 'B4', 
            'R7': 'B3', 'R8': 'B2', 'R9': 'B1',
        }

        # The order of sides to output cube status
        self.side_order = ['U', 'R', 'F', 'D', 'L', 'B']
        self.colors = ['B', 'G', 'O', 'R', 'W', 'Y']
        
        self.current_status = 'OOOOOOOOOBBBBBBBBBWWWWWWWWWRRRRRRRRRGGGGGGGGGYYYYYYYYY'
        self.side_to_color = {'O': 'O', 'B': 'B', 'W': 'W', 'R': 'R', 'G': 'G', 'Y': 'Y'}

        # Image size of each color, so each polygon has this size of frame to be detected
        self.img_width = 96  
        self.img_height = 96

        # Load models with error handling
        try:
            self.model = load_model('color_detection-v4-7.h5', compile=False)
        except Exception as e:
            print(f"Warning: Could not load CNN model: {e}")
            self.model = None
            
        try:
            self.decision_tree = load('decision_tree-v4-7.joblib')
        except Exception as e:
            print(f"Warning: Could not load Decision Tree: {e}")
            self.decision_tree = None

    def get_refined_polygons(self, img: np.ndarray, max_offset: int = 40, step: int = 6) -> Dict[str, Tuple[int,int,int,int]]:
        """
        Perform a local search around each nominal polygon to find the most saturated/bright patch.
        Returns a new polygons dict (label -> (x,y,w,h)).
        """
        h_img, w_img = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        refined = {}

        for label, (x, y, w, h) in self.polygons.items():
            x0 = max(0, int(x - max_offset))
            x1 = min(w_img - int(w), int(x + max_offset))
            y0 = max(0, int(y - max_offset))
            y1 = min(h_img - int(h), int(y + max_offset))
            best_score = -1.0
            best_xy = (int(x), int(y))

            # slide small window
            for sx in range(x0, x1 + 1, step):
                for sy in range(y0, y1 + 1, step):
                    crop = hsv[sy:sy + int(h), sx:sx + int(w)]
                    if crop.size == 0:
                        continue
                    # mean saturation * mean value (heuristic for vivid sticker center)
                    s_mean = float(np.mean(crop[:, :, 1]))
                    v_mean = float(np.mean(crop[:, :, 2]))
                    score = s_mean * v_mean
                    if score > best_score:
                        best_score = score
                        best_xy = (sx, sy)

            refined[label] = (best_xy[0], best_xy[1], int(w), int(h))

        return refined

    def predict_colors_with_cnn(self, img: np.ndarray, polygons: Dict[str, Tuple[int,int,int,int]]) -> Dict[str, int]:
        """Same as predict_colors_with_cnn but using the provided polygons dict"""
        if self.model is None:
            raise Exception("CNN model not loaded")

        predicts = {}
        for label, (x, y, width, height) in polygons.items():
            x, y, w, h = int(x), int(y), int(width), int(height)

            # Validate coordinates against image bounds
            if (y + h > img.shape[0] or x + w > img.shape[1] or y < 0 or x < 0 or h <= 0 or w <= 0):
                predicts[label] = 0
                continue

            try:
                rect = img[y:y + h, x:x + w]
                if rect.size == 0:
                    predicts[label] = 0
                    continue

                rect = cv2.resize(rect, (self.img_width, self.img_height))
                resized = np.reshape(rect, [-1, self.img_width, self.img_height, 3])
                predict = self.model.predict(resized)
                predicts[label] = int(np.argmax(predict))
            except Exception:
                predicts[label] = 0

        return predicts

    def predict_colors_with_decision_tree(self, img: np.ndarray, polygons: Dict[str, Tuple[int,int,int,int]]) -> Dict[str, int]:
        """Decision-tree prediction using given polygons"""
        if self.decision_tree is None:
            raise Exception("Decision tree model not loaded")

        inputs = []
        labels = []
        for label, (x, y, width, height) in polygons.items():
            labels.append(label)
            rect = img[round(y):round(y) + round(height), round(x):round(x) + round(width)]
            h_val, s_val, v_val = self.get_dominant_hsv(rect)
            inputs.append([h_val, s_val, v_val])

        predicts = self.decision_tree.predict(inputs)
        return {label: int(predicts[idx]) for idx, label in enumerate(labels)}

    def get_dominant_hsv(self, img: np.ndarray) -> Tuple[float, float, float]:
        """Get dominant HSV color from image region"""
        if img.size == 0:
            return (0, 0, 0)

        pixels = np.float32(img.reshape(-1, 3))
        n_colors = 1
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        try:
            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
            _, counts = np.unique(labels, return_counts=True)
            dominant_bgr = palette[np.argmax(counts)]
            b, g, r = dominant_bgr
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            return (h * 360, s * 100, v * 100)
        except Exception:
            return (0, 0, 0)

    def convert_to_status_input(self, predicts: Dict[str, int], 
                              label_mapping: Dict[str, str]) -> Dict[str, str]:
        """Convert regions in one picture to regions in cube"""
        return {
            label_mapping[l]: self.colors[predicts[l]] 
            for l in predicts if l in label_mapping
        }

    def generate_status_str(self, label_to_color: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
        """Generate status string for the whole cube"""
        # Extract center colors for each face
        side_to_color = {
            side: label_to_color.get(f'{side}5', 'W') 
            for side in self.side_order
        }

        # Create status string
        status_string = ''.join(
            label_to_color.get(f'{side}{i}', 'W')
            for side in self.side_order
            for i in range(1, 10)
        )

        return status_string, side_to_color

    def validate_color_count(self, status_list: str) -> Tuple[Dict[str, int], bool]:
        """Validate that each color appears exactly 9 times"""
        color_count = {}
        for c in status_list:
            color_count[c] = color_count.get(c, 0) + 1
        
        has_error = any(count != 9 for count in color_count.values())
        return color_count, has_error

    def detect_status_from_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[str, Dict[str, str], bool]:
        """Detect cube status from two frames using refined polygons"""
        # Convert frames to RGB (your rest of pipeline expects RGB)
        first_pic = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        second_pic = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # Refine polygons locally (small search window)
        refined_first = self.get_refined_polygons(first_pic, max_offset=40, step=6)
        refined_second = self.get_refined_polygons(second_pic, max_offset=40, step=6)

        # Predict colors (prefer CNN, fallback to decision tree)
        try:
            if self.model is not None:
                first_pic_predicts = self.predict_colors_with_cnn(first_pic, refined_first)
                second_pic_predicts = self.predict_colors_with_cnn(second_pic, refined_second)
            else:
                raise Exception("CNN model not available")
        except Exception:
            if self.decision_tree is not None:
                first_pic_predicts = self.predict_colors_with_decision_tree(first_pic, refined_first)
                second_pic_predicts = self.predict_colors_with_decision_tree(second_pic, refined_second)
            else:
                raise Exception("No prediction models available")

        # Merge predictions using the mappings (frame1 -> pic1 mapping, frame2 -> pic2 mapping)
        label_to_color = self.convert_to_status_input(first_pic_predicts, self.pic1_label_mapping)
        label_to_color.update(self.convert_to_status_input(second_pic_predicts, self.pic2_label_mapping))

        status_string, side_to_color = self.generate_status_str(label_to_color)
        color_count, has_error = self.validate_color_count(status_string)

        return status_string, side_to_color, has_error

    def change_status(self, input_status: str, moves: List[str]) -> str:
        """Apply moves to change cube status - FIXED"""
        if not moves:
            return input_status
            
        cube = list(input_status)
        
        for move in moves:
            cube = self._apply_move_to_list(cube, move)
            
        return ''.join(cube)
    
    def _apply_move_to_list(self, cube: List[str], move: str) -> List[str]:
        """Apply a single move to the cube list"""
        new_cube = cube.copy()
        
        if move == "U":
            self._rotate_face(new_cube, 0, True)  # U face clockwise
            # Rotate top layer
            temp = [new_cube[18], new_cube[19], new_cube[20]]  # F top
            new_cube[18], new_cube[19], new_cube[20] = new_cube[9], new_cube[10], new_cube[11]  # F <- R
            new_cube[9], new_cube[10], new_cube[11] = new_cube[45], new_cube[46], new_cube[47]  # R <- B  
            new_cube[45], new_cube[46], new_cube[47] = new_cube[36], new_cube[37], new_cube[38]  # B <- L
            new_cube[36], new_cube[37], new_cube[38] = temp[0], temp[1], temp[2]  # L <- F
            
        elif move == "U'":
            self._rotate_face(new_cube, 0, False)  # U face counterclockwise
            # Rotate top layer counterclockwise
            temp = [new_cube[18], new_cube[19], new_cube[20]]  # F top
            new_cube[18], new_cube[19], new_cube[20] = new_cube[36], new_cube[37], new_cube[38]  # F <- L
            new_cube[36], new_cube[37], new_cube[38] = new_cube[45], new_cube[46], new_cube[47]  # L <- B
            new_cube[45], new_cube[46], new_cube[47] = new_cube[9], new_cube[10], new_cube[11]  # B <- R
            new_cube[9], new_cube[10], new_cube[11] = temp[0], temp[1], temp[2]  # R <- F
            
        elif move == "U2":
            new_cube = self._apply_move_to_list(new_cube, "U")
            new_cube = self._apply_move_to_list(new_cube, "U")
            
        elif move == "R":
            self._rotate_face(new_cube, 9, True)  # R face clockwise
            # Rotate right layer
            temp = [new_cube[20], new_cube[23], new_cube[26]]  # F right
            new_cube[20], new_cube[23], new_cube[26] = new_cube[27], new_cube[30], new_cube[33]  # F <- D
            new_cube[27], new_cube[30], new_cube[33] = new_cube[47], new_cube[44], new_cube[41]  # D <- B (reversed)
            new_cube[47], new_cube[44], new_cube[41] = new_cube[2], new_cube[5], new_cube[8]  # B <- U (reversed)
            new_cube[2], new_cube[5], new_cube[8] = temp[0], temp[1], temp[2]  # U <- F
            
        elif move == "R'":
            self._rotate_face(new_cube, 9, False)  # R face counterclockwise
            # Rotate right layer counterclockwise
            temp = [new_cube[20], new_cube[23], new_cube[26]]  # F right
            new_cube[20], new_cube[23], new_cube[26] = new_cube[2], new_cube[5], new_cube[8]  # F <- U
            new_cube[2], new_cube[5], new_cube[8] = new_cube[47], new_cube[44], new_cube[41]  # U <- B (reversed)
            new_cube[47], new_cube[44], new_cube[41] = new_cube[27], new_cube[30], new_cube[33]  # B <- D (reversed)
            new_cube[27], new_cube[30], new_cube[33] = temp[0], temp[1], temp[2]  # D <- F
            
        elif move == "R2":
            new_cube = self._apply_move_to_list(new_cube, "R")
            new_cube = self._apply_move_to_list(new_cube, "R")
            
        elif move == "F":
            self._rotate_face(new_cube, 18, True)  # F face clockwise
            # Rotate front layer
            temp = [new_cube[6], new_cube[7], new_cube[8]]  # U bottom
            new_cube[6], new_cube[7], new_cube[8] = new_cube[36], new_cube[39], new_cube[42]  # U <- L (reversed)
            new_cube[36], new_cube[39], new_cube[42] = new_cube[29], new_cube[28], new_cube[27]  # L <- D (reversed)
            new_cube[29], new_cube[28], new_cube[27] = new_cube[11], new_cube[14], new_cube[17]  # D <- R (reversed)
            new_cube[11], new_cube[14], new_cube[17] = temp[0], temp[1], temp[2]  # R <- U
            
        elif move == "F'":
            self._rotate_face(new_cube, 18, False)  # F face counterclockwise
            # Rotate front layer counterclockwise
            temp = [new_cube[6], new_cube[7], new_cube[8]]  # U bottom
            new_cube[6], new_cube[7], new_cube[8] = new_cube[11], new_cube[14], new_cube[17]  # U <- R
            new_cube[11], new_cube[14], new_cube[17] = new_cube[29], new_cube[28], new_cube[27]  # R <- D (reversed)
            new_cube[29], new_cube[28], new_cube[27] = new_cube[36], new_cube[39], new_cube[42]  # D <- L (reversed)
            new_cube[36], new_cube[39], new_cube[42] = temp[0], temp[1], temp[2]  # L <- U (reversed)
            
        elif move == "F2":
            new_cube = self._apply_move_to_list(new_cube, "F")
            new_cube = self._apply_move_to_list(new_cube, "F")
            
        elif move == "D":
            self._rotate_face(new_cube, 27, True)  # D face clockwise
            # Rotate bottom layer
            temp = [new_cube[24], new_cube[25], new_cube[26]]  # F bottom
            new_cube[24], new_cube[25], new_cube[26] = new_cube[38], new_cube[41], new_cube[44]  # F <- L
            new_cube[38], new_cube[41], new_cube[44] = new_cube[45], new_cube[48], new_cube[51]  # L <- B
            new_cube[45], new_cube[48], new_cube[51] = new_cube[15], new_cube[12], new_cube[9]  # B <- R (reversed)
            new_cube[15], new_cube[12], new_cube[9] = temp[0], temp[1], temp[2]  # R <- F
            
        elif move == "D'":
            self._rotate_face(new_cube, 27, False)  # D face counterclockwise
            # Rotate bottom layer counterclockwise
            temp = [new_cube[24], new_cube[25], new_cube[26]]  # F bottom
            new_cube[24], new_cube[25], new_cube[26] = new_cube[15], new_cube[12], new_cube[9]  # F <- R (reversed)
            new_cube[15], new_cube[12], new_cube[9] = new_cube[45], new_cube[48], new_cube[51]  # R <- B
            new_cube[45], new_cube[48], new_cube[51] = new_cube[38], new_cube[41], new_cube[44]  # B <- L
            new_cube[38], new_cube[41], new_cube[44] = temp[0], temp[1], temp[2]  # L <- F
            
        elif move == "D2":
            new_cube = self._apply_move_to_list(new_cube, "D")
            new_cube = self._apply_move_to_list(new_cube, "D")
            
        elif move == "L":
            self._rotate_face(new_cube, 36, True)  # L face clockwise
            # Rotate left layer
            temp = [new_cube[18], new_cube[21], new_cube[24]]  # F left
            new_cube[18], new_cube[21], new_cube[24] = new_cube[0], new_cube[3], new_cube[6]  # F <- U
            new_cube[0], new_cube[3], new_cube[6] = new_cube[53], new_cube[50], new_cube[47]  # U <- B (reversed)
            new_cube[53], new_cube[50], new_cube[47] = new_cube[33], new_cube[30], new_cube[27]  # B <- D (reversed)
            new_cube[33], new_cube[30], new_cube[27] = temp[0], temp[1], temp[2]  # D <- F
            
        elif move == "L'":
            self._rotate_face(new_cube, 36, False)  # L face counterclockwise
            # Rotate left layer counterclockwise
            temp = [new_cube[18], new_cube[21], new_cube[24]]  # F left
            new_cube[18], new_cube[21], new_cube[24] = new_cube[33], new_cube[30], new_cube[27]  # F <- D
            new_cube[33], new_cube[30], new_cube[27] = new_cube[53], new_cube[50], new_cube[47]  # D <- B (reversed)
            new_cube[53], new_cube[50], new_cube[47] = new_cube[0], new_cube[3], new_cube[6]  # B <- U (reversed)
            new_cube[0], new_cube[3], new_cube[6] = temp[0], temp[1], temp[2]  # U <- F
            
        elif move == "L2":
            new_cube = self._apply_move_to_list(new_cube, "L")
            new_cube = self._apply_move_to_list(new_cube, "L")
            
        elif move == "B":
            self._rotate_face(new_cube, 45, True)  # B face clockwise
            # Rotate back layer
            temp = [new_cube[0], new_cube[1], new_cube[2]]  # U top
            new_cube[0], new_cube[1], new_cube[2] = new_cube[17], new_cube[14], new_cube[11]  # U <- R (reversed)
            new_cube[17], new_cube[14], new_cube[11] = new_cube[35], new_cube[34], new_cube[33]  # R <- D (reversed)
            new_cube[35], new_cube[34], new_cube[33] = new_cube[42], new_cube[39], new_cube[36]  # D <- L (reversed)
            new_cube[42], new_cube[39], new_cube[36] = temp[0], temp[1], temp[2]  # L <- U
            
        elif move == "B'":
            self._rotate_face(new_cube, 45, False)  # B face counterclockwise
            # Rotate back layer counterclockwise
            temp = [new_cube[0], new_cube[1], new_cube[2]]  # U top
            new_cube[0], new_cube[1], new_cube[2] = new_cube[42], new_cube[39], new_cube[36]  # U <- L
            new_cube[42], new_cube[39], new_cube[36] = new_cube[35], new_cube[34], new_cube[33]  # L <- D (reversed)
            new_cube[35], new_cube[34], new_cube[33] = new_cube[17], new_cube[14], new_cube[11]  # D <- R (reversed)
            new_cube[17], new_cube[14], new_cube[11] = temp[0], temp[1], temp[2]  # R <- U (reversed)
            
        elif move == "B2":
            new_cube = self._apply_move_to_list(new_cube, "B")
            new_cube = self._apply_move_to_list(new_cube, "B")
            
        return new_cube
    
    def _rotate_face(self, cube: List[str], face_start: int, clockwise: bool):
        """Rotate a face clockwise or counterclockwise"""
        indices = [
            face_start, face_start+1, face_start+2,
            face_start+3, face_start+4, face_start+5, 
            face_start+6, face_start+7, face_start+8
        ]
        
        if clockwise:
            # Clockwise rotation
            temp = cube[indices[0]]
            cube[indices[0]] = cube[indices[6]]
            cube[indices[6]] = cube[indices[8]]
            cube[indices[8]] = cube[indices[2]]
            cube[indices[2]] = temp
            
            temp = cube[indices[1]]
            cube[indices[1]] = cube[indices[3]]
            cube[indices[3]] = cube[indices[7]]
            cube[indices[7]] = cube[indices[5]]
            cube[indices[5]] = temp
        else:
            # Counterclockwise rotation
            temp = cube[indices[0]]
            cube[indices[0]] = cube[indices[2]]
            cube[indices[2]] = cube[indices[8]]
            cube[indices[8]] = cube[indices[6]]
            cube[indices[6]] = temp
            
            temp = cube[indices[1]]
            cube[indices[1]] = cube[indices[5]]
            cube[indices[5]] = cube[indices[7]]
            cube[indices[7]] = cube[indices[3]]
            cube[indices[3]] = temp

    def update_sticker(self, side: str, sticker_pos: str, color: str):
        """Update a single sticker color"""
        try:
            face_index = ['U', 'R', 'F', 'D', 'L', 'B'].index(side)
            sticker_num = int(sticker_pos[1:]) - 1
            status_index = face_index * 9 + sticker_num
            
            status_list = list(self.current_status)
            status_list[status_index] = color
            self.current_status = ''.join(status_list)
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid sticker position: {side}{sticker_pos}") from e
    
    def validate_state(self) -> Tuple[bool, List[str]]:
        """Validate cube state has correct color distribution"""
        color_counts = {}
        for color in self.current_status:
            color_counts[color] = color_counts.get(color, 0) + 1
        
        issues = []
        for color, count in color_counts.items():
            if count != 9:
                issues.append(f"Color {color} has {count} stickers (should be 9)")
        
        return len(issues) == 0, issues
    
    def convert_to_face_status(self, color_status: str) -> str:
        """Convert color-based status to face-based status for kociemba solver"""
        # Use the same mapping as before
        face_mapping = {
            'O': 'U',  # Orange to Up
            'B': 'R',  # Blue to Right  
            'W': 'F',  # White to Front
            'R': 'D',  # Red to Down
            'G': 'L',  # Green to Left
            'Y': 'B'   # Yellow to Back
        }
        
        return ''.join(face_mapping.get(color, 'U') for color in color_status)