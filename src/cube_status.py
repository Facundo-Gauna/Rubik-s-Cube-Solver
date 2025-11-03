import os, json, time
from typing import Counter, Optional, Tuple, Dict, List
from pathlib import Path

from config import config
from app_types import CubeState, DetectionResult
from manual_detector import ManualDetectionWorkflow


class CubeStatus:
    def __init__(self):
        self.manual_detector = ManualDetectionWorkflow()
        self.cube_state = CubeState()
        self.side_order = config.SIDE_ORDER

    def detect_status_manual(self) -> DetectionResult:
        try:
            result = self.manual_detector.run()
            ok,sol,color_str = result

            return DetectionResult(
                color_str==color_str,
                solution_str=sol,
                has_errors=ok,
            )    
        except Exception as e:
            print(f"Manual detection failed: {e}")
            raise

    def update_sticker(self, side: str, sticker_pos: str, color: str):
        try:
            face_index = self.side_order.index(side)
            sticker_num = int(sticker_pos[1:]) - 1
            status_index = face_index * 9 + sticker_num
            
            status_list = list(self.cube_state.color_status)
            status_list[status_index] = color
            self.cube_state.color_status = ''.join(status_list)
            
            print(f"Updated {side}{sticker_pos} to {color}")
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid sticker position: {side}{sticker_pos}") from e

    def validate_state(self) -> Tuple[bool, List[str]]:
        """Validate cube state has correct color distribution"""
        from collections import Counter
        color_counts = Counter(self.cube_state.color_status)
        
        issues = []
        for color, count in color_counts.items():
            if count != 9:
                issues.append(f"Color {color} has {count} stickers (should be 9)")
        
        return len(issues) == 0, issues

    def convert_to_face_status(self, color_status: str = None) -> str:
        """Convert color-based status to face-based status for kociemba solver"""
        if color_status is None:
            color_status = self.cube_state.color_status
            
        face_color_order = ['R','G','W','O','B','Y']
        face_mapping = {
            'R': 'U',  # Red to Up
            'G': 'R',  # Green to Right  
            'W': 'F',  # White to Front
            'O': 'D',  # Orange to Down
            'B': 'L',  # Blue to Left
            'Y': 'B'   # Yellow to Back
        }
        
        return ''.join(face_mapping.get(color, 'U') for color in color_status)

    @property
    def current_status(self) -> str:
        return self.cube_state.color_status

    @current_status.setter
    def current_status(self, value: str):
        self.cube_state.color_status = value

    @property
    def side_to_color(self) -> Dict[str, str]:
        return self.cube_state.side_to_color

    @side_to_color.setter
    def side_to_color(self, value: Dict[str, str]):
        self.cube_state.side_to_color = value
    
    # To UI movements.
    def change_status(self, input_status: str, moves: List[str]) -> str:
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
   
