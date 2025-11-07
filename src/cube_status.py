import threading
from typing import Counter, Optional, Tuple, Dict, List

import kociemba
try:
    import kociemba_module as kociemba_mod
except Exception:
    kociemba_mod = None

from config import CENTER_INDICES, FACE_ORDER, FACE_ORDER_IMG_1, FACE_ORDER_IMG_2, IMG1_PATH, IMG2_PATH, POLYGON_POSITIONS_PATH_1, POLYGON_POSITIONS_PATH_2, CubeState, DetectionResult
from detector import EXPECTED_COLORS, ColorDetector, ParallelCubeDetector, SynchronizedColorCorrectionUI, SynchronizedPolygonDetector, rotate_block_list, build_color_net_text, brute_force_face_orientations


class CubeStatus:
    def __init__(self):
        self.cube_state = CubeState()
        self.cube_detector = ParallelCubeDetector()
        self.sync_detector = SynchronizedPolygonDetector()
        self.detector = ColorDetector()

        self.pos1 = None
        self.pos2 = None
        self.det1 = None
        self.det2 = None
        self.labs1 = None
        self.labs2 = None
        self.corrected1 = None
        self.corrected2 = None
        
        self.have_sol : bool = False


    def detect(self):
        print("=== STEP 1: Defining positions for both images ===")
        self.pos1, self.pos2 = self.sync_detector.define_positions_both()
        if not  self.pos1 or not self.pos2:
            raise Exception("Failed to get positions for both images")

        print("\n=== STEP 2: Detecting colors in parallel ===")
        def job1():
            self.det1, self.labs1 = self.detector.detect_single_image(IMG1_PATH, self.pos1)
            print("✓ Image 1 colors detected")

        def job2():
            self.det2, self.labs2 = self.detector.detect_single_image(IMG2_PATH, self.pos2)
            print("✓ Image 2 colors detected")

        t1 = threading.Thread(target=job1)
        t2 = threading.Thread(target=job2)
        t1.start(); t2.start()
        t1.join(); t2.join()
        
        print("✓ Both image colors detected successfully!")
        
        print("\n=== STEP 3: Color Correction for both images ===")
        if not (self.pos1 and self.det1 and self.labs1 and self.pos2 and self.det2 and self.labs2):
            raise Exception("Must run detection before color correction")

        sync_correction = SynchronizedColorCorrectionUI(
            self.pos1, self.pos2,
            self.det1, self.labs1,
            self.det2, self.labs2
        )

        self.corrected1, corrected_labs1, self.corrected2, corrected_labs2 = sync_correction.run_both()
        print("✓ Both image colors corrected successfully!")


    def build_facelets_and_solve(self):
        """
        Build the canonical 54-color string and facelet string using corrected mappings.
        First image contains U,F,L (provided in this file), second contains D,R,B.
        The canonical face order needed by kociemba is U,R,F,D,L,B.
        """
        if not (self.corrected1 and self.corrected2):
            raise Exception("Run color correction first")

        mapping = {}
        mapping.update(self.corrected1 or {})
        mapping.update(self.corrected2 or {})

        # Build color_str in U,R,F,D,L,B order
        colors = []
        missing = []
        for face in FACE_ORDER:
            for i in range(1, 10):
                k = f"{face}{i}"
                if k not in mapping:
                    missing.append(k)
                    colors.append('X')
                else:
                    colors.append(mapping[k])
        if missing:
            raise ValueError(f"Missing stickers: {missing}")
        color_str = ''.join(colors)
        print("Built color string:", color_str)
        print(build_color_net_text(color_str))

        # Quick counts
        ctr = Counter(color_str)
        print("Color counts:", ctr)
        if any(ctr.get(c, 0) != 9 for c in EXPECTED_COLORS):
            print("WARNING: not all colors have 9 occurrences")

        # Map center colors -> faces
        color_to_face = {}
        for face_letter, idx in CENTER_INDICES.items():
            center_color = color_str[idx]
            if center_color in color_to_face:
                print("ERROR: center duplicated:", center_color)
            color_to_face[center_color] = face_letter
        if len(color_to_face) != 6:
            print("ERROR: Centers mapping invalid:", color_to_face)
            candidate_facelets, sol = brute_force_face_orientations(color_str)
            if candidate_facelets:
                print("Brute-force rotation produced valid facelet string")
                return color_str, candidate_facelets, sol
            raise ValueError("Invalid centers mapping - cannot proceed")

        facelet_str = ''.join(color_to_face.get(c, '?') for c in color_str)
        print("Converted facelet string:", facelet_str)

        if kociemba is not None:
            try:
                sol = kociemba.solve(facelet_str)
                print("kociemba solution:", sol)
                return color_str, facelet_str, sol
            except Exception as e:
                print("kociemba rejected facelet string:", e)
                # try brute force of face rotations
                candidate_facelets, sol = brute_force_face_orientations(color_str)
                if candidate_facelets:
                    return color_str, candidate_facelets, sol
                raise ValueError("Solver rejected the facelet string and brute-force failed.")
        else:
            print("kociemba not available; returning facelets for inspection")
            return color_str, facelet_str, None


    def detect_status(self) -> DetectionResult:
        try:
            self.detect()
            color_str, facelet,sol = self.build_facelets_and_solve()
            self.have_sol = sol != None 
            return DetectionResult(
                color_str=color_str,
                solution_str=sol,
                face_str=facelet,
                has_errors= sol != None,
            )
        except Exception as e:
            print(f"Detection failed: {e}")
            return DetectionResult(
                color_str='',
                solution_str='',
                face_str='',
                has_errors=True,
            )


    def solve(self) -> Tuple[bool,str]:
        if self.have_sol:
            return True,"Solved"
        self.have_sol = True
        try:
            sol = kociemba.solve(self.cube_state.face_status)
            self.cube_state.solution = sol
            return True,"Solved"
        except Exception as e:
            candidate_facelets, sol = brute_force_face_orientations(self.cube_state.color_status)
            self.cube_state.face_status = candidate_facelets
            self.cube_state.solution = sol
            if candidate_facelets:
                return True,"Solved"
            self.have_sol = False
            return False,"Solver rejected the facelet string and brute-force failed."

    def update_sticker(self, side: str, sticker_pos: str, color: str):
        try:
            face_index = FACE_ORDER.index(side)
            sticker_num = int(sticker_pos[1:]) - 1
            status_index = face_index * 9 + sticker_num
            
            status_list = list(self.cube_state.color_status)
            status_list[status_index] = color
            self.cube_state.colors_status_set(''.join(status_list))
            
            self.have_sol = False # if a cube state suffer a change and it had a solution to make, it is delted
            
            print(f"Updated {side}{sticker_pos} to {color}")
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid sticker position: {side}{sticker_pos}") from e


    def validate_state(self) -> Tuple[bool, str]:
        try:
            fc = kociemba_mod.FaceCube(self.cube_state.face_status)
            status = fc.toCubieCube().verify()
            match status:
                case -2: return False, "Not all 12 edges exist exactly once"
                case -3: return False, "Flip error: One edge has to be flipped"
                case -4: return False, "Not all corners exist exactly once"
                case -5: return False, "Twist error: One corner has to be twisted"
                case -6: return False, "Parity error: Two corners ore two edges have to be exchanged"
                case _: return True,"Cube OK"
        except Exception as e:
            return False,e

        return False,"Unexpected...."

    # --------- Manual changer of status ----------

    def change_status(self, moves: List[str]):
        if not moves:
            return 

        self.have_sol = False
        facelet_str = self.cube_state.face_status
        print("moves : "+ ("".join(moves)))
        print("facelet init : "+ facelet_str)
        try:
            fc = kociemba_mod.FaceCube(facelet_str)
            cc = fc.toCubieCube()
            move_index = {'U':0, 'R':1, 'F':2, 'D':3, 'L':4, 'B':5}
            for mv in moves:
                if not mv: 
                    continue
                base = mv[0].upper()
                if base not in move_index:
                    continue
                idx = move_index[base]
                if mv.endswith("2"):
                    times = 2
                elif mv.endswith("'"):
                    times = 3  # inverse = apply 3 times
                else:
                    times = 1
                for _ in range(times):
                    cc.multiply(kociemba_mod.moveCube[idx])
            new_facelet = cc.toFaceCube().to_String()
            self.cube_state.face_status_set(new_facelet)
        except Exception as e:
            print("kociemba_module move application failed, falling back:", e)

