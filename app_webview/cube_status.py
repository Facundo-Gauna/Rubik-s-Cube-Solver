import threading
from typing import Counter, Optional, Tuple, Dict, List

import kociemba
import kociemba_module as kociemba_mod


from config import CENTER_INDICES, FACE_ORDER, CubeState
from detector import  brute_force_face_orientations

EXPECTED_COLORS = ['R','O','Y','G','B','W']

class CubeStatus:
    def __init__(self):
        self.cube_state = CubeState()
        self.have_sol : bool = False

    def build_facelets_and_solve(self, mapping : Dict[str,str]):
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

    def change_status(self, seq: str):
        if not seq:
            return 

        self.have_sol = False
        facelet_str = self.cube_state.face_status
        moves = seq.split(" ")
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

