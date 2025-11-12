import time
from typing import Counter, Optional, Tuple, Dict, List

import kociemba
import kociemba_module as kociemba_mod

from config import CENTER_INDICES, FACE_ORDER, CubeState

EXPECTED_COLORS = ['R','O','Y','G','B','W']

ROT90 = [6, 3, 0, 7, 4, 1, 8, 5, 2]


def rotate_block_list(vals9: List, times: int) -> List:
    out = list(vals9)
    for _ in range(times % 4):
        out = [out[i] for i in ROT90]
    return out


def build_color_net_text(color_str: str) -> str:
    faces = ['U', 'R', 'F', 'D', 'L', 'B']
    out = []
    for fi, face in enumerate(faces):
        out.append(f"\n{face}:")
        block = color_str[fi * 9:(fi + 1) * 9]
        for r in range(3):
            out.append(' '.join(block[r * 3:(r + 1) * 3]))
    return '\n'.join(out)


def brute_force_face_orientations(face_color_string: str):
    if kociemba is None:
        print("kociemba not installed; cannot brute force.")
        return None, None
    blocks = [list(face_color_string[i * 9:(i + 1) * 9]) for i in range(6)]
    import itertools
    t0 = time.time()
    print("Brute-force: trying all face rotations (4^6)...")
    for comb in itertools.product(range(4), repeat=6):
        candidate_blocks = [''.join(rotate_block_list(b, comb[i])) for i, b in enumerate(blocks)]
        candidate_color_str = ''.join(candidate_blocks)
        color_to_face = {}
        ok = True
        for face_letter, idx in CENTER_INDICES.items():
            c = candidate_color_str[idx]
            if c in color_to_face:
                ok = False
                break
            color_to_face[c] = face_letter
        if not ok:
            continue
        candidate_facelets = ''.join(color_to_face[c] for c in candidate_color_str)
        try:
            sol = kociemba.solve(candidate_facelets)
            print(f"Found valid orientation (brute-force) after {time.time() - t0:.2f}s.")
            return candidate_facelets, sol
        except Exception:
            continue
    print("Brute-force exhausted; no valid orientation found.")
    return None, None


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

    def find_solution(self, mapping: Dict[str,str], timeout_seconds: float = 3.0, max_iters: int = 2000):
        """
        Buscar la "mejor" solución intentando variantes por rotación de bloque de cada cara.
        Criterio de "mejor": minimizar la cantidad de movimientos 'U' en la solución.
        Parámetros:
            mapping: label->color_letter (ej. 'U1':'B', ...)
            timeout_seconds: tiempo máximo a gastar en búsqueda (segundos)
            max_iters: máximo de solve attempts (si es menor que 4^6)
        Retorna: (color_str, best_facelet_str, best_solution) o lanza ValueError si falla.
        """
        import itertools
        t0 = time.time()

        # 1) construyo color_str igual que build_facelets_and_solve (U,R,F,D,L,B order)
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
        print("find_solution: Built color string:", color_str)

        # conteos rápidos
        ctr = Counter(color_str)
        print("find_solution: Color counts:", ctr)
        if any(ctr.get(c, 0) != 9 for c in EXPECTED_COLORS):
            print("find_solution: WARNING: not all colors have 9 occurrences")

        # mapear colores centrales -> caras
        color_to_face = {}
        for face_letter, idx in CENTER_INDICES.items():
            center_color = color_str[idx]
            if center_color in color_to_face:
                print("find_solution: ERROR: center duplicated:", center_color)
            color_to_face[center_color] = face_letter
        if len(color_to_face) != 6:
            print("find_solution: ERROR: Centers mapping invalid:", color_to_face)
            # intenta la brute-force que tenés ya (rota bloques y prueba)
            candidate_facelets, sol = brute_force_face_orientations(color_str)
            if candidate_facelets:
                return color_str, candidate_facelets, sol
            raise ValueError("Invalid centers mapping - cannot proceed")

        # blocks = list de 6 strings de 9 caracteres (por cara)
        blocks = [list(color_str[i * 9:(i + 1) * 9]) for i in range(6)]

        best = {
            "u_count": None,
            "moves_len": None,
            "facelets": None,
            "solution": None,
            "orientation": None,
        }

        # iterador sobre combinaciones; control por timeout y max_iters
        total_possible = 4 ** 6
        it = 0
        print(f"find_solution: Starting search (max {min(max_iters, total_possible)} attempts or {timeout_seconds}s)...")

        # Recomendación: probar primero la identidad (sin rotación) para tener baseline
        combos = itertools.product(range(4), repeat=6)

        for comb in combos:
            it += 1
            # control de iteraciones/tiempo
            if it > max_iters or (time.time() - t0) > timeout_seconds:
                break

            # aplicar rotaciones por bloque según comb
            candidate_blocks = [''.join(rotate_block_list(blocks[i], comb[i])) for i in range(6)]
            candidate_color_str = ''.join(candidate_blocks)

            # convertir candidate_color_str -> facelet string usando color_to_face (centros permanecen)
            try:
                candidate_facelets = ''.join(color_to_face.get(c, '?') for c in candidate_color_str)
            except Exception as e:
                # improbable, pero saltar
                continue

            # pedir solución al solver (kociemba)
            try:
                sol = kociemba.solve(candidate_facelets)
            except Exception:
                # solver rechazó; seguir
                continue

            # analizar la solución: contar U-moves y longitud total
            tokens = [tok for tok in sol.split() if tok]
            u_count = sum(1 for tok in tokens if tok[0].upper() == 'U')
            moves_len = len(tokens)

            # guardo si es mejor: prioridad menos U, luego menor longitud
            improved = False
            if best["solution"] is None:
                improved = True
            else:
                if u_count < best["u_count"]:
                    improved = True
                elif u_count == best["u_count"] and moves_len < best["moves_len"]:
                    improved = True

            if improved:
                best.update({
                    "u_count": u_count,
                    "moves_len": moves_len,
                    "facelets": candidate_facelets,
                    "solution": sol,
                    "orientation": comb,
                })
                print(f"find_solution: New best (it={it}, time={time.time()-t0:.2f}s): U_count={u_count}, len={moves_len}, comb={comb}")

                # si encontramos 0 U moves, podemos parar inmediatamente (óptimo según criterio)
                if u_count == 0:
                    break

        # fin loop
        elapsed = time.time() - t0
        print(f"find_solution: Finished search iters={it}, elapsed={elapsed:.2f}s")

        if best["solution"] is None:
            # fallback: intentar resolver el facelet original (sin rotaciones)
            try:
                facelet_str = ''.join(color_to_face.get(c, '?') for c in color_str)
                sol = kociemba.solve(facelet_str)
                print("find_solution: Fallback original solution found")
                return color_str, facelet_str, sol
            except Exception as e:
                raise ValueError("Solver failed for all attempted orientations and original.") from e

        # devolvemos la mejor encontrada
        return color_str, best["facelets"], best["solution"]

    def solve_from_detect(self,mapping : Dict[str,str]) -> bool:
        try:
            color_str, facelets, sol = self.find_solution(mapping)
            self.cube_state.color_status = color_str
            if facelets == None or sol == None:
                return False
            self.cube_state.face_status = facelets
            self.cube_state.solution = sol
            return True
        except Exception as e:
            self.cube_state.solution = ""
            print("Imposible to find a solution :"+e)
            print(mapping)
            return False

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

