"""
cube_solver.py — Robust Kociemba Solver with Orientation Correction
===========================================================================

This module implements a comprehensive and reliable interface for solving a
Rubik's cube state using the Kociemba algorithm. It is specifically designed
for robotic/hardware environments where reliable initial cube scanning can be
challenging, incorporating automatic orientation detection as a powerful fallback.

### Core Classes

* **CubeState**: A dataclass that manages the 54-character state of the cube
  (both color and facelet status), along with its internal Kociemba
  representations (`FaceCube` and `CubieCube`). It provides core methods for
  applying moves (`move`, `apply_sequence`) and state synchronization.

* **CubeSolver**: The primary class responsible for the solving workflow. It
  handles input conversion, validation, caching, and the core solving logic implementation.

### Key Features & Robustness

1.  **Input Handling and Static Rotations**: Automatically converts sticker
    mappings (e.g., {'U1': 'B', ...}) into the 54-character Kociemba facelet
    string, applying any configured static face rotations from the `config` module.

2.  ** 4^6 Orientation Search (Fallback)**: If the initial facelet string fails
    Kociemba's internal validation (indicating twist, flip, or parity errors),
    the solver automatically attempts to correct the issue by iterating through
    all  4^6 (4096) combinations of 90-degree face-local rotations until a
    solvable state is found. This dramatically increases robustness against
    camera misalignment or manual rotation errors.

3.  **Efficiency**: Utilizes precomputed 54-index rotation maps (`_ROT54`) for
    high-speed application of face rotations during the search fallback.

--------------------------------------------------------------------------

Copyright (c) 2025 Facundo Gauna & Ulises Carnevale. Licensed under MIT License.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import logging
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import kociemba
import kociemba_module as kociemba_mod

from config import (
    CENTER_INDICES,
    FACE_ORDER,
    COLOR_INIT_STATE,
    FACE_TO_COLOR,
    FACES_INIT_STATE,
    MOVE_INDEX,
    FACE_ROTATIONS,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# rotation maps & precomputed 54-index mappings
# rotation index maps for a 3x3 face (index 0..8)
# mapping: _ROT_MAP[r][i] = source index (0..8) of sticker that should go to position i
_ROT_MAP = {
    0: [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 0°
    1: [6, 3, 0, 7, 4, 1, 8, 5, 2],  # 90° CW
    2: [8, 7, 6, 5, 4, 3, 2, 1, 0],  # 180°
    3: [2, 5, 8, 1, 4, 7, 0, 3, 6],  # 270° CW
}

# Face offsets in the 54-length facelet/color strings (FACE_ORDER assumed U,R,F,D,L,B)
FACE_OFFSETS = [i * 9 for i in range(6)]  # [0, 9, 18, 27, 36, 45]

# Precompute mapping from (face_idx, rot) -> mapping54 (list length 54)
# mapping54[dest_index] = source_index (from original color_str) that should be copied to dest_index
_ROT54: Dict[Tuple[int, int], List[int]] = {}
for face_idx in range(6):
    base = FACE_OFFSETS[face_idx]
    for rot, small_map in _ROT_MAP.items():
        mapping54 = list(range(54))  # identity by default
        # replace only the 9 positions for this face
        for i in range(9):
            dest = base + i
            src = base + small_map[i]
            mapping54[dest] = src
        _ROT54[(face_idx, rot)] = mapping54


# Thread pool for asynchronous solves (small, single-worker by default)
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def apply_face_rotations_to_colorstr(color_str: str, rotations: Dict[str, int]) -> str:
    """
    Fast application of per-face rotations to a 54-char color_str using _ROT54 maps.
    rotations: mapping face_letter -> 0..3 rotations (90deg steps clockwise).
    """
    if not color_str or len(color_str) != 54:
        raise ValueError("color_str must be 54 characters")
    chars = list(color_str)
    new_chars = chars.copy()
    for face_idx, face in enumerate(FACE_ORDER):
        rot = int(rotations.get(face, 0)) % 4
        mapping54 = _ROT54[(face_idx, rot)]
        base = FACE_OFFSETS[face_idx]
        # copy only face's 9 positions
        for i in range(9):
            dest = base + i
            src = mapping54[dest]
            new_chars[dest] = chars[src]
    return ''.join(new_chars)


# CubeState: thin wrapper around kociemba FaceCube/CubieCube and simple move
@dataclass
class CubeState:
    color_status: str = COLOR_INIT_STATE  # color letters per sticker (54)
    face_status: str = FACES_INIT_STATE  # face letters per sticker (54)
    solution: str = ""  # last-known solution string

    face_cube: Optional[Any] = field(default=None, repr=False)
    cubie_cube: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self):
        self._sync_from_face_status()

    def _sync_from_face_status(self) -> None:
        """
        Synchronize face_cube / cubie_cube and color_status from face_status.
        If construction fails, cubes are set to None.
        """
        try:
            self.color_status = "".join(FACE_TO_COLOR[f] for f in self.face_status)
            self.face_cube = kociemba_mod.FaceCube(self.face_status)
            self.cubie_cube = self.face_cube.toCubieCube()
        except Exception:
            self.face_cube = None
            self.cubie_cube = None

    def face_status_set(self, face_status_new: str) -> None:
        """
        Set face_status (54-char string, U,R,F,D,L,B order), and sync cubes.
        """
        if not face_status_new or len(face_status_new) != 54:
            raise ValueError(
                "face_status must be a 54-character string (U,R,F,D,L,B each 9 stickers)."
            )
        self.face_status = face_status_new
        self._sync_from_face_status()

    def reset_state(self) -> None:
        """
        Reset to canonical solved state.
        """
        self.color_status = COLOR_INIT_STATE
        self.face_status = FACES_INIT_STATE
        self.solution = ""
        self._sync_from_face_status()

    def move(self, mv: str) -> None:
        """
        Apply a single move token (e.g. "R", "U'", "F2") to the internal cubie_cube
        and update face_status/color_status accordingly.
        """
        if not mv:
            return
        base = mv[0].upper()
        if base not in MOVE_INDEX:
            raise ValueError(f"Unknown move token: {mv!r}")
        idx = MOVE_INDEX[base]
        times = 1
        if mv.endswith("2"):
            times = 2
        elif mv.endswith("'"):
            times = 3
        if self.cubie_cube is None:
            raise RuntimeError("CubieCube not initialized")
        for _ in range(times):
            # multiply modifies cubie_cube in-place
            self.cubie_cube.multiply(kociemba_mod.moveCube[idx])
        # rebuild face string
        fc = self.cubie_cube.toFaceCube()
        new_facelet = fc.to_String()
        self.face_status = new_facelet
        self.color_status = "".join(FACE_TO_COLOR[f] for f in self.face_status)
        # update face_cube object
        self.face_cube = kociemba_mod.FaceCube(self.face_status)

    def apply_sequence(self, seq: str):
        if not seq:
            return []
        tokens = [tok for tok in seq.strip().split() if tok]
        for tok in tokens:
            self.move(tok)
        return tokens


class CubeSolver:
    def __init__(self):
        self.cube_state = CubeState()
        self._solve_cache: Dict[str, Optional[str]] = {}
        self._lock = threading.Lock()


    def _solve(self, facelets: str) -> Optional[str]:
        """
        Solve using kociemba (with caching).
        Returns solution string or None on failure.
        """
        try:
            cached = self._solve_cache.get(facelets, None)
            if cached is not None:
                logger.debug("Solver cache hit for facelets")
                return cached
            sol = kociemba.solve(facelets)
            # cache result (can be None if solver raised or returned None)
            self._solve_cache[facelets] = sol
            return sol
        except Exception as e:
            logger.exception("Solver exception: %s", e)
            # avoid repeated expensive failures (cache negative result)
            self._solve_cache[facelets] = None
            return None


    # asynchronous convenience wrapper: returns Future
    def solve_async(self, mapping: Optional[Dict[str, str]] = None, facelets: Optional[str] = None):
        return _EXECUTOR.submit(self.solve, mapping, facelets)


    # -----------------------
    # Face-orientation fallback (4^6)
    # -----------------------
    def _attempt_fix_by_face_orientations(self, color_str: str, max_tries: int = 4096):
        """
        Try combinations of local face rotations (0,90,180,270 deg) for each of the 6 faces.
        Returns (facelet_candidate, rotations_map, debug) on success, or (None, None, debug) on failure.

        Uses a cheap Counter prune before calling the kociemba verifier.
        """
        debug = {}
        # invert CENTER_INDICES to index->face_letter
        center_index_to_face = {idx: face for face, idx in CENTER_INDICES.items()}
        # read centers
        centers = {}
        for idx, face_letter in center_index_to_face.items():
            if idx < 0 or idx >= len(color_str):
                debug["err"] = f"center index {idx} out of range"
                return None, None, debug
            centers[face_letter] = color_str[idx]
        # Ensure centers are unique
        if len(set(centers.values())) != 6:
            debug["centers"] = centers
            debug["reason"] = "centers_not_unique"
            return None, None, debug
        color_to_face = {centers[f]: f for f in centers}

        old_chars = list(color_str)
        face_count = 6
        tried = 0

        # iterate combinations (product of 6 faces each rotation 0..3)
        for combo in itertools.product(range(4), repeat=face_count):
            tried += 1
            if tried > max_tries:
                break

            # build new color list applying per-face rotation via _ROT54 quickly
            new_chars = old_chars.copy()
            for face_idx in range(face_count):
                rot = combo[face_idx] % 4
                mapping54 = _ROT54[(face_idx, rot)]
                base = FACE_OFFSETS[face_idx]
                for i in range(9):
                    dest = base + i
                    src = mapping54[dest]
                    new_chars[dest] = old_chars[src]

            new_color_str = "".join(new_chars)

            # Build facelet candidate mapping colors->face letters based on centers
            facelet_candidate = "".join(color_to_face.get(ch, "?") for ch in new_color_str)
            if "?" in facelet_candidate:
                continue

            # cheap prune: check that each face letter appears exactly 9 times
            cnt = Counter(facelet_candidate)
            if any(cnt.get(f, 0) != 9 for f in FACE_ORDER):
                continue

            # quick validation using kociemba_module FaceCube verify
            valid, _msg = self.validate_facelet(facelet_candidate)
            if not valid:
                continue

            # final attempt to solve
            sol = self._solve(facelet_candidate)
            if sol:
                rotations_map = {FACE_ORDER[i]: combo[i] for i in range(face_count)}
                debug["tried"] = tried
                debug["centers"] = centers
                return facelet_candidate, rotations_map, debug

        debug["tried"] = tried
        debug["final"] = "no_face_orientation_fix_found"
        return None, None, debug


    def _build_facelets_from_mapping(self, mapping: Dict[str, str]) -> Tuple[str, str]:
        """
        Build color_str (54 color letters) and facelet_str (54 face letters U/R/F/D/L/B)
        from mapping: keys 'U1'..'B9' -> color letters (e.g. 'R','O','Y','G','B','W').

        Applies per-face rotation fixes using config FACE_ROTATIONS and persisted overrides.
        Raises ValueError if missing stickers or centers cannot be disambiguated.
        Returns (color_str, facelet_str).
        """
        colors_per_face: Dict[str, List[str]] = {}
        missing: List[str] = []
        for face in FACE_ORDER:  # expected order: U,R,F,D,L,B
            arr = []
            for i in range(1, 10):
                k = f"{face}{i}"
                if k not in mapping:
                    missing.append(k)
                    arr.append("X")
                else:
                    arr.append(mapping[k])
            colors_per_face[face] = arr
        if missing:
            raise ValueError(f"Missing stickers: {missing}")

        # Build initial flattened color_str in FACE_ORDER without rotations
        base_colors = []
        for face in FACE_ORDER:
            base_colors.extend(colors_per_face[face])
        base_color_str = "".join(base_colors)

        # Determine effective rotations
        eff_rot = {}
        for f in FACE_ORDER:
            eff_rot[f] = int(FACE_ROTATIONS.get(f, 0)) % 4
        logger.debug("Effective FACE_ROTATIONS used: %s", eff_rot)
        
        if any((eff_rot.get(face, 0) % 4) != 0 for face in FACE_ORDER):
            # Apply rotations efficiently using precomputed _ROT54 maps
            color_str = apply_face_rotations_to_colorstr(base_color_str, eff_rot)
        else:
            color_str = base_color_str

        # Map center colors -> face letters using CENTER_INDICES
        color_to_face: Dict[str, str] = {}
        for face_letter, idx in CENTER_INDICES.items():
            c = color_str[idx]
            if c in color_to_face:
                raise ValueError(
                    f"Duplicate center color {c!r} between {color_to_face[c]!r} and {face_letter!r}"
                )
            color_to_face[c] = face_letter
        if len(color_to_face) != 6:
            raise ValueError("Invalid center mapping: could not map six distinct centers.")

        facelet_str = "".join(color_to_face.get(c, "?") for c in color_str)
        return color_str, facelet_str


    def validate_facelet(self, facelets: str) -> Tuple[bool, str]:
        """
        Validate a 54-char facelet string using kociemba_module.FaceCube verification.
        Returns (ok:bool, message:str)
        """
        try:
            fc = kociemba_mod.FaceCube(facelets)
            status = fc.toCubieCube().verify()
            if status == 0:
                return True, "Cube OK"
            if status == -2:
                return False, "Not all 12 edges exist exactly once"
            if status == -3:
                return False, "Flip error: One edge has to be flipped"
            if status == -4:
                return False, "Not all corners exist exactly once"
            if status == -5:
                return False, "Twist error: One corner has to be twisted"
            if status == -6:
                return False, "Parity error: Two corners or two edges have to be exchanged"
            return False, f"Verify returned status {status}"
        except Exception as e:
            return False, f"Validation exception: {e}"


    def solve(self, mapping: Optional[Dict[str, str]] = None, facelets: Optional[str] = None) -> Tuple[bool, str]:
        """
        High-level solve interface.

        If mapping provided it will be converted to facelets (applying configured/persisted per-face rotations).
        Returns (ok: bool, message: str). On success, solver.cube_state is updated (face_status and solution).
        """
        try:
            if mapping is not None:
                logger.debug("Solving from mapping : %s", mapping)
                color_str, facelet_str = self._build_facelets_from_mapping(mapping)
                logger.debug("Built color string:   %s", color_str)
                logger.debug("Built facelet string: %s", facelet_str)

                ok, msg = self.validate_facelet(facelet_str)
                if not ok:
                    logger.debug("Facelet validation failed : %s", msg)
                    # fallback: try face-local rotations search (4^6)
                    candidate_face, rotations_map, dbg = self._attempt_fix_by_face_orientations(color_str)
                    if candidate_face:
                        logger.debug("Auto-corrected by rotating faces: %s", rotations_map)
                        facelet_str = candidate_face
                    else:
                        logger.debug("No face-orientation fix found. debug: %s", dbg)
                        return False, f"Facelets invalid: {msg}"

                sol = self._solve(facelet_str)
                if not sol:
                    return False, "kociemba failed or timed out"
                # Update cube state
                with self._lock:
                    self.cube_state.face_status_set(facelet_str)
                    self.cube_state.solution = sol
                return True, "Solved"

            # no mapping: use provided facelets or current cube_state
            if facelets is None:
                facelets = self.cube_state.face_status
            if not isinstance(facelets, str) or len(facelets) != 54:
                return False, "facelets must be a 54-character string"
            ok, msg = self.validate_facelet(facelets)
            if not ok:
                return False, f"Facelets invalid: {msg}"
            sol = self._solve(facelets)
            if not sol:
                return False, "kociemba failed or timed out"
            with self._lock:
                self.cube_state.face_status_set(facelets)
                self.cube_state.solution = sol
            return True, "Solved"

        except Exception as e:
            logger.exception("Solver error: %s", e)
            return False, str(e)


    def clear_cache(self) -> None:
        """Clear the internal solve cache."""
        self._solve_cache.clear()


# Module self-test when executed directly
if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="cube_solver.py")
    parser.add_argument("--facelets", help="Provide a 54-char facelet string to solve")
    parser.add_argument("--timeout", type=float, default=3.0, help="Solver timeout seconds")
    args = parser.parse_args()

    # instantiate solver (constructor accepts no args in this version)
    solver = CubeSolver()
    print("CubeSolver self-check...")

    # permutation-free sanity: check solver on solved cube
    solved_face = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    ok, msg = solver.solve(facelets=solved_face)
    print("Solved-cube test:", ok, msg)
    if ok:
        print("Solution (should be empty or trivial):", repr(solver.cube_state.solution))

    # If user passed facelets, try solve them
    if args.facelets:
        try:
            s = args.facelets.strip()
            ok, msg = solver.solve(facelets=s)
            print("User facelets solve:", ok, msg)
            if ok:
                print("Solution:", solver.cube_state.solution)
        except Exception as e:
            print("Failed solving provided facelets:", e)
            sys.exit(1)

    print("Interactive mode: paste 54-char facelets (or 'x' to quit)")
    while True:
        try:
            line = input("Facelets> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not line:
            continue
        if line.lower() == "x":
            break
        if len(line) != 54:
            print("Please input exactly 54 characters (U,R,F,D,L,B order).")
            continue
        ok, msg = solver.solve(facelets=line)
        print("Result:", ok, msg)
        if ok:
            print("Solution:", solver.cube_state.solution)
    print("Exiting.")
