# debug_kociemba_tools.py
# Utilities to generate solvable facelet strings (from random scrambles),
# run kociemba.solve on them, measure performance and optionally compare with
# CubeStatus.find_solution if available.

import random
import time
import statistics
from typing import List, Tuple, Dict, Optional

import kociemba
import kociemba_module as kociemba_mod

# move index mapping consistent con tu código
MOVE_INDEX = {'U':0, 'R':1, 'F':2, 'D':3, 'L':4, 'B':5}

# solved facelet string in Kociemba order (U, R, F, D, L, B)
SOLVED_FACELET = "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9

# allowed face letters for generating scrambles
FACES = ['U','R','F','D','L','B']
MODS = ['', "'", '2']  # normal, inverse, double

def random_scramble_moves(length: int = 25, avoid_cancel: bool = True) -> List[str]:
    """
    Generate a random scramble (list of move tokens) of given length.
    avoid_cancel True evita repetir la misma cara consecutivamente (mejor scramble).
    """
    moves = []
    prev_face = None
    for _ in range(length):
        face = random.choice(FACES)
        if avoid_cancel:
            # try a few times to avoid repeating face
            tries = 0
            while face == prev_face and tries < 5:
                face = random.choice(FACES)
                tries += 1
        mod = random.choices(MODS, weights=(70,15,15))[0]  # bias to single turns
        moves.append(face + mod)
        prev_face = face
    return moves

def apply_moves_to_facelet(facelet: str, moves: List[str]) -> str:
    """
    Applies moves (notations like 'R', "U'", 'F2') to a facelet string using kociemba_module.
    Returns the resulting facelet string (54 chars).
    """
    # create FaceCube from facelet string and get CubieCube
    fc = kociemba_mod.FaceCube(facelet)
    cc = fc.toCubieCube()
    for mv in moves:
        if not mv:
            continue
        base = mv[0].upper()
        if base not in MOVE_INDEX:
            continue
        idx = MOVE_INDEX[base]
        # determine times ('' -> 1, '2' -> 2, "'" -> 3)
        if mv.endswith("2"):
            times = 2
        elif mv.endswith("'"):
            times = 3
        else:
            times = 1
        for _ in range(times):
            cc.multiply(kociemba_mod.moveCube[idx])
    # convert back to FaceCube string
    new_facelet = cc.toFaceCube().to_String()
    return new_facelet

def generate_facelet_samples(n: int = 10, scramble_len: int = 25, seed: Optional[int] = None) -> List[Dict]:
    """
    Generate `n` solvable facelet strings by applying random scrambles to the solved state.
    Returns list of dicts: { 'scramble':str, 'moves':List[str], 'facelet':str }
    """
    if seed is not None:
        random.seed(seed)
    samples = []
    for i in range(n):
        moves = random_scramble_moves(scramble_len)
        facelet = apply_moves_to_facelet(SOLVED_FACELET, moves)
        samples.append({
            "scramble_notation": " ".join(moves),
            "moves_list": moves,
            "facelet": facelet
        })
    return samples

def solve_facelet_with_kociemba(facelet: str, timeout: Optional[float] = None) -> Tuple[Optional[str], float, Optional[Exception]]:
    """
    Solve facelet with kociemba.solve; returns (solution_string_or_None, elapsed_seconds, exception_or_None).
    If timeout is given, we do a naive timeout by checking elapsed time (kociemba.solve is blocking,
    so this cannot preempt the solver — this param is informative only).
    """
    t0 = time.time()
    try:
        sol = kociemba.solve(facelet)
        elapsed = time.time() - t0
        return sol, elapsed, None
    except Exception as e:
        elapsed = time.time() - t0
        return None, elapsed, e

def verify_facelet(facelet: str) -> Tuple[bool, Optional[int]]:
    """
    Verify using kociemba_module.FaceCube(...).toCubieCube().verify()
    Returns (is_valid, verify_status) where verify_status is the verify() return code or None on error.
    """
    try:
        fc = kociemba_mod.FaceCube(facelet)
        status = fc.toCubieCube().verify()
        return (status == 0), status
    except Exception as e:
        return False, None

def debug_run(num_samples: int = 10, scramble_len: int = 25, seed: Optional[int] = None,
              try_find_solution: bool = False, cube_status_obj = None):
    """
    Main debug function:
      - genera `num_samples` facelets
      - resuelve cada uno con kociemba (midiendo tiempos)
      - opcional: intenta cube_status_obj.find_solution(...) si se pasa la instancia (y try_find_solution True)
    Imprime resumen estadístico y devuelve los resultados.
    """
    print(f"[debug_run] Generating {num_samples} samples (scramble_len={scramble_len}, seed={seed})")
    samples = generate_facelet_samples(num_samples, scramble_len, seed)

    results = []
    times = []
    lengths = []
    failures = 0

    for idx, s in enumerate(samples, start=1):
        facelet = s["facelet"]
        scramble = s["scramble_notation"]
        print(f"\n[Sample {idx}/{num_samples}] scramble: {scramble[:120]}{'...' if len(scramble)>120 else ''}")
        valid, status = verify_facelet(facelet)
        print(f"  verify -> valid={valid}, status={status}")
        sol, elapsed, exc = solve_facelet_with_kociemba(facelet)
        if sol is None:
            print(f"  kociemba.solve FAILED in {elapsed:.3f}s: {exc}")
            failures += 1
            results.append({
                "facelet": facelet,
                "scramble": scramble,
                "valid": valid,
                "verify_status": status,
                "solution": None,
                "solve_time": elapsed,
                "error": str(exc)
            })
            continue
        tok_count = len([t for t in sol.split() if t])
        print(f"  kociemba.solve OK ({elapsed:.3f}s) moves={tok_count}: {sol[:160]}{'...' if len(sol)>160 else ''}")
        times.append(elapsed)
        lengths.append(tok_count)

        entry = {
            "facelet": facelet,
            "scramble": scramble,
            "valid": valid,
            "verify_status": status,
            "solution": sol,
            "solve_time": elapsed,
            "solution_len": tok_count
        }

        # optionally try your CubeStatus.find_solution to compare (if provided)
        if try_find_solution and cube_status_obj is not None:
            try:
                # your find_solution expects mapping label->color_letter; build mapping:
                mapping = {}
                # face order is U,R,F,D,L,B and each face has 1..9 positions
                faces = ['U','R','F','D','L','B']
                for fi, face in enumerate(faces):
                    block = facelet[fi*9:(fi+1)*9]
                    for i_ch, ch in enumerate(block, start=1):
                        mapping[f"{face}{i_ch}"] = ch
                t0f = time.time()
                cs_color_str, cs_facelets, cs_solution = cube_status_obj.find_solution(mapping)
                t_find = time.time() - t0f
                entry["find_solution_result"] = {
                    "facelets_used": cs_facelets,
                    "solution": cs_solution,
                    "elapsed": t_find
                }
                print(f"  find_solution(...) returned in {t_find:.3f}s; solution_len={len(cs_solution.split()) if cs_solution else 'N/A'}")
            except Exception as e:
                entry["find_solution_error"] = str(e)
                print(f"  find_solution(...) ERROR: {e}")

        results.append(entry)

    # summary
    if times:
        print("\n[SUMMARY] Solved samples:", len(times), "Failed:", failures)
        print(f"  solve time: mean={statistics.mean(times):.3f}s median={statistics.median(times):.3f}s max={max(times):.3f}s")
        print(f"  solution length: mean={statistics.mean(lengths):.1f} moves median={statistics.median(lengths):.1f} max={max(lengths)}")
    else:
        print("\n[SUMMARY] No successful solves.")

    return {
        "samples": samples,
        "results": results,
        "stats": {
            "num_samples": num_samples,
            "solved": len(times),
            "failed": failures,
            "solve_time_mean": statistics.mean(times) if times else None,
            "solve_time_median": statistics.median(times) if times else None,
            "solve_time_max": max(times) if times else None,
        }
    }

# Quick CLI usage if run as script (práctico para pruebas rápidas)
if __name__ == "__main__":
    # ejemplo rápido: 6 estados con scrambles de 20 movimientos
    out = debug_run(num_samples=6, scramble_len=20, seed=1234, try_find_solution=True, cube_status_obj=None)
    # guardá / inspeccioná 'out' para ver los facelets y soluciones
