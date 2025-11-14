from dataclasses import dataclass, field
import math
import time
from typing import Counter, Optional, Tuple, Dict, List
import threading

import kociemba
import kociemba_module as kociemba_mod
FaceCube = kociemba_mod.FaceCube
CubieCube = kociemba_mod.CubieCube

from config import CENTER_INDICES, COLOR_TO_FACE, FACE_ORDER, COLOR_INIT_STATE, FACE_TO_COLOR,FACES_INIT_STATE,MOVE_INDEX

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


@dataclass
class CubeState:
    color_status: str = COLOR_INIT_STATE
    face_status: str = FACES_INIT_STATE
    solution: str = ''

    face_cube: Optional[object] = field(default=None, repr=False)
    cubie_cube: Optional[object] = field(default=None, repr=False)

    def __post_init__(self):
        self._sync_from_face_status()

    def _sync_from_face_status(self):
        """Sincroniza face_cube/cubie_cube y color_status a partir de self.face_status."""
        try:
            self.color_status = ''.join(FACE_TO_COLOR[f] for f in self.face_status)
            self.face_cube = FaceCube(self.face_status)
            self.cubie_cube = self.face_cube.toCubieCube()
        except Exception as e:
                
                self.face_cube = None
                self.cubie_cube = None

    def _sync_from_color_status(self):
        """Sincroniza face_status/cubes a partir de color_status usando COLOR_TO_FACE."""
        faces = ''.join(COLOR_TO_FACE[c] for c in self.color_status)
        self.face_status = faces
        self._sync_from_face_status()

    def face_status_set(self, face_status_new: str):
        if not face_status_new or len(face_status_new) != 54:
            raise ValueError("face_status must be a 54-character string (faces U,R,F,D,L,B each 9 stickers).")
        self.face_status = face_status_new
        self._sync_from_face_status()

    def colors_status_set(self, colors_status_new: str):
        if not colors_status_new or len(colors_status_new) != 54:
            raise ValueError("colors_status must be a 54-character string (colors per sticker).")
        self.color_status = colors_status_new
        self._sync_from_color_status()

    def reset_state(self):
        self.color_status = COLOR_INIT_STATE
        self.face_status = FACES_INIT_STATE
        self.solution = ''
        self._sync_from_face_status()

    def move(self, mv: str):
        #print("curr Facelet"+self.face_status+" to Move "+mv)

        if not mv: return
        base = mv[0].upper()
        if base not in MOVE_INDEX: return

        idx = MOVE_INDEX[base]
        if mv.endswith("2"):
            times = 2
        elif mv.endswith("'"):
            times = 3  # inverse = apply 3 times
        else:
            times = 1

        for _ in range(times):
            self.cubie_cube.multiply(kociemba_mod.moveCube[idx])

        try:
            # muchas implementaciones: cubie_cube.toFaceCube().to_String()
            fc = self.cubie_cube.toFaceCube()
            new_facelet = fc.to_String()
            self.face_status = new_facelet
            self.color_status = ''.join(FACE_TO_COLOR[f] for f in self.face_status)
            self.face_cube = FaceCube(self.face_status)
        except Exception as e:
            raise RuntimeError(f"Failed to derive face string from cubie_cube after applying move: {e}") from e
        #print("Resulted in : "+self.face_status)
        #RLFUUFUUBLRULRULRRRLDFFDBBUDDBDDBFFFUFFULLLRRLBBRBBDDD


    def apply_sequence(self, seq: str):
        if not seq:
            return []
        tokens = [tok for tok in seq.strip().split() if tok]
        for tok in tokens:
            self.move(tok)
        return tokens

    def validate_facelet(self, facelets: str) -> Tuple[bool,str]:
        try:
            fc = kociemba_mod.FaceCube(facelets)
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


class CubeSolver:
    def __init__(self):
        self.cube_state = CubeState()
        self._permutations_24_cache: Optional[List[Dict]] = None
        self.lock = threading.Lock()


    def _generate_24_permutations(self):
        """
        Genera las 24 permutaciones (una lista de dicts) donde cada dict contiene:
          - 'perm': lista de 54 enteros tal que sticker original i -> nueva posición perm[i]
          - 'rotated_to_original': dict mapa cara_orientada_letter -> original_letter
          - 'name': nombre corto de orientación
        """
        if self._permutations_24_cache is not None:
            return self._permutations_24_cache

        # Ejes y normales
        X = (1.0, 0.0, 0.0)
        Y = (0.0, 1.0, 0.0)
        Z = (0.0, 0.0, 1.0)
        axes = [X, Y, Z]
        unit_vecs = [
            (1.0,0.0,0.0), (-1.0,0.0,0.0),
            (0.0,1.0,0.0), (0.0,-1.0,0.0),
            (0.0,0.0,1.0), (0.0,0.0,-1.0),
        ]

        faces = FACE_ORDER  # ['U','R','F','D','L','B']

        # Definición de centros (normal hacia afuera)
        face_normals = {
            'U': (0.0, 1.0, 0.0),
            'D': (0.0, -1.0, 0.0),
            'F': (0.0, 0.0, 1.0),
            'B': (0.0, 0.0, -1.0),
            'R': (1.0, 0.0, 0.0),
            'L': (-1.0, 0.0, 0.0),
        }

        # Para cada cara definimos vector "right" y "down" tal que el orden 3x3 quede
        # coherente con la convención de facelets (fila superior a inferior, izquierda a derecha).
        face_axes = {
            'U': ((1,0,0), (0,0,1)),     # right = +x, down = +z
            'D': ((1,0,0), (0,0,-1)),    # right = +x, down = -z
            'F': ((1,0,0), (0,-1,0)),    # right = +x, down = -y
            'B': ((-1,0,0), (0,-1,0)),   # right = -x, down = -y
            'R': ((0,0,-1), (0,-1,0)),   # looking from +x: right = -z, down = -y
            'L': ((0,0,1), (0,-1,0)),    # looking from -x: right = +z, down = -y
        }

        # Construir lista de coordenadas 3D para cada sticker index 0..53 (face blocks de 9)
        sticker_coords = []  # list of (x,y,z) for index i
        sticker_face_index = []  # face index 0..5 for sticker i
        spacing = 1.0  # escala arbitraria (solo relación importa)
        offset = 1.0  # distancia desde el centro a la cara
        # indices: por cada cara en FACE_ORDER, posiciones r=0..2, c=0..2
        for fi, face in enumerate(faces):
            center = face_normals[face]  # (cx,cy,cz)
            right = face_axes[face][0]
            down = face_axes[face][1]
            for r in range(3):
                for c in range(3):
                    # r: 0..2 (top->bottom), c: 0..2 (left->right)
                    # pos = center*offset + ((c-1)*right + (r-1)*down) * spacing/2
                    cx, cy, cz = center
                    rx, ry, rz = right
                    dx, dy, dz = down
                    # factor to separate stickers well: use 2/3 of face size (arbitrary)
                    f = 0.6
                    x = cx*offset + ((c-1) * rx + (r-1) * dx) * f
                    y = cy*offset + ((c-1) * ry + (r-1) * dy) * f
                    z = cz*offset + ((c-1) * rz + (r-1) * dz) * f
                    sticker_coords.append((x,y,z))
                    sticker_face_index.append(fi)

        # función utilitaria: cross product y normalización
        def cross(a,b):
            return (a[1]*b[2] - a[2]*b[1],
                    a[2]*b[0] - a[0]*b[2],
                    a[0]*b[1] - a[1]*b[0])
        def dot(a,b):
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
        def norm(a):
            return math.sqrt(dot(a,a))
        def normalize(a):
            n = norm(a)
            if n == 0:
                return (0.0,0.0,0.0)
            return (a[0]/n, a[1]/n, a[2]/n)

        # Construir las 24 rotaciones (matrices) como elección de nuevos ejes ortonormales
        rotations = []
        unit_choices = unit_vecs
        for ux in unit_choices:
            # pick uy perpendicular to ux
            for uy in unit_choices:
                if abs(dot(ux,uy)) > 1e-9:
                    continue
                uz = cross(ux, uy)
                # asegurarnos determinante positivo (uz debe ser unit ±)
                det = dot(cross(ux, uy), uz)
                # uz ya es perpendicular; debemos normalizar uz and ensure right-handed
                # Si uz sea in the set of unit vecs up to sign, it's fine; check det sign:
                # compute determinant via scalar triple product of (ux,uy,uz)
                # but if uz is cross(ux,uy), determinant > 0.
                # Guardar la matriz (3x3) col-wise: [ux,uy,uz]
                rot = (ux, uy, uz)
                # Normalizar uz por seguridad
                rotations.append(rot)
        # Filtrar duplicados (puede haber repeticiones)
        unique_rots = []
        seen = set()
        for rot in rotations:
            # redondear componentes a tolerancia para comparar
            key = tuple(round(x,5) for v in rot for x in v)
            if key in seen:
                continue
            seen.add(key)
            unique_rots.append(rot)
        # Debemos quedar con 24
        # (En práctica unique_rots tendrá 24)
        rotations = unique_rots

        results = []
        # para cada rotación calculo permutación: original index i -> new index j
        for r_idx, rot in enumerate(rotations):
            ux, uy, uz = rot
            # función para aplicar rotación (matrix with columns ux,uy,uz)
            def apply_rot(vec):
                # vec expressed in original basis; rotated = [ux uy uz] * vec_coords
                # But our rot is orthonormal basis; to rotate point p we compute:
                # p' = (dot(p, ux), dot(p, uy), dot(p, uz))  (coordenadas en nueva base)
                return (dot(vec, ux), dot(vec, uy), dot(vec, uz))

            # Aplicar rot a cada sticker coord y asignar a una cara destino (por normal más cercana)
            dest_positions = [None]*54  # dest_positions[i] = j (index destino)
            # para reconstruir cara centers y detectar mapping de caras
            # normal de destino face k is simply rotated original normal:
            rotated_face_normals = {}
            for face in faces:
                n0 = face_normals[face]
                nr = apply_rot(n0)
                rotated_face_normals[face] = nr

            # para cada sticker i, rotamos su coords y luego decidimos a qué cara destino pertenece:
            # elegimos la cara cuyo rotated_face_normal tenga mayor dot con rotated sticker vector.
            rotated_coords = [apply_rot(p) for p in sticker_coords]

            # construir array de destino: para cada sticker i, encontrar target face and r,c
            dest_index_for_i = [-1]*54
            # Para poder determinar r,c en target face: necesitamos proyectar en base right/down del target face (que también están rotados)
            # calcular right/down vectores rotados para las caras:
            rotated_face_axes = {}
            for face in faces:
                right0, down0 = face_axes[face]
                right_r = apply_rot(right0)
                down_r = apply_rot(down0)
                # normalize small numerical errors
                right_r = normalize(right_r)
                down_r = normalize(down_r)
                rotated_face_axes[face] = (right_r, down_r)

            # para cada rotated coord, decidir target face
            for i, p in enumerate(rotated_coords):
                # pick face with maximum dot(p, face_normal)
                best_face = None
                best_dp = -1e9
                for face in faces:
                    fn = rotated_face_normals[face]
                    dp = dot(normalize(p), normalize(fn))
                    if dp > best_dp:
                        best_dp = dp
                        best_face = face
                # ahora calcular r,c: proyectar p onto right and down of that face, comparar con center
                # Obtener center rotated coord (apply_rot of face center normal)
                face_center = rotated_face_normals[best_face]
                right_v, down_v = rotated_face_axes[best_face]
                # vector from face center to sticker point:
                vec = (p[0] - face_center[0], p[1] - face_center[1], p[2] - face_center[2])
                # coordenadas en base [right_v, down_v]
                rc = dot(vec, right_v)
                rd = dot(vec, down_v)
                # convertir rc,rd a índice 0,1,2: recuerden que en construcción original
                # usamos offset (c-1)*right + (r-1)*down scaling ~ f*1
                # por tanto hacemos round a nearest int of (rc / f) + 1
                f = 0.6
                c_pos = int(round(rc / f + 1))
                r_pos = int(round(rd / f + 1))
                # clamp
                if r_pos < 0: r_pos = 0
                if r_pos > 2: r_pos = 2
                if c_pos < 0: c_pos = 0
                if c_pos > 2: c_pos = 2
                # calcular índice destino j = face_index*9 + r_pos*3 + c_pos
                face_idx = faces.index(best_face)
                j = face_idx*9 + r_pos*3 + c_pos
                dest_index_for_i[i] = j

            # asegurarnos que dest_index_for_i es permutación (no colisiones)
            if len(set(dest_index_for_i)) != 54:
                # si hay colisiones, intentar pequeña corrección: usar nearest neighbor asignando por distancia
                assigned = {}
                dest_index_for_i = [-1]*54
                for i,p in enumerate(rotated_coords):
                    best_face = None
                    best_dp = -1e9
                    for face in faces:
                        fn = rotated_face_normals[face]
                        dp = dot(normalize(p), normalize(fn))
                        if dp > best_dp:
                            best_dp = dp
                            best_face = face
                    face_center = rotated_face_normals[best_face]
                    right_v, down_v = rotated_face_axes[best_face]
                    vec = (p[0] - face_center[0], p[1] - face_center[1], p[2] - face_center[2])
                    rc = dot(vec, right_v); rd = dot(vec, down_v)
                    f = 0.6
                    c_pos = int(round(rc / f + 1)); r_pos = int(round(rd / f + 1))
                    c_pos = max(0, min(2, c_pos)); r_pos = max(0, min(2, r_pos))
                    face_idx = faces.index(best_face)
                    j = face_idx*9 + r_pos*3 + c_pos
                    # si ocupado, buscar nearest free in 3x3 grid of that face
                    if j in assigned.values():
                        # buscar por distancia simple
                        best_j = None; best_dist = 1e9
                        for rr in range(3):
                            for cc in range(3):
                                candidate = face_idx*9 + rr*3 + cc
                                if candidate in assigned.values():
                                    continue
                                # centro of candidate:
                                ccx = face_center[0] + (cc-1)*right_v[0] + (rr-1)*down_v[0]
                                ccy = face_center[1] + (cc-1)*right_v[1] + (rr-1)*down_v[1]
                                ccz = face_center[2] + (cc-1)*right_v[2] + (rr-1)*down_v[2]
                                dx = p[0]-ccx; dy = p[1]-ccy; dz = p[2]-ccz
                                d = dx*dx+dy*dy+dz*dz
                                if d < best_dist:
                                    best_dist = d; best_j = candidate
                        j = best_j
                    dest_index_for_i[i] = j
                    assigned[i] = j

                if len(set(dest_index_for_i)) != 54:
                    # si aún así falla, saltar esta rotación
                    continue

            # perm: lista tal que original index i -> new index perm[i]
            perm = list(dest_index_for_i)

            # construir rotated_to_original: para cada target face (0..5) ver que centro original terminó ahí
            # center original index for face fi: fi*9 + 4
            rotated_to_original = {}
            for orig_fi, orig_face in enumerate(faces):
                orig_center_idx = orig_fi*9 + 4
                dest_center_idx = perm[orig_center_idx]
                dest_face_idx = dest_center_idx // 9
                dest_face_letter = faces[dest_face_idx]
                # original face orig_face ended up in dest_face_letter
                # queremos mapping oriented_face -> original_face (inverse)
                rotated_to_original[dest_face_letter] = orig_face

            results.append({
                'perm': perm,
                'rotated_to_original': rotated_to_original,
                'name': f"rot_{r_idx}"
            })

        # dedupe por perm
        unique = []
        seen_perms = set()
        for r in results:
            key = tuple(r['perm'])
            if key in seen_perms:
                continue
            seen_perms.add(key)
            unique.append(r)
        self._permutations_24_cache = unique
        return unique

    @staticmethod
    def _remap_solution_tokens(sol_str: str, rotated_to_original_map: Dict[str,str]) -> str:
        tokens = [tok for tok in sol_str.strip().split() if tok]
        out_tokens = []
        for tok in tokens:
            base = tok[0].upper()
            suffix = tok[1:] if len(tok) > 1 else ""
            if base in rotated_to_original_map and rotated_to_original_map[base] != '?':
                mapped = rotated_to_original_map[base] + suffix
                out_tokens.append(mapped)
            else:
                out_tokens.append(tok)
        return ' '.join(out_tokens)

    def _find_solution_from_facelet(self, facelet_str: str, timeout_seconds: float = 3.0, max_iters: int = 2000) -> Tuple[str,str]:
        """
        Recibe un facelet string (54 chars, orden U,R,F,D,L,B) y devuelve (best_facelet, best_solution)
        donde best_solution es la cadena de movimientos remapeada al sistema de caras original,
        y best_facelet es el facelet (orientado) correspondiente a esa solución.
        Lanza ValueError si falla en encontrar solución válida.
        """
        if not isinstance(facelet_str, str) or len(facelet_str) != 54:
            raise ValueError("facelet_str must be a 54-character string in order U,R,F,D,L,B")

        # quick validation: centers deben ser U,R,F,D,L,B (o pueden ser colores si así trabajas)
        # Aquí asumimos que facelet_str usa letras de caras (U,R,F,D,L,B) o equivalentes; dejamos al solver verificar
        perms = self._generate_24_permutations()

        t0 = time.time()
        tries = 0

        best = {
            "u_count": None,
            "moves_len": None,
            "facelets": None,
            "solution": None,
            "orientation": None,
        }

        # First, try the original facelet as baseline
        candidates = []
        candidates.append({
            'perm': None,
            'rotated_to_original': {c:c for c in FACE_ORDER},
            'facelet': facelet_str,
            'name': 'original'
        })

        # Append permutations (limit to max_iters)
        for pdesc in perms:
            candidates.append({
                'perm': pdesc['perm'],
                'rotated_to_original': pdesc['rotated_to_original'],
                'facelet': None,
                'name': pdesc['name']
            })

        # Iterate candidates
        for c in candidates:
            tries += 1
            if tries > max_iters or (time.time() - t0) > timeout_seconds:
                break

            if c['perm'] is None:
                oriented = facelet_str
            else:
                # build oriented by placing sticker i of original into pos perm[i]
                perm = c['perm']
                arr = ['?']*54
                for i,ch in enumerate(facelet_str):
                    j = perm[i]
                    arr[j] = ch
                oriented = ''.join(arr)
            c['facelet'] = oriented

            # try solver
            try:
                sol = kociemba.solve(oriented)
            except Exception:
                continue

            # remap tokens back to original labeling (si la orientación no es original)
            mapped_sol = self._remap_solution_tokens(sol, c['rotated_to_original'])

            # contar U (en mapped_sol)
            toks = [tok for tok in mapped_sol.split() if tok]
            u_count = sum(1 for tok in toks if tok[0].upper() == 'U')
            moves_len = len(toks)

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
                    "facelets": oriented,
                    "solution": mapped_sol,
                    "orientation": c['name'],
                })
                # early exit if optimal
                if u_count == 0:
                    break

        if best["solution"] is None:
            raise ValueError("Solver failed for original and all orientations within limits.")

        return best["facelets"], best["solution"]

    def _build_facelets(self, mapping : Dict[str,str]):
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

    def solve(self,mapping : Dict[str,str] = None) -> Tuple[bool,str]:
        try:  
            if mapping != None:
                color_str, facelets, sol = self._build_facelets(mapping)
                if facelets == None or sol == None:
                    return False,"Imposible to convert the mapping"
                facelets_imp , sol_imp = self._find_solution_from_facelet(facelets)
                self.cube_state.face_status_set(facelets_imp)
                self.cube_state.solution = sol_imp
            else:
                with self.lock:
                    face_snapshot = str(self.cube_state.face_status)
                print("Face status prev solved : " + face_snapshot)
                facelets , sol = self._find_solution_from_facelet(face_snapshot)
                with self.lock:
                    self.cube_state.face_status_set(facelets)
                    self.cube_state.solution = sol
        except Exception as e:
            self.cube_state.solution = ""
            return False,e
        return True,"Solved"
    

if __name__ == "__main__":
    cb = CubeSolver()
    while True:
        facelets = input("Facelets: ")
        if facelets == "x": break

        facelets_imp , sol_imp = cb._find_solution_from_facelet(facelets,100,9999999)
        print("Facelets Result : "+facelets_imp)
        print("Solution : "+sol_imp)


#    `UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB`

 #   `UBULURUFURURFRBRDRFUFLFRFDFDFDLDRDBDLULBLFLDLBUBRBLBDB`

  #  `FBLLURRFBUUFBRFDDFUULLFRDDLRFBLDRFBLUUBFLBDDBUURRBLDDR`