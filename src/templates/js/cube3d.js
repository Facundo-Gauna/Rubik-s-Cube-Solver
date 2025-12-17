/*
cube3d.js — 3D Rubik's cube renderer and UI controller
=================================================================

Purpose & responsibilities:
This module implements the 3D cube renderer and interactive cube controls used
in the project's frontend. It uses WebGL via the three.js library to draw a
rotatable, animatable Rubik's-cube, and exposes programmatic hooks that the
rest of the UI uses to apply moves, run animations, and synchronize the
visual state with the solver/model.

Major responsibilities:
- Build and render a 3D cube mesh composed of 27 cubies (3x3x3 logical grid).
- Translate high-level move tokens (e.g. "R", "U'", "F2") into visual
rotations of the corresponding slice of cubies, including smooth animations.
- Provide functions to apply sequences, scramble, reset, and export current
facelet string/state.
- Offer user interaction: mouse/touch drag to rotate the whole cube and UI
controls to rotate faces.
- Interoperate with the `rubik` object (global) to keep model and view
synchronized: when moves are applied via the UI, the view updates and viceversa.

Design notes:
- The module favors clarity over extreme optimization. Rotations are implemented
by grouping cubie meshes into temporary parent Objects to perform slice
rotations; after the rotation completes the meshes are reattached to the
main cube root and transforms are normalized.
- Animations use requestAnimationFrame loop integrated with three.js rendering.
Moves are animated by interpolating rotation angles over time according to
an easing function, and a queue serializes concurrent move requests.
- The rendering pipeline sets devicePixelRatio with a capped maximum to avoid
extremely large canvases on high-DPI displays.

Synchronization with model (rubik):
- This module expects a `rubik` global that exposes methods like `move()`,
`apply_sequence()` or `resetCube()` depending on the rest of the app.
- When a user triggers a move visually, the module should call `rubik` to
update the logical cube state and forward the move to the backend motors 
via pywebview API.

Accessibility & performance:
- Rendering uses WebGL. Ensure the target environment supports WebGL (modern 
browsers / electron / pywebview).
- For slower machines consider lowering `RENDER_MAX_DPR` or reducing canvas
resolution to improve framerate.

Extensibility suggestions:
- Extract a small public event emitter for move start/finish so other modules
can react (play sound, update move history) instead of relying on globals.
- Add a configuration object to the public init function to avoid hardcoded
constants (colors, cubie size, padding) inside the module.
- Add optional VR/AR hooks to visualize the cube in immersive contexts.

------------------------------------------------------------------

Copyright (c) 2025 Facundo Gauna & Ulises Carnevale. MIT License.
*/

function initRubik3D(container) {
  // ---------- Scene ----------
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera.position.set(5, 5, 5);

  const canvas = document.createElement('canvas');
  let context = canvas.getContext && canvas.getContext('webgl2', {
    antialias: false,           
    alpha: true,              
    powerPreference: 'high-performance',
    preserveDrawingBuffer: false
  });
  
  // webgl2 fallback a webgl
  if (!context) {
    context = canvas.getContext('webgl', {
      antialias: false,
      alpha: true,
      powerPreference: 'high-performance'
    });
  }

  const renderer = new THREE.WebGLRenderer({ canvas,context });
  renderer.setSize(container.clientWidth, container.clientHeight);
  
  const MAX_PIXEL_RATIO = 1.5; // 1.5-2
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO));
  renderer.setClearColor(0x0d1117, 0);
  renderer.autoClear = true;
  renderer.sortObjects = false;
  renderer.shadowMap.enabled = false;  


  container.appendChild(renderer.domElement);
  renderer.domElement.style.display = 'block';
  renderer.domElement.style.width = '100%';
  renderer.domElement.style.height = '100%';
  renderer.domElement.style.background = 'transparent';

  window.rubikRenderer = renderer;

  // ---------- Lights ----------
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));
  const hemi = new THREE.HemisphereLight(0xe0f7ff, 0xffe8c0, 0.5);
  scene.add(hemi);

  const dir1 = new THREE.DirectionalLight(0xffffff, 0.8);
  dir1.position.set(5, 8, 5);

  const dir2 = new THREE.DirectionalLight(0xffe8c0, 0.4);
  dir2.position.set(-5, 8, -5);

  const dir3 = new THREE.DirectionalLight(0xc0d8ff, 0.4);
  dir3.position.set(5, -3, -5);

  scene.add(dir1, dir2, dir3);

  // ---------- Controls ----------
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.enableZoom = false;
  controls.target.set(0, 0, 0);
  controls.update();

  const BASE_CUBE_GEOMETRY = new THREE.BoxGeometry(0.95, 0.95, 0.95, 1, 1, 1);
  const STICKER_GEOMETRY = new THREE.BoxGeometry(0.84, 0.84, 0.02, 1, 1, 1);

  // ---------- Materials ----------
  const PLASTIC_BASE  = (color) => new THREE.MeshStandardMaterial({
    color,
    roughness: 0.35,
    metalness: 0.05,
  });

  const BASE_COLOR_HEX = {
    W: 0xffffff,
    Y: 0xffff00,
    R: 0xff0000,
    O: 0xff6600,
    B: 0x0000ff,
    G: 0x00aa00,
  };

  const BASE_MATERIALS = {
    W: PLASTIC_BASE(BASE_COLOR_HEX.W),
    Y: PLASTIC_BASE(BASE_COLOR_HEX.Y),
    R: PLASTIC_BASE(BASE_COLOR_HEX.R),
    O: PLASTIC_BASE(BASE_COLOR_HEX.O),
    B: PLASTIC_BASE(BASE_COLOR_HEX.B),
    G: PLASTIC_BASE(BASE_COLOR_HEX.G),
  };

  const BLACK_PLASTIC = new THREE.MeshPhysicalMaterial({
    color: 0x000000,
    roughness: 0.4,
    metalness: 0.2,
    clearcoat: 1,
    clearcoatRoughness: 0.3,
  });

  // ---------- Create cubelets ----------
  const cubelets = [];
  const STEP = 0.95;
  let _stickerIdCounter = 0;

  const stickerMeshes = []; // cache of sticker meshes for faster traversal

  function makeSticker(material, face) {
    const mesh = new THREE.Mesh(STICKER_GEOMETRY, material);
    mesh.userData.face = face;
    mesh.userData.stickerId = _stickerIdCounter++;
    stickerMeshes.push(mesh);
    return mesh;
  }

  for (let x = -1; x <= 1; x++) {
    for (let y = -1; y <= 1; y++) {
      for (let z = -1; z <= 1; z++) {
        const base = new THREE.Mesh(BASE_CUBE_GEOMETRY, BLACK_PLASTIC );
        base.position.set(x * STEP, y * STEP, z * STEP);

        // store initial transforms for reset
        base.userData.initialPosition = base.position.clone();
        base.userData.initialRotation = base.rotation.clone();

        if (y === 1) {
          const s = makeSticker(BASE_MATERIALS.B, 'U');
          s.rotation.x = -Math.PI / 2;
          s.position.y = 0.485;
          base.add(s);
        }
        if (y === -1) {
          const s = makeSticker(BASE_MATERIALS.G, 'D');
          s.rotation.x = Math.PI / 2;
          s.position.y = -0.485;
          base.add(s);
        }
        if (z === 1) {
          const s = makeSticker(BASE_MATERIALS.Y, 'F');
          s.position.z = 0.485;
          base.add(s);
        }
        if (z === -1) {
          const s = makeSticker(BASE_MATERIALS.W, 'B');
          s.rotation.y = Math.PI;
          s.position.z = -0.485;
          base.add(s);
        }
        if (x === 1) {
          const s = makeSticker(BASE_MATERIALS.O, 'R');
          s.rotation.y = Math.PI / 2;
          s.position.x = 0.485;
          base.add(s);
        }
        if (x === -1) {
          const s = makeSticker(BASE_MATERIALS.R, 'L');
          s.rotation.y = -Math.PI / 2;
          s.position.x = -0.485;
          base.add(s);
        }

        scene.add(base);
        cubelets.push(base);
      }
    }
  }

  // Original mapping array kept for compatibility with backend mapping logic
  const ORIGINAL_ORDER = [
    18, 16, 13, 31, 30, 28, 51, 49, 46,
    19, 32, 52, 11, 27, 44, 6, 25, 39,
    15, 17, 20, 9, 10, 12, 2, 4, 7,
    33, 36, 38, 21, 23, 24, 0, 3, 5,
    40, 37, 35, 45, 43, 42, 53, 50, 48,
    34, 22, 1, 41, 26, 8, 47, 29, 14
  ];

  // cached move map used by move/sequence/drawCube functions
  const MOVES_MAP = {
    "D": { axis: "y", index: -1, dir: 1 },
    "D'": { axis: "y", index: -1, dir: -1 },
    "D2": { axis: "y", index: -1, dir: 2 },
    "F": { axis: "z", index: 1, dir: -1 },
    "F'": { axis: "z", index: 1, dir: 1 },
    "F2": { axis: "z", index: 1, dir: 2 },
    "B": { axis: "z", index: -1, dir: 1 },
    "B'": { axis: "z", index: -1, dir: -1 },
    "B2": { axis: "z", index: -1, dir: 2 },
    "R": { axis: "x", index: 1, dir: -1 },
    "R'": { axis: "x", index: 1, dir: 1 },
    "R2": { axis: "x", index: 1, dir: 2 },
    "L": { axis: "x", index: -1, dir: 1 },
    "L'": { axis: "x", index: -1, dir: -1 },
    "L2": { axis: "x", index: -1, dir: 2 },
    "U": { axis: "y", index: 1, dir: -1 },
    "U'": { axis: "y", index: 1, dir: 1 },
    "U2": { axis: "y", index: 1, dir: 2 },
  };

  const FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B'];

  function computeKociembaStickerIds() {
    const facesCollected = { U: [], R: [], F: [], D: [], L: [], B: [] };

    for (const sticker of stickerMeshes) {
      const parent = sticker.parent;
      if (!parent) continue;
      const face = sticker.userData.face || inferFromParentPosition(parent.position);
      // use parent's grid coords to compute u/v easily
      const p = parent.position;
      // compute u,v scalar depending on face (use parent's x,y,z)
      let u = 0, v = 0;
      switch (face) {
        case 'U': u = p.x; v = -p.z; break;
        case 'D': u = p.x; v = p.z; break;
        case 'F': u = p.x; v = -p.y; break;
        case 'B': u = -p.x; v = -p.y; break;
        case 'R': u = p.z; v = -p.y; break;
        case 'L': u = -p.z; v = -p.y; break;
      }
      facesCollected[face].push({ id: sticker.userData.stickerId, u, v });
    }

    const result = [];
    for (const f of FACE_ORDER) {
      const list = facesCollected[f] || [];
      list.sort((A,B) => {
        const dv = B.v - A.v;
        if (Math.abs(dv) > 1e-6) return dv;
        return A.u - B.u;
      });
      if (list.length !== 9) console.warn(`Face ${f} has ${list.length} stickers`);
      list.forEach(x => result.push(x.id));
    }
    return result;
  }

  // initial computation of kociembaIDs (once)
  let kociembaIDs = computeKociembaStickerIds();
  if (!kociembaIDs || kociembaIDs.length !== 54) {
    console.warn("kociembaIDs length unexpected:", kociembaIDs && kociembaIDs.length);
  }

  // permutation mapping 
  const permutation = new Array(54).fill(-1);
  for (let j = 0; j < ORIGINAL_ORDER.length && j < 54; j++) {
    const stickerId = ORIGINAL_ORDER[j];
    const i = kociembaIDs.indexOf(stickerId);
    permutation[j] = (i >= 0 ? i : -1);
  }

  // ---------- Rendering loop ----------
  function renderLoop() {
    requestAnimationFrame(renderLoop);
    controls.update();
    renderer.render(scene, camera);
  }
  renderLoop();

  // resize handling (debounced simple approach)
  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }, 100);
  });

  // ---------- Rotation animation helper and move handling ----------
  let busy = false;
  let sequenceRunning = false;
  let DURATION = 400;
  let rubik_keys_attached = true;

  // Keydown handler for manual cube turns
  window.addEventListener('keydown', async (e) => {
    if (!rubik_keys_attached) return;
    if (busy || sequenceRunning) return;

    let notation = e.key.toUpperCase();
    if (!notation) return;

    const validKeys = ["F", "B", "D", "R", "L", "U"];
    if (!validKeys.includes(notation)) return;

    if (e.shiftKey) notation += "'";

    await sequence(notation);
  });


  function turnKeys(enable) {
    if (enable) {
      rubik_keys_attached = true;
      console.log("[rubik] manual keys ENABLED");
    } else {
      rubik_keys_attached = false;
      console.log("[rubik] manual keys DISABLED");
    }
  }

  // animation routine: returns a Promise that resolves after animation finishes
  function animateRotation(group, axis, start, target, duration) {
    return new Promise(resolve => {
      const t0 = performance.now();
      (function anim(now) {
        const t = Math.min(1, (now - t0) / duration);
        const eased = 1 - Math.pow(1 - t, 3);
        group.rotation[axis] = start + (target - start) * eased;
        if (t < 1) requestAnimationFrame(anim);
        else {
          group.rotation[axis] = target;
          resolve();
        }
      })(t0);
    });
  }

  const pi_div2 = Math.PI / 2;

  async function move(notation) {
    if (busy) return;
    busy = true;
    console.log("rotating:", notation);

    // ensure backend call is a Promise; wait for backend before animating for lock-step
    let backendPromise = Promise.resolve();
    try {
      backendPromise = window.pywebview.api.send_sequence(notation);
    } catch (e) {
      console.warn("pywebview API send_sequence error:", e);
      backendPromise = Promise.resolve();
    }

    const moveSpec = MOVES_MAP[notation];
    if (!moveSpec) {
      busy = false;
      return;
    }
    const {
      axis,
      index,
      dir
    } = moveSpec;

    const group = new THREE.Group();
    scene.add(group);

    cubelets.forEach(c => {
      if (Math.round(c.position[axis] / STEP) === index) group.attach(c);
    });
    const start = group.rotation[axis];
    const target = start + (dir === 2 ? Math.PI : dir * pi_div2);

    // Wait for backend confirmation (lock-step)
    try {
      await backendPromise;
    } catch (e) {
      console.warn("Backend rejected or errored move; continuing animation locally:", e);
    }

    // perform animation (global render loop will render frames)
    await animateRotation(group, axis, start, target, DURATION);

    // finalize: detach children and normalize transforms
    const children = [...group.children];
    children.forEach(ch => {
      ch.updateMatrixWorld(true);
      scene.attach(ch);

      ch.position.x = Math.round(ch.position.x / STEP) * STEP;
      ch.position.y = Math.round(ch.position.y / STEP) * STEP;
      ch.position.z = Math.round(ch.position.z / STEP) * STEP;

      ch.rotation.x = Math.round(ch.rotation.x / (pi_div2)) * (pi_div2);
      ch.rotation.y = Math.round(ch.rotation.y / (pi_div2)) * (pi_div2);
      ch.rotation.z = Math.round(ch.rotation.z / (pi_div2)) * (pi_div2);

      ch.updateMatrixWorld(true);
    });

    scene.remove(group);
    busy = false;
  }
  const sequencePanel = document.getElementById('sequence-summary');
  
  // Half replacement of U movement.
  // U: replaceOfU + D + replaceOfU
  // U': replaceOfU + D' + replaceOfU
  // U2: replaceOfU + D2 + replaceOfU
  const replaceOfU = ["L","R","B2","F2","L'","R'"];

  async function sequence(sequenceStr) {
    if (sequenceRunning) return;
    sequenceRunning = true;
    if (busy) {
      sequenceRunning = false;
      return;
    }

    console.log("sequence:", sequenceStr);
    sequencePanel.textContent = sequenceStr || '—';

    const moves = sequenceStr.trim().split(/\s+/).filter(Boolean);

    if(!window.UsageU){
      for (let mv of moves) {
        if (mv[0] == 'U') {
          for (let s of replaceOfU) await move(s);
          await move("D" + (mv.length > 1 ? mv[1] : ""));
          for (let s of replaceOfU) await move(s);
        } else {
          await move(mv);
        }
      }
    }else{
      for (let mv of moves) {await move(mv);}
    }

    sequenceRunning = false;
  }

  async function drawCube(sequenceStr) {
    // instant (non-animated) application of moves to show a static resulting state
    sequencePanel.textContent = sequenceStr || '—';
    const moves = sequenceStr.trim().split(/\s+/);
    for (const notation of moves) {
      const moveSpec = MOVES_MAP[notation];
      if (!moveSpec) continue;
      const {
        axis,
        index,
        dir
      } = moveSpec;
      const group = new THREE.Group();
      scene.add(group);

      cubelets.forEach(c => {
        if (Math.round(c.position[axis] / STEP) === index) group.attach(c);
      });
      const angle = dir === 2 ? Math.PI : dir * pi_div2;
      group.rotation[axis] += angle;

      const children = [...group.children];
      children.forEach(ch => {
        ch.updateMatrixWorld(true);
        scene.attach(ch);
        ch.position.x = Math.round(ch.position.x / STEP) * STEP;
        ch.position.y = Math.round(ch.position.y / STEP) * STEP;
        ch.position.z = Math.round(ch.position.z / STEP) * STEP;

        ch.rotation.x = Math.round(ch.rotation.x / (pi_div2)) * pi_div2;
        ch.rotation.y = Math.round(ch.rotation.y / (pi_div2)) * pi_div2;
        ch.rotation.z = Math.round(ch.rotation.z / (pi_div2)) * pi_div2;
        ch.updateMatrixWorld(true);
      });

      scene.remove(group);
    }
    renderer.render(scene, camera);
  }

  function invert(seq) {
    const moves = seq.trim().split(/\s+/).reverse();
    return moves.map(move => {
      if (move.endsWith("2")) return move;
      else if (move.endsWith("'")) return move.slice(0, -1);
      else return move + "'";
    }).join(' ');
  }

  async function customMoves(movesStr) {
    if (!movesStr || typeof movesStr !== 'string') {
      console.warn("customMoves: invalid input");
      return;
    }
    const customSeq = movesStr.trim().split(/\s+/)

    if (customSeq.length == 0){
      sequencePanel.textContent = '—';
      return;
    }

    for(let s of customSeq){
      if (!MOVES_MAP[s]){
        sequencePanel.textContent = "invalid notation: "+ s +" (Valid: U D L R F B with optional ' or 2)";
        console.warn("customMoves: invalid notation:", s);
        return;
      }
    }

    await sequence(movesStr);
  }

  async function resetCube() {
    busy = true;
    sequenceRunning = false;
    try {
      await window.pywebview.api.reset_cube_state();
    } catch (e) {
      // ignore if API not available
    }

    // reset visual cube to initial transforms
    cubelets.forEach(c => {
      c.position.copy(c.userData.initialPosition);
      c.rotation.copy(c.userData.initialRotation);
      c.updateMatrixWorld(true);
    });

    turnKeys(true);
    busy = false;
  }

  const rubik = {
    busy,
    sequenceRunning,

    setMoveDuration: (ms) => {
      DURATION = Math.max(10, Number(ms) || DURATION);
    },
    getMoveDuration: () => DURATION,

    sequence,
    resetCube,
    drawCube,
    customMoves,

    invert,
    turnKeys,
  };

  return rubik;
}