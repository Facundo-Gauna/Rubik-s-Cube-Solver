// cube3d.js
// Inicializa el cubo 3D con funciones para pintar y rotar.
// Cambios importantes: mapeo robusto Kociemba -> orden interno (stickerId).

function initRubik3D(container) {
  // ---------- Escena ----------
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera.position.set(5,5,5);

  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0xffffff, 1);
  container.appendChild(renderer.domElement);

  // ---------- Luces ----------
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));
  const hemi = new THREE.HemisphereLight(0xe0f7ff, 0xffe8c0, 0.5);
  scene.add(hemi);

  const dir1 = new THREE.DirectionalLight(0xffffff, 0.8);
  dir1.position.set(5,8,5);
  const dir2 = new THREE.DirectionalLight(0xffe8c0,0.4);
  dir2.position.set(-5,8,-5);
  const dir3 = new THREE.DirectionalLight(0xc0d8ff,0.4);
  dir3.position.set(5,-3,-5);
  scene.add(dir1,dir2,dir3);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.enableZoom = false;
  controls.target.set(0, 0, 0);
  controls.update();

  // ---------- Materiales ----------
  const plasticSticker = (color) => new THREE.MeshPhysicalMaterial({
    color,
    roughness: 0.25,
    metalness: 0.05,
    clearcoat: 1,
    clearcoatRoughness: 0.1,
    reflectivity: 0.6,
  });

  const ColoresBaseHex = {
    W: 0xffffff,
    Y: 0xffff00,
    R: 0xff0000,
    O: 0xff6600,
    B: 0x0000ff,
    G: 0x00aa00,
  };

  const ColoresBase = {
    W: plasticSticker(ColoresBaseHex.W),
    Y: plasticSticker(ColoresBaseHex.Y),
    R: plasticSticker(ColoresBaseHex.R),
    O: plasticSticker(ColoresBaseHex.O),
    B: plasticSticker(ColoresBaseHex.B),
    G: plasticSticker(ColoresBaseHex.G),
  };

  const blackPlastic = new THREE.MeshPhysicalMaterial({
    color: 0x000000,
    roughness: 0.4,
    metalness: 0.2,
    clearcoat: 1,
    clearcoatRoughness: 0.3,
  });

  // ---------- Crear cubitos ----------
  const cubelets = [];
  const STEP = 0.95;
  let stickerIdCounter = 0;

  function makeSticker(material, face){
    const geo = new THREE.BoxGeometry(0.84,0.84,0.02,8,8,1);
    const mesh = new THREE.Mesh(geo, material.clone());
    mesh.castShadow = false;
    mesh.userData.face = face;            // útil como ayuda, pero NO fiarse sólo de esto
    mesh.userData.stickerId = stickerIdCounter++;
    return mesh;
  }

  for(let x=-1;x<=1;x++){
    for(let y=-1;y<=1;y++){
      for(let z=-1;z<=1;z++){
        const baseGeo = new THREE.BoxGeometry(0.95,0.95,0.95);
        const base = new THREE.Mesh(baseGeo, blackPlastic);
        base.position.set(x*STEP, y*STEP, z*STEP);

        // 🔹 Guardar posición y rotación inicial
        base.userData.initialPosition = base.position.clone();
        base.userData.initialRotation = base.rotation.clone();

        if (y === 1) { const s = makeSticker(ColoresBase.B, 'U'); s.rotation.x=-Math.PI/2; s.position.y=0.485; base.add(s);}
        if (y === -1){ const s = makeSticker(ColoresBase.G , 'D'); s.rotation.x=Math.PI/2; s.position.y=-0.485; base.add(s);}
        if (z === 1) { const s = makeSticker(ColoresBase.Y, 'F'); s.position.z=0.485; base.add(s);}
        if (z === -1){ const s = makeSticker(ColoresBase.W, 'B'); s.rotation.y=Math.PI; s.position.z=-0.485; base.add(s);}
        if (x === 1) { const s = makeSticker(ColoresBase.O, 'R'); s.rotation.y=Math.PI/2; s.position.x=0.485; base.add(s);}
        if (x === -1){ const s = makeSticker(ColoresBase.R, 'L'); s.rotation.y=-Math.PI/2; s.position.x=-0.485; base.add(s);}

        scene.add(base);
        cubelets.push(base);
      }
    }
  }

  const ORDEN = [
        18,16,13,31,30,28,51,49,46,
        19,32,52,11,27,44,6,25,39,
        15,17,20,9,10,12,2,4,7,
        33,36,38,21,23,24,0,3,5,
        40,37,35,45,43,42,53,50,48,
        34,22,1,41,26,8,47,29,14 ];
  
  const faceNormals = {
    U: new THREE.Vector3(0, 1, 0),
    R: new THREE.Vector3(1, 0, 0),
    F: new THREE.Vector3(0, 0, 1),
    D: new THREE.Vector3(0, -1, 0),
    L: new THREE.Vector3(-1, 0, 0),
    B: new THREE.Vector3(0, 0, -1),
  };

  // función helper para generar ejes u (left->right) y v (top->bottom)
  function buildAxesForNormal(n) {
    // pick a world "up" that is not parallel to n
    let worldUp = new THREE.Vector3(0, 1, 0);
    if (Math.abs(worldUp.dot(n)) > 0.9) {
      worldUp.set(0, 0, 1);
    }
    // u = worldUp x n  (approx pointing left->right)
    const u = new THREE.Vector3().crossVectors(worldUp, n).normalize();
    // v = n x u  (top->bottom)
    const v = new THREE.Vector3().crossVectors(n, u).normalize();
    return { u, v };
  }

  function computeKociembaStickerIds() {
    const faces = { U:[], R:[], F:[], D:[], L:[], B:[] };
    const tmpPos = new THREE.Vector3();
    const tmpQuat = new THREE.Quaternion();
    const normalWorld = new THREE.Vector3();

    // recolectar stickers
    scene.traverse(obj => {
      if (!obj.isMesh) return;
      if (obj.userData && obj.userData.stickerId !== undefined) {
        obj.getWorldPosition(tmpPos);
        obj.getWorldQuaternion(tmpQuat);
        normalWorld.set(0,0,1).applyQuaternion(tmpQuat).normalize();
        // identificar cara por la normal
        let face;
        const ax = Math.abs(normalWorld.x), ay = Math.abs(normalWorld.y), az = Math.abs(normalWorld.z);
        if (ay >= ax && ay >= az) {
          face = normalWorld.y > 0 ? 'U' : 'D';
        } else if (ax >= ay && ax >= az) {
          face = normalWorld.x > 0 ? 'R' : 'L';
        } else {
          face = normalWorld.z > 0 ? 'F' : 'B';
        }
        faces[face].push({
          id: obj.userData.stickerId,
          pos: tmpPos.clone(),
          normal: normalWorld.clone()
        });
      }
    });

    const FACE_ORDER = ['U','R','F','D','L','B'];
    const resultIds = [];

    // build axes on the fly and sort each face using projected coordinates
    for (const f of FACE_ORDER) {
      const n = faceNormals[f];
      const { u, v } = buildAxesForNormal(n);

      const list = faces[f].map(item => {
        const projU = item.pos.dot(u);
        const projV = item.pos.dot(v);
        return { id: item.id, u: projU, v: projV };
      });

      // sort: v desc (top first), then u asc (left first)
      list.sort((A,B) => {
        const dv = B.v - A.v;
        if (Math.abs(dv) > 1e-6) return dv;
        return A.u - B.u;
      });

      if (list.length !== 9) {
        console.warn(`Face ${f} tiene ${list.length} stickers (esperado 9).`, list.map(x=>x.id));
      }
      list.forEach(x => resultIds.push(x.id));
    }

    return resultIds;
  }

  // calcular kociembaIDs y la permutación (como antes)
  let kociembaIDs = computeKociembaStickerIds(); // array length 54
  if (!kociembaIDs || kociembaIDs.length !== 54) {
    console.warn("kociembaIDs length unexpected:", kociembaIDs && kociembaIDs.length);
  }
  const permutation = new Array(54).fill(-1);
  for (let j = 0; j < ORDEN.length && j < 54; j++) {
    const stickerId = ORDEN[j];
    const i = kociembaIDs.indexOf(stickerId);
    permutation[j] = (i >= 0 ? i : -1);
  }
  // ---------- Loop ----------
  function loop(){
    requestAnimationFrame(loop);
    controls.update();
    renderer.render(scene,camera);
  }
  loop();

  // resize handler
  window.addEventListener('resize', ()=>{
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  // ---------- Rotaciones (idéntico a tu implementación) ----------
  let busy=false;
  let sequenceRunning = false;
  const DURATION=400;
  const easeOut=t=>1-Math.pow(1-t,3);

  function handleKeyDown(e) {
      if (busy || sequenceRunning) return;
      let notation = e.key.toUpperCase();
      if (!notation) return;
      if (e.shiftKey) notation += "'";
      RotarCara(notation);
  }
  window.addEventListener('keydown', handleKeyDown);

  async function RotarCara(notation) {
    if(busy) return;
    window.removeEventListener('keydown', handleKeyDown);
    busy = true;

    if(notation == "U") { busy = false; await SecuenciaDeGiros("L R B2 F2 L' R' D L R B2 F2 L' R'");  window.addEventListener('keydown', handleKeyDown); return; }
    else if(notation == "U'") { busy = false; await SecuenciaDeGiros("L' R' B2 F2 L R D' L' R' B2 F2 L R");  window.addEventListener('keydown', handleKeyDown); return; }
    else if(notation == "U2") { busy = false; await SecuenciaDeGiros("L R B2 F2 L' R' D2 L R B2 F2 L' R'"); window.addEventListener('keydown', handleKeyDown); return; }

    let backendPromise = window.pywebview.api.send_sequence(notation);

    const map = {
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
    };

    const move = map[notation];
    if (!move) { busy = false;  window.addEventListener('keydown', handleKeyDown); return; }

    const { axis, index, dir } = move;
    const group = new THREE.Group();
    scene.add(group);

    cubelets.forEach(c => {
        if (Math.round(c.position[axis] / STEP) === index) group.attach(c);
    });

    const start = group.rotation[axis];
    const target = start + (dir === 2 ? Math.PI : dir * Math.PI / 2);
    const t0 = performance.now();

    const animPromise = new Promise(resolve => {
      function anim(now) {
          const t = Math.min(1, (now - t0) / DURATION);
          group.rotation[axis] = start + (target - start) * easeOut(t);
          renderer.render(scene, camera);
          if (t < 1) requestAnimationFrame(anim);
          else {
              group.rotation[axis] = target;
              const hijos = [...group.children];
              hijos.forEach(ch => {
                  ch.updateMatrixWorld(true);
                  scene.attach(ch);
                  ch.position.x = Math.round(ch.position.x / STEP) * STEP;
                  ch.position.y = Math.round(ch.position.y / STEP) * STEP;
                  ch.position.z = Math.round(ch.position.z / STEP) * STEP;
              });
              scene.remove(group);
              busy = false;
              window.addEventListener('keydown', handleKeyDown);
              resolve();
          }
      }
      requestAnimationFrame(anim);
  });

    await Promise.all([backendPromise, animPromise]);
    window.addEventListener('keydown', handleKeyDown);
  }

  
  async function SecuenciaDeGiros(sequenceStr) {
    if (sequenceRunning) return;
    sequenceRunning = true;
    if (busy) { sequenceRunning = false; return; }

    const moves = sequenceStr.split(/\s+/);
    let i = 0;

    async function nextMove() {
        if (i >= moves.length) {
            sequenceRunning = false;
            return;
        }
        await RotarCara(moves[i]);
        const check = setInterval(() => {
            if (!busy) {
                clearInterval(check);
                i++;
                nextMove();
            }
        }, 20);
    }

    await nextMove();
  }
  

  async function SecuenciaInstantanea(sequenceStr) {
    // No usar busy ni API ni animaciones
    const moves = sequenceStr.trim().split(/\s+/);

    const map = {
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

    for (const notation of moves) {
      const move = map[notation];
      if (!move) continue;

      const { axis, index, dir } = move;

      // Crear grupo temporal para rotar instantáneamente
      const group = new THREE.Group();
      scene.add(group);

      cubelets.forEach(c => {
        if (Math.round(c.position[axis] / STEP) === index) group.attach(c);
      });

      // Aplicar rotación directa (sin animación)
      const angle = dir === 2 ? Math.PI : dir * Math.PI / 2;
      group.rotation[axis] += angle;

      // Actualizar posiciones y desprender los cubies del grupo
      const hijos = [...group.children];
      hijos.forEach(ch => {
        ch.updateMatrixWorld(true);
        scene.attach(ch);
        ch.position.x = Math.round(ch.position.x / STEP) * STEP;
        ch.position.y = Math.round(ch.position.y / STEP) * STEP;
        ch.position.z = Math.round(ch.position.z / STEP) * STEP;
      });

      scene.remove(group);
    }

    renderer.render(scene, camera);
  }

  // ---------- Cambiar colores ----------
  function CambiarColorSticker(id, colorHex){
    for(const cubito of cubelets){
        const sticker = cubito.children.find(s => s.userData.stickerId === id);
        if(sticker){
            sticker.material.color.setHex(colorHex);
            return true;
        }
    }
    return false;
  }

  function CambiarColoresDelCubo(coloresArray) {
    if (!Array.isArray(coloresArray) || coloresArray.length !== 54) {
        console.error("CambiarColoresDelCubo: El array debe contener exactamente 54 colores.");
        return;
    }
    coloresArray.forEach((colorHex, index) => {
        const stickerId = ORDEN[index];
        CambiarColorSticker(stickerId, colorHex);
    });
    ResetearCubo();
  }

  function ResetearCubo() {
  busy = true;
  sequenceRunning = false;

  cubelets.forEach(c => {
    // Restaurar posición y rotación
    c.position.copy(c.userData.initialPosition);
    c.rotation.copy(c.userData.initialRotation);
    c.updateMatrixWorld(true);
  });

  busy = false;
  }

  function applyFacesTo3D(faces) {
    console.log("Faces str : ",faces)
    const FACE_ORDER = {'U':'B','F':'Y','L':'R','D':'G','R':'O','B':'W'};

    const listaFromColorStr = new Array(54);
    for (let i = 0; i < 54; i++) {
      const F = faces[i].toUpperCase();
      listaFromColorStr[i] = ColoresBaseHex[FACE_ORDER[F]] ?? 0xffffff;
    }
    try {
      CambiarColoresDelCubo(listaFromColorStr);
      return true;
    } catch (e) {
      console.error("applyColorStrTo3D -> CambiarColoresDelCubo failed:", e);
      return false;
    }
  }
  function applyColorStrTo3D(colorStr) {
    console.log("Color str : ",colorStr)
    
    const desiredFaceLetter = {
      U: 'B', 
      R: 'O',
      F: 'Y',
      D: 'G',
      L: 'R',
      B: 'W'
    };

    // índices centrales en el orden Kociemba (caras U,R,F,D,L,B)
    const centerIndices = [4, 13, 22, 31, 40, 49];
    const FACE_ORDER = ['U','F','L','D','R','B'];

    // 1) Inferir qué letra usa el backend para cada cara, mirando los centros
    const translation = {}; // backendLetter -> desiredLetter
    for (let k = 0; k < FACE_ORDER.length; k++) {
      const face = FACE_ORDER[k];
      const idx = centerIndices[k];
      const backendLetter = colorStr[idx].toUpperCase();
      translation[backendLetter] = desiredFaceLetter[face];
    }

    // debug rápido (opcional)
    console.log("applyColorStrTo3D: inferred center translation:", translation);

    // 2) Construir listaFromColorStr remapeando cada letra según translation
    const listaFromColorStr = new Array(54);
    for (let i = 0; i < 54; i++) {
      const L = colorStr.charAt(i).toUpperCase();
      const mapped = translation[L] || L; // si no está en translation, dejarla tal cual
      listaFromColorStr[i] = ColoresBaseHex[mapped] ?? 0xffffff;
    }

    const coloresRemapped = new Array(54).fill(0xffffff);
    for (let j = 0; j < 54; j++) {
      const i = permutation[j];
      if (i >= 0 && i < 54) {
        coloresRemapped[j] = listaFromColorStr[i];
      } else {
        coloresRemapped[j] = 0xffffff;
      }
    }

    try {
      CambiarColoresDelCubo(coloresRemapped);
      return true;
    } catch (e) {
      console.error("applyColorStrTo3D -> CambiarColoresDelCubo failed:", e);
      return false;
    }
  }


  

  function getFaceStartIndex(face){
    switch(face){
      case 'U': return 0;
      case 'D': return 9;
      case 'F': return 18;
      case 'B': return 27;
      case 'R': return 36;
      case 'L': return 45;
    }
  }

  // ---------- Export API ----------
  const rubik = {
    cubelets,
    CambiarColoresDelCubo,
    ColoresBaseHex,
    RotarCara,
    SecuenciaDeGiros,
    applyColorStrTo3D,
    applyFacesTo3D,
    ResetearCubo,
    SecuenciaInstantanea,
    // debug: inspecciona en consola para verificar mapeo
    _debug: {
      ORDEN,
      kociembaIDs,
      permutation
    },
    get busy() { return busy; }
  };

  return rubik;
}
