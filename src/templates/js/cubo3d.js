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
  controls.enablePan = false;    // ❌ No se puede mover
  controls.enableZoom = false;   // ❌ Opcional: no se puede hacer zoom
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
    Blanco: 0xffffff,
    Amarillo: 0xffff00,
    Rojo: 0xff0000,
    Naranja: 0xff6600,
    Azul: 0x0000ff,
    Verde: 0x00aa00,
  };

  const ColoresBase = {
    Blanco: plasticSticker(0xffffff),
    Amarillo: plasticSticker(0xffff00),
    Rojo: plasticSticker(0xff0000),
    Naranja: plasticSticker(0xff6600),
    Azul: plasticSticker(0x0000ff),
    Verde: plasticSticker(0x00aa00),
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
    const mesh = new THREE.Mesh(geo, material.clone()); // <- clonar material
    mesh.castShadow = false;
    mesh.userData.face = face;
    mesh.userData.stickerId = stickerIdCounter++;
    return mesh;
}



  for(let x=-1;x<=1;x++){
    for(let y=-1;y<=1;y++){
      for(let z=-1;z<=1;z++){
        const baseGeo = new THREE.BoxGeometry(0.95,0.95,0.95);
        const base = new THREE.Mesh(baseGeo, blackPlastic);
        base.position.set(x*STEP, y*STEP, z*STEP);

        if (y === 1) { const s = makeSticker(ColoresBase.Blanco, 'U'); s.rotation.x=-Math.PI/2; s.position.y=0.485; base.add(s);}
        if (y === -1){ const s = makeSticker(ColoresBase.Amarillo, 'D'); s.rotation.x=Math.PI/2; s.position.y=-0.485; base.add(s);}
        if (z === 1) { const s = makeSticker(ColoresBase.Rojo, 'F'); s.position.z=0.485; base.add(s);}
        if (z === -1){ const s = makeSticker(ColoresBase.Naranja, 'B'); s.rotation.y=Math.PI; s.position.z=-0.485; base.add(s);}
        if (x === 1) { const s = makeSticker(ColoresBase.Azul, 'R'); s.rotation.y=Math.PI/2; s.position.x=0.485; base.add(s);}
        if (x === -1){ const s = makeSticker(ColoresBase.Verde, 'L'); s.rotation.y=-Math.PI/2; s.position.x=-0.485; base.add(s);}


        scene.add(base);
        cubelets.push(base);
      }
    }
  }

  // ---------- Loop ----------
  function loop(){
    requestAnimationFrame(loop);
    controls.update();
    renderer.render(scene,camera);
  }
  loop();

  // ---------- Ajuste al tamaño del contenedor ----------
  window.addEventListener('resize', ()=>{
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  // ---------- Rotación de las caras ----------
  let busy=false;
  const DURATION=100;
  const easeOut=t=>1-Math.pow(1-t,3);

  function RotarCara(notation) {
    if(busy) return;
    busy = true;

    const map = {
        "U": { axis: "y", index: 1, dir: 1 },
        "U'": { axis: "y", index: 1, dir: -1 },
        "U2": { axis: "y", index: 1, dir: 2 },

        "D": { axis: "y", index: -1, dir: -1 },
        "D'": { axis: "y", index: -1, dir: 1 },
        "D2": { axis: "y", index: -1, dir: 2 },

        "F": { axis: "z", index: 1, dir: 1 },
        "F'": { axis: "z", index: 1, dir: -1 },
        "F2": { axis: "z", index: 1, dir: 2 },

        "B": { axis: "z", index: -1, dir: -1 },
        "B'": { axis: "z", index: -1, dir: 1 },
        "B2": { axis: "z", index: -1, dir: 2 },

        "R": { axis: "x", index: 1, dir: 1 },
        "R'": { axis: "x", index: 1, dir: -1 },
        "R2": { axis: "x", index: 1, dir: 2 },

        "L": { axis: "x", index: -1, dir: -1 },
        "L'": { axis: "x", index: -1, dir: 1 },
        "L2": { axis: "x", index: -1, dir: 2 },
    };

    const move = map[notation];
    if (!move) { console.error("Notación inválida:", notation); busy = false; return; }

    const { axis, index, dir } = move;
    const group = new THREE.Group();
    scene.add(group);

    cubelets.forEach(c => {
        if (Math.round(c.position[axis] / STEP) === index) group.attach(c);
    });

    const start = group.rotation[axis];
    const target = start + (dir === 2 ? Math.PI : dir * Math.PI / 2); // 2 significa 180º
    const t0 = performance.now();

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
        }
    }

    requestAnimationFrame(anim);
  }

  // Función para ejecutar una secuencia de movimientos
  function SecuenciaDeGiros(sequenceStr) {
    if (busy) return; // Si ya hay un movimiento, no hacer nada
    const moves = sequenceStr.split(/\s+/); // Separa por espacios

    let i = 0;

    function nextMove() {
        if (i >= moves.length) return; // Terminar secuencia
        RotarCara(moves[i]);     // Ejecuta el movimiento actual

        // Espera hasta que termine el movimiento antes de pasar al siguiente
        const check = setInterval(() => {
            if (!busy) {
                clearInterval(check);
                i++;
                nextMove();
            }
        }, 20);
    }

    nextMove();
  }



// ---------- Controles de la rotación de las caras ----------
  window.addEventListener('keydown', e => {
    if (busy) return;

    let notation = '';
    switch (e.key.toUpperCase()) {
        case 'U': notation = 'U'; break;
        case 'D': notation = 'D'; break;
        case 'L': notation = 'L'; break;
        case 'R': notation = 'R'; break;
        case 'F': notation = 'F'; break;
        case 'B': notation = 'B'; break;
    }

    if (notation) {
        // Si se mantiene Shift, hacemos el movimiento invertido
        if (e.shiftKey) notation += "'";
        RotarCara(notation); // Ejecuta el movimiento usando la secuencia
    }
  });

  // ---------- Cambiar colores ----------
  function CambiarColorSticker(id, colorHex){
    for(const cubito of cubelets){
        const sticker = cubito.children.find(s => s.userData.stickerId === id);
        if(sticker){
            sticker.material.color.setHex(colorHex);
            return true; // encontrado y cambiado
        }
    }
    return false; // no se encontró el ID
  }

 function CambiarColoresDelCubo(coloresArray) {
    const orden = [
        18,16,13,31,30,28,51,49,46,
        19,32,52,11,27,44,6,25,39,
        15,17,20,9,10,12,2,4,7,
        33,36,38,21,23,24,0,3,5,
        40,37,35,45,43,42,53,50,48,
        34,22,1,41,26,8,47,29,14
    ];

    if (!Array.isArray(coloresArray) || coloresArray.length !== 54) {
        console.error("El array debe contener exactamente 54 colores.");
        return;
    }

    // Aplicar los colores según el orden especificado
    coloresArray.forEach((colorHex, index) => {
        const stickerId = orden[index];
        CambiarColorSticker(stickerId, colorHex);
    });
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

  // Exponer funciones que queremos que el proyecto principal use
  return { cubelets, CambiarColoresDelCubo, ColoresBaseHex, RotarCara, SecuenciaDeGiros, get busy() { return busy; } };
}


