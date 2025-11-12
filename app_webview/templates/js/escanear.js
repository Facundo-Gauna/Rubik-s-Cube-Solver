document.getElementById("scan-btn").addEventListener("click", showScanView);

function hasPywebview() {
  return typeof window.pywebview !== "undefined" && window.pywebview.api && typeof window.pywebview.api.get_frame === "function";
}

async function fetchPreviewFrame() {
  if (hasPywebview()) {
    try {
      return await window.pywebview.api.get_frame();
    } catch (e) {
      console.warn("Error al obtener frame:", e);
      return "";
    }
  } else {
    return "";
  }
}

let previewInterval = null;
function startPreviewLoop(previewImgElement) {
  stopPreviewLoop();
  (async function frameLoop() {
    const frame = await fetchPreviewFrame();
    if (previewImgElement) previewImgElement.src = frame || "";
    previewInterval = setTimeout(frameLoop, 100);
  })();
}
function stopPreviewLoop() {
  if (previewInterval) {
    clearTimeout(previewInterval);
    previewInterval = null;
  }
}

let photos = [null, null];
let selectedPhotoBox = 0;
let solution = "";

function drawPreviewPoints(imgElement) {
  if (!imgElement) return;
  const positionsObj = {
    "U1": [702, 276], "U2": [886, 185], "U3": [1038, 117],
    "U4": [851, 356], "U5": [1034, 270], "U6": [1193, 186],
    "U7": [1031, 476], "U8": [1215, 353], "U9": [1368, 265],
    "F1": [1112, 629], "F2": [1299, 503], "F3": [1447, 412],
    "F4": [1127, 804], "F5": [1285, 681], "F6": [1422, 581],
    "F7": [1132, 949], "F8": [1273, 825], "F9": [1403, 727],
    "L1": [628, 421], "L2": [768, 514], "L3": [940, 634],
    "L4": [668, 589], "L5": [800, 691], "L6": [962, 809],
    "L7": [705, 740], "L8": [825, 835], "L9": [988, 964]
  };
  const container = imgElement.parentElement;
  const containerWidth = container.clientWidth;
  const containerHeight = container.clientHeight;
  const scaleX = containerWidth / 1920;
  const scaleY = containerHeight / 1080;

  const prev = container.querySelectorAll(".point");
  prev.forEach(p => p.remove());

  const letterToCss1 = { "U":"#0000FF","F":"#FFFF00","L":"#FF0000" };
  const letterToCss2 = { "L":"#FFFFFF","F":"#FFA500","U":"#008000" };

  for (const key in positionsObj) {
    const [x, y] = positionsObj[key];
    const point = document.createElement("div");
    point.className = "point";
    point.style.position = "absolute";
    point.style.width = "14px";
    point.style.height = "14px";
    point.style.background = selectedPhotoBox == 0 ? letterToCss1[key.charAt(0)] || "#808080" : letterToCss2[key.charAt(0)] || "#808080" ;
    point.style.borderRadius = "50%";
    point.style.left = `${x * scaleX - 7}px`;
    point.style.top = `${y * scaleY - 7}px`;
    point.style.zIndex = "10";
    container.appendChild(point);
  }
}

function update_box(img,idx = null){
  if (idx == null) selectedPhotoBox = !selectedPhotoBox;
  else selectedPhotoBox = idx;
  drawPreviewPoints(img);
}

function showScanView() {
  photos = [null, null];

  panelTitle.textContent = "Escanear — Paso 1";
  panelContent.innerHTML = `
    <div class="camera-view" style="position:relative;">
      <img id="previewImg" alt="preview" style="width:100%; display:block;">
    </div>
    <button class="action-btn capture-btn" id="captureBtn">Capturar imagen</button>
    <div class="photos-container">
      <div class="photo-box selected" data-index="0"></div>
      <div class="photo-box" data-index="1"></div>
    </div>
    <button class="action-btn next-btn" id="step1Next" disabled>Siguiente</button>
  `;

  const previewImg = document.getElementById("previewImg");
  const captureBtn = document.getElementById("captureBtn");
  const photoBoxes = panelContent.querySelectorAll(".photo-box");
  const nextBtn = document.getElementById("step1Next");

  update_box(previewImg,0);
  const onPreviewLoadOrResize = () => drawPreviewPoints(previewImg);
  window.removeEventListener("resize", onPreviewLoadOrResize);
  window.addEventListener("resize", onPreviewLoadOrResize);

  startPreviewLoop(previewImg);

  photoBoxes.forEach(b => b.classList.remove("selected"));
  photoBoxes[0].classList.add("selected");

  photoBoxes.forEach(box => {
    box.addEventListener("click", () => {
      photoBoxes.forEach(b => b.classList.remove("selected"));
      box.classList.add("selected");
      update_box(previewImg,parseInt(box.dataset.index));
    });
  });

  captureBtn.addEventListener("click", async () => {
    let res;
    try {
      res = await window.pywebview.api.save_frame(selectedPhotoBox);
    } catch (e) {
      console.error("save_frame fallo:", e);
      res = null;
    }

    if (!res || !res.ok) {
      console.warn("No se guardó el frame:", res);
      alert("No hay frame disponible para guardar.");
      return;
    }

    const dataUrl = res.data_url;
    const box = photoBoxes[selectedPhotoBox];
    box.style.backgroundImage = `url("${dataUrl}")`;
    box.style.backgroundSize = "cover";
    box.style.backgroundPosition = "center";

    photos[selectedPhotoBox] = dataUrl;

    const otherIndex = selectedPhotoBox === 0 ? 1 : 0;
    if (!photos[otherIndex]) {
      photoBoxes.forEach(b => b.classList.remove("selected"));
      photoBoxes[otherIndex].classList.add("selected");
      update_box(previewImg,otherIndex);
    }

    nextBtn.disabled = !(photos[0] && photos[1]);
  });

  nextBtn.addEventListener("click", showAdjustPointsView);
}

// -- Step2 / coloreado / validación (idéntico comportamiento que tenías, con saveColorsClick actualizado) --

let pointsData = [];
let pointElements = [{}, {}];
let pointsEditable = true;
let colorModeEnabled = false;
let selectedColor = null;

function setupCanvasFor(wrapper, idx, positions) {
  const img = wrapper.querySelector("img");
  const canvas = wrapper.querySelector("canvas");
  const ctx = canvas.getContext("2d");

  function resizeCanvas() {
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    drawPoints(idx, canvas, ctx, positions);
  }
  img.removeEventListener("load", resizeCanvas);
  img.addEventListener("load", resizeCanvas);

  window.removeEventListener("resize", resizeCanvas);
  window.addEventListener("resize", resizeCanvas);

  resizeCanvas();
  return { img, canvas, ctx, resizeCanvas };
}

function showAdjustPointsView() {
  pointsEditable = true;
  colorModeEnabled = false;
  selectedColor = null;

  panelTitle.textContent = "Ajustar puntos — Paso 2";
  panelContent.innerHTML = `
    <div class="image-step2" style="display:flex;gap:10px;">
        <div class="img-wrapper" data-index="0" style="flex:1; position:relative;">
            <img src="${photos[0] || './patrones/patron0.svg'}" alt="Imagen 1" style="width:100%;display:block;">
            <canvas class="points-canvas" style="position:absolute;left:0;top:0;"></canvas>
        </div>
        <div class="img-wrapper" data-index="1" style="flex:1; position:relative;">
            <img src="${photos[1] || './patrones/patron0.svg'}" alt="Imagen 2" style="width:100%;display:block;">
            <canvas class="points-canvas" style="position:absolute;left:0;top:0;"></canvas>
        </div>
    </div>
    <div class="color-buttons-container" style="display:none; margin:1rem 0; text-align:center;">
        <button class="color-btn" data-color="W" style="background:white;"></button>
        <button class="color-btn" data-color="Y" style="background:yellow;"></button>
        <button class="color-btn" data-color="R" style="background:red;"></button>
        <button class="color-btn" data-color="O" style="background:orange;"></button>
        <button class="color-btn" data-color="B" style="background:blue;"></button>
        <button class="color-btn" data-color="G" style="background:green;"></button>
    </div>
    <button class="action-btn next-btn" id="step2Next">Siguiente</button>
  `;

  const wrappers = panelContent.querySelectorAll(".img-wrapper");

  const positionsStep2 = [
    {
      "U1":[702,276], "U2":[886,185], "U3":[1038,117],
      "U4":[851,356], "U5":[1034,270], "U6":[1193,186],
      "U7":[1031,476], "U8":[1215,353], "U9":[1368,265],
      "F1":[1112,629], "F2":[1299,503], "F3":[1447,412],
      "F4":[1127,804], "F5":[1285,681], "F6":[1422,581],
      "F7":[1132,949], "F8":[1273,825], "F9":[1403,727],
      "L1":[628,421], "L2":[768,514], "L3":[940,634],
      "L4":[668,589], "L5":[800,691], "L6":[962,809],
      "L7":[705,740], "L8":[825,835], "L9":[988,964]
    },
    {
      "D1":[702,276], "D2":[886,185], "D3":[1038,117],
      "D4":[851,356], "D5":[1034,270], "D6":[1193,186],
      "D7":[1031,476], "D8":[1215,353], "D9":[1368,265],
      "R1":[628,421], "R2":[768,514], "R3":[940,634],
      "R4":[668,589], "R5":[800,691], "R6":[962,809],
      "R7":[705,740], "R8":[825,835], "R9":[988,964],
      "B1":[1112,629], "B2":[1299,503], "B3":[1447,412],
      "B4":[1127,804], "B5":[1285,681], "B6":[1422,581],
      "B7":[1132,949], "B8":[1273,825], "B9":[1403,727],
    }
  ];

  pointsData = positionsStep2;
  pointElements = [{}, {}];

  wrappers.forEach((wrapper, idx) => {
    const { img, canvas, ctx } = setupCanvasFor(wrapper, idx, positionsStep2[idx]);

    for (const key in positionsStep2[idx]) {
      pointElements[idx][key] = { color: "null" };
    }

    if (pointsEditable) {
      attachDragHandlers(idx, canvas, positionsStep2[idx], ctx);
    }
  });

  const step2Next = document.getElementById("step2Next");
  step2Next.addEventListener("click", async function activateColorModeOnce() {
    if (!hasPywebview()) return;

    try {
      
      // Antes de: await window.pywebview.api.save_json(JSON.stringify(pointsData[0]), "positions1.json");
      console.log(">>> DEBUG saving positions BEFORE API call");
      console.log("positionsData[0] sample keys:", Object.keys(pointsData[0]).slice(0,12));
      console.log("positionsData[1] sample keys:", Object.keys(pointsData[1]).slice(0,12));
      // Also dump full coords for visual inspection (but be careful with verbosity)
      console.log("positions1 full:", pointsData[0]);
      console.log("positions2 full:", pointsData[1]);

      await window.pywebview.api.save_json(JSON.stringify(pointsData[0]), "positions1.json");
      await window.pywebview.api.save_json(JSON.stringify(pointsData[1]), "positions2.json");

      panelTitle.textContent = "Pintar puntos — Paso 3";
      enableColorMode();
      await loadAndApplyPointColors();

      step2Next.removeEventListener("click", activateColorModeOnce);
      step2Next.addEventListener("click", async () => {
        if (!hasPywebview()) return;
        const ok = await saveColorsClick();
        if (!ok) {
          // validación fallida -> no avanzar
          return;
        }
      });
    } catch (e) {
      console.error(e);
      alert("Error guardando coordenadas");
    };
  });
}

// Reemplazar / insertar dentro de scaner.js: función saveColorsClick() mejorada
async function saveColorsClick() {
  if (!hasPywebview()) return false;
  try {
    // construir lettersMap como antes
    const lettersMap = {};
    for (const idx in pointElements) {
      for (const key in pointElements[idx]) {
        lettersMap[key] = (pointElements[idx][key].color || "W").toUpperCase().charAt(0);
      }
    }

    // --- VALIDACIONES Y DIAGNÓSTICO LOCAL ---
    // Keys esperadas (orden Kociemba U,R,F,D,L,B)
    const expectedKeys = [];
    ['U','R','F','D','L','B'].forEach(f => { for (let i=1;i<=9;i++) expectedKeys.push(f+i); });
    const keys = Object.keys(lettersMap).sort();
    const missing = expectedKeys.filter(k => !keys.includes(k));
    const extra = keys.filter(k => !expectedKeys.includes(k));
    console.log("lettersMap keys count:", keys.length, "missing:", missing, "extra:", extra);

    // Mostrar muestra
    console.log("sample lettersMap:", Object.fromEntries(keys.slice(0,18).map(k => [k, lettersMap[k]])));

    // Obtener centros detectados (el sticker X5 es el centro de cada cara)
    const faceCenters = {};
    ['U','R','F','D','L','B'].forEach(f => {
      const k = `${f}5`;
      faceCenters[f] = lettersMap[k] || null;
    });
    console.log("Detected face centers (from lettersMap):", faceCenters);

    // --- Corrección: intercambiar caras R y B ---
    const tempR = {};
    const tempB = {};

    for (let i = 1; i <= 9; i++) {
      tempR[`R${i}`] = lettersMap[`R${i}`];
      tempB[`B${i}`] = lettersMap[`B${i}`];
    }

    // Swap
    for (let i = 1; i <= 9; i++) {
      lettersMap[`R${i}`] = tempB[`B${i}`];
      lettersMap[`B${i}`] = tempR[`R${i}`];
    }
    console.log("✅ Intercambio aplicado: caras R y B corregidas.");

    // --- Corrección: intercambiar colores W y O ---
    for (const k in lettersMap) {
      if (lettersMap[k] === "W") lettersMap[k] = "O";
      else if (lettersMap[k] === "O") lettersMap[k] = "W";
    }
    console.log("✅ Intercambio aplicado: colores W ↔ O corregidos.");


    const val = await window.pywebview.api.validate_cube_state(lettersMap);

    if (!val || !val.ok) {
      alert("Error en validación backend: " + ((val && val.error) || "desconocido"));
      console.warn("validate_cube_state returned:", val && val.debug ? val.debug : val);
      return false;
    }

    solution = val.solution;
    
    const moves = solution.trim().split(/\s+/).reverse();

    const inverso = moves.map(move => {
      if (move.endsWith("2")) return move; // giros dobles son su propio inverso
      else if (move.endsWith("'")) return move.slice(0, -1); // U' → U
      else return move + "'"; // U → U'
    }).join(' ');

    rubik.SecuenciaInstantanea(inverso);

    EscaneoCompleto();
    return true;
  } catch (e) {
    console.error("Error guardando colores:", e);
    alert("Error guardando colores: " + e);
    return false;
  }
}


async function loadAndApplyPointColors() {
  if (!hasPywebview()) return;
  try {
    const res = await window.pywebview.api.color_detector();
    if (!res.ok) {
      alert("Detección de colores falló: " + (res.error || "Desconocido. Mejora iluminación/ajusta posiciones."));
      console.warn("color_detector:", res);
      return;
    }

    const mapping = res.colors || {}; // expected { "U1":"W", ... }
    console.log(`Initialy mapping colors from color detector : `,mapping);


    for (let idx = 0; idx < pointElements.length; idx++) {
      for (const k in pointElements[idx]) {
        if (mapping && mapping[k]) {
          console.log(`Applying colors to image index ${idx}: keys count`, Object.keys(pointElements[idx]).length);
          pointElements[idx][k].color = mapping[k].toUpperCase().charAt(0);
        } else {
          const unexpected = Object.keys(mapping).filter(k => k.startsWith(idx===0 ? 'D' : 'U')); // example check
          console.log("unexpected keys for this image (simple heuristic):", unexpected.slice(0,8));
          pointElements[idx][k].color = "W";
        }
      }
      const canvas = panelContent.querySelectorAll(".img-wrapper canvas")[idx];
      if (canvas) {
        drawPoints(idx, canvas, canvas.getContext("2d"), pointsData[idx]);
      }
    }
    console.log(`Finlay mapping colors from color detector : `,mapping);

  } catch (e) {
    console.error("Error cargando colores:", e);
    alert("Error al obtener colores del backend: " + e);
  }
}

async function validatePointColors() {
  if (!hasPywebview()) return false;
  try {
    const lettersMap = {};
    for (const idx in pointElements) {
      for (const key in pointElements[idx]) {
        lettersMap[key] = (pointElements[idx][key].color || "W").toUpperCase().charAt(0);
      }
    }
   // console.log(">>> DEBUG final lettersMap about to send (sample 24):", Object.fromEntries(Object.keys(lettersMap).slice(0,24).map(k => [k, lettersMap[k]])));

    const res = await window.pywebview.api.validate_cube_state(lettersMap);
    if (!res || !res.ok) {
      alert("Error en validación: " + ((res && res.error) || "desconocido"));
      return false;
    }

    solution = Array.isArray(res.solution) ? res.solution.join(" ") : (res.solution || "");
    alert("Colores válidos");
    return true;
  } catch (e) {
    console.error("Error guardando colores:", e);
    alert("Error guardando colores: " + e);
    return false;
  }
}

function EscaneoCompleto() {
  panelTitle.textContent = "Escaneo completo";
  panelContent.innerHTML = `
    <div class="scan-complete-container">
        <img src="./patrones/patron0.svg" alt="Logo Escaneo">
        <h2>¡Escaneo completado exitosamente!</h2>
        <p>Todos los puntos han sido capturados y guardados.</p>
        <button class="action-btn" id="rescanBtn">Escanear nuevamente</button>
    </div>
  `;
  const rescanBtn = document.getElementById("rescanBtn");
  rescanBtn.addEventListener("click", () => {
    showScanView();
  });
}

document.getElementById("solve-btn").addEventListener("click", async () => {
  if (!solution || solution.length === 0) {
    alert("No hay solución calculada");
    return;
  }
  const seqStr = solution.trim();
  if (!seqStr) return;
  const moves = seqStr.split(/\s+/).filter(Boolean);

  try {
    for (const move of moves) {
      await rubik.RotarCara(move);
    }
    console.log("Solve sequence completed");
    solution = ""
  } catch (e) {
    console.error("Error during solve sequence:", e);
    alert("Error al ejecutar solución: " + (e.message || e));
  }
});

document.getElementById("scramble-btn").addEventListener("click", async () => {
  try {
    const seq = await window.pywebview.api.scramble();
    for (const move of seq) {
      await rubik.RotarCara(move);
    }

    const res = await window.pywebview.api.validate_cube_state();
    if (!res.ok) {
      alert("Error en validación: " + (res.error || "desconocido"));
      return;
    }
    solution = Array.isArray(res.solution) ? res.solution.join(" ") : (res.solution || "");
    alert("Cubo mezclado");
  } catch (e) {
    console.error("Error al ejecutar scramble:", e);
    alert("Error en scramble: " + (e.message || e));
  }
});

// --- drawPoints, attachDragHandlers, enableColorMode (las mantuve iguales) ---
function drawPoints(idx, canvas, ctx, positions) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const scaleX = canvas.width / 1920;
  const scaleY = canvas.height / 1080;
  const letterToCss = { "W":"white", "Y":"yellow", "R":"red", "O":"orange", "B":"blue", "G":"green","null":"gray" };

  for (const key in positions) {
    const [x, y] = positions[key];
    const letter = (pointElements[idx][key]?.color || "null").toUpperCase();
    ctx.fillStyle = letterToCss[letter] || "grey";
    ctx.beginPath();
    ctx.arc(x * scaleX, y * scaleY, 7, 0, Math.PI*2);
    ctx.fill();
  }
}

function attachDragHandlers(idx, canvas, positions, ctx) {
  let dragging = null;

  canvas.addEventListener("pointerdown", (e) => {
    if (!pointsEditable) return;
    canvas.setPointerCapture(e.pointerId);
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    for (const key in positions) {
      const [x, y] = positions[key];
      const scaledX = x * canvas.width / 1920;
      const scaledY = y * canvas.height / 1080;
      if (Math.hypot(mouseX - scaledX, mouseY - scaledY) < 10) {
        dragging = key;
        break;
      }
    }
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const baseX = (mouseX / canvas.width) * 1920;
    const baseY = (mouseY / canvas.height) * 1080;
    positions[dragging] = [
      Math.max(0, Math.min(1920, baseX)),
      Math.max(0, Math.min(1080, baseY))
    ];
    drawPoints(idx, canvas, ctx, positions);
  });

  const endDrag = (e) => { dragging = null; };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("pointerleave", endDrag);
}

function enableColorMode() {
  colorModeEnabled = true;
  pointsEditable = false;
  const container = panelContent.querySelector(".color-buttons-container");
  container.style.display = "block";

  const buttons = container.querySelectorAll(".color-btn");
  buttons.forEach(btn => {
    btn.disabled = false;
    btn.removeEventListener("click", btn._handler);
    btn._handler = () => {
      buttons.forEach(b => b.classList.remove("selected"));
      btn.classList.add("selected");
      console.log("Color elegido : ", btn.dataset.color);
      selectedColor = btn.dataset.color;
    };
    btn.addEventListener("click", btn._handler);
  });

  const wrappers = panelContent.querySelectorAll(".img-wrapper");
  wrappers.forEach((wrapper, idx) => {
    const canvas = wrapper.querySelector("canvas");
    canvas.addEventListener("click", (e) => {
      if (!selectedColor) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      for (const key in pointsData[idx]) {
        const [x, y] = pointsData[idx][key];
        const scaledX = x * canvas.width / 1920;
        const scaledY = y * canvas.height / 1080;
        if (Math.hypot(mouseX - scaledX, mouseY - scaledY) < 10) {
          pointElements[idx][key].color = selectedColor;
          drawPoints(idx, canvas, canvas.getContext("2d"), pointsData[idx]);
          break;
        }
      }
    });
  });
}
