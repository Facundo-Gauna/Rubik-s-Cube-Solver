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
    // backend throttles encoding; 100ms is fine
    previewInterval = setTimeout(frameLoop, 100);
  })();
}
function stopPreviewLoop() {
  if (previewInterval) {
    clearTimeout(previewInterval);
    previewInterval = null;
  }
}

// -------------- Step 1 (captura) --------------
let photos = [null, null];
let selectedPhotoBox = 0;
let solution = "";

const _globalDrag = {
  activePointEl: null,
  activePointKey: null,
  container: null,
  offsetX: 0,
  offsetY: 0,
  scaleX: 1,
  scaleY: 1,
  positionsObj: null
};

window.addEventListener("mousemove", (e) => {
  if (!_globalDrag.activePointEl) return;
  const rect = _globalDrag.container.getBoundingClientRect();
  let newX = e.clientX - rect.left - _globalDrag.offsetX;
  let newY = e.clientY - rect.top - _globalDrag.offsetY;
  newX = Math.max(0, Math.min(newX, rect.width));
  newY = Math.max(0, Math.min(newY, rect.height));
  _globalDrag.activePointEl.style.left = `${newX}px`;
  _globalDrag.activePointEl.style.top = `${newY}px`;
});

window.addEventListener("mouseup", (e) => {
  if (!_globalDrag.activePointEl) return;
  _globalDrag.activePointEl.style.cursor = "grab";
  const left = parseFloat(_globalDrag.activePointEl.style.left) + 7;
  const top = parseFloat(_globalDrag.activePointEl.style.top) + 7;
  // save back into positions object (scaled to 1920x1080)
  _globalDrag.positionsObj[_globalDrag.activePointKey] = [
    Math.round(left / _globalDrag.scaleX),
    Math.round(top / _globalDrag.scaleY)
  ];
  _globalDrag.activePointEl = null;
  _globalDrag.activePointKey = null;
  _globalDrag.container = null;
  _globalDrag.positionsObj = null;
});

function drawPreviewPoints(imgElement, positionsObj) {
  if (!imgElement || !positionsObj) return;
  const container = imgElement.parentElement;
  const containerWidth = container.clientWidth;
  const containerHeight = container.clientHeight;
  const scaleX = containerWidth / 1920;
  const scaleY = containerHeight / 1080;

  // store scale for drag finalization
  // remove previous
  const prev = container.querySelectorAll(".point");
  prev.forEach(p => p.remove());

  // color shown in the small preview points (by face letter)
  const letterToCss = { "U":"#0000FF","F":"#FFFF00","L":"#FF0000","B":"#FFFFFF","R":"#FFA500","D":"#008000" };

  // create points (but use global drag handlers for move)
  for (const key in positionsObj) {
    const [x, y] = positionsObj[key];
    const point = document.createElement("div");
    point.className = "point";
    point.style.position = "absolute";
    point.style.width = "14px";
    point.style.height = "14px";
    point.style.background = letterToCss[key.charAt(0)] || "#808080";
    point.style.borderRadius = "50%";
    point.style.left = `${x * scaleX - 7}px`;
    point.style.top = `${y * scaleY - 7}px`;
    point.style.cursor = "grab";
    point.style.zIndex = "10";

    point.addEventListener("mousedown", (e) => {
      e.stopPropagation();
      _globalDrag.activePointEl = point;
      _globalDrag.activePointKey = key;
      _globalDrag.container = container;
      _globalDrag.offsetX = e.offsetX;
      _globalDrag.offsetY = e.offsetY;
      _globalDrag.scaleX = scaleX;
      _globalDrag.scaleY = scaleY;
      _globalDrag.positionsObj = positionsObj;
      point.style.cursor = "grabbing";
    });

    container.appendChild(point);
  }
}

function showScanView() {
  photos = [null, null];
  selectedPhotoBox = 0;

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

  // default positions (visual only; user can adjust in step 2)
  const positions = {
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

  // draw preview points when image loads or window resizes
  const onPreviewLoadOrResize = () => drawPreviewPoints(previewImg, positions);
  previewImg.addEventListener("load", onPreviewLoadOrResize);
  window.removeEventListener("resize", onPreviewLoadOrResize); // avoid duplicates
  window.addEventListener("resize", onPreviewLoadOrResize);

  startPreviewLoop(previewImg);

  photoBoxes.forEach(b => b.classList.remove("selected"));
  photoBoxes[0].classList.add("selected");

  photoBoxes.forEach(box => {
    box.addEventListener("click", () => {
      photoBoxes.forEach(b => b.classList.remove("selected"));
      box.classList.add("selected");
      selectedPhotoBox = parseInt(box.dataset.index);
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
      selectedPhotoBox = otherIndex;
    }

    nextBtn.disabled = !(photos[0] && photos[1]);
  });

  nextBtn.addEventListener("click", showAdjustPointsView);
}

// -------------- Step 2 (ajustar puntos + colorear) --------------
let pointsData = [];
let pointElements = [{}, {}];
let pointsEditable = true;
let colorModeEnabled = false;
let selectedColor = null;

// helper: safe query for canvas and keep one resize handler per canvas
function setupCanvasFor(wrapper, idx, positions) {
  const img = wrapper.querySelector("img");
  const canvas = wrapper.querySelector("canvas");
  const ctx = canvas.getContext("2d");

  function resizeCanvas() {
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    drawPoints(idx, canvas, ctx, positions);
  }
  // ensure no duplicate handlers
  img.removeEventListener("load", resizeCanvas);
  img.addEventListener("load", resizeCanvas);

  window.removeEventListener("resize", resizeCanvas);
  window.addEventListener("resize", resizeCanvas);

  // initial
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
        <!-- data-color uses letters -->
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
      "B1":[1112,629], "B2":[1299,503], "B3":[1447,412],
      "B4":[1127,804], "B5":[1285,681], "B6":[1422,581],
      "B7":[1132,949], "B8":[1273,825], "B9":[1403,727],
      "R1":[628,421], "R2":[768,514], "R3":[940,634],
      "R4":[668,589], "R5":[800,691], "R6":[962,809],
      "R7":[705,740], "R8":[825,835], "R9":[988,964]
    }
  ];

  pointsData = positionsStep2;
  pointElements = [{}, {}];

  wrappers.forEach((wrapper, idx) => {
    const { img, canvas, ctx } = setupCanvasFor(wrapper, idx, positionsStep2[idx]);

    for (const key in positionsStep2[idx]) {
      // store letters by default (unknown = "W" as safe default)
      pointElements[idx][key] = { color: "W" };
    }

    if (pointsEditable) {
      attachDragHandlers(idx, canvas, positionsStep2[idx], ctx);
    }
  });

  const step2Next = document.getElementById("step2Next");
  step2Next.addEventListener("click", async function activateColorModeOnce() {
    if (!hasPywebview()) return;

    try {
      await window.pywebview.api.save_json(JSON.stringify(pointsData[0]), "positions1.json");
      await window.pywebview.api.save_json(JSON.stringify(pointsData[1]), "positions2.json");

      panelTitle.textContent = "Pintar puntos — Paso 3";
      enableColorMode();
      await loadAndApplyPointColors();

      step2Next.removeEventListener("click", activateColorModeOnce);
      step2Next.addEventListener("click", async () => { await saveColorsClick(); });
    } catch (e) {
      console.error(e);
      alert("Error guardando coordenadas");
    }
  });
}

async function saveColorsClick() {
  if (!hasPywebview()) return;
  try {
    // gather letter codes directly from pointElements
    const lettersMap = {};
    for (const idx in pointElements) {
      for (const key in pointElements[idx]) {
        // ensure uppercase single-letter code
        lettersMap[key] = (pointElements[idx][key].color || "W").toUpperCase().charAt(0);
      }
    }

    // send lettersMap directly to backend
    const val = await window.pywebview.api.validate_cube_state(lettersMap);
    if (!val.ok) {
      alert("Error en validación backend: " + (val.error || "desconocido"));
      return;
    }

    // backend returns solution array or string
    solution = Array.isArray(val.solution) ? val.solution.join(" ") : (val.solution || "");

    // update 3D if available
    if (typeof rubik !== "undefined" && typeof rubik.CambiarColoresDelCubo === "function") {
      // rubik should expose ColoresBaseHex keyed by letters (W,Y,R,O,B,G)
      const ColoresBaseHex = (rubik && rubik.ColoresBaseHex) ? rubik.ColoresBaseHex : {
        W: 0xffffff, Y: 0xffff00, R: 0xff0000, O: 0xff6600, B: 0x0000ff, G: 0x00aa00
      };

      // build listaColores using the same iteration order as lettersMap keys
      const listaColores = [];
      for (const k in lettersMap) {
        const letter = lettersMap[k] || "W";
        listaColores.push(ColoresBaseHex[letter] ?? 0xffffff);
      }

      // CambiarColoresDelCubo expects an array of 54 hex colors (must match your order)
      rubik.CambiarColoresDelCubo(listaColores);
    }

    EscaneoCompleto();
  } catch (e) {
    console.error("Error guardando colores:", e);
    alert("Error guardando colores: " + e);
  }
}

function drawPoints(idx, canvas, ctx, positions) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const scaleX = canvas.width / 1920;
  const scaleY = canvas.height / 1080;

  // letter -> css color for display
  const letterToCss = { "W":"white", "Y":"yellow", "R":"red", "O":"orange", "B":"blue", "G":"green" };

  for (const key in positions) {
    const [x, y] = positions[key];
    const letter = (pointElements[idx][key]?.color || "W").toUpperCase();
    ctx.fillStyle = letterToCss[letter] || "gray";
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
    btn.removeEventListener("click", btn._handler); // avoid duplicates
    btn._handler = () => {
      buttons.forEach(b => b.classList.remove("selected"));
      btn.classList.add("selected");
      selectedColor = btn.dataset.color; // letter like "W"
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
          pointElements[idx][key].color = selectedColor; // store letter
          drawPoints(idx, canvas, canvas.getContext("2d"), pointsData[idx]);
          break;
        }
      }
    });
  });
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

    const mapping = res.colors || {}; // expected: { "U1":"W", ... }

    for (let idx = 0; idx < pointElements.length; idx++) {
      for (const k in pointElements[idx]) {
        if (mapping && mapping[k]) {
          // mapping already letters (R,O,Y,G,B,W)
          pointElements[idx][k].color = mapping[k].toUpperCase().charAt(0);
        } else {
          pointElements[idx][k].color = "W"; // default safe
        }
      }
      const canvas = panelContent.querySelectorAll(".img-wrapper canvas")[idx];
      if (canvas) {
        drawPoints(idx, canvas, canvas.getContext("2d"), pointsData[idx]);
      }
    }
  } catch (e) {
    console.error("Error cargando colores:", e);
    alert("Error al obtener colores del backend: " + e);
  }
}

async function validatePointColors() {
  if (!hasPywebview()) return;
  try {
    const lettersMap = {};
    for (const idx in pointElements) {
      for (const key in pointElements[idx]) {
        lettersMap[key] = (pointElements[idx][key].color || "W").toUpperCase().charAt(0);
      }
    }

    const res = await window.pywebview.api.validate_cube_state(lettersMap);
    if (!res.ok) {
      alert("Error en validación: " + (res.error || "desconocido"));
      return;
    }
    solution = Array.isArray(res.solution) ? res.solution.join(" ") : (res.solution || "");
    alert("Colores válidos");
  } catch (e) {
    console.error("Error guardando colores:", e);
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

// helper: espera hasta que rubik.busy sea false (o timeout)
function waitRubikIdle(timeoutMs = 5000) {
  return new Promise((resolve, reject) => {
    const start = performance.now();
    const check = setInterval(() => {
      if (typeof rubik === "undefined" || !rubik) {
        clearInterval(check);
        resolve(); // no rubik -> nothing to wait
        return;
      }
      try {
        if (!rubik.busy) {
          clearInterval(check);
          resolve();
          return;
        }
      } catch (e) {
        // getter error -> stop waiting
        clearInterval(check);
        resolve();
        return;
      }
      if (performance.now() - start > timeoutMs) {
        clearInterval(check);
        reject(new Error("Timeout waiting for rubik idle"));
      }
    }, 20);
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
      // send and animate per-move (backend should ack each move)
      const backendPromise = window.pywebview.api.send_sequence(move);
      if (typeof rubik !== "undefined" && rubik && typeof rubik.RotarCara === "function") {
        try { rubik.RotarCara(move); } catch (e) { console.warn("rubik.RotarCara error:", e); }
      }

      try {
        await backendPromise;
      } catch (e) {
        console.warn("backend send_sequence failed for move", move, e);
        throw e;
      }

      try {
        await waitRubikIdle(3000);
      } catch (e) {
        console.warn("Rubik busy timeout after move", move);
      }
    }
    console.log("Solve sequence completed");
  } catch (e) {
    console.error("Error during solve sequence:", e);
    alert("Error al ejecutar solución: " + (e.message || e));
  }
});

document.getElementById("scramble-btn").addEventListener("click", async () => {
  try {
    const seq = await window.pywebview.api.scramble(); // expect array or string
    let moves = [];
    if (Array.isArray(seq)) moves = seq;
    else if (typeof seq === "string") moves = seq.split(/\s+/).filter(Boolean);

    for (const move of moves) {
      const backendPromise = window.pywebview.api.send_sequence(move);
      if (typeof rubik !== "undefined" && rubik && typeof rubik.RotarCara === "function") {
        try { rubik.RotarCara(move); } catch (e) { console.warn("rubik.RotarCara error:", e); }
      }
      await backendPromise;
      try { await waitRubikIdle(2000); } catch (_) {}
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
