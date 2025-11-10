/* script.js actualizado con Scan 3 pasos, Canvas 1080x1920, puntos movibles y pintables.
   Depende de `rubik` declarado globalmente y de window.pywebview.api.get_frame() si existe. */

// panel refs
const panelTitle = document.getElementById("panel-title");
const panelContent = document.getElementById("panel-content");

// estado global del scan
let photos = [null, null];         // URLs de las imágenes capturadas (o null)
let selectedPhotoBox = 0;         // 0 o 1 -> cuál está seleccionada en paso 1
let pointsData = [null, null];    // arrays de 27 puntos por imagen: {x:0-1080,y:0-1920,color:null}
let previewInterval = null;       // id del interval de preview de camara
let previewSrc = null;            // último frame de preview
let pointsEditable = true;        // control para arrastrar o no
let activeColor = null;           // color seleccionado para pintar en paso2-colorear

// Colores base (hex)
const palette = {
  Blanco: "#ffffff",
  Amarillo: "#ffff00",
  Rojo: "#ff0000",
  Naranja: "#ff6600",
  Azul: "#0000ff",
  Verde: "#00aa00"
};
const paletteOrder = ["Blanco","Amarillo","Rojo","Naranja","Azul","Verde"];

// ---------------------- UTILIDADES ----------------------
function hasPyWebview() {
  return typeof window.pywebview !== "undefined" && window.pywebview.api && typeof window.pywebview.api.get_frame === "function";
}

// obtiene preview (simulado si no hay pywebview)
async function fetchPreviewFrame() {
  if (hasPyWebview()) {
    try {
      const data = await window.pywebview.api.get_frame(); // esperamos base64 o url
      return data;
    } catch (e) {
      console.warn("pywebview get_frame falló:", e);
      return "patrones/patron0.svg";
    }
  } else {
    // simulación: imagen de ejemplo
    return "patrones/patron0.svg";
  }
}

// lee los puntos desde el JSON usando la API Python
async function ensurePointsForIndex(idx) {
  const data = await window.pywebview.api.load_points(idx);
  pointsData[idx] = [];

  for (const [name, arr] of Object.entries(data)) {
    const [x, y, color] = arr;
    // asegurar que existan los 3 valores
    pointsData[idx].push({
      name,
      x: Number(x) || 0,
      y: Number(y) || 0,
      color: color || "#cccccc",
    });
  }
}


// convierte coordenadas del mouse a coordenadas internas del canvas
function mouseToCanvasCoords(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top) * scaleY;
  return { x: mx, y: my };
}

// ---------------------- Renders STEP 1 ----------------------
function showScanView() {
  renderScanStep1();
}

function startPreviewLoop(previewImgElement) {
  stopPreviewLoop();
  // setear inmediatamente y luego cada 150ms
  (async function frame() {
    previewSrc = await fetchPreviewFrame();
    if (previewImgElement) previewImgElement.src = previewSrc;
    previewInterval = setTimeout(frame, 150);
  })();
}

function stopPreviewLoop() {
  if (previewInterval) {
    clearTimeout(previewInterval);
    previewInterval = null;
  }
}

function renderScanStep1() {
  // Step 1 UI
  panelTitle.textContent = "Escanear — Paso 1";
  panelContent.innerHTML = `
    <div class="camera-view"><img id="previewImg" alt="preview"></div>
    <button class="action-btn capture-btn" id="captureBtn">Capturar imagen</button>
    <div class="photos-container">
      <div class="photo-box selected" data-index="0"></div>
      <div class="photo-box" data-index="1"></div>
    </div>
    <button class="action-btn next-btn" id="step1Next" disabled>Siguiente</button>
  `;

  // referencias
  const previewImg = panelContent.querySelector("#previewImg");
  const captureBtn = panelContent.querySelector("#captureBtn");
  const photoBoxes = panelContent.querySelectorAll(".photo-box");
  const nextBtn = panelContent.querySelector("#step1Next");

  // iniciar preview
  startPreviewLoop(previewImg);

  // asegurarnos selección por defecto
  selectedPhotoBox = 0;
  photoBoxes.forEach((b)=>b.classList.remove("selected"));
  photoBoxes[0].classList.add("selected");

  // click en cada caja
  photoBoxes.forEach(box => {
    box.addEventListener("click", () => {
      // si ya hay una secuencia rubik en curso, no interfiere => no bloqueo aquí
      photoBoxes.forEach(b => b.classList.remove("selected"));
      box.classList.add("selected");
      selectedPhotoBox = parseInt(box.dataset.index);
    });
  });

  // captura: reemplaza la imagen del cuadro seleccionado (usa previewSrc)
  captureBtn.addEventListener("click", async () => {
    // tomar frame actual (previewSrc puede ser base64 o ruta)
    const frame = previewSrc || await fetchPreviewFrame();
    
    // ------------------------------------------------------
    try {
      await window.pywebview.api.save_frame(selectedPhotoBox, frame);
      console.log("Guardado solicitado OK");
    } catch (e) {
      console.warn("Error guardando frame:", e);
    }
    // ------------------------------------------------------


    photos[selectedPhotoBox] = frame;

    // pintar el fondo del cuadro
    const box = photoBoxes[selectedPhotoBox];
    box.style.backgroundImage = `url(${frame})`;

    // si el otro está vacío => cambiar automaticamente selección al vacío
    const otherIndex = selectedPhotoBox === 0 ? 1 : 0;
    if (!photos[otherIndex]) {
      photoBoxes.forEach(b => b.classList.remove("selected"));
      photoBoxes[otherIndex].classList.add("selected");
      selectedPhotoBox = otherIndex;
    }

    // habilitar siguiente si ambos tienen imagen
    if (photos[0] && photos[1]) {
      nextBtn.disabled = false;
    } else {
      nextBtn.disabled = true;
    }
  });

  // ir a paso 2
  nextBtn.addEventListener("click", () => {
    if (!photos[0] || !photos[1]) return;
    stopPreviewLoop();
    renderScanStep2();
  });
}

// ---------------------- STEP 2: ajustar puntos ----------------------
async function renderScanStep2() {
  const res = await window.pywebview.api.start_color_detection_and_prepare_correction();
  if (!res.ok) { alert("Detection failed: " + res.error); return; }
  console.log(res);

  panelTitle.textContent = "Escanear — Paso 2";
  panelContent.innerHTML = `
    <div class="canvas-wrapper" id="wrap0"><canvas id="canvas0" class="point-canvas"></canvas></div>
    <div style="height:12px;"></div>
    <div class="canvas-wrapper" id="wrap1"><canvas id="canvas1" class="point-canvas"></canvas></div>
    <div style="height:8px;"></div>
    <div style="display:flex;gap:10px;align-items:center;flex-direction:column;width:100%">
      <div id="paletteContainer" style="display:none"></div>
      <button class="action-btn" id="step2Next">Siguiente</button>
    </div>
  `;

  // leer los puntos desde los archivos JSON
  await ensurePointsForIndex(0);
  await ensurePointsForIndex(1);
  pointsEditable = true;
  activeColor = null;

  // canvases
  const canvas0 = panelContent.querySelector("#canvas0");
  const canvas1 = panelContent.querySelector("#canvas1");
  const canvases = [canvas0, canvas1];

  canvases.forEach((canvas, idx) => {
    // resolución interna
    canvas.width = 1920;
    canvas.height = 1080;
    canvas.style.width = "100%";
    canvas.style.height = "auto"; // mantiene proporción


    const ctx = canvas.getContext("2d");

    // dibujar imagen de fondo
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = photos[idx] || "patrones/patron0.svg";
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      drawPointsOnCanvas(idx);
    };
    img.onerror = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawPointsOnCanvas(idx);
    };

    // --- DRAGGING DE PUNTOS ---
    let dragging = null;

    canvas.addEventListener("mousedown", (e) => {
      if (!pointsEditable) return;
      const pos = mouseToCanvasCoords(e, canvas);
      const pts = pointsData[idx];
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        const d = Math.hypot(p.x - pos.x, p.y - pos.y);
        if (d < 20) {
          dragging = p;
          break;
        }
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const coords = mouseToCanvasCoords(e, canvas);
      dragging.x = Math.max(0, Math.min(canvas.width, coords.x));
      dragging.y = Math.max(0, Math.min(canvas.height, coords.y));
      redrawCanvas(idx);
    });

    window.addEventListener("mouseup", async (e) => {
      if (dragging) {
        // guardar coordenadas en JSON solo si son válidas
        if (
          !isNaN(dragging.x) &&
          !isNaN(dragging.y) &&
          typeof dragging.color === "string"
        ) {
          await window.pywebview.api.save_point(
            idx,
            dragging.name,
            dragging.x,
            dragging.y,
            dragging.color
          );
        }
        dragging = null;
      }
    });

    // --- CLICK PARA CAMBIAR COLOR (modo colorear) ---
    canvas.addEventListener("click", async (e) => {
      if (pointsEditable) return;
      if (!activeColor) return;
      const pos = mouseToCanvasCoords(e, canvas);
      const pts = pointsData[idx];
      let nearest = null,
        nd = 9999;
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        const d = Math.hypot(p.x - pos.x, p.y - pos.y);
        if (d < nd) {
          nd = d;
          nearest = p;
        }
      }
      if (nearest && nd < 30) {
        nearest.color = palette[activeColor];
        redrawCanvas(idx);
        await window.pywebview.api.save_point(
          idx,
          nearest.name,
          nearest.x,
          nearest.y,
          nearest.color
        );
      }
    });
  });

  // --- FUNCIONES DE DIBUJO ---
  function redrawCanvas(idx) {
      const canvas = canvases[idx];
      const ctx = canvas.getContext("2d");
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = photos[idx] || "patrones/patron0.svg";
      img.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        drawPointsOnCanvas(idx);
      };
      img.onerror = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawPointsOnCanvas(idx);
      };
    }

    function drawPointsOnCanvas(idx) {
    const canvas = canvases[idx];
    const ctx = canvas.getContext("2d");
    const pts = pointsData[idx] || [];

    pts.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 28, 0, Math.PI * 2);
      ctx.fillStyle = p.color || "#cccccc";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#000000";
      ctx.stroke();
    });
  }



  // --- BOTÓN SIGUIENTE ---
  const step2Next = panelContent.querySelector("#step2Next");
  step2Next.addEventListener("click", () => {
    for (let idx = 0; idx < 2; idx++) {
      pointsData[idx].forEach(async (p) => {
        if (!p.color) p.color = palette[paletteOrder[Math.floor(Math.random() * 6)]];
        if (p.x && p.y)
          await window.pywebview.api.save_point(idx, p.name, p.x, p.y, p.color);
      });
      redrawCanvas(idx);
    }

    pointsEditable = false;
    showColorPalette();
  });

  // --- MOSTRAR PALETA DE COLORES ---
  function showColorPalette() {
    const container = panelContent.querySelector("#paletteContainer");
    container.style.display = "flex";
    container.innerHTML = "";
    paletteOrder.forEach((key) => {
      const b = document.createElement("button");
      b.className = "color-btn";
      b.style.background = palette[key];
      b.title = key;
      b.addEventListener("click", () => {
        container.querySelectorAll(".color-btn").forEach((x) => x.classList.remove("selected"));
        b.classList.add("selected");
        activeColor = key;
      });
      container.appendChild(b);
    });

    // cambiar función del botón siguiente
    const oldBtn = document.querySelector("#step2Next");
    const newBtn = oldBtn.cloneNode(true);
    oldBtn.parentNode.replaceChild(newBtn, oldBtn);
    newBtn.textContent = "Siguiente";
    newBtn.addEventListener("click", () => {
      renderScanStep3();
    });
  }
}

// ---------------------- STEP 3 (final) ----------------------
function renderScanStep3() {  
  panelTitle.textContent = "Escanear — Paso 3";
  panelContent.innerHTML = `
    <div style="display:flex;flex-direction:column;align-items:center;gap:1rem;width:100%;">
      <div style="font-size:1.1rem;font-weight:600;">Cubo escaneado exitosamente 🎉</div>
      <div style="width:120px;height:120px;border-radius:12px;background:linear-gradient(135deg,#e0f7ff,#cfe8ff);display:flex;align-items:center;justify-content:center">✓</div>
      <button class="action-btn" id="rescanBtn">Volver a escanear</button>
    </div>
  `;

  document.getElementById("rescanBtn").addEventListener("click", () => {
    // resetear todo menos puntosData coordenadas (según tu pedido)
    photos = [null, null];
    selectedPhotoBox = 0;
    // NOTA: pointsData conserva las coordenadas; mantendremos colors también si querés, pero
    // pediste que se mantengan las coordenadas, así que no las borramos.
    // limpiar preview y volver a paso 1
    renderScanStep1();
  });
}

// ---------------------- Patrones y Ajustes (sin cambios conceptuales) ----------------------
function showPatternsView() {
  panelTitle.textContent = "Patrones";
  panelContent.innerHTML = `<div class="patterns-container"></div>`;
  const container = panelContent.querySelector(".patterns-container");

  let selectedImg = null;

  for (let i = 0; i <= 19; i++) {
    const img = document.createElement("img");
    img.className = "pattern";
    img.src = `patrones/patron${i}.svg`;
    img.alt = `Patrón ${i}`;

    img.style.transition = "transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease";
    img.addEventListener("mouseover", () => {
      if (img !== selectedImg) {
        img.style.transform = "scale(1.07)";
        img.style.boxShadow = "0 4px 12px rgba(0,0,0,0.25)";
        img.style.borderColor = "#4CAF50";
      }
    });
    img.addEventListener("mouseout", () => {
      if (img !== selectedImg) {
        img.style.transform = "scale(1)";
        img.style.boxShadow = "0 2px 6px rgba(0,0,0,0.15)";
        img.style.borderColor = "#ccc";
      }
    });

    img.addEventListener("click", () => {
      if (selectedImg === img) return;
      if (typeof rubik !== "undefined" && rubik.busy) return;

      if (selectedImg) {
        selectedImg.style.borderColor = "#ccc";
        selectedImg.style.boxShadow = "0 2px 6px rgba(0,0,0,0.15)";
        selectedImg.style.transform = "scale(1)";
      }

      selectedImg = img;
      img.style.borderColor = "#4CAF50";
      img.style.boxShadow = "0 4px 16px rgba(76, 175, 80, 0.5)";
      img.style.transform = "scale(1.1)";

      if (typeof rubik !== "undefined") rubik.SecuenciaDeGiros("U2 D2 L2 R2 B2 F2");
    });

    container.appendChild(img);
  }
}

function showSettingsView() {
  panelTitle.textContent = "Ajustes";
  panelContent.innerHTML = `
    <button class="action-btn">Opción 1</button>
    <button class="action-btn">Opción 2</button>
    <button class="action-btn">Opción 3</button>
  `;
}

// ---------------------- Botones principales ----------------------
document.getElementById("scan-btn").addEventListener("click", showScanView);
document.getElementById("patterns-btn").addEventListener("click", showPatternsView);
document.getElementById("settings-btn").addEventListener("click", showSettingsView);
document.getElementById("scramble-btn").addEventListener("click", () => {
    window.pywebview.api.rotar_cara();
});


// inicial: vacío
panelContent.innerHTML = "";


const ColoresBaseHex = {
    Blanco: 0xffffff,
    Amarillo: 0xffff00,
    Rojo: 0xff0000,
    Naranja: 0xff6600,
    Azul: 0x0000ff,
    Verde: 0x00aa00,
  };
const listaDeColores = [
  // U (Arriba) - Blanco
  ColoresBaseHex.Rojo, ColoresBaseHex.Blanco, ColoresBaseHex.Blanco,
  ColoresBaseHex.Blanco, ColoresBaseHex.Blanco, ColoresBaseHex.Blanco,
  ColoresBaseHex.Blanco, ColoresBaseHex.Blanco, ColoresBaseHex.Blanco,

  // D (Abajo) - Amarillo
  ColoresBaseHex.Amarillo, ColoresBaseHex.Amarillo, ColoresBaseHex.Amarillo,
  ColoresBaseHex.Amarillo, ColoresBaseHex.Amarillo, ColoresBaseHex.Amarillo,
  ColoresBaseHex.Amarillo, ColoresBaseHex.Amarillo, ColoresBaseHex.Amarillo,

  // F (Frente) - Verde
  ColoresBaseHex.Verde, ColoresBaseHex.Verde, ColoresBaseHex.Verde,
  ColoresBaseHex.Verde, ColoresBaseHex.Verde, ColoresBaseHex.Verde,
  ColoresBaseHex.Verde, ColoresBaseHex.Verde, ColoresBaseHex.Verde,

  // B (Atrás) - Azul
  ColoresBaseHex.Azul, ColoresBaseHex.Azul, ColoresBaseHex.Azul,
  ColoresBaseHex.Azul, ColoresBaseHex.Azul, ColoresBaseHex.Azul,
  ColoresBaseHex.Azul, ColoresBaseHex.Azul, ColoresBaseHex.Azul,

  // R (Derecha) - Rojo
  ColoresBaseHex.Blanco, ColoresBaseHex.Rojo, ColoresBaseHex.Rojo,
  ColoresBaseHex.Rojo, ColoresBaseHex.Rojo, ColoresBaseHex.Rojo,
  ColoresBaseHex.Rojo, ColoresBaseHex.Rojo, ColoresBaseHex.Rojo,

  // L (Izquierda) - Naranja
  ColoresBaseHex.Naranja, ColoresBaseHex.Naranja, ColoresBaseHex.Naranja,
  ColoresBaseHex.Naranja, ColoresBaseHex.Naranja, ColoresBaseHex.Naranja,
  ColoresBaseHex.Naranja, ColoresBaseHex.Naranja, ColoresBaseHex.Naranja,
];
