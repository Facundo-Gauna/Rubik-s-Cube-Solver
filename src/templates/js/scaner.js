/*
scaner.js — camera preview, snapshot and detection UI
================================================================

This module manages the camera preview, MJPEG streaming integration, snapshot
capture, and interactions required to run the color detection pipeline from the
web UI. It is a large UI module that coordinates with the Python backend via
`window.pywebview.api` and with other frontend controllers like rubik.

High-level responsibilities
- Connect to a local MJPEG preview server (Flask) and display the live stream
in an <img> element (or canvas) by fetching the latest JPEG frames.
- Provide controls to take snapshots, save images to the backend, and trigger
color detection/calibration flows using `window.pywebview.api` methods.
- Manage UI state for detection: show progress, results, error messages and
overlay detection markers on a canvas for debugging/visualization.
- Support saving/loading position files (positions1.json, positions2.json)
and allow the user to edit or re-run detection with the saved polygons.
- Offer convenience functions like `startPreview`, `stopPreview`, `takeSnapshot`,
and high-level actions bound to UI buttons.

Public API (attached to `window`)
- window.startPreviewLoopCam() -> start MJPEG fetching loop
- window.stopPreviewLoopCam() -> stop MJPEG fetching loop and release resources
- window.savePositions(json_str, filename) -> save polygon positions via backend
- window.loadPoints(idx) -> load positions from saved JSON
- window.detectFromSavedFrames() -> trigger the detector using saved frames

Notes on architecture
- The module uses a pull-based approach to MJPEG: it repeatedly requests a
JPEG from the backend `snapshot` endpoint instead of embedding an <img>
with multipart stream to avoid cross-origin or resource contention issues.
(This module also supports using an <img> source for live preview if desired.)
- It keeps two saved frame slots (img1,img2) for the two-photo detection flow.
- Detection is run in a non-blocking manner using async/await. While detection
runs, UI controls are disabled to prevent concurrent changes.
- Error handling: the module surfaces backend errors in UI dialogs and console
logs. In many places a failure is recoverable and the UI returns to idle.

Precautions & gotchas
- The module assumes the MJPEG server runs on localhost:5001 — change the
constants at the top if your server differs.
- Image coordinate systems: positions saved in positions1/2.json are pixel
coordinates relative to the original saved images (not the scaled preview).
Be careful when sampling ROI from the preview image vs saved full-resolution
image: the detector expects consistent coordinate spaces.
- Long-running detection: detection may take noticeable time for high-res
images. The UI shows a spinner and blocks some controls during detection.

------------------------------------------------------------------

Copyright (c) 2025 Facundo Gauna & Ulises Carnevale. MIT License.
*/

(() => {
  "use strict";

  const MJPEG_HOST = "127.0.0.1";
  const MJPEG_PORT = 5001;

  const BASE_DEFAULT_W = 1920;
  const BASE_DEFAULT_H = 1080;

  // small helpers
  const q = (sel, root = document) => root.querySelector(sel);
  const qa = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const on = (el, ev, fn, opts) => el && el.addEventListener(ev, fn, opts);

  function debounce(fn, wait) {
    let t = null;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), wait);
    };
  }

  // shared state for the module
  let photos = [null, null];
  let selectedPhotoBox = 0;
  let solution = "";
  let useForCalibration = false;
  let BASE_RES = { w: BASE_DEFAULT_W, h: BASE_DEFAULT_H, idx: 0 };
  let bottom_active = 0;
  window.bottom_active = bottom_active;

  // default positions to detect colors in every photo resolution.
  const DEFAULT_POSITIONS = [[
          {
              "U1": [702, 276], "U2": [886, 185], "U3": [1038, 117],
              "U4": [851, 356], "U5": [1034, 270], "U6": [1193, 186],
              "U7": [1031, 476], "U8": [1215, 353], "U9": [1368, 265],
              "F1": [1112, 629], "F2": [1299, 503], "F3": [1447, 412],
              "F4": [1127, 804], "F5": [1285, 681], "F6": [1422, 581],
              "F7": [1132, 949], "F8": [1273, 825], "F9": [1403, 727],
              "L1": [628, 421], "L2": [768, 514], "L3": [940, 634],
              "L4": [668, 589], "L5": [800, 691], "L6": [962, 809],
              "L7": [705, 740], "L8": [825, 835], "L9": [988, 964]
          },
          {
              "D1": [702, 276], "D2": [886, 185], "D3": [1038, 117],
              "D4": [851, 356], "D5": [1034, 270], "D6": [1193, 186],
              "D7": [1031, 476], "D8": [1215, 353], "D9": [1368, 265],
              "R1": [1112, 629], "R2": [1299, 503], "R3": [1447, 412],
              "R4": [1127, 804], "R5": [1285, 681], "R6": [1422, 581],
              "R7": [1132, 949], "R8": [1273, 825], "R9": [1403, 727],
              "B1": [628, 421], "B2": [768, 514], "B3": [940, 634],
              "B4": [668, 589], "B5": [800, 691], "B6": [962, 809],
              "B7": [705, 740], "B8": [825, 835], "B9": [988, 964]
          }
      ],
      [
          {
              "U1": [468, 184], "U2": [591, 123], "U3": [692, 78],
              "U4": [567, 237], "U5": [689, 180], "U6": [795, 124],
              "U7": [687, 317], "U8": [810, 235], "U9": [912, 177],
              "F1": [741, 419], "F2": [866, 335], "F3": [965, 275],
              "F4": [751, 536], "F5": [857, 454], "F6": [948, 387],
              "F7": [755, 633], "F8": [849, 550], "F9": [935, 485],
              "L1": [419, 281], "L2": [512, 343], "L3": [627, 423],
              "L4": [445, 393], "L5": [533, 461], "L6": [641, 540],
              "L7": [470, 493], "L8": [550, 557], "L9": [659, 643]
          },
          {
              "D1": [468, 184], "D2": [591, 123], "D3": [692, 78],
              "D4": [567, 237], "D5": [689, 180], "D6": [795, 124],
              "D7": [687, 317], "D8": [810, 235], "D9": [912, 177],
              "R1": [741, 419], "R2": [866, 335], "R3": [965, 275],
              "R4": [751, 536], "R5": [857, 454], "R6": [948, 387],
              "R7": [755, 633], "R8": [849, 550], "R9": [935, 485],
              "B1": [419, 281], "B2": [512, 343], "B3": [627, 423],
              "B4": [445, 393], "B5": [533, 461], "B6": [641, 540],
              "B7": [470, 493], "B8": [550, 557], "B9": [659, 643]
          }
      ],
      [
          {
              "U1": [351, 138], "U2": [443, 92], "U3": [519, 58],
              "U4": [425, 178], "U5": [517, 135], "U6": [596, 93],
              "U7": [515, 238], "U8": [607, 176], "U9": [684, 132],
              "F1": [556, 314], "F2": [649, 251], "F3": [723, 206],
              "F4": [563, 402], "F5": [642, 340], "F6": [711, 290],
              "F7": [566, 474], "F8": [636, 412], "F9": [701, 363],
              "L1": [314, 210], "L2": [384, 257], "L3": [470, 317],
              "L4": [334, 294], "L5": [400, 345], "L6": [481, 404],
              "L7": [352, 370], "L8": [412, 417], "L9": [494, 482]
          },
          {
              "D1": [351, 138], "D2": [443, 92], "D3": [519, 58],
              "D4": [425, 178], "D5": [517, 135], "D6": [596, 93],
              "D7": [515, 238], "D8": [607, 176], "D9": [684, 132],
              "R1": [556, 314], "R2": [649, 251], "R3": [723, 206],
              "R4": [563, 402], "R5": [642, 340], "R6": [711, 290],
              "R7": [566, 474], "R8": [636, 412], "R9": [701, 363],
              "B1": [314, 210], "B2": [384, 257], "B3": [470, 317],
              "B4": [334, 294], "B5": [400, 345], "B6": [481, 404],
              "B7": [352, 370], "B8": [412, 417], "B9": [494, 482]
          }
      ]
    ];

  // points / elements local
  let pointsData = [];
  let pointElements = [{}, {}];
  let pointsEditable = true;
  let selectedColor = null;
  let _currentStreamUrl = null;
  let _previewImgElement = null;

  async function setStreamParams({ width, height, fps = 10, quality = 70 } = {}) {
    try {
      const url = `http://${MJPEG_HOST}:${MJPEG_PORT}/set_stream_params`;
      const body = { width: Number(width), height: Number(height), fps: Number(fps), quality: Number(quality) };
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const json = await resp.json();
      if (!json.ok) {
        console.warn("setStreamParams failed:", json);
        return null;
      }
      console.log("Stream params updated:", json);

      // Update stream URL only if changed
      const newUrl = `http://${MJPEG_HOST}:${MJPEG_PORT}/stream`;
      if (_currentStreamUrl !== newUrl) {
        _currentStreamUrl = newUrl;
        // If a preview image is attached, update its src once (this will open one connection)
        if (_previewImgElement) {
          // avoid reassigning same src repeatedly
          if (!_previewImgElement.src || !_previewImgElement.src.includes(newUrl)) {
            _previewImgElement.src = newUrl;
            // add cache-busting only when we *really* need to force a reload:
            // _previewImgElement.src = newUrl + '?v=' + Date.now();
          }
        }
      }

      return json;
    } catch (e) {
      console.warn("setStreamParams error:", e);
      return null;
    }
  }

  // Start preview: assign src ONCE and keep a reference to the img element.
  // Avoid appending a timestamp every call.
  function startPreviewLoop(previewImgElement) {
    if (!previewImgElement) return;
    _previewImgElement = previewImgElement;

    // If stream url not configured yet, use default base
    if (!_currentStreamUrl) _currentStreamUrl = `http://${MJPEG_HOST}:${MJPEG_PORT}/stream`;

    // Only set src if it's not already using the current stream url (prevents re-opening)
    try {
      const cur = previewImgElement.src || "";
      if (!cur.includes(_currentStreamUrl)) {
        previewImgElement.src = _currentStreamUrl;
      }
    } catch (e) {
      previewImgElement.src = _currentStreamUrl;
    }
  }

  function stopPreviewLoopCam() {
    try {
      if (_previewImgElement) {
        _previewImgElement.src = "";
        _previewImgElement.removeAttribute("src");
        _previewImgElement = null;
      }
    } catch (e) {
      console.warn("stopPreviewLoopCam error:", e);
    }
  }

  window.stopPreviewLoopCam = stopPreviewLoopCam;

  // canvas drawing
  function drawPoints(canvas, positions, pointElems = {}) {
    if (!canvas || !positions) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const scaleX = canvas.width / BASE_RES.w;
    const scaleY = canvas.height / BASE_RES.h;

    // letterToCss mapping
    const letterToCss = { "W":"white","Y":"yellow","R":"red","O":"orange","B":"blue","G":"green","null":"gray" };

    for (const key in positions) {
      const [x, y] = positions[key];
      const letter = (pointElems && pointElems[key] && pointElems[key].color) ? String(pointElems[key].color).toUpperCase() : "null";
      ctx.fillStyle = letterToCss[letter] || "gray";
      ctx.beginPath();
      ctx.arc(x * scaleX, y * scaleY, 7, 0, Math.PI*2);
      ctx.fill();
    }
  }

  function drawPreviewPoints(imgElement) {
    if (!imgElement) return;
    const container = imgElement.parentElement;
    if (!container) return;
    const positionsObj = DEFAULT_POSITIONS[BASE_RES.idx][selectedPhotoBox];
    // remove existing .point nodes
    container.querySelectorAll(".point").forEach(n => n.remove());

    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const scaleX = containerWidth / BASE_RES.w;
    const scaleY = containerHeight / BASE_RES.h;

    const letterToCss = { "U":"#0000FF","F":"#FFFF00","L":"#FF0000","B":"#FFFFFF","R":"#FFA500","D":"#008000" };

    for (const key in positionsObj) {
      const [x,y] = positionsObj[key];
      const div = document.createElement("div");
      div.className = "point";
      div.style.position = "absolute";
      div.style.width = "14px";
      div.style.height = "14px";
      div.style.borderRadius = "50%";
      div.style.left = `${x * scaleX - 7}px`;
      div.style.top = `${y * scaleY - 7}px`;
      div.style.zIndex = 10;
      div.style.background = letterToCss[key.charAt(0)] || "#808080";
      container.appendChild(div);
    }
  }

  function update_box(img, idx = null) {
    if (idx === null) selectedPhotoBox = 1 - selectedPhotoBox;
    else selectedPhotoBox = idx;
    drawPreviewPoints(img);
  }

  // setup canvas: one resize handler per wrapper; returns cleanup when needed
  function setupCanvasFor(wrapper, idx, positions) {
    const img = wrapper.querySelector("img");
    const canvas = wrapper.querySelector("canvas");
    if (!img || !canvas) return { img, canvas: null, resize: () => {} };

    const ctx = canvas.getContext("2d");

    const resizeCanvas = () => {
      const width = img.clientWidth || wrapper.clientWidth || 640;
      const height = img.clientHeight || wrapper.clientHeight || 480;
      const newW = Math.max(1, Math.round(width));
      const newH = Math.max(1, Math.round(height));
      if (canvas.width !== newW || canvas.height !== newH) {
        canvas.width = newW;
        canvas.height = newH;
        drawPoints(canvas, positions, pointElements[idx]);
      }
    };

    const debouncedResize = debounce(resizeCanvas, 120);

    // attach handlers safely (avoid multiple attaches)
    img.removeEventListener("load", resizeCanvas);
    img.addEventListener("load", () => { setTimeout(resizeCanvas, 8); });
    // use debounced on window resize
    window.removeEventListener("resize", debouncedResize);
    window.addEventListener("resize", debouncedResize);
    if (img.complete) setTimeout(() => { resizeCanvas(); }, 10);

    return { img, canvas, ctx, resize: resizeCanvas };
  }

  function attachDragHandlers(idx, canvas, positions) {
    if (!canvas) return;
    let dragging = null;

    const onDown = (e) => {
      if (!pointsEditable) return;
      canvas.setPointerCapture?.(e.pointerId);
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      for (const key in positions) {
        const [x, y] = positions[key];
        const scaledX = x * canvas.width / BASE_RES.w;
        const scaledY = y * canvas.height / BASE_RES.h;
        if (Math.hypot(mouseX - scaledX, mouseY - scaledY) < 10) {
          dragging = key;
          break;
        }
      }
    };

    const onMove = (e) => {
      if (!dragging) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const baseX = (mouseX / canvas.width) * BASE_RES.w;
      const baseY = (mouseY / canvas.height) * BASE_RES.h;
      positions[dragging] = [
        Math.max(0, Math.min(BASE_RES.w, baseX)),
        Math.max(0, Math.min(BASE_RES.h, baseY))
      ];
      // only redraw the canvas that changed
      drawPoints(canvas, positions, pointElements[idx]);
    };

    const endDrag = () => { dragging = null; };

    on(canvas, "pointerdown", onDown);
    on(canvas, "pointermove", onMove);
    on(canvas, "pointerup", endDrag);
    on(canvas, "pointercancel", endDrag);
    on(canvas, "pointerleave", endDrag);
  }

  // try load saved positions - returns array or null
  async function tryLoadSavedPositions() {
    try {
      const p0 = await window.pywebview.api.load_points(0);
      const p1 = await window.pywebview.api.load_points(1);
      const ok0 = p0 && typeof p0 === "object" && Object.keys(p0).length >= 9;
      const ok1 = p1 && typeof p1 === "object" && Object.keys(p1).length >= 9;
      if (ok0 && ok1) return [p0, p1];
      if (ok0 && !ok1) return [p0, DEFAULT_POSITIONS[BASE_RES.idx][1]];
      if (!ok0 && ok1) return [DEFAULT_POSITIONS[BASE_RES.idx][0], p1];
      return null;
    } catch (e) {
      console.warn("Could not load saved positions:", e);
      return null;
    }
  }

  // View 1: Scan preview + capture
  function showScanView() {

    photos = [null, null];
    selectedPhotoBox = 0;

    if (!document.getElementById("panel-title") || !document.getElementById("panel-content")) {
      console.error("panel elements missing");
      return;
    }

    const panelTitle = document.getElementById("panel-title");
    const panelContent = document.getElementById("panel-content");

    panelTitle.textContent = "Scan — Step 1";
    panelContent.innerHTML = `
      <div class="camera-view" style="position:relative;">
        <img id="previewImg" alt="preview" style="width:100%; display:block;">
      </div>
      <button class="action-btn capture-btn" id="captureBtn">Capture image</button>
      <div class="photos-container">
        <div class="photo-box selected" data-index="0"></div>
        <div class="photo-box" data-index="1"></div>
      </div>
      <button class="action-btn next-btn" id="step1Next" disabled>Next</button>
    `;

    const previewImg = document.getElementById("previewImg");
    const captureBtn = document.getElementById("captureBtn");
    const photoBoxes = panelContent.querySelectorAll(".photo-box");
    const nextBtn = document.getElementById("step1Next");

    // ensure preview draws
    if(_previewImgElement == null) startPreviewLoop(previewImg);

    // redraw preview points on resize
    const onPreviewResize = () => drawPreviewPoints(previewImg);
    window.removeEventListener("resize", onPreviewResize);
    window.addEventListener("resize", onPreviewResize);
    previewImg.addEventListener("load", onPreviewResize);
    if (previewImg.complete) setTimeout(onPreviewResize, 10);

    // photo box click handlers
    photoBoxes.forEach(box => {
      box.addEventListener("click", () => {
        photoBoxes.forEach(b => b.classList.remove("selected"));
        box.classList.add("selected");
        update_box(previewImg, Number(box.dataset.index));
      });
    });

    // capture snapshot
    on(captureBtn, "click", async () => {
      try {
        const resp = await window.pywebview.api.save_frame(selectedPhotoBox);
        if (!resp || !resp.ok) throw new Error(resp && resp.error ? resp.error : "save_frame failed");
      
        const dataUrl = resp.data_url || "";
        const box = photoBoxes[selectedPhotoBox];
        if (box && dataUrl) {
          box.style.backgroundImage = `url("${dataUrl}")`;
          box.style.backgroundSize = "cover";
          box.style.backgroundPosition = "center";
        }
      
        photos[selectedPhotoBox] = dataUrl;
        nextBtn.disabled = !(photos[0] && photos[1]);
      
        update_box(previewImg);
        photoBoxes.forEach(b => b.classList.remove("selected"));
        const nextBox = photoBoxes[selectedPhotoBox];
        if (nextBox) nextBox.classList.add("selected");
      } catch (e) {
        console.error("capture failed:", e);
        alert("Failed to capture snapshot: " + e);
      }
    });


    on(nextBtn, "click", showAdjustPointsView);
  }

  // View 2: Adjust points
  async function showAdjustPointsView() {

    pointsEditable = true;
    selectedColor = null;

    const panelTitle = document.getElementById("panel-title");
    const panelContent = document.getElementById("panel-content");
    panelTitle.textContent = "Adjust points — Step 2";
    panelContent.innerHTML = `
      <div class="image-step2" style="display:flex;gap:10px;">
        <div class="img-wrapper" data-index="0" style="flex:1; position:relative;">
          <img src="${photos[0] || ""}" alt="Image 1" style="width:100%;display:block;">
          <canvas class="points-canvas" style="position:absolute;left:0;top:0;right:0;bottom:0;"></canvas>
        </div>
        <div class="img-wrapper" data-index="1" style="flex:1; position:relative;">
          <img src="${photos[1] || ""}" alt="Image 2" style="width:100%;display:block;">
          <canvas class="points-canvas" style="position:absolute;left:0;top:0;right:0;bottom:0;"></canvas>
        </div>
      </div>

      <div class="controls-column">
        <div class="color-buttons-container" style="display:none; text-align:center;">
          <button class="color-btn" data-color="W" style="background:white;"></button>
          <button class="color-btn" data-color="Y" style="background:yellow;"></button>
          <button class="color-btn" data-color="R" style="background:red;"></button>
          <button class="color-btn" data-color="O" style="background:orange;"></button>
          <button class="color-btn" data-color="B" style="background:blue;"></button>
          <button class="color-btn" data-color="G" style="background:green;"></button>
        </div>

        <div class="calibrate-row">
          <button id="restore-defaults" class="action-btn">Restore default points</button>
          <button id="use-calibrate-btn" class="action-btn calibrate-btn" aria-pressed="false" style="display:none;">Use to calibrate</button>
        </div>

        <button class="action-btn next-btn" id="step2Next">Next</button>
      </div>
    `;

    stopPreviewLoopCam();
    
    // load saved or default positions
    const saved = await tryLoadSavedPositions();
    if (saved && Array.isArray(saved) && saved.length === 2) pointsData = saved;
    else pointsData = DEFAULT_POSITIONS[BASE_RES.idx].map(x => ({ ...x }));

    // init pointElements
    pointElements = [{}, {}];
    for (let i = 0; i < 2; i++) {
      for (const key in pointsData[i]) pointElements[i][key] = { color: "null" };
    }

    const wrappers = qa(".img-wrapper", document.getElementById("panel-content"));
    wrappers.forEach((wrapper, idx) => {
      const { img, canvas, resize } = setupCanvasFor(wrapper, idx, pointsData[idx]);
      if (canvas) attachDragHandlers(idx, canvas, pointsData[idx]);
      if (img && img.complete) setTimeout(() => resize(), 10);
    });

    const restoreBtn = q("#restore-defaults", document.getElementById("panel-content"));
    if (restoreBtn) {
      const handler = async () => {
        try {
          console.log("Restore defaults: resetting points...");
        
          // 1) Reset logical positions (deep clone each object)
          pointsData = DEFAULT_POSITIONS[BASE_RES.idx].map(x => ({ ...x }));
        
          // 2) Reset pointElements (colors -> "null")
          pointElements = [{}, {}];
          for (let i = 0; i < 2; i++) {
            for (const k in pointsData[i]) {
              pointElements[i][k] = { color: "null" };
            }
          }
        
          // 3) For each wrapper: replace canvas node to remove old listeners, set sizes, redraw, reattach handlers
          const freshWrappers = qa(".img-wrapper", document.getElementById("panel-content"));
          freshWrappers.forEach((wrapper, idx) => {
            const img = wrapper.querySelector("img");
            const oldCanvas = wrapper.querySelector("canvas");
          
            // If there is an old canvas, replace it with a fresh clone (this removes all listeners)
            let canvas;
            if (oldCanvas) {
              const newCanvas = document.createElement("canvas");
              newCanvas.className = oldCanvas.className || "points-canvas";
              // ensure style/position preserved
              newCanvas.style.cssText = oldCanvas.style.cssText || "";
              oldCanvas.replaceWith(newCanvas);
              canvas = newCanvas;
            } else {
              // create one if missing
              canvas = document.createElement("canvas");
              canvas.className = "points-canvas";
              canvas.style.position = "absolute";
              canvas.style.top = 0;
              canvas.style.left = 0;
              canvas.style.right = 0;
              canvas.style.bottom = 0;
              wrapper.appendChild(canvas);
            }
          
            // 4) ensure canvas size and draw initial default points
            const resizeAndDraw = () => {
              const iw = img.clientWidth || wrapper.clientWidth || 640;
              const ih = img.clientHeight || wrapper.clientHeight || 480;
              canvas.width = Math.max(1, Math.round(iw));
              canvas.height = Math.max(1, Math.round(ih));
              drawPoints(canvas, pointsData[idx], pointElements[idx]);
            };
          
            // 5) (Re)attach handlers via setupCanvasFor / attachDragHandlers to get consistent behavior
            // setupCanvasFor will add img load & window resize listeners; attachDragHandlers will add pointer handlers
            const { img: returnedImg, canvas: returnedCanvas, resize } = setupCanvasFor(wrapper, idx, pointsData[idx]);
            if (returnedCanvas) {
              // Make sure drag handlers are bound to the new canvas (setupCanvasFor created a fresh canvas)
              attachDragHandlers(idx, returnedCanvas, pointsData[idx]);
            }
          
            // If image is already loaded, trigger an immediate resize/draw
            if (img && img.complete) {
              setTimeout(resizeAndDraw, 10);
            } else if (img) {
              // if not loaded, will be drawn by the load listener inside setupCanvasFor
            }
          });
        
          console.log("Restore defaults: done.");
        } catch (err) {
          console.error("Error restoring defaults:", err);
          alert("Error restoring defaults: " + err);
        }
      };
    
      // bind (removing previous listener if exists to avoid duplicates)
      restoreBtn.removeEventListener("click", restoreBtn._handler);
      restoreBtn._handler = handler;
      restoreBtn.addEventListener("click", restoreBtn._handler);
      restoreBtn.style.display = "";
    }


    // step2 Next -> enable color mode and move to step 3
    const step2Next = q("#step2Next", document.getElementById("panel-content"));
    on(step2Next, "click", async function activateColorModeOnce() {
      try {
        await window.pywebview.api.save_json(JSON.stringify(pointsData[0]), "positions1.json");
        await window.pywebview.api.save_json(JSON.stringify(pointsData[1]), "positions2.json");
        document.getElementById("panel-title").textContent = "Paint points — Step 3";
        enableColorMode();
        await loadAndApplyPointColors();

        step2Next.removeEventListener("click", activateColorModeOnce);
        step2Next.addEventListener("click", async () => {
          const ok = await saveColorsClick();
          if (!ok) return;
        });
      } catch (e) {
        console.error(e);
        alert("Error saving coordinates: " + e);
      }
    });
  }

  // enable color painting
  function enableColorMode() {
    pointsEditable = false;
    const panelContent = document.getElementById("panel-content");
    const restoreBtn = q("#restore-defaults", panelContent);
    if (restoreBtn) { restoreBtn.style.display = "none"; }

    const calibrateBtn = q("#use-calibrate-btn", panelContent);
    if (calibrateBtn) {
      calibrateBtn.style.display = "";
      calibrateBtn.classList.toggle("active", !!useForCalibration);
      calibrateBtn.setAttribute("aria-pressed", useForCalibration ? "true" : "false");
      calibrateBtn.textContent = useForCalibration ? "Calibrating: ON" : "Use to calibrate";
      if (!calibrateBtn._handler) {
        calibrateBtn._handler = () => {
          useForCalibration = !useForCalibration;
          calibrateBtn.classList.toggle("active", useForCalibration);
          calibrateBtn.setAttribute("aria-pressed", useForCalibration ? "true" : "false");
          calibrateBtn.textContent = useForCalibration ? "Calibrating: ON" : "Use to calibrate";
        };
        calibrateBtn.addEventListener("click", calibrateBtn._handler);
      }
    }

    const container = q(".color-buttons-container", panelContent);
    if (container) container.style.display = "block";
    const buttons = qa(".color-btn", container);
    buttons.forEach(btn => {
      btn.disabled = false;
      if (btn._handler) btn.removeEventListener("click", btn._handler);
      btn._handler = () => {
        buttons.forEach(b => b.classList.remove("selected"));
        btn.classList.add("selected");
        selectedColor = btn.dataset.color;
      };
      btn.addEventListener("click", btn._handler);
    });

    // canvas color-assign click
    const wrappers = qa(".img-wrapper", panelContent);
    wrappers.forEach((wrapper, idx) => {
      const canvas = wrapper.querySelector("canvas");
      if (!canvas) return;
      const handler = (e) => {
        if (!selectedColor) return;
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        for (const key in pointsData[idx]) {
          const [x, y] = pointsData[idx][key];
          const scaledX = x * canvas.width / BASE_RES.w;
          const scaledY = y * canvas.height / BASE_RES.h;
          if (Math.hypot(mouseX - scaledX, mouseY - scaledY) < 10) {
            pointElements[idx][key].color = selectedColor;
            drawPoints(canvas, pointsData[idx], pointElements[idx]);
            break;
          }
        }
      };
      canvas.addEventListener("click", handler);
    });
  }

  async function saveColorsClick() {
    try {
      const lettersMap = {};
      for (const idx in pointElements) {
        for (const key in pointElements[idx]) lettersMap[key] = (pointElements[idx][key].color || "W").toUpperCase().charAt(0);
      }
      const val = await window.pywebview.api.solve(lettersMap);
      if (!val || !val.ok) {
        alert("Backend validation error: " + ((val && val.error) || "unknown"));
        console.warn("validate_cube_state returned:", val && val.debug ? val.debug : val);
        return false;
      }
      solution = val.solution;
      rubik.drawCube(rubik.invert(solution));
      rubik.turnKeys(false);


      if(useForCalibration){
        console.debug("Calibraing colors ...")
        const ans = await window.pywebview.api.calibrate_colors(lettersMap);
        if (!ans){
          console.error("Calibration couldnt be saved");
        }
        useForCalibration = false;
      }

      completeScan();
      return true;
    } catch (e) {
      console.error("Error saving colors:", e);
      alert("Error saving colors: " + e);
      return false;
    }
  }

  async function loadAndApplyPointColors() {
    try {
      const res = await window.pywebview.api.color_detector();
      if (!res.ok) {
        alert("Color detection failed: " + (res.error || "Unknown. Improve lighting/adjust positions."));
        console.warn("color_detector:", res);
        return;
      }
      const mapping = res.colors || {};
      for (let idx = 0; idx < pointElements.length; idx++) {
        for (const k in pointElements[idx]) {
          pointElements[idx][k].color = mapping && mapping[k] ? mapping[k].toUpperCase().charAt(0) : "W";
        }
        const canvas = document.querySelectorAll(".img-wrapper canvas")[idx];
        if (canvas) drawPoints(canvas, pointsData[idx], pointElements[idx]);
      }
    } catch (e) {
      console.error("Error loading colors:", e);
      alert("Error getting colors from backend: " + e);
    }
  }

  function completeScan() {
    window.setCalibratedStatus(true);
    window.bottom_active = 1;
    const panelTitle = document.getElementById("panel-title");
    const panelContent = document.getElementById("panel-content");
    panelTitle.textContent = "Scan complete";
    panelContent.innerHTML = `
      <div class="scan-complete-container">
        <img src="patterns/pattern0.svg" alt="Scan Complete" style="max-width:100%;display:block;margin:0 auto;">
        <h2>Scan completed successfully</h2>
        <p>All points were captured and saved.</p>
        <button class="action-btn" id="rescanBtn">Rescan</button>
      </div>
    `;
    const rescanBtn = document.getElementById("rescanBtn");
    if (rescanBtn) rescanBtn.addEventListener("click", showScanView);
  }

  // Bind external UI buttons
  document.addEventListener("DOMContentLoaded", () => {
    const scanBtn = document.getElementById("scan-btn");
    if (scanBtn) scanBtn.addEventListener("click", showScanView);

    const solveBtn = document.getElementById("solve-btn");
    if (solveBtn) {
      solveBtn.addEventListener("click", async () => {
        if (window.bottom_active == 2) {
          await window.doPatternZero();
          return;
        }

        if (!solution) {
          const ans = await window.pywebview.api.solve();
          if (!ans || !ans.ok) {
            alert("Error getting solution: " + ((ans && ans.error) || "unknown"));
            console.warn("solve returned:", ans && ans.debug ? ans.debug : ans);
            return;
          }
          solution = ans.solution;
        }

        const seqStr = solution;
        if (!seqStr) return;

        try {
          await rubik.sequence(seqStr);
        } catch (e) {
          console.error("Error during solve sequence:", e);
          alert("Error executing solution: " + (e.message || e));
        }
        solution = "";
        window.bottom_active = 0;
        rubik.turnKeys(true);
      });
    }

    const scrambleBtn = document.getElementById("scramble-btn");
    if (scrambleBtn) {
      scrambleBtn.addEventListener("click", async () => {
        if (rubik.busy || rubik.sequenceRunning || solution !== "" || window.bottom_active !== 0) return;
        try {
          window.bottom_active = 1;
          rubik.turnKeys(false);
          const scrambleSequence = await window.pywebview.api.scramble();
          await rubik.sequence(scrambleSequence);
          solution = "";
        } catch (e) {
          console.error("Error while executing scramble:", e);
          const errorMessage = e && e.message ? e.message : e;
          alert("An error occurred generating or executing the scramble: " + errorMessage);
        }
      });
    }
  });

  // expose functions
  window.showScanView = showScanView;

  window.setBaseResolution = async function(w, h) {
    BASE_RES.w = Number(w) || BASE_DEFAULT_W;
    BASE_RES.h = Number(h) || BASE_DEFAULT_H;
    BASE_RES.idx = (BASE_RES.w === 1920) ? 0 : (BASE_RES.w === 1280 ? 1 : 2);

    const streamW = (BASE_RES.w >= 1920) ? 1280 : (BASE_RES.w >= 1280 ? 960 : 640);
    const streamH = Math.round(streamW * (BASE_RES.h / BASE_RES.w));

    const res = await setStreamParams({ width: streamW, height: streamH, fps: 12, quality: 70 });
    if (!res) {
      console.warn("Could not update stream params; leaving previous stream resolution.");
    }

    document.querySelectorAll(".points-canvas").forEach((c, idx) => {
      const wrapper = c.parentElement;
      const img = wrapper ? wrapper.querySelector("img") : null;
      if (img && img.complete) {
        const newW = img.clientWidth;
        const newH = img.clientHeight;
        if (c.width !== newW || c.height !== newH) {
          c.width = newW;
          c.height = newH;
        }
        drawPoints(c, pointsData[idx], pointElements[idx]);
      }
    });

    console.log("Base resolution set to:", BASE_RES);
  };

  // close preview when window/tab is hidden or before unload
  window.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
      stopPreviewLoopCam();
    }
  });

  window.addEventListener('beforeunload', () => {
    stopPreviewLoopCam();
  });


  window.sleep = function(ms) { return new Promise(res => setTimeout(res, ms)); };
})();
          