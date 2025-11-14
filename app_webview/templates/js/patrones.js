document.getElementById("patterns-btn").addEventListener("click", showPatternsView);

let indice = 0;

const patterns = [
  "L2 L2",
  "F2 B2 U2 D2 R2 L2",
  "R' L' U2 F2 D2 F2 R L B2 U2 B2 U2",
  "U B D' F2 D B' U' R2 D F2 D' R2 D F2 D' R2",
  "L' B' D U R U' R' D2 R2 D L D' L' R' F U",
  "U2 R2 L2 U2 R2 L2",
  "U D R L' F' B U D' R2 U R2 L2 D2 F2 B2 D",
  "F U F R L2 B D' R D2 L D' B R2 L F U F",
  "R2 L' D F2 R' D' R' L U' D R D B2 R' U D2",
  "F L F U' R U F2 L2 U' L' B D' B' L2 U",
  "U D' R L' F B' U D'",
  "F R' U L F' L' F U' R U L' U' L F'",
  "F D F' D2 L' B' U L D R U L' F' U L U2",
  "F L' D F' U' B U F U' F R' F2 L U' R' D2",
  "R2 F2 L2 R2 F2 L2",
  "L2 F2 D' L2 B2 D' U' R2 B2 U' L' B2 L D L B' D L' U",
  "F2 D2 F' L2 D2 U2 R2 B' U2 F2",
  "F B' U F U F U L B L2 B' U F' L U L' B"
];

function click_out(img, selectedImg){
  if (img !== selectedImg) {
    img.style.transform = "scale(1.07)";
    img.style.boxShadow = "0 4px 12px rgba(255, 215, 0, 0.4)";
    img.style.borderColor = "#FFD700"; // 🔸 Amarillo en hover
  }
}

function showPatternsView() {
  panelTitle.textContent = "Patrones";
  panelContent.innerHTML = `<div class="patterns-container"></div>`;
  const container = panelContent.querySelector(".patterns-container");
  // usamos la selectedImg global para que la vidriera y la vista compartan estado
  // (antes tenías una variable local aquí que provocaba desincronías)
  // let selectedImg = null;   <-- removido

  for (let i = 0; i < 19; i++) {
    const img = document.createElement("img");
    img.className = "pattern";
    img.src = `patrones/patron${i}.svg`;
    img.alt = `Patrón ${i}`;
    img.id = i;
    img.style.transition = "transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease";
    img.style.border = "2px solid #ccc";
    img.style.boxShadow = "0 2px 6px rgba(0,0,0,0.15)";
    img.style.borderRadius = "10px";

    img.addEventListener("mouseover", () => {
      click_out(img, selectedImg);
    });

    img.addEventListener("mouseout", () => {
      if (img !== selectedImg) {
        img.style.transform = "scale(1)";
        img.style.boxShadow = "0 2px 6px rgba(0,0,0,0.15)";
        img.style.borderColor = "#ccc";
      }
    });

    img.addEventListener("click", async () => {
      if (selectedImg === img) return;
      if (typeof rubik !== "undefined" && rubik.busy) return;

      if (selectedImg) {
        selectedImg.style.borderColor = "#ccc";
        selectedImg.style.boxShadow = "0 2px 6px rgba(0,0,0,0.15)";
        selectedImg.style.transform = "scale(1)";
      }

      selectedImg = img;
      img.style.borderColor = "#FFD700"; // 🔸 Amarillo seleccionado
      img.style.boxShadow = "0 4px 16px rgba(255, 215, 0, 0.6)";
      img.style.transform = "scale(1.1)";

      await hacerpatron(img.id);
    });

    container.appendChild(img);

    if (i == indice) {
      selectedImg = img;
      img.style.borderColor = "#FFD700";
      img.style.boxShadow = "0 4px 16px rgba(255, 215, 0, 0.6)";
      img.style.transform = "scale(1.1)";
    }
  }
}

let selectedImg = null;              // img DOM actualmente resaltada (o null)
let vidrieraRunning = false;         // true mientras el loop async esté en marcha
let vidrieraStopRequested = false;   // true si se pidió detener la vidriera (ESC / toggle)
let _vidrieraEscHandler = null;      // referencia al handler de teclado

function sleep(ms) { return new Promise(res => setTimeout(res, ms)); }

async function hacerpatron(id) {
  if (id == 0) {
    if (indice == 0) return;
    await rubik.SecuenciaDeGiros(invert(patterns[indice]));
  } else if (id == 18) {
    if (vidrieraRunning && !vidrieraStopRequested) {
      stopVidriera();
      return;
    } else if (vidrieraRunning && vidrieraStopRequested) {
      return;
    } else {
      await startVidriera();
      return;
    }
  } else {
    if (indice != 0) {
      const str = invert(patterns[indice]) + " " + patterns[id];
      await rubik.SecuenciaDeGiros(str);
    } else {
      await rubik.SecuenciaDeGiros(patterns[id]);
    }
  }
  indice = id;
}

async function startVidriera() {
  if (vidrieraRunning) return;
  vidrieraRunning = true;
  vidrieraStopRequested = false;

  // Forzamos un pequeño delay para asegurar que la UI esté lista
  await sleep(120);

  // Handler ESC: marca la solicitud de parada (no interrumpe lo que esté en curso)
  _vidrieraEscHandler = function (ev) {
    if (ev.key === "Escape") {
      if (!vidrieraRunning) return;
      if (!vidrieraStopRequested) {
        vidrieraStopRequested = true;
        // feedback visual: outline en la miniatura actual
        if (selectedImg) selectedImg.style.outline = "3px dashed #ff5252";
      }
    }
  };
  window.addEventListener("keydown", _vidrieraEscHandler);

  // Loop principal (se ejecuta hasta que vidrieraStopRequested === true)
  (async () => {
    let prevIndex = 0; // índice del patrón previamente aplicado (0 = resuelto)
    try {
      while (!vidrieraStopRequested) {
        for (let i = 0; i < patterns.length; i++) {
          if (vidrieraStopRequested) break;
          if (i === 18) continue; // 18 es el activador; lo saltamos

          // Ejecutar patrón usando la lógica de transición
          try {
            if (prevIndex !== 0) {
              const transition = invert(patterns[prevIndex]) + " " + patterns[i];
              await rubik.SecuenciaDeGiros(transition);
            } else {
              await rubik.SecuenciaDeGiros(patterns[i]);
            }
          } catch (e) {
            console.error("Vidriera: error ejecutando patrón", i, e);
            // solicitamos parada responsable para evitar seguir en mal estado
            vidrieraStopRequested = true;
          }

          // *** Actualizamos UI DESPUÉS de que la secuencia terminó (mantener sincronía) ***
          try {
            // des-resaltamos anterior
            if (selectedImg && selectedImg.id !== String(i)) {
              selectedImg.style.borderColor = "#ccc";
              selectedImg.style.boxShadow = "0 2px 6px rgba(0,0,0,0.15)";
              selectedImg.style.transform = "scale(1)";
              selectedImg.style.outline = "none";
            }

            // resaltamos la miniatura actual si existe
            const imgEl = document.getElementById(String(i));
            if (imgEl) {
              imgEl.style.borderColor = "#FFD700";
              imgEl.style.boxShadow = "0 4px 16px rgba(255, 215, 0, 0.6)";
              imgEl.style.transform = "scale(1.08)";
              imgEl.style.outline = "none";
              selectedImg = imgEl;
            } else {
              selectedImg = null;
            }
          } catch (e) {
            console.warn("Vidriera: error actualizando miniatura:", e);
            selectedImg = null;
          }

          // actualizar índices para coherencia global
          prevIndex = i;
          indice = i;

          // esperar 5 segundos mostrando este patrón; durante la espera
          // chequeamos vidrieraStopRequested periódicamente para salir tras terminar patrón
          const totalWait = 5000;
          const chunk = 200;
          let waited = 0;
          while (!vidrieraStopRequested && waited < totalWait) {
            await sleep(chunk);
            waited += chunk;
          }
          if (vidrieraStopRequested) break;
        }
        // Si completó el for sin stopRequested, vuelve al while y repite (bucle infinito)
      }
    } finally {
      try {
        const lastIdx = prevIndex; // índice del último patrón aplicado

        // si no hay patrón aplicable, sólo limpiamos UI
        if (lastIdx && lastIdx !== 0 && typeof patterns[lastIdx] === "string" && patterns[lastIdx].trim() !== "") {
          // esperar a que rubik termine cualquier trabajo interno
          // (es habitual que rubik.busy todavía sea true justo después de await)
          const maxWait = 8000; // timeout de seguridad (ms)
          let waited = 0;
          const waitChunk = 100;

          while (typeof rubik !== "undefined" && rubik.busy && waited < maxWait) {
            await sleep(waitChunk);
            waited += waitChunk;
          }

          // retardo final para estabilidad
          await sleep(10000);

          const rev = invert(patterns[lastIdx]);
          if (rev && rev.trim() !== "") {
            try {
              await rubik.SecuenciaDeGiros(rev);
            } catch (e) {
              console.error("Vidriera: error aplicando inversa del último patrón", lastIdx, e);
            }
          } else {
            console.warn("Vidriera: inversa calculada vacía para index", lastIdx);
          }
        } else {
          // nothing to invert (ya estaba en 0 o índice inválido)
        }

        // Limpiamos resaltados previos
        if (selectedImg) {
          selectedImg.style.borderColor = "#ccc";
          selectedImg.style.boxShadow = "0 2px 6px rgba(0,0,0,0.15)";
          selectedImg.style.transform = "scale(1)";
          selectedImg.style.outline = "none";
          selectedImg = null;
        }

        // Marcamos índice 0 (resuelto) y resaltamos la miniatura 0 si existe
        indice = 0;
        const img0 = document.getElementById("0");
        if (img0) {
          img0.style.borderColor = "#FFD700";
          img0.style.boxShadow = "0 4px 16px rgba(255, 215, 0, 0.6)";
          img0.style.transform = "scale(1.08)";
          selectedImg = img0;
        }
      } catch (e) {
        console.warn("Vidriera: error durante la secuencia de cierre:", e);
      } finally {
        // Reset flags y remover listener
        vidrieraRunning = false;
        vidrieraStopRequested = false;
        if (_vidrieraEscHandler) {
          window.removeEventListener("keydown", _vidrieraEscHandler);
          _vidrieraEscHandler = null;
        }
      }
    }

  })();
}

function stopVidriera() {
  if (!vidrieraRunning) return;
  vidrieraStopRequested = true;
  if (selectedImg) selectedImg.style.outline = "3px dashed #ff5252";
}

