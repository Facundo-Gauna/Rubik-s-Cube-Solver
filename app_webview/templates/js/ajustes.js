const panelTitle = document.getElementById("panel-title");
const panelContent = document.getElementById("panel-content");
panelContent.innerHTML = "";

document.getElementById("settings-btn").addEventListener("click", showSettingsView);

const statusCalDot = document.getElementById("status-calibrated-dot");
const statusCalText = document.getElementById("status-calibrated-text");
const statusArdDot = document.getElementById("status-arduino-dot");
const statusArdText = document.getElementById("status-arduino-text");

function hasPywebview() {
  return typeof window.pywebview !== "undefined" && window.pywebview.api;
}

function setArduinoStatus(connected) {
  if (connected) {
    statusArdDot.classList.add("connected");
    statusArdText.textContent = "Arduino Conectado";
  } else {
    statusArdDot.classList.remove("connected");
    statusArdText.textContent = "Arduino desconectado";
  }
}

function setCalibratedStatus(calibrated) {
  if (calibrated) {
    statusCalDot.classList.add("connected");
    statusCalText.textContent = "Cubo Calibrado";
  } else {
    statusCalDot.classList.remove("connected");
    statusCalText.textContent = "Cubo No calibrado";
  }
}

if (statusArdDot && statusArdText) setArduinoStatus(false);
if (statusCalDot && statusCalText) setCalibratedStatus(false);

function showSettingsView() {
  panelTitle.textContent = "Ajustes";

  panelContent.innerHTML = `
    <div class="settings-panel">
        <div class="settings-options">
        <button id="reset-btn" class="settings-btn-panel">Resetear Cubo</button>
        <button id="arduino-btn" class="settings-btn-panel">Conectar Arduino</button>
        <button id="cam-btn" class="settings-btn-panel">Cambiar Cámara</button>
        <button id="darkmode-btn" class="settings-btn-panel">☀️ Modo claro</button>
        </div>
        <button id="exit-btn" class="settings-btn-panel settings-exit">Salir</button>
    </div>
    `;

  const resetBtn = document.getElementById("reset-btn");
  const arduinoBtn = document.getElementById("arduino-btn");
  const camBtn = document.getElementById("cam-btn");
  const themeButton = document.getElementById("darkmode-btn");
  const exitBtn = document.getElementById("exit-btn");

  resetBtn.onclick = async () => {
    try {
      if (typeof rubik !== "undefined" && rubik.ResetearCubo) rubik.ResetearCubo();
      if (hasPywebview() && window.pywebview.api.reset_cube_state) {
        try { await window.pywebview.api.reset_cube_state(); } catch(e) { /* no crítico */ }
      }
      setCalibratedStatus(false);
    } catch (e) {
      console.error("Reset error:", e);
      alert("Error al resetear el cubo: " + e);
    }
  };

  arduinoBtn.onclick = async () => {
    if (!hasPywebview()) {
      alert("La función Arduino solo funciona en la versión de escritorio (pywebview).");
      return;
    }
    try {
      arduinoBtn.disabled = true;
      arduinoBtn.textContent = "Conectando...";

      // el backend devuelve true/false por tu API actual
      const ok = await window.pywebview.api.connect_arduino();
      setArduinoStatus(!!ok);
      arduinoBtn.textContent = ok ? "Arduino Conectado" : "Intentar conectar Arduino";
    } catch (e) {
      console.error("connect_arduino error:", e);
      alert("Error conectando Arduino: " + e);
      setArduinoStatus(false);
      arduinoBtn.textContent = "Intentar conectar Arduino";
    } finally {
      arduinoBtn.disabled = false;
    }
  };

  camBtn.onclick = async () => {
    if (!hasPywebview()) {
      alert("Cambio de cámara solo disponible en la versión de escritorio (pywebview).");
      return;
    }
    try {
      camBtn.disabled = true;
      const ok = await window.pywebview.api.switch_camera();
      camBtn.textContent = ok ? "Cámara cambiada" : "Cambio fallido";
      setTimeout(() => { camBtn.textContent = "Cambiar Cámara"; }, 1400);
    } catch (e) {
      console.error("switch_camera error:", e);
      alert("Error cambiando cámara: " + e);
      camBtn.textContent = "Cambiar Cámara";
    } finally {
      camBtn.disabled = false;
    }
  };

  exitBtn.onclick = async () => {
    if (!hasPywebview()) {
      alert("Cerrar aplicación disponible solo en la versión de escritorio.");
      return;
    }
    try {
      await window.pywebview.api.shutdown();
    } catch (e) {
      console.error("shutdown error:", e);
      // no hacemos nada más
    }
  };

  const darkModeActiveInit = document.body.classList.contains('dark-mode');
  themeButton.textContent = darkModeActiveInit ? '☀️ Modo claro' : '🌙 Modo oscuro';

  themeButton.onclick = () => {
    document.body.classList.toggle('dark-mode');
    const darkModeActive = document.body.classList.contains('dark-mode');
    themeButton.textContent = darkModeActive ? '☀️ Modo claro' : '🌙 Modo oscuro';

    // detectar renderer global (cubo3d.js define window.rubikRenderer)
    const renderer = (typeof window.rubikRenderer !== 'undefined' && window.rubikRenderer)
        ? window.rubikRenderer
        : null;

    if (renderer && typeof renderer.setClearColor === "function") {
      try {
        // usar hex number para setClearColor
        renderer.setClearColor(darkModeActive ? 0x0d1117 : 0xffffff, 1);
      } catch (e) {
        console.warn("No se pudo cambiar clearColor del renderer:", e);
      }
    }
  };
}
