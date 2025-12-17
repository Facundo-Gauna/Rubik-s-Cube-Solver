/*
settings.js — UI settings panel and local persistence
====================================================

This module builds and manages the "Settings" panel of the web UI
and exposes a small public API used by other modules.

Responsibilities
- Render the Settings panel HTML and wire its event handlers.
- Persist user preferences to localStorage (resolution, velocity, theme, etc.).
- Provide small helper functions used by the UI (theme toggle, status dots).
- Interact with the Python backend via `window.pywebview.api` for actions
  such as changing camera resolution, setting move duration, connecting the
  Arduino or requesting shutdown.
- Expose simple status updaters for other modules to call: `setArduinoStatus`
  and `setCalibratedStatus`.

Public API (attached to window)
- window.showSettingsView()        -> Build and show settings panel (main entry)
- window.setArduinoStatus(bool)    -> Update Arduino connection indicator
- window.setCalibratedStatus(bool) -> Update calibration indicator

Design notes / behaviour
- Settings are stored under localStorage key "rubik_app_settings" as JSON.
- Sliders use a debounced backend update strategy to avoid spamming the API.
- The module is resilient: it checks DOM nodes before binding and tries to
  avoid adding duplicate listeners by storing temporary handler references.
- Accessibility: resolution buttons and the panel container include basic
  ARIA attributes; the theme button is updated with `aria-pressed`.

Usage
- The file attaches a DOMContentLoaded handler which binds the settings button
  to `showSettingsView()` and wires the theme toggle in the top bar.

------------------------------------------------------------------

Copyright (c) 2025 Facundo Gauna & Ulises Carnevale. MIT License.
*/

(() => {
  "use strict";

  // Local storage key and default values for the settings panel.
  // Keep defaults in JS to allow the UI to function without backend state.
  const STORE_KEY = "rubik_app_settings";
  const DEFAULTS = {
    resolution: "1920x1080",
    durationMs: 400,
    galleryDisplayMs: 45000,
    theme: "dark",
    useRealU: false
  };

  // Supported streaming/base resolutions presented in the UI.
  const RESOLUTIONS = ["1920x1080", "1280x720", "960x540"];
  const DEBOUNCE_DELAY_MS = 1000; // debounce for sending duration updates

  /* ---------------------- Utilities ---------------------- */
  // Small DOM helpers used throughout the module.
  const q = (sel, root = document) => root.querySelector(sel);
  const qa = (sel, root = document) => Array.from((root || document).querySelectorAll(sel));
  const on = (el, ev, fn, opts) => el && el.addEventListener(ev, fn, opts);

  // Debounce with clear method (used for coalescing backend calls)
  function debounce(fn, wait) {
    let t = null;
    const wrapper = (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), wait);
    };
    wrapper.clear = () => { clearTimeout(t); t = null; };
    return wrapper;
  }

  /* ---------------------- Storage helpers ---------------------- */
  // readSettings / writeSettings: tiny wrappers around localStorage with
  // defensive JSON parsing to avoid throwing on corrupt data.
  function readSettings() {
    try {
      return JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
    } catch (e) {
      console.warn("readSettings parse error:", e);
      return {};
    }
  }
  function writeSettings(obj) {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(obj || {}));
    } catch (e) {
      console.warn("writeSettings error:", e);
    }
  }

  // Convenience getters/setters that fall back to DEFAULTS when keys don't exist.
  function getSetting(key) {
    const s = readSettings();
    return (s && s[key] !== undefined) ? s[key] : DEFAULTS[key];
  }
  function setSetting(key, val) {
    const s = readSettings();
    s[key] = val;
    writeSettings(s);
  }

  /* ---------------------- Panel elements ---------------------- */
  // Cached references to the central panel title/content nodes. The UI may
  // call showSettingsView() before the application fully initializes, so
  // findPanelElements is defensive.
  let panelTitleEl = null;
  let panelContentEl = null;
  function findPanelElements() {
    panelTitleEl = document.getElementById("panel-title") || document.querySelector(".panel-title");
    panelContentEl = document.getElementById("panel-content") || document.querySelector(".panel-content");
    return !!(panelTitleEl && panelContentEl);
  }

  /* ---------------------- Status Updaters (public) ---------------------- */
  // Simple helpers that update status indicators on the page. They are
  // intentionally tolerant: if the elements don't exist the call is a no-op.
  function setArduinoStatus(connected) {
    const statusArdDot = document.getElementById("status-arduino-dot");
    const statusArdText = document.getElementById("status-arduino-text");
    if (statusArdDot && statusArdText) {
      statusArdDot.style.backgroundColor = connected ? "rgba(22, 210, 41, 0.49)"  : "rgba(210, 22, 22, 0.49)";
      statusArdText.textContent = connected ? "Arduino Connected" : "Arduino Disconnected";
    } else {
      console.warn("setArduinoStatus: status elements not found.");
    }
  }

  function setCalibratedStatus(calibrated) {
    const statusCalDot = document.getElementById("status-calibrated-dot");
    const statusCalText = document.getElementById("status-calibrated-text");
    if (statusCalDot && statusCalText) {
      statusCalDot.style.backgroundColor = calibrated ? "rgba(22, 210, 41, 0.49)" : "rgba(210, 22, 22, 0.49)";
      statusCalText.textContent = calibrated ? "Cube Calibrated" : "Cube Not Calibrated";
    } else {
      console.warn("setCalibratedStatus: status elements not found.");
    }
  }

  /* ---------------------- Theme helpers ---------------------- */
  // Toggle dark/light mode and persist preference.
  function applyTheme(theme) {
    const t = (theme === "light") ? "light" : "dark";
    if (t === "light") {
      document.body.classList.add("light-mode");
      document.body.classList.remove("dark-mode");
    } else {
      document.body.classList.add("dark-mode");
      document.body.classList.remove("light-mode");
    }
    setSetting("theme", t);
    const themeBtn = document.getElementById("theme-toggle-btn");
    if (themeBtn) {
      themeBtn.setAttribute("aria-pressed", t === "light" ? "true" : "false");
      themeBtn.classList.toggle("theme-light", t === "light");
    }
  }

  // Apply initial theme on module load (defensive wrapper).
  (function _applyInitialTheme() {
    try {
      const saved = getSetting("theme");
      applyTheme(saved || DEFAULTS.theme);
    } catch (e) {
      console.warn("applyInitialTheme error:", e);
    }
  })();

  /* ---------------------- HTML builder ---------------------- */
  // Build the inner HTML for the settings panel. Kept as a template string
  // so it is easy to insert by showSettingsView(). Buttons have ARIA hints.
  function buildSettingsHtml() {
    const resButtonsHtml = RESOLUTIONS.map(r => `<button type="button" class="res-btn" data-res="${r}" aria-pressed="false">${r}</button>`).join("");

     return `
      <div class="settings-panel compact-wrap" id="settings-root" role="region" aria-label="Settings panel">
        <div class="settings-grid">
          <div class="full">
            <button id="reset-btn" class="main-btn btn-compact" type="button">Reset cube</button>
          </div>

          <div><button id="arduino-btn" class="main-btn btn-compact" type="button">Connect Arduino</button></div>
          <div><button id="cam-btn" class="main-btn btn-compact" type="button">Switch Camera</button></div>

          <div class="full">
            <div class="res-buttons" role="group" aria-label="Resolutions">
              ${resButtonsHtml}
            </div>
          </div>

          <div class="full">
            <div class="vel-row">
              <button id="vel-minus" class="vel-small" type="button" aria-label="Decrease velocity">-</button>
              <div class="vel-pill">
                <input id="vel-slider" type="range" min="50" max="1200" step="10" />
                <div class="vel-center">
                  <div class="vel-name">Rotation Velocity</div>
                  <div class="vel-value" id="vel-value">--</div>
                </div>
              </div>
              <button id="vel-plus" class="vel-small vel-blue" type="button" aria-label="Increase velocity">+</button>
            </div>
          </div>

          <div class="full">
            <div class="vel-row">
              <div style="width:100%;">
                <div class="vel-pill gallery-pill">
                  <input id="gallery-slider" type="range" min="1000" max="120000" step="500" />
                  <div class="vel-center">
                    <div class="vel-name">Gallery time</div>
                    <div class="vel-value" id="gallery-value">--</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="full">
            <div class="custom-moves-panel">
              <input id="custom-moves-input" type="text" placeholder="Sequence of Moves (e.g. U F2 R')" aria-label="Custom moves string" />
              <div style="display:flex; gap:8px;">
                <button id="custom-moves-apply" class="main-btn btn-compact" type="button">Apply</button>
                <button id="custom-moves-cancel" class="main-btn btn-compact" type="button">Clear</button>
              </div>
            </div>
          </div>

          <button id="test-motors-btn" class="main-btn btn-compact" type="button">Test Motors</button>
          <button id="u-presence-btn" class="main-btn btn-compact" type="button" aria-pressed="false">U: ?</button>

          <div class="full sanjo-grow">
            <div class="sanjo-image" role="img" aria-label="San Jose logo">
              <img src="patterns/sanjo.png" alt="San Jose logo" />
            </div>
          </div>

          <div class="full settings-footer">
            <div class="footer-buttons" role="group" aria-label="Footer actions">
              <button id="info-btn" class="main-btn info-btn btn-compact" type="button">Info / GitHub</button>
              <button id="exit-btn" class="main-btn exit-btn btn-compact" type="button">Exit</button>
            </div>
          </div>

        </div>
      </div>
    `;
  }

  /* ---------------------- Main: showSettingsView (builds HTML + wires handlers) ---------------------- */
  function showSettingsView() {
    // Stop preview loop (if running in other modules) so camera connection isn't
    // double-used while settings are open.
    window.stopPreviewLoopCam();

    if (!findPanelElements()) {
      console.error("settings: missing panel-title or panel-content elements");
      return;
    }

    panelTitleEl.textContent = "Settings";
    panelContentEl.innerHTML = buildSettingsHtml();

    // Keep layout consistent while settings is open
    panelContentEl.classList.add("settings-open");

    // Cache newly created elements we will wire handlers for
    const resetBtn = q("#reset-btn", panelContentEl);
    const arduinoBtn = q("#arduino-btn", panelContentEl);
    const camBtn = q("#cam-btn", panelContentEl);
    const testBtn = q("#test-motors-btn", panelContentEl);
    const uBtn = q("#u-presence-btn", panelContentEl);
    const resBtns = qa(".res-btn", panelContentEl);
    const velSlider = q("#vel-slider", panelContentEl);
    const velValueEl = q("#vel-value", panelContentEl);
    const velMinus = q("#vel-minus", panelContentEl);
    const velPlus = q("#vel-plus", panelContentEl);
    const infoBtn = q("#info-btn", panelContentEl);
    const exitBtn = q("#exit-btn", panelContentEl);
    const customInput = q("#custom-moves-input", panelContentEl);
    const customApply = q("#custom-moves-apply", panelContentEl);
    const customCancel = q("#custom-moves-cancel", panelContentEl);
    const gallerySlider = q("#gallery-slider", panelContentEl);
    const galleryValueEl = q("#gallery-value", panelContentEl);

    // -------- init resolution UI --------
    const selectedRes = getSetting("resolution") || DEFAULTS.resolution;
    resBtns.forEach(b => b.classList.toggle("selected", b.dataset.res === selectedRes));
    const resContainer = q(".res-buttons", panelContentEl);
    if (resContainer) {
      // Click handler changes the stored value and asks backend to change
      // the camera resolution. If succeeded we update the local base resolution
      resContainer.addEventListener("click", async (ev) => {
        const btn = ev.target.closest(".res-btn");
        if (!btn) return;
        resBtns.forEach(x => x.classList.remove("selected"));
        btn.classList.add("selected");
        setSetting("resolution", btn.dataset.res);
        try {
          const [w, h] = btn.dataset.res.split("x").map(Number);
          await window.pywebview.api.set_camera_resolution(w, h);
          // update global base resolution helper used by other UI modules
          window.setBaseResolution(Number(w), Number(h));
        } catch (e) {
          console.warn("Resolution change handler error:", e);
        }
      });
    }

    // -------- velocity slider (rotation duration) --------
    // Duration is stored in ms. We reflect the saved value in the slider and
    // propagate changes to both the front-end Rubik controller and the
    // backend via a debounced call to avoid spamming the API.
    let dur = Math.max(10, Number(getSetting("durationMs") || DEFAULTS.durationMs));

    if (velSlider) velSlider.value = dur;
    if (velValueEl) velValueEl.textContent = `${dur} ms`;

    const sendDurationToBackend = debounce((ms) => {
      try { rubik.setMoveDuration(ms); } catch (e) { console.warn("rubik.setMoveDuration error:", e); }
      try {
        window.pywebview.api.set_move_duration(ms).catch(e => console.warn("set_move_duration error:", e));
      } catch (err) {
        console.warn("set_move_duration call failed:", err);
      }
    }, DEBOUNCE_DELAY_MS);

    function setDuration(ms, sendImmediately = false) {
      dur = Math.min(10000, Math.max(10, Math.round(ms)));
      if (velSlider) velSlider.value = dur;
      if (velValueEl) velValueEl.textContent = `${dur} ms`;
      setSetting("durationMs", dur);
      try { rubik.setMoveDuration(dur); } catch (e) { console.warn("rubik.setMoveDuration error:", e); }
      if (sendImmediately) {
        // clear any pending debounce and trigger immediately
        sendDurationToBackend.clear && sendDurationToBackend.clear();
        sendDurationToBackend(dur);
      } else {
        sendDurationToBackend(dur);
      }
    }

    // initialize control with stored value and send initial update
    setDuration(dur,true);


    if (velSlider) {
      on(velSlider, "input", (e) => {
        const ms = Number(e.target.value);
        if (velValueEl) velValueEl.textContent = `${ms} ms`;
        sendDurationToBackend(ms);
      });
      on(velSlider, "change", (e) => setDuration(Number(e.target.value)));
      on(velSlider, "mouseup", () => setDuration(Number(velSlider.value), true));
      on(velSlider, "touchend", () => setDuration(Number(velSlider.value), true));
    }
    if (velMinus) on(velMinus, "click", () => setDuration(dur - 50, true));
    if (velPlus) on(velPlus, "click", () => setDuration(dur + 50, true));

    // -------- gallery slider (pattern gallery timing) --------
    window.galleryMs =  Number(getSetting("galleryDisplayMs") || DEFAULTS.galleryDisplayMs);

    if (gallerySlider) gallerySlider.value = String(window.galleryMs);
    if (galleryValueEl) galleryValueEl.textContent = formatMsReadable(window.galleryMs);

    const saveGallerySettingDebounced = debounce((v) => setSetting("galleryDisplayMs", v), 250);

    function setGalleryMs(ms) {
      const v = Math.min(120000, Math.max(100, Math.round(Number(ms))));
      window.galleryMs = v;
    
      if (gallerySlider) gallerySlider.value = String(v);
      if (galleryValueEl) galleryValueEl.textContent = formatMsReadable(v);
    
      saveGallerySettingDebounced(v);
    }

    if (gallerySlider) {
      on(gallerySlider, "input", (e) => {
        const ms = Number(e.target.value);
        if (galleryValueEl) galleryValueEl.textContent = formatMsReadable(ms);
        setGalleryMs(ms);
      });
    
      on(gallerySlider, "change", (e) => setGalleryMs(Number(e.target.value)));
      on(gallerySlider, "mouseup", () => setGalleryMs(Number(gallerySlider.value)));
      on(gallerySlider, "touchend", () => setGalleryMs(Number(gallerySlider.value)));
    }

    function formatMsReadable(ms) {
      if (ms >= 60000) return (ms / 60000).toFixed(2) + " min";
      if (ms >= 1000) return (ms / 1000).toFixed(1) + " s";
      return ms + " ms";
    }

    // -------- reset button (reset cube state) --------
    if (resetBtn) {
      on(resetBtn, "click", async () => {
        try {
          await rubik.resetCube();
          window.setCalibratedStatus(false);
        } catch (e) {
          console.error("Reset error:", e);
          alert("Error resetting the cube: " + e);
        }
      });
    }

    // -------- arduino connect --------
    if (arduinoBtn) {
      on(arduinoBtn, "click", async () => {
        try {
          arduinoBtn.disabled = true;
          arduinoBtn.textContent = "Connecting...";
          const ok = await window.pywebview.api.connect_arduino();
          setArduinoStatus(ok);
          arduinoBtn.textContent = ok ? "Arduino Connected" : "Try connect Arduino";
        } catch (e) {
          console.error("connect_arduino error:", e);
          alert("Error connecting Arduino: " + e);
          setArduinoStatus(false);
          arduinoBtn.textContent = "Try connect Arduino";
        } finally {
          arduinoBtn.disabled = false;
          setTimeout(() => { arduinoBtn.textContent = "Connect Arduino"; }, 1200);
        }
      });
    }

    // -------- camera switch --------
    if (camBtn) {
      on(camBtn, "click", async () => {
        try {
          camBtn.disabled = true;
          const ok = await window.pywebview.api.switch_camera();
          camBtn.textContent = ok ? "Camera switched" : "Switch failed";
        } catch (e) {
          console.error("switch_camera error:", e);
          alert("Error switching camera: " + e);
          camBtn.textContent = "Switch Camera";
        } finally {
          camBtn.disabled = false;
          setTimeout(() => { camBtn.textContent = "Switch Camera"; }, 1400);
        }
      });
    }

    // -------- test motors --------
    if (testBtn) {
      on(testBtn, "click", async () => {
        try {
          if (!arduinoBtn || arduinoBtn.disabled) {
            alert("Please connect the Arduino first.");
            return;
          }
          if (customApply) customApply.disabled = true;
          if (customCancel) customCancel.disabled = true;
          rubik.turnKeys(false);
          const prev = window.bottom_active;
          window.bottom_active = -1; // block other interactions while testing
          const ok = await window.pywebview.api.test_motors();
          if (!ok) alert("Error during motors test. Please check the connections.");
          else alert("Motors test completed successfully.");
          window.bottom_active = prev;
          rubik.turnKeys(true);
        } catch (e) {
          alert("Error during motors test: " + e);
        } finally {
          if (customApply) customApply.disabled = false;
          if (customCancel) customCancel.disabled = false;
          setTimeout(() => rubik.turnKeys(false), 20);
        }
      });
    }

    // -------- U move Activation (local preference) --------
    // The U-toggle is an entirely local preference stored in settings and used
    // by the UI logic (does not require backend changes).
    window.UsageU = Boolean(getSetting("useRealU"));

    if (uBtn) {
      function updateUToggleUI() {
        uBtn.setAttribute("aria-pressed", window.UsageU ? "true" : "false");
        if (window.UsageU) {
          uBtn.textContent = "U: ON ✓";
          uBtn.style.background = "#4caf50";
        } else {
          uBtn.textContent = "U: OFF ✕";
          uBtn.style.background = "#d72828ff";
        }
      }

      // initialize UI
      updateUToggleUI();

      // toggle handler (pure local preference change)
      on(uBtn, "click", (ev) => {
        try {
          if(rubik.busy || rubik.sequenceRunning) return;
          window.UsageU = !window.UsageU;
          setSetting("useRealU", window.UsageU);
          updateUToggleUI();
        } catch (e) {
          console.warn("u-toggle click handler error:", e);
        }
      });
    }

    // -------- info / github -> open repository in a new tab/window --------
    const REPOSITORY = "https://github.com/Facruck3/-Rubik-s-Cube-Solver-San-Jose";
    if (infoBtn) on(infoBtn, "click", () => { try { window.open(REPOSITORY, "_blank"); } catch (e) { location.href = REPOSITORY; } });

    // -------- exit -> request Python backend to shutdown application --------
    if (exitBtn) on(exitBtn, "click", async () => {
      try {
        if (panelContentEl) panelContentEl.classList.remove("settings-open");
        await window.pywebview.api.shutdown();
      } catch (e) {
        console.warn("shutdown error:", e);
      }
    });

    // -------- custom Moves input handling (text box & apply/cancel) --------
    if (customInput) {
      on(customInput, "focus", () => { rubik.turnKeys(false); panelContentEl.classList.add("typing-moves"); });
      on(customInput, "blur", () => { setTimeout(() => rubik.turnKeys(true), 40); panelContentEl.classList.remove("typing-moves"); });
      on(customInput, "keydown", (ev) => {
        ev.stopPropagation();
        if (ev.key === "Enter") { ev.preventDefault(); rubik.customMoves(customInput.value || ""); }
        else if (ev.key === "Escape") { ev.preventDefault(); customInput.blur(); }
      });
    }
    if (customCancel) on(customCancel, "click", () => { if (customInput) { customInput.value = ""; customInput.focus(); } });
    if (customApply) on(customApply, "click", async () => { await rubik.customMoves(customInput ? customInput.value || "" : ""); rubik.turnKeys(false); });
  }

  /* ---------------------- Theme toggle button binding ---------------------- */
  function bindThemeToggleButton() {
    const themeBtn = document.getElementById("theme-toggle-btn");
    if (!themeBtn) return;
    const current = getSetting("theme") || DEFAULTS.theme;
    themeBtn.setAttribute("aria-pressed", current === "light" ? "true" : "false");
    themeBtn.classList.toggle("theme-light", current === "light");
    themeBtn.addEventListener("click", () => {
      const now = getSetting("theme") || DEFAULTS.theme;
      const next = now === "light" ? "dark" : "light";
      applyTheme(next);
    });
  }

  // Bind the main settings button once the DOM is ready.
  document.addEventListener("DOMContentLoaded", () => {
    const trigger = document.getElementById("settings-btn");
    if (trigger && !trigger._settingsBound) {
      trigger.addEventListener("click", showSettingsView);
      trigger._settingsBound = true;
    }
    bindThemeToggleButton();
  });

  /* ---------------------- Expose public API ---------------------- */
  window.showSettingsView = showSettingsView;
  window.setCalibratedStatus = setCalibratedStatus;
  window.setArduinoStatus = setArduinoStatus;

  // initialize status indicators to disconnected / not calibrated
  setArduinoStatus(false);
  setCalibratedStatus(false);

})();
