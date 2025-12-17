/*
patterns.js — present patterns / gallery
===============================================

This module defines a collection of predefined cube move patterns and
implements a small UI "gallery" (vidriera) that allows the user to browse
and apply those patterns. The gallery can run automatically as a slideshow
(vidriera mode).

Responsibilities
- Maintain the `patterns` array: strings representing move sequences.
- Render a thumbnail grid of patterns in the settings panel area and allow
users to click to apply a pattern via the main UI `rubik` controller.
- Provide a "vidriera" (gallery/slideshow) mode that cycles through the
patterns at a configurable interval with start/stop controls and ESC key
to exit.

Design notes
- This module manipulates global `window` state for simple persistence
across navigations (pattern_idx, selectedImg, vidrieraRunning...).
- It integrates with the existing `rubik` frontend controller (global) to
apply move sequences; if `rubik` is not present the action is a no-op.
- Accessibility: thumbnail images receive a `title` and `aria` hints for
the vidriera activator. Focus/keyboard interaction is limited (ESC to exit).
- UI behavior is implemented imperatively and expects the presence of
`panel-title` / `panel-content` containers used across the app.

Limitations & suggestions
- The module assumes a global `rubik` object — consider passing an API
object to decouple UI from implementation for easier testing.
- Patterns are simple strings; a future enhancement could store metadata
(name, thumbnail, difficulty) alongside the sequence.
- Keyboard navigation inside the gallery is minimal. Adding arrow-key
navigation and focus management would improve accessibility.

------------------------------------------------------------------

Copyright (c) 2025 Facundo Gauna & Ulises Carnevale. MIT License.
*/

(function () {
  "use strict";

  window.pattern_idx = (typeof window.pattern_idx === "number") ? window.pattern_idx : 0;
  window.selectedImg = window.selectedImg || null;
  window.vidrieraRunning = window.vidrieraRunning || false;
  window.vidrieraStopRequested = window.vidrieraStopRequested || false;
  window._vidrieraEscHandler = window._vidrieraEscHandler || null;

  const patterns = [
    "", // sorted cube
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
    "F B' U F U F U L B L2 B' U F' L U L' B",
    "" // galley activator
  ];
  const inverted = [
    "", // sorted cube
    "L2 R2 D2 U2 B2 F2",
    "U2 B2 U2 B2 L' R' F2 D2 F2 U2 L R",
    "R2 D F2 D' R2 D F2 D' R2 U B D' F2 D B' U'",
    "U' F' R L D L' D' R2 D2 R U R' U' D' B L",
    "L2 R2 U2 L2 R2 U2",
    "D' B2 F2 D2 L2 R2 U' R2 D U' B' F L R' D' U'",
    "F' U' F' L' R2 B' D L' D2 R' D B' L2 R' F' U' F'",
    "D2 U' R B2 D' R' D' U L' R D R F2 D' L R2",
    "U' L2 B D B' L U L2 F2 U' R' U F' L' F'",
    "D U' B F' L R' D U'",
    "F L' U L U' R' U F' L F L' U' R F'",
    "U2 L' U' F L U' R' D' L' U' B L D2 F D' F'",
    "D2 R U L' F2 R F' U F' U' B' U F D' L F'",
    "L2 F2 R2 L2 F2 R2",
    "U' L D' B L' D' L' B2 L U B2 R2 U D B2 L2 D F2 L2",
    "F2 U2 B R2 U2 D2 L2 F D2 F2",
    "B' L U' L' F U' B L2 B' L' U' F' U' F' U' B F'",
    "" // galley activator
  ];  


  function sleep(ms) { return new Promise(res => setTimeout(res, ms)); }

  function markThumbnail(img, active) {
    if (!img) return;
    img.classList.toggle("pattern-active", !!active);
    if (active) {
      img.setAttribute("aria-pressed", "true");
      img.style.borderColor = "#FFD700";
      img.style.boxShadow = "0 6px 18px rgba(255,215,0,0.6)";
      img.style.transform = "scale(1.08)";
    } else {
      img.removeAttribute("aria-pressed");
      img.style.borderColor = "#ccc";
      img.style.boxShadow = "0 2px 6px rgba(0,0,0,0.15)";
      img.style.transform = "scale(1)";
    }

  }

  function showPatternsView() {
    window.stopPreviewLoopCam();
    
    const panelTitle = document.getElementById("panel-title");
    const panelContent = document.getElementById("panel-content");
    if (!panelTitle || !panelContent) return console.warn("panel-title/panel-content are null");

    panelTitle.textContent = "Patterns";
    panelContent.innerHTML = `<div class="patterns-container" role="list"></div>`;
    const container = panelContent.querySelector(".patterns-container");

    // ensure scroll is enabled
    panelContent.style.overflowY = "auto";

    for (let i = 0; i < patterns.length; i++) {
      const img = document.createElement("img");
      img.className = "pattern pattern-img";
      img.src = `patterns/pattern${i}.svg`;
      img.alt = `Pattern ${i}`;
      img.id = String(i);
      img.loading = "lazy";
      img.setAttribute("role", "listitem");
      img.style.outline = "none";
      
      // set ARIA and tooltip when last index = activator
      if (i === patterns.length - 1) {
        img.title = "Activate / Stop Gallery Mode";
        img.dataset.vidrieraActivator = "1";
      }

      // hover visuals
      img.addEventListener("mouseenter", () => {
        if (window.selectedImg !== img) {
          img.style.transform = "scale(1.03)";
          img.style.boxShadow = "0 8px 18px rgba(0,0,0,0.12)";
        }
      });
      img.addEventListener("mouseleave", () => {
        if (window.selectedImg !== img) {
          img.style.transform = "scale(1)";
          img.style.boxShadow = "0 2px 6px rgba(0,0,0,0.15)";
        }
      });

      img.addEventListener("click", async () => {
        if(window.bottom_active == 1 || window.bottom_active > 3) return;
        
        const idx = Number(img.id);

        if (idx === patterns.length - 1) {
          if (window.vidrieraRunning && !window.vidrieraStopRequested) {
            await stopVidriera();
            return;
          }
          await startVidriera();
          return;
        }
        
        if(window.vidrieraRunning) return;

        if (window.selectedImg === img) return;        
        if (rubik.busy || rubik.sequenceRunning) return; 

        if (window.selectedImg) markThumbnail(window.selectedImg, false);
        
        window.selectedImg = img;
        markThumbnail(img, true);

        await doPattern(idx);
      });

      container.appendChild(img);

      // mark image
      if (window.pattern_idx === i) {
        window.selectedImg = img;
        markThumbnail(img, true);
      }
    }
  }
  // Apply the pattern at index i by sending moves to the UI controller
  async function doPattern(id) {
    if (id === 0) {
      if (window.pattern_idx === 0) return;
      await rubik.sequence(inverted[window.pattern_idx]);
      window.pattern_idx = 0;
      window.bottom_active = 0;
      return;
    }

    window.bottom_active = 2;
    try {
      if (window.pattern_idx && window.pattern_idx !== 0) {
        const seq = (inverted[window.pattern_idx] + " " + (patterns[id]));
        await rubik.sequence(seq);
      } else {
        await rubik.sequence(patterns[id]);
      }
      window.pattern_idx = id;
    } catch (e) {
      console.error("doPattern error:", e);
    }
  }

  async function doPatternZero(){
    const img = document.getElementById("0");
    if(rubik.busy || rubik.sequenceRunning) return;
    
    if(window.selectedImg != null){
      markThumbnail(window.selectedImg,false);
    }

    window.selectedImg = img;
    markThumbnail(img,true);

    if(window.pattern_idx == 0) return;
    await rubik.sequence(inverted[window.pattern_idx]);
    window.pattern_idx = 0;
  }

  window.doPatternZero = doPatternZero;

  async function startVidriera() {
    if (window.vidrieraRunning) return;
    window.bottom_active = 3; // lock bottom panel

    if (rubik.busy || rubik.sequenceRunning) {
      console.debug("Gallery cannot be open. rubik is busy");
      return;
    }

    window.vidrieraRunning = true;
    window.vidrieraStopRequested = false;

    window._vidrieraEscHandler = async function (ev) {
      if (ev.key === "Escape") {
        if (!window.vidrieraRunning) return;
        if (!window.vidrieraStopRequested) {
          window.vidrieraStopRequested = true;
        }
      }
    };

    window.addEventListener("keydown", window._vidrieraEscHandler);

    if (window.pattern_idx !== 0) {
      try {
        await rubik.sequence(inverted[window.pattern_idx]);
      } catch (e) {
        console.warn("Start Gallery: error returning 0 (sort position) :", e);
      }
      window.pattern_idx = 0;
    }

    (async () => {
      let prevIndex = 0;
      try {
        while (!window.vidrieraStopRequested) {
          for (let i = 0; i < patterns.length - 1; i++) {
            if (window.vidrieraStopRequested) break;

            if (patterns[i] === "") continue;

            try {
              if (prevIndex !== 0) {
                const transition = (inverted[prevIndex] + " " + (patterns[i]));
                await rubik.sequence(transition);
              } else {
                await rubik.sequence(patterns[i]);
              }
            } catch (e) {
              console.error("Gallery: error running sequence : ", e);
              window.vidrieraStopRequested = true;
            }

            // Update UI after of finished pattern
            let str_idx = String(i);
            try {
              if (window.selectedImg && window.selectedImg.id !== str_idx) {
                markThumbnail(window.selectedImg, false);
              }

              const newThumb = document.getElementById(str_idx);
              if (newThumb) {
                markThumbnail(newThumb, true);
                window.selectedImg = newThumb;
              } else {
                window.selectedImg = null;
              }
            } catch (uiErr) {
              console.warn("Gallery UI update error:", uiErr);
            }

            prevIndex = i;
            window.pattern_idx = i;

            const total = window.galleryMs || 3000;
            console.debug("wait time :",total);
            await sleep(total);

            if (window.vidrieraStopRequested) break;
          }
        }
      } finally {
        // at the moment of stoping the cube tries to be sort.
        try {
          if (prevIndex !== 0 &&  patterns[prevIndex] !== "") {
            // Wait untill cube is free.        
            while ( rubik.busy || rubik.sequenceRunning ) { await sleep(100); }
            await sleep(800);
          }
          try { await doPatternZero(); } catch (e) { console.error("Gallery error to invert pattern : ", e); }
          
        } catch (e) {
          console.warn("Gallery close error : ", e);
        } finally {
          window.vidrieraRunning = false;
          window.vidrieraStopRequested = false;
          if (window._vidrieraEscHandler) { window.removeEventListener("keydown", window._vidrieraEscHandler); window._vidrieraEscHandler = null; }
        }
      }
    })();
  }

  async function stopVidriera() {
    if (!window.vidrieraRunning) return;
    window.bottom_active = 0;
    window.vidrieraStopRequested = true;
  }

  // exports
  window.showPatternsView = showPatternsView;
  window.doPattern = doPattern;
  window.startVidriera = startVidriera;
  window.stopVidriera = stopVidriera;

  // auto-attach trigger button if present
  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("patterns-btn");
    if (btn) btn.addEventListener("click", showPatternsView);
  });

})();
