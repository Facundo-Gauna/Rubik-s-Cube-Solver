document.getElementById("patterns-btn").addEventListener("click", showPatternsView);

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

      if (typeof rubik !== "undefined") rubik.SecuenciaInstantanea("R2 F' D L B2 U' L2 F R' U2 B D' R U' L2 F2 D R' F' U L2 B2 U' R D2 F R' U2");
    });

    container.appendChild(img);
  }
}