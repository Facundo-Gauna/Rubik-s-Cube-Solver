const panelTitle = document.getElementById("panel-title");
const panelContent = document.getElementById("panel-content");
panelContent.innerHTML = "";

document.getElementById("settings-btn").addEventListener("click", showSettingsView);

function showSettingsView() {
    panelTitle.textContent = "Ajustes";
    panelContent.innerHTML = `
        <button id="dark-mode-btn" class="action-btn">Modo oscuro</button>
        <button class="action-btn">Opción 2</button>
        <button class="action-btn">Opción 3</button>
    `;
}
