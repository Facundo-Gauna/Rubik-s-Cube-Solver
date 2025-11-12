import os
import traceback
import webview
from api import API  # Importamos la API con las funciones
try:

    api = API()
    window = webview.create_window("Rubik's Cube Solver", "templates/index.html", js_api=api, resizable=True, fullscreen=True)

    api.window = window
    webview.start(debug=True)
    api.shutdown()
except Exception as e:
    print("Se produjo un error:", e)
    with open("error.log", "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
        
