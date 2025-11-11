import os
import webview
from api import API  # Importamos la API con las funciones

api = API()
html_path = os.path.abspath("templates/index.html")
window = webview.create_window("Rubik's Cube Solver", f"file://{html_path}", js_api=api, resizable=True, fullscreen=True)
api.window = window
webview.start(debug=True)
api.shutdown()
