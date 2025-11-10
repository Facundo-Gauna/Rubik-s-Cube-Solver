import webview
from api import WebAPI
import os

api = WebAPI()
html_path = os.path.abspath("templates/index.html")
window = webview.create_window("Rubik's Cube Solver", f"file://{html_path}", js_api=api, resizable=True, fullscreen=False)
api.window = window
webview.start(debug=True)
api.shutdown()

