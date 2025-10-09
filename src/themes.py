# Constants
DARK_THEME = """
    QMainWindow, QWidget, QDialog {
        background-color: #2b2b2b;
        color: #ffffff;
    }
    
    QGroupBox {
        color: #ffffff;
        font-weight: bold;
        border: 2px solid #555555;
        border-radius: 5px;
        margin-top: 1ex;
        padding-top: 10px;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px 0 5px;
        color: #ffffff;
    }
    
    QPushButton {
        background-color: #404040;
        color: #ffffff;
        border: 1px solid #555555;
        border-radius: 3px;
        padding: 5px 10px;
        font-weight: bold;
        min-height: 20px;
    }
    
    QPushButton:hover {
        background-color: #505050;
        border: 1px solid #666666;
    }
    
    QPushButton:pressed {
        background-color: #606060;
    }
    
    QPushButton:disabled {
        background-color: #333333;
        color: #777777;
    }
    
    QLabel {
        color: #ffffff;
    }
    
    QStatusBar {
        background-color: #2b2b2b;
        color: #ffffff;
    }
"""
