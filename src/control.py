import random
import serial
import time
import serial.tools.list_ports
from typing import List, Optional

class MotorController:
    
    def __init__(self):
        self.gcode_map = {
            # Format: "Motor<Step><Direction>"
            # Step: 0=90°, 1=180°
            # Direction: 0=negative, 1=positive
            "D": "D01", "D'": "D00", "D2": "D11",
            "F": "F01", "F'": "F00", "F2": "F11",
            "R": "R01", "R'": "R00", "R2": "R11",
            "L": "L01", "L'": "L00", "L2": "L11",
            "B": "B01", "B'": "B00", "B2": "B11",
            "U": "U01", "U'": "U00", "U2": "U11"
        }
        
        self.serial_connection = None
        self.is_connected = False

    def connect(self) -> bool:
        port = self._find_arduino_port()
        if not port:
            print("Arduino not found.")
            return False
            
        try:
            self.serial_connection = serial.Serial(port, 9600, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            self.is_connected = True
            print("Connected to Arduino!")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            self.is_connected = False
            return False

    def _find_arduino_port(self) -> Optional[str]:
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if any(keyword in port.description for keyword in ["Arduino", "CH340"]):
                return port.device
        return None

    def execute_move(self, move: str) -> bool:
        if not self.is_connected:
            print("Not connected to hardware")
            return False
            
        if move not in self.gcode_map:
            print(f"Unsupported move: {move}")
            return False
        
        gcode = self.gcode_map[move]
        try:
            for char in gcode:
                self._send_gcode(char)
            return True
        except Exception as e:
            print(f"Move execution failed: {e}")
            return False

    def execute_sequence(self, moves: List[str]) -> bool:
        for move in moves:
            if not self.execute_move(move):
                return False
            time.sleep(0.5)  # Brief pause between moves
        return True

    def generate_scramble(self, num_moves: int = 20) -> List[str]:
        all_moves = list(self.gcode_map.keys())
        return random.choices(all_moves, k=num_moves)

    def test_all_motors(self) -> bool:
        if not self.is_connected:
            return False            
        success = True
        
        for move in self.gcode_map.keys():
            if not self.execute_move(move):
                success = False
            time.sleep(1)
            
        return success

    def _send_gcode(self, gcode: str):
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.write(bytes(gcode + '\n', 'UTF-8'))
            time.sleep(0.1)

    def disconnect(self):
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        self.is_connected = False