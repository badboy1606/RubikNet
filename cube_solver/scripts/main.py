import numpy as np
import torch
import serial
import time
from cube import Cube
from state_capture import RubiksCubeScanner
from a_star import AStar
from adi import ADI


scanned=RubiksCubeScanner()
cube=scanned.scan()
state=cube.state.copy()


def save_solution(moves, filename="solution.txt"):
    
    with open(filename, "w") as f:
        if moves:
            f.write(" ".join(moves))
        else:
            f.write("No solution found.")
    print(f"Solution saved to {filename}")


# Load model
model = ADI()
model.load_state_dict(torch.load("deepcube_adi_model.pth", map_location="cpu"))
model.eval()

    # Initialize solvers
astar_solver = AStar(model="deepcube_adi_model.pth", device='cpu')

move, solved= astar_solver.a_star_search(start_state=state, max_nodes=150000, max_depth=20)

if solved:
    print("solution found", move)
    save_solution(move, "moves.txt")


PORT = "COM18"       
BAUD = 115200

# Read moves from file
with open("moves.txt", "r") as f:
    moves = [line.strip() for line in f if line.strip()]

# Send over serial
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  

for move in moves:
    print(f"Sending: {move}")
    ser.write((move + "\n").encode())
    time.sleep(1)

ser.close()