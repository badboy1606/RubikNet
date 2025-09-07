# Rubik's Cube Solver

A comprehensive neural network-based Rubik's Cube solver using Audidactic Iteration (ADI) with multiple search algorithms including A*, Beam Search, and Monte Carlo Tree Search (MCTS).

## Overview

This project implements a system capable of solving Rubik's Cube puzzles using deep learning techniques. The system includes:

- **ADI** neural network for state evaluation
- **Multiple search algorithms** for solution finding
- **Computer vision** for physical cube state capture
- **Comprehensive testing and analysis** tools

## Project Structure

```
├── adi.py                  # ADI neural network model and training
├── adi_standalone.py       # Standalone model testing and analysis
├── adi results             # Results obtained by ADI standalone
├── analysis.py             # Performance comparison between algorithms
├── a_star.py              # A* search implementation
├── beam.py                # Beam search implementation
├── cube.py                # Rubik's Cube state representation and operations
├── dataset.py             # Training dataset generation
├── main.py                # Main integration script
├── mcts.py                # Monte Carlo Tree Search implementation
└── state_capture.py       # Computer vision for cube scanning
```

## Installation

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd RubikNet

# Install required packages
pip install torch torchvision numpy pandas matplotlib seaborn opencv-python pyserial
```

## Usage

### 1. Training the Neural Network

Generate training data and train the ADI model:

```python
python adi.py
```

Configuration parameters in `adi.py`:
- `NUM_EPOCHS`: Number of training epochs (default: 75)
- `BATCH_SIZE`: Training batch size (default: 64)
- `SCRAMBLE_DEPTH`: Maximum scramble depth for training (default: 20)
- `SCRAMBLE_RUNS_PER_EPOCH`: Training samples per epoch (default: 1000)

#### GPU Training

The system automatically detects and uses CUDA if available. For CPU-only training, the system will fall back automatically.

### 2. Testing the Trained Model

Run comprehensive performance analysis:

```python
python adi_standalone.py
```

This will:
- Test the model on cubes scrambled to various depths
- Generate performance visualizations
- Save detailed results to JSON format

### 3. Solving Cubes with Different Algorithms

#### A* Search
```python
from a_star import AStar
from cube import Cube

solver = AStar("deepcube_adi_model.pth")
cube = Cube()
cube.scramble(10)
moves, solved = solver.a_star_search(cube.state, max_nodes=50000, max_depth=25)
```

#### Beam Search
```python
from beam import BeamSearch
from cube import Cube

solver = BeamSearch("deepcube_adi_model.pth")
cube = Cube()
cube.scramble(10)
moves, solved = solver.search(cube.state, beam_width=200, max_depth=50)
```

#### MCTS
```python
from mcts import solve_cube
from cube import Cube

cube = Cube()
cube.scramble(10)
solution, success = solve_cube(cube, num_simulations=15000, max_depth=25)
```

### 4. Computer Vision Integration

Capture a physical cube's state using your webcam:

```python
from state_capture import RubiksCubeScanner

scanner = RubiksCubeScanner()
cube_colors = scanner.scan()
```

The system will:
1. Guide you through capturing all 6 faces
2. Automatically detect colors using HSV analysis
3. Allow manual correction for misdetected colors
4. Return the cube state as a color array

### 5. Complete Solving Pipeline

Use the main script for end-to-end cube solving:

```python
python main.py
```

This integrates:
- Computer vision capture
- A* solving algorithm
- Serial communication to robot hardware

## Algorithm Comparison

Run comparative analysis between different solving approaches:

```python
python analysis.py
```

This benchmarks A* vs Beam Search across various scramble depths and generates performance plots.

## Model Architecture

The ADI neural network consists of:

```
Input: 324 features (54 stickers × 6 colors one-hot encoded)
├── Shared Layers
│   ├── FC1: 324 → 4096 (ELU activation)
│   └── FC2: 4096 → 2048 (ELU activation)
├── Policy Head
│   ├── FC3_1: 2048 → 512 (ELU activation)
│   └── Output: 512 → 12 (move probabilities)
└── Value Head
    ├── FC3_2: 2048 → 512 (ELU activation)
    └── Output: 512 → 1 (state value, tanh activation)
```

## Configuration

### Training Parameters

Adjust these parameters in `adi.py` for different training configurations:

```python
NUM_EPOCHS = 75              # Training epochs
BATCH_SIZE = 64              # Batch size
BATCH_ITERATIONS = 10        # Iterations per batch
SCRAMBLE_DEPTH = 20          # Max training scramble depth
SCRAMBLE_RUNS_PER_EPOCH = 1000  # Samples per epoch
```

### Search Algorithm Parameters

#### A* Search
- `max_nodes`: Maximum nodes to expand (default: 50,000)
- `max_depth`: Maximum search depth (default: 25)

#### Beam Search
- `beam_width`: Number of best candidates to keep (default: 200)
- `max_depth`: Maximum search depth (default: 50)

#### MCTS
- `num_simulations`: Number of MCTS simulations (default: 15,000)
- `max_depth`: Maximum tree depth (default: 25)
- `c_puct`: UCB exploration constant (default: 4.0)

## Troubleshooting

### Common Issues

1. **Model file not found**: Ensure `deepcube_adi_model.pth` exists after training
2. **CUDA out of memory**: Reduce `BATCH_SIZE` in training configuration
3. **Poor color detection**: Ensure good lighting when using computer vision
4. **Slow solving**: Reduce `max_nodes` or `num_simulations` for faster results

## Hardware

### STL Files

All the used .stl files inclusing the customised files are included in this folder. Ready to be 3D printed if you want to see your own solver come to life!

### Bill of Materials

Your go-to shopping list with all the links to all the components required to save you the hassle of surfing through sites or market places.

### Uart
C code for ESP32 that controls 6 stepper motors to physically manipulate a Rubik's cube based on serial commands. Available in scripts.

#### Motor Control
- **`step_motor()`**: Controls individual stepper motors with step/direction pins
- **50 steps = 90° rotation** with 800μs pulse timing
- **6 motor functions**: `rotate_R()`, `rotate_L()`, `rotate_U()`, `rotate_D()`, `rotate_F()`, `rotate_B()`

#### Usage Flow
1. **Flash** this code to ESP32 after hardware connections
2. **Run** `main.py` on computer
3. **Python sends** move commands (`"R"`, `"U'"`, etc.) via serial
4. **ESP32 executes** corresponding motor rotations

#### Integration
- Works with the neural network solver from the main project
- Receives solution moves from A*/Beam Search/MCTS algorithms
- Physically executes the cube solving sequence
