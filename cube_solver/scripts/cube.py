import random
import torch 
import numpy as np

# Color mapping for encoding cube states
color_map = {
    'w': [1, 0, 0, 0, 0, 0],  # White
    'y': [0, 1, 0, 0, 0, 0],  # Yellow
    'b': [0, 0, 1, 0, 0, 0],  # Blue
    'g': [0, 0, 0, 1, 0, 0],  # Green
    'r': [0, 0, 0, 0, 1, 0],  # Red
    'o': [0, 0, 0, 0, 0, 1]   # Orange
}

def find_action_index(parent_state, child_state):
    # Find which move transforms parent_state into child_state
    test_cube = Cube(state=parent_state.copy())  
    for i, move in enumerate(test_cube.moves):
        new_cube = Cube(state=parent_state.copy())  
        new_cube.move(move)

        if new_cube.state == child_state:
            return i , move 

    return -1   # Return -1 if no move produces the child state


def encode_cube_state(state_str):
    # Convert cube state string into one-hot encoded vector
    encoded = []
    for color in state_str:
        encoded.extend(color_map[color])
    return encoded


def decode_cube_state(encoded_state):
    # Convert one-hot vector back to cube state string
    color_list = ['w', 'y', 'b', 'g', 'r', 'o']
    decoded = []

    # Ensure tensor is converted to numpy
    if isinstance(encoded_state, torch.Tensor):
        encoded_state = encoded_state.cpu().detach().numpy()

    for i in range(0, len(encoded_state), 6):
        color_vec = encoded_state[i:i+6]
        color_i = np.argmax(color_vec)
        decoded.append(color_list[color_i])
    return decoded


# Cube model representing state and operations
class Cube:
    def __init__(self, state=None):
        self.reset_cube()  # Initialize solved cube
        self.move_history = []  # Track applied moves
        self.scramble_states = []  # Store states during scrambling
        self.moves = ["U", "D", "F", "B", "R", "L", "U'", "D'", "F'", "B'", "R'", "L'"]
        if state is not None:
            self.state = state.copy()

    def reset_cube(self):
        # Reset cube to solved state
        self.state = ["w"]*9 + ["y"]*9 + ["b"]*9 + ["g"]*9 + ["r"]*9 + ["o"]*9
        self.history = []

    def is_solved(self):
        # Check if all faces have uniform color
        if self.state is None:
            return False
        for i in range(0, 54, 9):
            if len(set(self.state[i:i+9])) != 1:
                return False
        return True

    def scramble(self, n):
        # Apply n random moves to scramble the cube
        moves = self.moves
        self.scramble_states = []
        self.move_history = []
        for turn in range(n):
            move = random.choice(moves)
            self.move(move)
            self.scramble_states.append({
                "state": self.state.copy(),
                "moves": self.move_history.copy(),
                "turns": turn + 1
            })

    def move(self, move_name):
        # Apply a move and update state
        self.move_history.append(move_name)
        self.state = self.switch(move_name, self.state)

    def print_cube(self):
        # Print cube faces for debugging
        print("F:", self.state[0:9])
        print("B:", self.state[9:18])
        print("U:", self.state[18:27])
        print("D:", self.state[27:36])
        print("R:", self.state[36:45])
        print("L:", self.state[45:54])

    def get_reward(self):
        # Return +1 if solved, -1 otherwise (for RL)
        return 1 if self.is_solved() else -1

    def switch(self, move, cube):
        # Apply face rotations and side swaps for a given move
        cube = cube.copy()

        def rotate_face(c, a,b,c_,d,e,f,g,h):
            # Clockwise rotation of a face
            c[a],c[b],c[c_],c[d],c[e],c[f],c[g],c[h] = c[c_],c[e],c[h],c[b],c[g],c[a],c[d],c[f]

        def rotate_face_anti(c, a,b,c_,d,e,f,g,h):
            # Counterclockwise rotation of a face
            c[a],c[b],c[c_],c[d],c[e],c[f],c[g],c[h] = c[f],c[d],c[a],c[g],c[b],c[h],c[e],c[c_]

        # Handle all 12 possible moves
        if move == "F":
            rotate_face_anti(cube, 0,1,2,3,5,6,7,8)
            cube[24],cube[25],cube[26],cube[36],cube[39],cube[42],cube[33],cube[34],cube[35],cube[53],cube[50],cube[47] = \
            cube[53],cube[50],cube[47],cube[24],cube[25],cube[26],cube[36],cube[39],cube[42],cube[33],cube[34],cube[35]

        elif move == "F'":
            rotate_face(cube, 0,1,2,3,5,6,7,8)
            cube[24],cube[25],cube[26],cube[36],cube[39],cube[42],cube[33],cube[34],cube[35],cube[53],cube[50],cube[47] = \
            cube[36],cube[39],cube[42],cube[33],cube[34],cube[35],cube[53],cube[50],cube[47],cube[24],cube[25],cube[26]

        elif move == "B":
            rotate_face_anti(cube, 9,10,11,12,14,15,16,17)
            cube[18],cube[19],cube[20],cube[45],cube[48],cube[51],cube[29],cube[28],cube[27],cube[44],cube[41],cube[38] = \
            cube[44],cube[41],cube[38],cube[18],cube[19],cube[20],cube[45],cube[48],cube[51],cube[29],cube[28],cube[27]

        elif move == "B'":
            rotate_face(cube, 9,10,11,12,14,15,16,17)
            cube[18],cube[19],cube[20],cube[45],cube[48],cube[51],cube[29],cube[28],cube[27],cube[44],cube[41],cube[38] = \
            cube[45],cube[48],cube[51],cube[29],cube[28],cube[27],cube[44],cube[41],cube[38],cube[18],cube[19],cube[20]

        elif move == "U":
            rotate_face_anti(cube, 18,19,20,21,23,24,25,26)
            cube[0],cube[1],cube[2],cube[36],cube[37],cube[38],cube[9],cube[10],cube[11],cube[45],cube[46],cube[47] = \
            cube[36],cube[37],cube[38],cube[9],cube[10],cube[11],cube[45],cube[46],cube[47],cube[0],cube[1],cube[2]

        elif move == "U'":
            rotate_face(cube, 18,19,20,21,23,24,25,26)
            cube[0],cube[1],cube[2],cube[36],cube[37],cube[38],cube[9],cube[10],cube[11],cube[45],cube[46],cube[47] = \
            cube[45],cube[46],cube[47],cube[0],cube[1],cube[2],cube[36],cube[37],cube[38],cube[9],cube[10],cube[11]

        elif move == "D":
            rotate_face_anti(cube, 27,28,29,30,32,33,34,35)
            cube[6],cube[7],cube[8],cube[51],cube[52],cube[53],cube[15],cube[16],cube[17],cube[42],cube[43],cube[44] = \
            cube[42],cube[43],cube[44],cube[6],cube[7],cube[8],cube[51],cube[52],cube[53],cube[15],cube[16],cube[17]

        elif move == "D'":
            rotate_face(cube, 27,28,29,30,32,33,34,35)
            cube[6],cube[7],cube[8],cube[51],cube[52],cube[53],cube[15],cube[16],cube[17],cube[42],cube[43],cube[44] = \
            cube[51],cube[52],cube[53],cube[15],cube[16],cube[17],cube[42],cube[43],cube[44],cube[6],cube[7],cube[8]

        elif move == "R":
            rotate_face_anti(cube, 36,37,38,39,41,42,43,44)
            cube[2],cube[5],cube[8],cube[26],cube[23],cube[20],cube[11],cube[14],cube[17],cube[29],cube[32],cube[35] = \
            cube[29],cube[32],cube[35],cube[2],cube[5],cube[8],cube[26],cube[23],cube[20],cube[11],cube[14],cube[17]

        elif move == "R'":
            rotate_face(cube, 36,37,38,39,41,42,43,44)
            cube[2],cube[5],cube[8],cube[26],cube[23],cube[20],cube[11],cube[14],cube[17],cube[29],cube[32],cube[35] = \
            cube[26],cube[23],cube[20],cube[11],cube[14],cube[17],cube[29],cube[32],cube[35],cube[2],cube[5],cube[8]

        elif move == "L":
            rotate_face_anti(cube, 45,46,47,48,50,51,52,53)
            cube[0],cube[3],cube[6],cube[18],cube[21],cube[24],cube[9],cube[12],cube[15],cube[33],cube[30],cube[27] = \
            cube[18],cube[21],cube[24],cube[9],cube[12],cube[15],cube[33],cube[30],cube[27],cube[0],cube[3],cube[6]

        elif move == "L'":
            rotate_face(cube, 45,46,47,48,50,51,52,53)
            cube[0],cube[3],cube[6],cube[18],cube[21],cube[24],cube[9],cube[12],cube[15],cube[33],cube[30],cube[27] = \
            cube[33],cube[30],cube[27],cube[0],cube[3],cube[6],cube[18],cube[21],cube[24],cube[9],cube[12],cube[15]

        else:
            print(f"Invalid move: {move}")

        return cube

    def get_child_states_at_all_steps(self):
        # Generate child states for every scramble step
        all_children = []
        for scramble_state in self.scramble_states:
            base_state = scramble_state["state"]
            children = []
            for move in self.moves:
                new_cube = Cube(state=base_state.copy())
                new_cube.move(move)
                children.append(new_cube)
            all_children.append(children)
        return all_children