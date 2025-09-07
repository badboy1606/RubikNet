import torch
import numpy as np
import heapq
from adi import ADI
from cube import Cube


class AStar:
    def __init__(self, model_path="deepcube_adi_model.pth", device="cpu"):
        # Load the trained ADI model for cube solving
        self.model = ADI()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        # One-hot encoding for cube face colors
        self.color_map = {
            'w': [1, 0, 0, 0, 0, 0],
            'y': [0, 1, 0, 0, 0, 0],
            'b': [0, 0, 1, 0, 0, 0],
            'g': [0, 0, 0, 1, 0, 0],
            'r': [0, 0, 0, 0, 1, 0],
            'o': [0, 0, 0, 0, 0, 1]
        }

    def get_cube_child_states(self, cube):
        # Generate all possible next states from current cube state
        children = []
        original_state = cube.state.copy()

        for move in cube.moves:
            child_cube = Cube(state=original_state.copy())
            child_cube.move(move)
            children.append(child_cube.state.copy())

        return children

    def encode_cube_state(self, state_str):
        # Convert cube state (string) to one-hot encoded vector
        encoded = []
        for color in state_str:
            encoded.extend(self.color_map[color])
        return encoded

    def decode_cube_state(self, encoded_state):
        # Convert one-hot encoded vector back to cube color list
        color_list = ['w', 'y', 'b', 'g', 'r', 'o']
        decoded = []

        if isinstance(encoded_state, torch.Tensor):
            encoded_state = encoded_state.cpu().detach().numpy().flatten()

        for i in range(0, len(encoded_state), 6):
            color_vec = encoded_state[i:i + 6]
            color_i = np.argmax(color_vec)
            decoded.append(color_list[color_i])
        return decoded

    def _state_key_from_list(self, state_list):
        # Convert cube state list to a unique string key
        return ''.join(state_list)

    def _to_state_list(self, state):
        # Ensure state is returned as a list of 54 color characters
        if isinstance(state, list) and len(state) == 54 and all(isinstance(x, str) for x in state):
            return state
        if isinstance(state, torch.Tensor):
            return self.decode_cube_state(state)
        if isinstance(state, str):
            return list(state)
        return list(state)

    def _inverse_move(self, m):
        # Return inverse of a cube move (e.g., R -> R')
        if isinstance(m, str):
            return m[:-1] if m.endswith("'") else m + "'"
        return m

    def _tensor_from_state_list(self, state_list):
        # Convert state list to a PyTorch tensor for model input
        enc = self.encode_cube_state(''.join(state_list))
        return torch.FloatTensor(enc).unsqueeze(0)

    def get_model_value(self, state_list):
        """Get value from neural network (higher = better/closer to solved)"""
        try:
            state_tensor = self._tensor_from_state_list(state_list)
            with torch.no_grad():
                policy_logits, value = self.model(state_tensor)
            return float(value.item() if torch.is_tensor(value) else value)
        except Exception as e:
            print(f"Model error: {e}")
            return 0.0

    def count_misplaced_pieces(self, state_list):
        # Count number of stickers that do not match their face center color
        if len(state_list) != 54:
            return 54

        misplaced = 0
        for face in range(6):
            base = face * 9
            center_color = state_list[base + 4]
            for i in range(9):
                if state_list[base + i] != center_color:
                    misplaced += 1
        return misplaced

    def a_star_search(self, start_state, max_nodes=50000, max_depth=25):
        # Perform A* search to find solution moves from start_state
        start_list = self._to_state_list(start_state)
        start_cube = Cube(state=start_list)

        if start_cube.is_solved():
            return [], True

        open_set = []  # Priority queue for A* nodes
        start_key = self._state_key_from_list(start_list)

        g_score = 0  # Cost from start node
        h_score = -self.get_model_value(start_list)  # Heuristic (model value)
        f_score = g_score + h_score

        heapq.heappush(open_set, (f_score, g_score, start_key, start_list, []))

        best_g_score = {start_key: 0}
        nodes_expanded = 0

        print(f"Starting A* search. Initial h_score: {h_score:.3f}")

        while open_set and nodes_expanded < max_nodes:
            current_f, current_g, current_key, current_state, current_path = heapq.heappop(open_set)

            # Skip if this path is worse than previously known
            if current_key in best_g_score and current_g > best_g_score[current_key]:
                continue

            # Depth limit check
            if current_g >= max_depth:
                continue

            nodes_expanded += 1

            if nodes_expanded % 1000 == 0:
                print(f"Expanded {nodes_expanded} nodes, depth {current_g}, f_score {current_f:.3f}")

            cube = Cube(state=current_state)
            child_states = self.get_cube_child_states(cube)
            moves = getattr(cube, "moves", list(range(len(child_states))))

            for idx, child_state in enumerate(child_states):
                if not isinstance(child_state, list) or len(child_state) != 54:
                    continue

                move = moves[idx] if idx < len(moves) else idx

                # Avoid immediately undoing previous move
                if (len(current_path) > 0 and isinstance(move, str) and
                        isinstance(current_path[-1], str) and self._inverse_move(move) == current_path[-1]):
                    continue

                child_key = self._state_key_from_list(child_state)
                tentative_g = current_g + 1

                # Skip if a better path to this state already exists
                if child_key in best_g_score and tentative_g >= best_g_score[child_key]:
                    continue

                child_cube = Cube(state=child_state)
                if child_cube.is_solved():
                    print(f"Solution found! Nodes expanded: {nodes_expanded}")
                    return current_path + [move], True

                model_val = self.get_model_value(child_state)
                misplaced = self.count_misplaced_pieces(child_state)

                # Heuristic combines model prediction and misplaced pieces
                h_score = -model_val + 0.1 * misplaced
                f_score = tentative_g + h_score

                best_g_score[child_key] = tentative_g
                heapq.heappush(open_set, (f_score, tentative_g, child_key, child_state, current_path + [move]))

        print(f"Search terminated. Nodes expanded: {nodes_expanded}")
        return [], False