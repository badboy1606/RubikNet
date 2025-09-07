import torch
import numpy as np
from adi import ADI
from cube import Cube


class BeamSearch:
    def __init__(self, model_path="deepcube_adi_model.pth", device="cpu"):
        # Load ADI model for state value estimation
        self.model = ADI()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        # One-hot color encoding for cube state
        self.color_map = {
            'w': [1, 0, 0, 0, 0, 0],
            'y': [0, 1, 0, 0, 0, 0],
            'b': [0, 0, 1, 0, 0, 0],
            'g': [0, 0, 0, 1, 0, 0],
            'r': [0, 0, 0, 0, 1, 0],
            'o': [0, 0, 0, 0, 0, 1]
        }

    # ---------- state helpers ----------
    def get_cube_child_states(self, cube):
        # Generate all child states by applying every move once
        children = []
        original_state = cube.state.copy()
        for move in cube.moves:
            child_cube = Cube(state=original_state.copy())
            child_cube.move(move)
            children.append(child_cube.state.copy())
        return children

    def encode_cube_state(self, state_str):
        # Convert cube state to one-hot encoded vector
        encoded = []
        for color in state_str:
            encoded.extend(self.color_map[color])
        return encoded

    def decode_cube_state(self, encoded_state):
        # Convert one-hot encoding back to cube state list
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
        # Make string key from state list (used for seen set)
        return ''.join(state_list)

    def _to_state_list(self, state):
        # Normalize state format to list of 54 color chars
        if isinstance(state, list) and len(state) == 54 and all(isinstance(x, str) for x in state):
            return state
        if isinstance(state, torch.Tensor):
            return self.decode_cube_state(state)
        if isinstance(state, str):
            return list(state)
        return list(state)

    def _inverse_move(self, m):
        # Return inverse of a given move
        return m[:-1] if m.endswith("'") else m + "'"

    # ---------- scoring ----------
    def _heuristic_by_centers(self, state_list):
        # Heuristic: count stickers matching center color on each face
        if not isinstance(state_list, list) or len(state_list) != 54:
            return 0.0
        score = 0
        for f in range(6):
            base = f * 9
            center = state_list[base + 4]
            face = state_list[base:base + 9]
            score += sum(1 for c in face if c == center)
        return float(score)

    def _model_value_score(self, state_tensor):
        # Get predicted value from ADI model for given state
        try:
            with torch.no_grad():
                _, value = self.model(state_tensor)
            v = value.item() if torch.is_tensor(value) else float(value)
            return float(v)
        except Exception:
            return None

    def _tensor_from_state_list(self, state_list):
        # Convert state list to model input tensor
        enc = self.encode_cube_state(''.join(state_list))
        return torch.FloatTensor(enc).unsqueeze(0)

    # ---------- main search ----------
    def search(
        self,
        start_state,
        beam_width=5,
        max_depth=20,
        prune_inverses=True,
        avoid_repeats=True
    ):
        """
        Perform beam search to solve Rubik's Cube.
        Returns (list_of_moves, solved_bool).
        """
        s_list = self._to_state_list(start_state)
        root_cube = Cube(state=s_list)
        if root_cube.is_solved():
            return [], True

        # Combined heuristic score: face center match + model value prediction
        def node_score(state_list):
            base = self._heuristic_by_centers(state_list)
            st = self._tensor_from_state_list(state_list)
            mv = self._model_value_score(st)
            if mv is None:
                return base
            return base + 50.0 * mv

        start_key = self._state_key_from_list(s_list)
        beam = [(s_list, [], None, node_score(s_list))]  # (state, path, last_move, score)
        seen = {start_key: 0}

        for depth in range(1, max_depth + 1):
            candidates = []

            for state_list, path, last_move, _ in beam:
                cube = Cube(state=state_list)
                child_states = self.get_cube_child_states(cube)
                moves = getattr(cube, "moves", list(range(len(child_states))))

                for idx, child in enumerate(child_states):
                    if not isinstance(child, list) or len(child) != 54:
                        continue
                    move = moves[idx] if idx < len(moves) else idx

                    # Skip immediate inverse move to avoid undoing last step
                    if prune_inverses and last_move is not None:
                        if isinstance(move, str) and self._inverse_move(move) == last_move:
                            continue

                    key = self._state_key_from_list(child)
                    # Skip already visited states
                    if avoid_repeats and key in seen and seen[key] <= depth:
                        continue

                    child_cube = Cube(state=child)
                    if child_cube.is_solved():
                        return path + [move], True

                    sc = node_score(child)
                    candidates.append((child, path + [move], move, sc))
                    if avoid_repeats:
                        seen[key] = depth

            if not candidates:
                return [], False  # Dead-end

            # Keep top beam_width candidates based on score
            candidates.sort(key=lambda x: x[3], reverse=True)
            beam = candidates[:beam_width]

        return [], False  # Max depth reached, not solved