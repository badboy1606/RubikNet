import torch
import numpy as np
import random
import time
from cube import Cube
from adi import ADI

# Color encoding for neural network input
COLOR_MAP = {
    'w': [1, 0, 0, 0, 0, 0],
    'y': [0, 1, 0, 0, 0, 0],
    'b': [0, 0, 1, 0, 0, 0],
    'g': [0, 0, 0, 1, 0, 0],
    'r': [0, 0, 0, 0, 1, 0],
    'o': [0, 0, 0, 0, 0, 1]
}

def encode_cube_state(state_str):
    # Convert cube state string to neural network format
    encoded = []
    for color in state_str:
        encoded.extend(COLOR_MAP[color])
    return encoded

def decode_cube_state(encoded_state):
    # Convert encoded state back to color list
    color_list = ['w', 'y', 'b', 'g', 'r', 'o']
    decoded = []
    if isinstance(encoded_state, torch.Tensor):
        encoded_state = encoded_state.cpu().detach().numpy().flatten()
    for i in range(0, len(encoded_state), 6):
        color_vec = encoded_state[i:i+6]
        color_i = np.argmax(color_vec)
        decoded.append(color_list[color_i])
    return decoded

def get_cube_children(cube):
    # Generate all possible child states from current state
    children = []
    original_state = cube.state.copy()
    for move in cube.moves:
        child_cube = Cube(state=original_state.copy())
        child_cube.move(move)
        children.append(child_cube.state.copy())
    return children

class MCTSNode:
    # MCTS tree node with UCB1 selection and neural network evaluation
    
    def __init__(self, state, model, parent=None, action_from_parent=None):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = {}
        self.expanded = False
        
        # Action statistics
        self.visit_counts = np.zeros(12)
        self.value_sums = np.zeros(12)
        self.virtual_losses = np.zeros(12)
        self.total_visits = 0
        
        # Neural network evaluation
        self.policy_probs, self.value = self._evaluate_state(model)
        self.is_solved = self._check_solved()
        
        # Create cube object for expansion
        try:
            decoded_state = decode_cube_state(state)
            self.cube = Cube(state=decoded_state)
        except Exception:
            self.cube = None
    
    def _evaluate_state(self, model):
        # Get policy and value from neural network
        try:
            with torch.no_grad():
                policy_logits, value = model(self.state)
            policy_probs = torch.softmax(policy_logits.squeeze(), dim=0).cpu().numpy()
            value_scalar = float(value.item())
        except Exception:
            policy_probs = np.ones(12) / 12.0
            value_scalar = 0.0
        return policy_probs, value_scalar
    
    def _check_solved(self):
        # Check if state represents solved cube
        try:
            decoded_state = decode_cube_state(self.state)
            test_cube = Cube(state=decoded_state)
            return test_cube.is_solved()
        except Exception:
            return False
    
    def select_action(self, c_puct=4.0):
        # UCB1 action selection with policy priors
        if self.total_visits == 0:
            return np.argmax(self.policy_probs)
        
        ucb_scores = []
        sqrt_total = np.sqrt(self.total_visits)
        
        for action in range(12):
            if self.visit_counts[action] == 0:
                ucb_scores.append(float('inf'))
            else:
                q_value = (self.value_sums[action] - self.virtual_losses[action]) / self.visit_counts[action]
                exploration = c_puct * self.policy_probs[action] * sqrt_total / (1 + self.visit_counts[action])
                ucb_scores.append(q_value + exploration)
        
        return np.argmax(ucb_scores)
    
    def expand(self, model):
        # Create all child nodes
        if self.expanded or self.is_solved or self.cube is None:
            return
        
        try:
            child_states = get_cube_children(self.cube)
            for action_idx, child_state in enumerate(child_states):
                if len(child_state) != 54:
                    continue
                child_str = ''.join(child_state)
                child_encoded = encode_cube_state(child_str)
                child_tensor = torch.FloatTensor(child_encoded).unsqueeze(0)
                child_node = MCTSNode(child_tensor, model, self, action_idx)
                self.children[action_idx] = child_node
            self.expanded = True
        except Exception as e:
            print(f"Error during expansion: {e}")
    
    def add_virtual_loss(self, action, loss=0.3):
        # Add virtual loss for parallel exploration
        self.virtual_losses[action] += loss
    
    def remove_virtual_loss(self, action, loss=0.3):
        # Remove virtual loss after simulation
        self.virtual_losses[action] = max(0, self.virtual_losses[action] - loss)
    
    def backup(self, action, value):
        # Update statistics after simulation
        self.visit_counts[action] += 1
        self.value_sums[action] += value
        self.total_visits += 1
        self.remove_virtual_loss(action)
    
    def get_best_action(self):
        # Get best action for final move selection
        if not self.children:
            return None
        
        best_action = None
        best_score = float('-inf')
        
        for action, child in self.children.items():
            if child.is_solved:
                return action
            
            if self.visit_counts[action] > 0:
                avg_value = self.value_sums[action] / self.visit_counts[action]
                visit_bonus = np.log(self.visit_counts[action] + 1)
                policy_bonus = self.policy_probs[action]
                score = avg_value + 0.01 * visit_bonus + 0.1 * policy_bonus
                
                if score > best_score:
                    best_score = score
                    best_action = action
        
        return best_action if best_action is not None else np.argmax(self.visit_counts)

def mcts_search(initial_state, model, num_simulations=10000, max_depth=20, c_puct=4.0):
    # Main MCTS search algorithm
    
    # Encode initial state
    try:
        if isinstance(initial_state, list):
            state_encoded = encode_cube_state(''.join(initial_state))
        else:
            state_encoded = encode_cube_state(initial_state)
        state_tensor = torch.FloatTensor(state_encoded).unsqueeze(0)
    except Exception as e:
        print(f"Error encoding initial state: {e}")
        return [], False
    
    root = MCTSNode(state_tensor, model)
    
    if root.is_solved:
        return [], True
    
    print(f"Running MCTS: {num_simulations} simulations, max depth {max_depth}")
    start_time = time.time()
    
    solutions_found = []
    
    for simulation in range(num_simulations):
        # Selection and expansion
        path = []
        actions = []
        current = root
        depth = 0
        
        # Tree traversal using UCB1
        while current.expanded and not current.is_solved and depth < max_depth:
            if not current.children:
                break
            
            action = current.select_action(c_puct)
            if action not in current.children:
                break
            
            current.add_virtual_loss(action)
            path.append(current)
            actions.append(action)
            current = current.children[action]
            depth += 1
            
            if current.is_solved:
                solutions_found.append(actions.copy())
                break
        
        # Expand leaf node
        if not current.is_solved and not current.expanded and depth < max_depth:
            current.expand(model)
            
            # Check immediate children for solutions
            for action_idx, child in current.children.items():
                if child.is_solved:
                    solution = actions + [action_idx]
                    solutions_found.append(solution)
                    break
        
        # Get value for backpropagation
        leaf_value = current.value
        
        # Sample child for evaluation if expanded
        if current.expanded and current.children and not current.is_solved:
            action_probs = current.policy_probs / np.sum(current.policy_probs)
            sampled_action = np.random.choice(12, p=action_probs)
            if sampled_action in current.children:
                leaf_value = max(leaf_value, current.children[sampled_action].value)
        
        # Update all nodes in path
        for i, (node, action) in enumerate(zip(path, actions)):
            node.backup(action, leaf_value)
        
        # Early termination for good solutions
        if solutions_found:
            best_solution = min(solutions_found, key=len)
            if len(best_solution) <= max(3, max_depth // 4):
                print(f"Early termination: found {len(best_solution)}-move solution")
                break
        
        # Progress reporting
        if simulation > 0 and simulation % (num_simulations // 10) == 0:
            elapsed = time.time() - start_time
            print(f"Simulation {simulation}/{num_simulations} ({elapsed:.1f}s)")
            if solutions_found:
                best_len = min(len(sol) for sol in solutions_found)
                print(f"Best solution so far: {best_len} moves")
    
    elapsed = time.time() - start_time
    print(f"MCTS completed in {elapsed:.1f}s")
    
    # Return best solution if found
    if solutions_found:
        best_solution = min(solutions_found, key=len)
        print(f"Solution found: {len(best_solution)} moves")
        return best_solution, True
    
    # Extract best path from tree
    print("No complete solution found, extracting best path...")
    path = extract_best_path(root, max_depth)
    return path, False

def extract_best_path(root, max_depth):
    # Extract most promising path from MCTS tree
    path = []
    current = root
    visited_states = set()
    
    for depth in range(max_depth):
        if current.is_solved:
            break
        
        state_key = tuple(current.state.cpu().numpy().flatten())
        if state_key in visited_states:
            break
        visited_states.add(state_key)
        
        if not current.children:
            break
        
        best_action = current.get_best_action()
        if best_action is None or best_action not in current.children:
            break
        
        path.append(best_action)
        current = current.children[best_action]
        
        if current.is_solved:
            break
    
    return path

def solve_cube(cube_state, model_path="deepcube_adi_model.pth", num_simulations=15000, max_depth=25):
    # Main cube solving function
    
    print("Loading neural network model...")
    model = ADI()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Extract state and moves
    if isinstance(cube_state, Cube):
        state = cube_state.state
        moves_list = cube_state.moves
    else:
        state = cube_state
        moves_list = Cube().moves
    
    # Run MCTS search
    solution_indices, solved = mcts_search(state, model, num_simulations, max_depth)
    
    if not solution_indices:
        return [], False
    
    # Convert indices to move strings
    solution_moves = []
    for move_idx in solution_indices:
        if move_idx < len(moves_list):
            solution_moves.append(moves_list[move_idx])
        else:
            print(f"Warning: Invalid move index {move_idx}")
            return [], False
    
    # Verify solution works
    if isinstance(cube_state, Cube):
        test_cube = Cube(state=cube_state.state.copy())
    else:
        decoded_state = decode_cube_state(state) if isinstance(state, torch.Tensor) else state
        test_cube = Cube(state=decoded_state)
    
    for move in solution_moves:
        test_cube.move(move)
    
    if test_cube.is_solved():
        print(f"Solution verified: {len(solution_moves)} moves")
        return solution_moves, True
    else:
        print("Solution verification failed")
        return solution_moves, False

# Test single scramble depth
def test_single_scramble(scramble_depth, num_simulations=None, max_depth=None):
    print(f"\n{'='*60}")
    print(f"TESTING {scramble_depth}-MOVE SCRAMBLE")
    print(f"{'='*60}")
    
    # Auto-select parameters based on difficulty
    if num_simulations is None:
        if scramble_depth <= 5:
            num_simulations = 8000
        elif scramble_depth <= 10:
            num_simulations = 12000
        elif scramble_depth <= 15:
            num_simulations = 18000
        else:
            num_simulations = 25000
    
    if max_depth is None:
        max_depth = max(scramble_depth + 5, 15)
    
    print(f"Parameters: {num_simulations} simulations, max depth {max_depth}")
    
    cube = Cube()
    cube.scramble(scramble_depth)
    print(f"Scramble: {' '.join(cube.move_history)}")
    
    start_time = time.time()
    solution, success = solve_cube(cube, num_simulations=num_simulations, max_depth=max_depth)
    solve_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    if success:
        print("SUCCESS!")
        print(f"Original: {' '.join(cube.move_history)} ({len(cube.move_history)} moves)")
        print(f"Solution: {' '.join(solution)} ({len(solution)} moves)")
        print(f"Time: {solve_time:.1f} seconds")
        print(f"Efficiency: {len(cube.move_history)/len(solution):.2f}")
        print(f"Speed: {len(solution)/solve_time:.1f} moves/sec")
    else:
        print("FAILED")
        print(f"Time: {solve_time:.1f} seconds")
        if solution:
            print(f"Partial solution: {len(solution)} moves")
    print(f"{'='*60}")
    
    return success, len(solution) if success else 0, solve_time

# Run comprehensive test suite
def run_test_suite():
    print("RUBIK'S CUBE MCTS SOLVER - TEST SUITE")
    print("=" * 70)
    
    test_depths = [3, 5, 7, 10, 12, 15, 20]
    results = {}
    
    for depth in test_depths:
        print(f"\n{'='*70}")
        print(f"TESTING {depth}-MOVE SCRAMBLES")
        print(f"{'='*70}")
        
        num_tests = 3 if depth <= 15 else 2
        successes = 0
        total_time = 0
        solution_lengths = []
        
        for test_i in range(num_tests):
            print(f"\n--- Test {test_i+1}/{num_tests} for {depth}-move scramble ---")
            
            success, sol_len, time_taken = test_single_scramble(depth)
            total_time += time_taken
            
            if success:
                successes += 1
                solution_lengths.append(sol_len)
                print(f"SUCCESS: {sol_len} moves in {time_taken:.1f}s")
            else:
                print(f"FAILED after {time_taken:.1f}s")
        
        # Calculate statistics
        success_rate = successes / num_tests
        avg_time = total_time / num_tests
        avg_solution_length = sum(solution_lengths) / len(solution_lengths) if solution_lengths else 0
        
        results[depth] = {
            'success_rate': success_rate,
            'successes': successes,
            'total_tests': num_tests,
            'avg_time': avg_time,
            'avg_solution_length': avg_solution_length,
            'solution_lengths': solution_lengths
        }
        
        print(f"\nRESULTS FOR {depth}-MOVE SCRAMBLES:")
        print(f"Success rate: {successes}/{num_tests} ({success_rate*100:.1f}%)")
        print(f"Average time: {avg_time:.1f}s")
        if solution_lengths:
            print(f"Average solution length: {avg_solution_length:.1f} moves")
            print(f"Compression ratio: {depth/avg_solution_length:.2f}")
            min_sol = min(solution_lengths)
            max_sol = max(solution_lengths)
            print(f"Solution range: {min_sol}-{max_sol} moves")
    
    print_summary(results)
    return results

# Print final summary
def print_summary(results):
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    print(f"{'Depth':<6} {'Success Rate':<12} {'Avg Time':<10} {'Avg Moves':<10} {'Compression':<12}")
    print("-" * 70)
    
    total_successes = 0
    total_tests = 0
    
    for depth, stats in results.items():
        success_rate = stats['success_rate'] * 100
        avg_time = stats['avg_time']
        avg_moves = stats['avg_solution_length']
        compression = depth / avg_moves if avg_moves > 0 else 0
        
        print(f"{depth:<6} {stats['successes']}/{stats['total_tests']} ({success_rate:4.1f}%) {avg_time:8.1f}s "
              f"{avg_moves:8.1f} {compression:10.2f}" if avg_moves > 0 
              else f"{depth:<6} {stats['successes']}/{stats['total_tests']} ({success_rate:4.1f}%) {avg_time:8.1f}s "
                   f"{'N/A':>8} {'N/A':>10}")
        
        total_successes += stats['successes']
        total_tests += stats['total_tests']
    
    overall_success_rate = (total_successes / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"Total success rate: {total_successes}/{total_tests} ({overall_success_rate:.1f}%)")
    
    # Performance by category
    categories = {
        "Easy (3-7 moves)": [3, 5, 7],
        "Medium (10-12 moves)": [10, 12], 
        "Hard (15-20 moves)": [15, 20]
    }
    
    for category, depths in categories.items():
        cat_successes = sum(results[d]['successes'] for d in depths if d in results)
        cat_total = sum(results[d]['total_tests'] for d in depths if d in results)
        if cat_total > 0:
            cat_rate = (cat_successes / cat_total) * 100
            print(f"{category}: {cat_successes}/{cat_total} ({cat_rate:.1f}%)")

# Test different parameter configurations
def benchmark_performance():
    print(f"\n{'='*70}")
    print("PERFORMANCE BENCHMARK")
    print(f"{'='*70}")
    
    scramble_depth = 10
    test_configs = [
        {"sims": 5000, "name": "Fast"},
        {"sims": 10000, "name": "Balanced"}, 
        {"sims": 20000, "name": "Thorough"}
    ]
    
    for config in test_configs:
        print(f"\n--- {config['name']} Configuration ({config['sims']} simulations) ---")
        
        successes = 0
        times = []
        solution_lengths = []
        
        for i in range(3):
            cube = Cube()
            cube.scramble(scramble_depth)
            
            start_time = time.time()
            solution, success = solve_cube(cube, num_simulations=config['sims'], max_depth=20)
            solve_time = time.time() - start_time
            
            times.append(solve_time)
            if success:
                successes += 1
                solution_lengths.append(len(solution))
        
        avg_time = sum(times) / len(times)
        avg_solution_length = sum(solution_lengths) / len(solution_lengths) if solution_lengths else 0
        success_rate = (successes / 3) * 100
        
        print(f"Success rate: {successes}/3 ({success_rate:.1f}%)")
        print(f"Average time: {avg_time:.1f}s")
        if solution_lengths:
            print(f"Average solution length: {avg_solution_length:.1f} moves")
            print(f"Moves per second: {avg_solution_length/avg_time:.1f}")

# Test extreme scrambles
def stress_test():
    print(f"\n{'='*70}")
    print("STRESS TEST - EXTREME SCRAMBLES")
    print(f"{'='*70}")
    
    extreme_depths = [25, 30]
    
    for depth in extreme_depths:
        print(f"\n--- {depth}-move scrambles ---")
        
        success, sol_len, time_taken = test_single_scramble(
            depth, 
            num_simulations=30000, 
            max_depth=40
        )
        
        if success:
            print(f"Extreme test passed! {sol_len} moves in {time_taken:.1f}s")
        else:
            print(f"Extreme test failed after {time_taken:.1f}s")

# Quick demonstration
def quick_demo():
    print(f"\n{'='*70}")
    print("QUICK DEMO")
    print(f"{'='*70}")
    
    scramble_depth = 8
    cube = Cube()
    cube.scramble(scramble_depth)
    
    print(f"Scrambled cube: {' '.join(cube.move_history)}")
    print("Solving...")
    
    start_time = time.time()
    solution, success = solve_cube(cube, num_simulations=10000, max_depth=15)
    solve_time = time.time() - start_time
    
    if success:
        print(f"\nDemo successful!")
        print(f"Scramble: {' '.join(cube.move_history)} ({len(cube.move_history)} moves)")
        print(f"Solution: {' '.join(solution)} ({len(solution)} moves)")
        print(f"Solved in {solve_time:.1f} seconds")
        print(f"Compression ratio: {len(cube.move_history)/len(solution):.2f}")
    else:
        print(f"\nDemo failed")
        print(f"Time taken: {solve_time:.1f} seconds")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "demo":
            quick_demo()
        elif command == "benchmark":
            benchmark_performance()
        elif command == "stress":
            stress_test()
        elif command == "suite":
            run_test_suite()
        elif command.isdigit():
            depth = int(command)
            test_single_scramble(depth)
        else:
            print("Usage: python mcts_complete.py [demo|benchmark|stress|suite|<depth>]")
            print("  demo      - Quick demonstration")
            print("  benchmark - Performance benchmark")
            print("  stress    - Extreme scramble test")  
            print("  suite     - Full test suite")
            print("  <depth>   - Test specific scramble depth")
    else:
        # Default example
        print("MCTS Rubik's Cube Solver")
        print("Run with 'python mcts_complete.py demo' for a quick demonstration")
        print("Or 'python mcts_complete.py suite' for full testing")
        
        cube = Cube()
        cube.scramble(10)
        print(f"\nExample scramble: {' '.join(cube.move_history)}")
        
        start_time = time.time()
        solution, success = solve_cube(cube, num_simulations=12000, max_depth=15)
        solve_time = time.time() - start_time
        
        if success:
            print(f"Solved in {len(solution)} moves: {' '.join(solution)}")
            print(f"Time: {solve_time:.1f}s")
        else:
            print(f"Failed to solve in {solve_time:.1f}s")