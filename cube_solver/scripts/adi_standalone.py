import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
import json
import os
from datetime import datetime

# Import your existing modules
from cube import Cube, encode_cube_state
from adi import ADI  # Import your ADI model class

# Set style for better plots with fallback
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

try:
    sns.set_palette("husl")
except:
    pass

class ADITester:
    def __init__(self, model_path='deepcube_adi_model.pth', device=None):
        """Initialize the tester with a trained ADI model."""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load the trained model
        self.model = ADI().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded successfully from {model_path}")
        
        # Results storage
        self.results = defaultdict(list)
        
    def solve_cube(self, cube, max_moves=50, verbose=False):
        """
        Attempt to solve a cube using the trained model.
        Returns (solved, moves_taken, move_sequence, values_sequence)
        """
        move_sequence = []
        values_sequence = []
        # Create a copy of the cube by copying its state
        cube_copy = Cube(state=cube.state.copy())
        
        for move_count in range(max_moves):
            # Check if solved
            if cube_copy.is_solved():
                return True, move_count, move_sequence, values_sequence
            
            # Get current state as string (join the state list)
            state_string = ''.join(cube_copy.state)
            state_encoded = encode_cube_state(state_string)
            state_tensor = torch.FloatTensor(state_encoded).unsqueeze(0).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                policy_logits, value = self.model(state_tensor)
                policy_probs = F.softmax(policy_logits, dim=-1)
                predicted_move_idx = torch.argmax(policy_probs).item()
                predicted_value = value.item()
            
            # Execute the predicted move
            move_name = cube_copy.moves[predicted_move_idx]
            cube_copy.move(move_name)
            
            move_sequence.append(move_name)
            values_sequence.append(predicted_value)
            
            if verbose:
                print(f"Move {move_count + 1}: {move_name} (value: {predicted_value:.4f})")
        
        # Not solved within max_moves
        return False, max_moves, move_sequence, values_sequence
    
    def test_scramble_depth(self, depth, num_tests=100, max_moves=50, verbose=False):
        """Test the model on cubes scrambled to a specific depth."""
        print(f"\nTesting scramble depth {depth} ({num_tests} cubes)...")
        
        results = {
            'depth': depth,
            'total_tests': num_tests,
            'solved_count': 0,
            'solve_rates': [],
            'moves_to_solve': [],
            'solve_times': [],
            'failed_attempts': 0,
            'avg_final_value': 0,
            'move_sequences': [],
            'value_sequences': []
        }
        
        start_time = time.time()
        
        for test_i in range(num_tests):
            if verbose or (test_i + 1) % 10 == 0:
                print(f"  Test {test_i + 1}/{num_tests}")
            
            # Create and scramble cube
            cube = Cube()
            if depth > 0:
                cube.scramble(depth)
            
            # Attempt to solve
            test_start = time.time()
            solved, moves_taken, move_seq, value_seq = self.solve_cube(
                cube, max_moves=max_moves, verbose=False
            )
            test_time = time.time() - test_start
            
            # Record results
            results['solve_times'].append(test_time)
            results['move_sequences'].append(move_seq)
            results['value_sequences'].append(value_seq)
            
            if solved:
                results['solved_count'] += 1
                results['moves_to_solve'].append(moves_taken)
            else:
                results['failed_attempts'] += 1
                if value_seq:
                    results['avg_final_value'] += value_seq[-1]
        
        # Calculate final statistics
        results['solve_rate'] = results['solved_count'] / num_tests * 100
        results['avg_moves_to_solve'] = np.mean(results['moves_to_solve']) if results['moves_to_solve'] else 0
        results['avg_solve_time'] = np.mean(results['solve_times'])
        results['total_time'] = time.time() - start_time
        
        if results['failed_attempts'] > 0:
            results['avg_final_value'] /= results['failed_attempts']
        
        print(f"  Results: {results['solved_count']}/{num_tests} solved ({results['solve_rate']:.1f}%)")
        if results['moves_to_solve']:
            print(f"  Average moves to solve: {results['avg_moves_to_solve']:.1f}")
        
        return results
    
    def comprehensive_test(self, max_depth=15, tests_per_depth=100, max_moves=50, save_results=True):
        """Run comprehensive tests across multiple scramble depths."""
        print("="*60)
        print("COMPREHENSIVE ADI MODEL TESTING")
        print("="*60)
        
        all_results = []
        
        for depth in range(max_depth + 1):
            results = self.test_scramble_depth(
                depth=depth,
                num_tests=tests_per_depth,
                max_moves=max_moves,
                verbose=False
            )
            all_results.append(results)
            self.results[depth] = results
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"adi_test_results_{timestamp}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = []
            for result in all_results:
                json_result = result.copy()
                for key, value in json_result.items():
                    if isinstance(value, (np.ndarray, np.float64, np.int64)):
                        json_result[key] = float(value) if isinstance(value, (np.float64, np.int64)) else value.tolist()
                json_results.append(json_result)
            
            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"\nResults saved to {filename}")
        
        return all_results
    
    def plot_results(self, results=None, save_plots=True):
        """Generate comprehensive visualizations of test results."""
        if results is None:
            results = [self.results[depth] for depth in sorted(self.results.keys())]
        
        if not results:
            print("No results to plot. Run tests first.")
            return
        
        # Extract data for plotting
        depths = [r['depth'] for r in results]
        solve_rates = [r['solve_rate'] for r in results]
        avg_moves = [r['avg_moves_to_solve'] for r in results if r['moves_to_solve']]
        solve_depths = [r['depth'] for r in results if r['moves_to_solve']]
        avg_times = [r['avg_solve_time'] for r in results]
        
        try:
            # Create comprehensive plot
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('ADI Model Performance Analysis', fontsize=16, fontweight='bold')
            
            # 1. Solve Rate vs Scramble Depth
            axes[0, 0].plot(depths, solve_rates, 'o-', linewidth=2, markersize=8, color='blue')
            axes[0, 0].set_xlabel('Scramble Depth')
            axes[0, 0].set_ylabel('Solve Rate (%)')
            axes[0, 0].set_title('Solve Rate vs Scramble Depth')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 105)
            
            # Add trend line if we have enough data points
            if len(depths) > 2:
                try:
                    z = np.polyfit(depths, solve_rates, min(2, len(depths)-1))
                    p = np.poly1d(z)
                    axes[0, 0].plot(depths, p(depths), '--', alpha=0.8, color='red')
                except:
                    pass
            
            # 2. Average Moves to Solve
            if solve_depths and avg_moves:
                axes[0, 1].plot(solve_depths, avg_moves, 's-', linewidth=2, markersize=8, color='green')
                axes[0, 1].set_xlabel('Scramble Depth')
                axes[0, 1].set_ylabel('Average Moves to Solve')
                axes[0, 1].set_title('Solving Efficiency')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'No solve data\navailable', ha='center', va='center', 
                               transform=axes[0, 1].transAxes, fontsize=14)
                axes[0, 1].set_title('Solving Efficiency')
            
            # 3. Solve Time Distribution
            axes[0, 2].plot(depths, avg_times, '^-', linewidth=2, markersize=8, color='purple')
            axes[0, 2].set_xlabel('Scramble Depth')
            axes[0, 2].set_ylabel('Average Solve Time (seconds)')
            axes[0, 2].set_title('Computational Performance')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Move Distribution Histogram (for a specific depth)
            target_depth = min(5, max(depths)) if depths else 0
            target_result = next((r for r in results if r['depth'] == target_depth), None)
            if target_result and target_result['moves_to_solve']:
                axes[1, 0].hist(target_result['moves_to_solve'], bins=20, alpha=0.7, 
                               color='orange', edgecolor='black')
                axes[1, 0].set_xlabel('Moves to Solve')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title(f'Move Distribution (Depth {target_depth})')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No solve data\navailable', ha='center', va='center', 
                               transform=axes[1, 0].transAxes, fontsize=14)
                axes[1, 0].set_title('Move Distribution')
            
            # 5. Success Rate Bar Chart
            colors = ['red' if rate < 50 else 'orange' if rate < 80 else 'green' for rate in solve_rates]
            axes[1, 1].bar(depths, solve_rates, color=colors, alpha=0.7)
            axes[1, 1].set_xlabel('Scramble Depth')
            axes[1, 1].set_ylabel('Solve Rate (%)')
            axes[1, 1].set_title('Success Rate by Depth')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 105)
            
            # 6. Summary Statistics Table
            axes[1, 2].axis('off')
            
            # Create summary table
            total_tests = sum(r['total_tests'] for r in results)
            total_solved = sum(r['solved_count'] for r in results)
            overall_rate = (total_solved / total_tests * 100) if total_tests > 0 else 0
            
            summary_text = f"""
PERFORMANCE SUMMARY

Total Tests: {total_tests:,}
Total Solved: {total_solved:,}
Overall Success Rate: {overall_rate:.1f}%

Best Depth: {depths[np.argmax(solve_rates)]} ({max(solve_rates):.1f}%)
Worst Depth: {depths[np.argmin(solve_rates)]} ({min(solve_rates):.1f}%)

Avg Solve Time: {np.mean(avg_times):.3f}s
            """
            
            axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                            fontsize=11, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            if save_plots:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'adi_performance_analysis_{timestamp}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Plots saved as {filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
            print("Continuing with text summary only...")
    
    def print_detailed_summary(self, results=None):
        """Print a detailed text summary of results."""
        if results is None:
            results = [self.results[depth] for depth in sorted(self.results.keys())]
        
        if not results:
            print("No results available.")
            return
        
        print("\n" + "="*80)
        print("DETAILED PERFORMANCE SUMMARY")
        print("="*80)
        
        print(f"{'Depth':<6} {'Tests':<6} {'Solved':<7} {'Rate':<8} {'Avg Moves':<10} {'Avg Time':<10}")
        print("-" * 60)
        
        for result in results:
            depth = result['depth']
            tests = result['total_tests']
            solved = result['solved_count']
            rate = result['solve_rate']
            avg_moves = result['avg_moves_to_solve']
            avg_time = result['avg_solve_time']
            
            print(f"{depth:<6} {tests:<6} {solved:<7} {rate:<7.1f}% {avg_moves:<9.1f} {avg_time:<9.3f}s")
        
        # Overall statistics
        total_tests = sum(r['total_tests'] for r in results)
        total_solved = sum(r['solved_count'] for r in results)
        overall_rate = (total_solved / total_tests * 100) if total_tests > 0 else 0
        
        print("-" * 60)
        print(f"TOTAL  {total_tests:<6} {total_solved:<7} {overall_rate:<7.1f}%")
        
        # Find best and worst performing depths
        rates = [r['solve_rate'] for r in results]
        best_idx = np.argmax(rates)
        worst_idx = np.argmin(rates)
        
        print(f"\nBest Performance: Depth {results[best_idx]['depth']} ({rates[best_idx]:.1f}%)")
        print(f"Worst Performance: Depth {results[worst_idx]['depth']} ({rates[worst_idx]:.1f}%)")


def main():
    """Main testing routine."""
    import os  # Ensure os is available
    
    print("Looking for model file...")
    print(f"Current directory: {os.getcwd()}")
    
    # List all .pth files in current directory
    pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    
    if pth_files:
        print(f"Found .pth files: {pth_files}")
        if len(pth_files) == 1:
            model_path = pth_files[0]
            print(f"Using: {model_path}")
        else:
            print("Multiple .pth files found. Using the first one:", pth_files[0])
            model_path = pth_files[0]
    else:
        print("No .pth files found in current directory.")
        print("Please ensure your model file (.pth) is in the current directory.")
        return
    
    # Configuration
    MAX_DEPTH = 15
    TESTS_PER_DEPTH = 100
    MAX_MOVES = 50
    
    print("\nADI Model Standalone Tester")
    print("="*40)
    print(f"Model: {model_path}")
    print(f"Testing depths: 0-{MAX_DEPTH}")
    print(f"Tests per depth: {TESTS_PER_DEPTH}")
    print(f"Max moves per solve attempt: {MAX_MOVES}")
    
    try:
        # Initialize tester
        tester = ADITester(model_path)
        
        # Run comprehensive tests
        results = tester.comprehensive_test(
            max_depth=MAX_DEPTH,
            tests_per_depth=TESTS_PER_DEPTH,
            max_moves=MAX_MOVES,
            save_results=True
        )
        
        # Generate visualizations
        tester.plot_results(results, save_plots=True)
        
        # Print detailed summary
        tester.print_detailed_summary(results)
        
        print("\nTesting completed!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check that:")
        print("1. Your model file exists and is accessible")
        print("2. The cube.py and adi.py modules are in the same directory")
        print("3. All required dependencies are installed")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()