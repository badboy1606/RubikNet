import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from adi import ADI
from cube import Cube
from a_star import AStar
from beam import BeamSearch


MAX_TIME_PER_TRIAL = 1200  # Maximum time allowed per trial in seconds


def run_experiments(max_scramble=4, num_trials=5):
    # Load trained ADI model
    model = ADI()
    model.load_state_dict(torch.load("deepcube_adi_model.pth", map_location="cpu"))
    model.eval()

    # Initialize solvers
    astar_solver = AStar("deepcube_adi_model.pth")
    beam_solver = BeamSearch("deepcube_adi_model.pth")

    scramble_lengths = list(range(max_scramble))  # Scramble depths to test
    print(scramble_lengths)

    # Store results for A* and Beam search
    astar_results = {"percent": [], "times": []}
    beam_results = {"percent": [], "times": []}

    for scramble_len in scramble_lengths:
        print(f"\n===== SCRAMBLE LENGTH: {scramble_len} =====")

        solved_astar = 0
        solved_beam = 0
        total_time_astar = 0.0
        total_time_beam = 0.0

        for trial in range(num_trials):
            # Create and scramble cube
            cube = Cube()
            cube.reset_cube()
            cube.scramble(scramble_len)
            state = cube.state.copy()

            # --- A* Search ---
            start_time = time.perf_counter()
            moves_astar, solved = None, False
            try:
                moves_astar, solved = astar_solver.a_star_search(
                    start_state=state,
                    max_nodes=70000,
                    max_depth=20
                )
            except Exception as e:
                print(f"A* failed: {e}")
                solved = False
            elapsed = time.perf_counter() - start_time

            if elapsed > MAX_TIME_PER_TRIAL:
                print("⏳ A* exceeded time limit, marking as unsolved.")
                solved = False
                elapsed = MAX_TIME_PER_TRIAL  # Cap the time for fairness

            total_time_astar += elapsed
            if solved:
                solved_astar += 1

            # --- Beam Search ---
            start_time = time.perf_counter()
            moves_beam, solved = None, False
            try:
                moves_beam, solved = beam_solver.search(
                    start_state=state,
                    beam_width=200,
                    max_depth=50
                )
            except Exception as e:
                print(f"Beam failed: {e}")
                solved = False
            elapsed = time.perf_counter() - start_time

            if elapsed > MAX_TIME_PER_TRIAL:
                print("Beam ka time hogyaya solve karne ka ")  # Debug message
                solved = False
                elapsed = MAX_TIME_PER_TRIAL  # Cap the time

            total_time_beam += elapsed
            if solved:
                solved_beam += 1

        # Store percentage solved and average time for this scramble length
        astar_results["percent"].append(100.0 * solved_astar / num_trials)
        beam_results["percent"].append(100.0 * solved_beam / num_trials)

        astar_results["times"].append(total_time_astar / num_trials)
        beam_results["times"].append(total_time_beam / num_trials)

        # Print summary for this scramble length
        print(f"A*: {astar_results['percent'][-1]:.1f}% solved, avg time {astar_results['times'][-1]:.3f}s")
        print(f"Beam: {beam_results['percent'][-1]:.1f}% solved, avg time {beam_results['times'][-1]:.3f}s")

    return scramble_lengths, astar_results, beam_results


def plot_results(scramble_lengths, astar_results, beam_results):
    # Plot percentage of cubes solved
    plt.figure(figsize=(10, 5))
    plt.plot(scramble_lengths, astar_results["percent"], marker="o", label="A* Search")
    plt.plot(scramble_lengths, beam_results["percent"], marker="s", label="Beam Search")
    plt.xlabel("Scramble Length")
    plt.ylabel("Solved Percentage (%)")
    plt.title("Solved Percentage vs Scramble Length")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot average time taken
    plt.figure(figsize=(10, 5))
    plt.plot(scramble_lengths, astar_results["times"], marker="o", label="A* Search")
    plt.plot(scramble_lengths, beam_results["times"], marker="s", label="Beam Search")
    plt.xlabel("Scramble Length")
    plt.ylabel("Average Solve Time (s)")
    plt.title("Average Solve Time vs Scramble Length")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Run experiments and visualize results
    scramble_lengths, astar_results, beam_results = run_experiments(
        max_scramble=7,   # Maximum scramble depth to test
        num_trials=25     # Number of trials per scramble length
    )

    plot_results(scramble_lengths, astar_results, beam_results)