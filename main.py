"""
ACO Maze Solver - Main Entry Point

This is the main entry point for the Ant Colony Optimization (ACO) maze pathfinding system.
It orchestrates maze generation, baseline A* computation, ACO solving with live visualization,
and exports results to the outputs/ folder.

Project Structure:
- config.py: Configuration parameters
- maze.py: Maze generation and utilities
- astar.py: A* baseline algorithm
- aco_solver.py: ACO algorithm implementation
- visualization.py: Live animation and image export
- main.py: Main entry point (this file)
"""

import random
import numpy as np
import json

from config import ACOConfig
from maze import generate_maze, find_start_goal
from astar import astar
from aco_solver import ACOSolver
from visualization import visualize, save_hero_image


def main():
    """Main execution function."""
    
    # Load configuration
    cfg = ACOConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    print("="*60)
    print("ACO Maze Solver - Nature-Inspired Pathfinding")
    print("="*60)
    print(f"Maze size: {cfg.rows}x{cfg.cols}")
    print(f"Ants per generation: {cfg.n_ants}")
    print(f"ACO parameters: α={cfg.alpha}, β={cfg.beta}, ρ={cfg.rho}")
    print("="*60)
    
    # Generate maze
    print("\n[1/5] Generating maze...")
    grid = generate_maze(cfg.rows, cfg.cols, cfg.seed)
    start, goal = find_start_goal(grid)
    print(f"  Start: {start}, Goal: {goal}")
    
    # A* baseline
    print("\n[2/5] Running A* baseline...")
    astar_path = astar(grid, start, goal)
    astar_len = len(astar_path) if astar_path else None
    print(f"  A* optimal path length: {astar_len}")
    
    # Initialize ACO solver
    print("\n[3/5] Initializing ACO solver...")
    solver = ACOSolver(grid, start, goal, cfg)
    print(f"  Solver initialized with {cfg.n_ants} ants")
    
    # Run visualization
    print("\n[4/5] Running ACO algorithm (live visualization)...")
    print("  Close window to continue after solution found\n")
    stats = visualize(grid, solver, cfg)
    
    # Save outputs
    print("\n[5/5] Saving outputs...")
    save_hero_image(grid, solver)
    print("  ✓ Hero image saved: outputs/hero.png")
    
    # Save statistics
    report = {
        'algorithm': 'Pure ACO (Ant Colony Optimization)',
        'config': cfg.__dict__,
        'results': {
            'astar_optimal_length': astar_len,
            'aco_best_length': solver.best_length,
            'solution_found_at_generation': solver.solution_found_at,
            'total_ants_released': solver.total_ants,
            'total_generations': solver.generation,
            'path_quality': round(solver.best_length / astar_len, 2) if astar_len else None,
        },
        'stats_sample': stats[-10:] if len(stats) >= 10 else stats,
    }
    
    with open('outputs/stats.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("  ✓ Statistics saved: outputs/stats.json")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"A* Optimal Path: {astar_len} cells")
    print(f"ACO Best Path: {solver.best_length} cells")
    if astar_len:
        quality = (solver.best_length / astar_len) * 100
        print(f"Path Quality: {quality:.1f}% of optimal")
    print(f"Solution Found: Generation {solver.solution_found_at}")
    print(f"Total Ants Released: {solver.total_ants}")
    print("="*60)
    print("\nDone! Check outputs/ folder for results.")


if __name__ == '__main__':
    main()
