"""
Configuration Module for ACO Maze Solver

This module contains all tunable parameters for the Ant Colony Optimization algorithm.
Adjust these values to experiment with different maze sizes, ant populations, and
pheromone dynamics. The default values represent a balanced configuration that works
well for 16×16 mazes.

Key Parameters:
- alpha/beta: Control balance between pheromone trails and heuristic guidance
- rho: Evaporation rate (higher = faster forgetting of old paths)
- penalty_factor: How aggressively to discourage dead-end paths
"""

from dataclasses import dataclass


@dataclass
class ACOConfig:
    """ACO Algorithm Configuration"""
    
    # Maze parameters
    rows: int = 16
    cols: int = 16
    
    # ACO parameters
    n_ants: int = 30  # Ants per generation
    max_generations: int = 100  # Safety limit
    
    # ACO hyperparameters
    alpha: float = 1.0  # Pheromone importance (τ^α)
    beta: float = 2.0  # Heuristic importance (η^β)
    rho: float = 0.1  # Evaporation rate (10% per iteration)
    Q: float = 100.0  # Pheromone deposit constant
    initial_pheromone: float = 0.1
    penalty_factor: float = 0.2  # Dead-end penalty (0.2 = 80% reduction)
    
    # Termination
    stop_on_solution: bool = True
    required_solutions: int = 1  # Stop when N ants find solution
    
    # Visualization
    animation_interval: int = 30  # ms between frames
    
    seed: int = 42
