"""
Ant Colony Optimization (ACO) Solver for Maze Pathfinding

This module implements the core ACO algorithm inspired by the foraging behavior of real ants.
Artificial ants explore the maze, deposit pheromones on successful paths, and collectively
converge toward optimal solutions through stigmergy (indirect communication via environment).

Key Features:
- Dual pheromone system (strong/weak) for distinguishing successful vs exploratory paths
- Probabilistic movement based on pheromone trails (τ) and heuristic distance (η)
- Anti-pheromone penalties for dead-end paths to guide future exploration
- Per-ant memory (visited set) to prevent infinite loops within a single path

The algorithm demonstrates swarm intelligence principles: simple individual rules
producing sophisticated collective behavior without centralized control.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from config import ACOConfig
from maze import get_neighbors, manhattan


class ACOSolver:
    """
    Ant Colony Optimization Solver for Maze Pathfinding
    
    Algorithm Overview:
    1. Initialize pheromone matrix with uniform small values
    2. Release N ants from start position each generation
    3. Each ant moves probabilistically based on: P ∝ τ^α × η^β
       - τ (tau): pheromone strength (collective memory)
       - η (eta): heuristic value (1 / distance_to_goal)
       - α: controls pheromone importance
       - β: controls heuristic importance
    4. After all ants finish:
       - Apply penalties to dead-end paths (anti-pheromone)
       - Deposit strong pheromones on successful paths (goal-reaching ants)
       - Deposit weak pheromones on all paths (exploration tracking)
       - Evaporate all pheromones to prevent unlimited accumulation
    5. Repeat until solution found or max generations reached
    """
    
    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], cfg: ACOConfig):
        """
        Initialize ACO solver.
        
        Args:
            grid: Binary maze grid
            start: Start position
            goal: Goal position
            cfg: ACO configuration
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        self.cfg = cfg
        
        self.h, self.w = grid.shape
        
        # Pheromone matrices
        self.pheromone = np.full((self.h, self.w), cfg.initial_pheromone, dtype=np.float32)
        self.strong_pheromone = np.zeros((self.h, self.w), dtype=np.float32)  # Successful paths
        self.weak_pheromone = np.zeros((self.h, self.w), dtype=np.float32)  # Exploration
        
        # Ants
        self.ants = []
        
        # Statistics
        self.generation = 0
        self.total_ants = 0
        self.best_path = None
        self.best_length = float('inf')
        self.solution_found_at = None
        
        self.init_ants()
    
    def init_ants(self):
        """Initialize new generation of ants at start position."""
        self.ants = []
        for _ in range(self.cfg.n_ants):
            self.ants.append({
                'pos': self.start,
                'path': [self.start],
                'visited': {self.start},  # Prevent loops
                'active': True,
                'found_goal': False,
                'hit_dead_end': False,
            })
        self.generation += 1
        self.total_ants += self.cfg.n_ants
    
    def choose_next(self, ant: Dict) -> Optional[Tuple[int, int]]:
        """
        ACO probabilistic movement: P ∝ τ^α × η^β
        
        Args:
            ant: Ant dictionary
        
        Returns:
            Next position to move to, or None if stuck
        """
        current = ant['pos']
        neighbors = get_neighbors(self.grid, current)
        
        if not neighbors:
            return None
        
        # Filter out visited cells (prevent loops)
        unvisited = [n for n in neighbors if n not in ant['visited']]
        
        if not unvisited:
            return None  # Stuck in loop
        
        # Calculate probabilities
        probabilities = []
        for neighbor in unvisited:
            # Pheromone component
            tau = self.pheromone[neighbor[0], neighbor[1]]
            
            # Heuristic component: closer to goal is better
            dist = manhattan(neighbor, self.goal)
            eta = 1.0 / (dist + 1)
            
            # ACO formula: τ^α × η^β
            prob = (tau ** self.cfg.alpha) * (eta ** self.cfg.beta)
            probabilities.append(prob)
        
        # Normalize
        total = sum(probabilities)
        if total == 0:
            return random.choice(unvisited)
        
        probabilities = [p / total for p in probabilities]
        
        # Roulette wheel selection
        return random.choices(unvisited, weights=probabilities, k=1)[0]
    
    def move_ants(self):
        """Move all active ants one step."""
        for ant in self.ants:
            if not ant['active']:
                continue
            
            # Check if reached goal
            if ant['pos'] == self.goal:
                ant['active'] = False
                ant['found_goal'] = True
                continue
            
            # Choose next position
            next_pos = self.choose_next(ant)
            
            if next_pos is None:
                # Dead end
                ant['active'] = False
                ant['hit_dead_end'] = True
            else:
                # Move ant
                ant['pos'] = next_pos
                ant['path'].append(next_pos)
                ant['visited'].add(next_pos)
                
                # Safety: max path length
                if len(ant['path']) > self.h * self.w:
                    ant['active'] = False
    
    def penalize_dead_ends(self):
        """
        Strongly reduce pheromones on paths that led to dead ends.
        This is the anti-pheromone effect.
        """
        for ant in self.ants:
            if ant.get('hit_dead_end', False):
                # Penalize last 10 cells in path (the dead-end area)
                dead_end_cells = ant['path'][-10:] if len(ant['path']) > 10 else ant['path']
                
                for cell in dead_end_cells:
                    # STRONG reduction (anti-pheromone effect)
                    self.pheromone[cell[0], cell[1]] *= self.cfg.penalty_factor
                    self.weak_pheromone[cell[0], cell[1]] *= self.cfg.penalty_factor * 0.5
                    
                    # Keep minimum level
                    self.pheromone[cell[0], cell[1]] = max(
                        self.pheromone[cell[0], cell[1]], 
                        self.cfg.initial_pheromone * 0.001
                    )
    
    def deposit_pheromones(self):
        """
        Deposit pheromones based on ant performance.
        - STRONG pheromones (white/hot): successful paths
        - WEAK pheromones (red): exploration
        """
        for ant in self.ants:
            path_length = len(ant['path'])
            
            if ant['found_goal']:
                # STRONG pheromone: successful path
                delta = self.cfg.Q / path_length  # Shorter paths = stronger pheromone
                
                for cell in ant['path']:
                    self.pheromone[cell[0], cell[1]] += delta
                    self.strong_pheromone[cell[0], cell[1]] += delta
                
                # Update best path
                if path_length < self.best_length:
                    self.best_length = path_length
                    self.best_path = ant['path'].copy()
                    if self.solution_found_at is None:
                        self.solution_found_at = self.generation
            
            else:
                # WEAK pheromone: exploration (including dead ends)
                delta = self.cfg.Q * 0.05 / path_length
                
                for cell in ant['path']:
                    self.pheromone[cell[0], cell[1]] += delta
                    self.weak_pheromone[cell[0], cell[1]] += delta
    
    def evaporate_pheromones(self):
        """
        Evaporate pheromones: τ = (1 - ρ) × τ
        Prevents unlimited pheromone accumulation.
        """
        self.pheromone *= (1.0 - self.cfg.rho)
        self.strong_pheromone *= (1.0 - self.cfg.rho)
        self.weak_pheromone *= (1.0 - self.cfg.rho)
        
        # Minimum pheromone level
        self.pheromone = np.maximum(self.pheromone, self.cfg.initial_pheromone * 0.01)
    
    def step(self) -> Dict:
        """
        Execute one iteration of the ACO algorithm.
        
        Returns:
            Dictionary with iteration statistics
        """
        # Move ants
        self.move_ants()
        
        # Check if all ants finished
        all_done = all(not a['active'] for a in self.ants)
        
        if all_done:
            # Count successful ants
            successful = sum(1 for a in self.ants if a['found_goal'])
            
            # Penalize dead-end paths
            self.penalize_dead_ends()
            
            # Deposit pheromones
            self.deposit_pheromones()
            
            # Evaporate pheromones
            self.evaporate_pheromones()
            
            # Check termination
            should_stop = (self.cfg.stop_on_solution and 
                          successful >= self.cfg.required_solutions and
                          self.best_path is not None)
            
            if not should_stop and self.generation < self.cfg.max_generations:
                # Start new generation
                self.init_ants()
            
            return {
                'generation': self.generation,
                'total_ants': self.total_ants,
                'successful': successful,
                'best_length': self.best_length if self.best_path else None,
                'should_stop': should_stop,
                'all_done': True,
            }
        else:
            # Still moving
            active = sum(1 for a in self.ants if a['active'])
            return {
                'generation': self.generation,
                'total_ants': self.total_ants,
                'active': active,
                'best_length': self.best_length if self.best_path else None,
                'should_stop': False,
                'all_done': False,
            }
