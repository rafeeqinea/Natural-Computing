"""
Maze Generation and Utility Functions

This module provides maze generation using iterative Depth-First Search (DFS) algorithm,
which creates "perfect mazes" - mazes with exactly one path between any two points.
Also includes utility functions for pathfinding operations.

Functions:
- generate_maze(): Creates random perfect maze using iterative DFS
- find_start_goal(): Locates start (top-left) and goal (bottom-right) positions
- get_neighbors(): Returns valid neighboring cells for movement
- manhattan(): Calculates Manhattan distance heuristic
"""

import numpy as np
import random
from typing import Tuple, List


def generate_maze(rows: int, cols: int, seed: int) -> np.ndarray:
    """
    Generate perfect maze using iterative DFS (Depth-First Search).
    
    Args:
        rows: Number of maze rows
        cols: Number of maze columns
        seed: Random seed for reproducibility
    
    Returns:
        Binary grid: 0 = path, 1 = wall
    """
    rng = random.Random(seed)
    h, w = 2 * rows + 1, 2 * cols + 1
    grid = np.ones((h, w), dtype=np.int8)
    
    # Start carving from (1,1)
    start_r, start_c = 1, 1
    grid[start_r, start_c] = 0
    
    stack = [(start_r, start_c)]
    
    while stack:
        r, c = stack[-1]
        
        # Find unvisited neighbors (2 cells away)
        neighbors = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 0 < nr < h and 0 < nc < w and grid[nr, nc] == 1:
                neighbors.append((nr, nc, r + dr // 2, c + dc // 2))
        
        if neighbors:
            nr, nc, wr, wc = rng.choice(neighbors)
            grid[wr, wc] = 0  # Remove wall
            grid[nr, nc] = 0  # Mark as visited
            stack.append((nr, nc))
        else:
            stack.pop()
    
    return grid


def find_start_goal(grid: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Find start (top-left) and goal (bottom-right) positions.
    
    Args:
        grid: Binary maze grid
    
    Returns:
        Tuple of (start_position, goal_position)
    """
    h, w = grid.shape
    
    # Start: top-left open cell
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 0:
                start = (i, j)
                break
        else:
            continue
        break
    
    # Goal: bottom-right open cell
    for i in range(h - 1, -1, -1):
        for j in range(w - 1, -1, -1):
            if grid[i, j] == 0:
                goal = (i, j)
                break
        else:
            continue
        break
    
    return start, goal


def get_neighbors(grid: np.ndarray, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Get valid neighbors (not walls) of a cell.
    
    Args:
        grid: Binary maze grid
        pos: Current position (row, col)
    
    Returns:
        List of valid neighbor positions
    """
    h, w = grid.shape
    r, c = pos
    neighbors = []
    
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
            neighbors.append((nr, nc))
    
    return neighbors


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    Calculate Manhattan distance between two points.
    
    Args:
        a: First position (row, col)
        b: Second position (row, col)
    
    Returns:
        Manhattan distance
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
