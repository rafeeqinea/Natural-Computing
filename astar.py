"""
A* Pathfinding Algorithm - Optimal Baseline

This module implements the A* algorithm, a classic graph search algorithm that guarantees
finding the shortest path. It's used as a baseline to evaluate ACO performance by
comparing path lengths and demonstrating the trade-off between optimality guarantees
and adaptive exploration.

A* maintains a priority queue of nodes to explore, using f(n) = g(n) + h(n):
- g(n): actual cost from start to current node
- h(n): estimated cost from current node to goal (Manhattan distance)
"""

from typing import Optional, List, Tuple, Dict
import math
from maze import get_neighbors, manhattan


def astar(grid, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding algorithm - finds optimal path.
    Used as baseline to compare against ACO performance.
    
    Args:
        grid: Binary maze grid
        start: Start position
        goal: Goal position
    
    Returns:
        List of positions forming the path, or None if no path exists
    """
    open_set = {start}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score = {start: 0}
    f_score = {start: manhattan(start, goal)}
    
    while open_set:
        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        open_set.remove(current)
        
        for neighbor in get_neighbors(grid, current):
            tentative_g = g_score[current] + 1
            
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + manhattan(neighbor, goal)
                open_set.add(neighbor)
    
    return None
