"""
Live Visualization Module for ACO Maze Solver

This module provides real-time animated visualization of the ACO algorithm in action,
showing pheromone trail emergence, ant movement, and convergence to optimal paths.

Visualization Color Scheme:
- GRAY: Maze walls and paths
- RED (weak pheromones): Exploration trails, including dead-end paths
- WHITE/YELLOW (strong pheromones): Successful paths that reached the goal
- ORANGE dots: Active ants currently exploring
- CYAN line: Current best path found
- LIME circle: Start position
- RED star: Goal position

The dual-pheromone visualization clearly shows the difference between exploration
(where ants have been) and exploitation (where successful solutions exist).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict
from config import ACOConfig
from aco_solver import ACOSolver


def visualize(grid: np.ndarray, solver: ACOSolver, cfg: ACOConfig) -> List[Dict]:
    """
    Animate ACO solving the maze with live pheromone visualization.
    
    Visualization features:
    - RED pheromones: Weak exploration trails
    - WHITE/HOT pheromones: Strong successful paths
    - Orange dots: Active ants
    - Cyan line: Best path found
    
    Args:
        grid: Binary maze grid
        solver: ACO solver instance
        cfg: Configuration
    
    Returns:
        List of statistics per iteration
    """
    stats = []
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off()
    fig.patch.set_facecolor('#111111')
    
    # Maze
    maze_display = grid * 0.5
    maze_img = ax.imshow(maze_display, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    
    # WEAK pheromone overlay (RED - exploration/dead ends)
    weak_display = solver.weak_pheromone.copy()
    weak_img = ax.imshow(weak_display, cmap='Reds', alpha=0.4, 
                         interpolation='bilinear', vmin=0, vmax=cfg.Q * 0.1)
    
    # STRONG pheromone overlay (WHITE/YELLOW - successful paths)
    strong_display = solver.strong_pheromone.copy()
    strong_img = ax.imshow(strong_display, cmap='hot', alpha=0.6, 
                           interpolation='bilinear', vmin=0, vmax=cfg.Q * 5)
    
    # Best path
    best_line, = ax.plot([], [], color='cyan', linewidth=2.5, alpha=0.9, zorder=5)
    
    # Ants
    ants_scatter = ax.scatter([], [], c='#FF4500', s=80, alpha=0.95, 
                             edgecolors='yellow', linewidths=1.5, marker='o', zorder=10)
    
    # Start and Goal markers
    ax.scatter([solver.start[1]], [solver.start[0]], c='lime', marker='o', 
              s=150, edgecolors='white', linewidths=2, zorder=10)
    ax.text(solver.start[1], solver.start[0]-1.5, 'START', color='lime', 
           fontweight='bold', ha='center', va='bottom', fontsize=10)
    
    ax.scatter([solver.goal[1]], [solver.goal[0]], c='red', marker='*', 
              s=250, edgecolors='white', linewidths=2, zorder=10)
    ax.text(solver.goal[1], solver.goal[0]+1.5, 'GOAL', color='red', 
           fontweight='bold', ha='center', va='top', fontsize=10)
    
    title = ax.text(0.5, 1.02, '', transform=ax.transAxes, ha='center', 
                   color='white', fontsize=11, fontweight='bold')
    
    iteration = [0]
    
    def update(frame):
        info = solver.step()
        stats.append(info)
        iteration[0] += 1
        
        # Update pheromone displays
        # WEAK pheromones (red - exploration)
        weak_display = solver.weak_pheromone.copy()
        weak_display[grid == 1] = 0  # Mask walls
        weak_img.set_data(weak_display)
        
        # STRONG pheromones (hot/white - successful paths)
        strong_display = solver.strong_pheromone.copy()
        strong_display[grid == 1] = 0  # Mask walls
        strong_img.set_data(strong_display)
        
        # Update ants
        active_ants = [a for a in solver.ants if a['active']]
        if active_ants:
            positions = [a['pos'] for a in active_ants]
            ys = [p[0] for p in positions]
            xs = [p[1] for p in positions]
            ants_scatter.set_offsets(np.c_[xs, ys])
        else:
            ants_scatter.set_offsets(np.empty((0, 2)))
        
        # Update best path
        if solver.best_path:
            ys = [p[0] for p in solver.best_path]
            xs = [p[1] for p in solver.best_path]
            best_line.set_data(xs, ys)
        
        # Update title
        if info.get('all_done'):
            title.set_text(
                f"Gen {info['generation']} | Ants: {info['total_ants']} | "
                f"Success: {info.get('successful', 0)}/{cfg.required_solutions} | "
                f"Best: {info['best_length']}"
            )
        else:
            title.set_text(
                f"Gen {info['generation']} | Ants: {info['total_ants']} | "
                f"Active: {info.get('active', 0)} | Best: {info['best_length']}"
            )
        
        # Stop if solution found
        if info['should_stop']:
            ani.event_source.stop()
            print(f"\n✓ Solution found!")
            print(f"  Generation: {solver.solution_found_at}")
            print(f"  Total ants: {solver.total_ants}")
            print(f"  Best path length: {solver.best_length}")
        
        return maze_img, weak_img, strong_img, best_line, ants_scatter, title
    
    ani = animation.FuncAnimation(fig, update, frames=cfg.max_generations * 1000,
                                 interval=cfg.animation_interval, blit=True, repeat=False)
    
    plt.show()
    plt.close(fig)
    
    return stats


def save_hero_image(grid: np.ndarray, solver: ACOSolver, path: str = 'outputs/hero.png'):
    """
    Save high-resolution image of the solution.
    
    Args:
        grid: Binary maze grid
        solver: ACO solver with solution
        path: Output file path
    """
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    ax.set_axis_off()
    fig.patch.set_facecolor('#111111')
    
    # Maze
    maze_display = grid * 0.5
    ax.imshow(maze_display, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    
    # Best path
    if solver.best_path:
        ys = [p[0] for p in solver.best_path]
        xs = [p[1] for p in solver.best_path]
        ax.plot(xs, ys, color='cyan', linewidth=3, alpha=0.9)
    
    # Markers
    ax.scatter([solver.start[1]], [solver.start[0]], c='lime', marker='o', 
              s=200, edgecolors='white', linewidths=2, zorder=10)
    ax.scatter([solver.goal[1]], [solver.goal[0]], c='red', marker='*', 
              s=300, edgecolors='white', linewidths=2, zorder=10)
    
    fig.savefig(path, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)
