"""Animate a differential-drive micromouse navigating a grid maze."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


MAZE = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=int,
)
CELL_SIZE = 0.1
START = (1, 1)
GOAL = (8, 8)
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W


def plan_path(
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Plan a non-revisiting path using relative right-hand priorities."""
    path = [start]
    visited = {start}
    position = start
    heading = 1  # East

    for _ in range(maze.size):
        if position == goal:
            return path

        for offset in (1, 0, -1, 2):  # right, forward, left, backward
            candidate_heading = (heading + offset) % 4
            row_step, col_step = DIRECTIONS[candidate_heading]
            candidate = (position[0] + row_step, position[1] + col_step)

            inside = 0 <= candidate[0] < maze.shape[0] and 0 <= candidate[1] < maze.shape[1]
            if inside and maze[candidate] == 0 and candidate not in visited:
                position = candidate
                heading = candidate_heading
                visited.add(candidate)
                path.append(candidate)
                break
        else:
            raise RuntimeError("The planner reached a dead end before finding the goal.")

    raise RuntimeError("The planner exceeded its maximum number of steps.")


def to_world(cell: tuple[int, int], maze: np.ndarray) -> tuple[float, float]:
    """Convert a grid cell to the centre point of that cell in metres."""
    row, col = cell
    return col * CELL_SIZE + CELL_SIZE / 2, (maze.shape[0] - 1 - row) * CELL_SIZE + CELL_SIZE / 2


def interpolate_path(
    grid_path: list[tuple[int, int]],
    maze: np.ndarray,
    frames_per_segment: int,
) -> list[tuple[float, float, float]]:
    """Interpolate grid waypoints and calculate the robot heading."""
    smooth_path: list[tuple[float, float, float]] = []
    for first, second in zip(grid_path, grid_path[1:]):
        x0, y0 = to_world(first, maze)
        x1, y1 = to_world(second, maze)
        heading = np.arctan2(y1 - y0, x1 - x0)
        for fraction in np.linspace(0, 1, frames_per_segment, endpoint=False):
            smooth_path.append(
                (x0 + (x1 - x0) * fraction, y0 + (y1 - y0) * fraction, heading)
            )

    final_x, final_y = to_world(grid_path[-1], maze)
    smooth_path.append((final_x, final_y, smooth_path[-1][2]))
    return smooth_path


def draw_maze(ax: plt.Axes, maze: np.ndarray) -> None:
    for row, col in np.argwhere(maze == 1):
        ax.add_patch(
            patches.Rectangle(
                (col * CELL_SIZE, (maze.shape[0] - 1 - row) * CELL_SIZE),
                CELL_SIZE,
                CELL_SIZE,
                color="#111827",
            )
        )


def draw_robot(ax: plt.Axes, x: float, y: float, heading: float) -> None:
    parts = [
        patches.Rectangle((-0.03, -0.02), 0.06, 0.04, ec="#ef4444", fc="#9ca3af"),
        patches.Rectangle((-0.04, -0.02), 0.01, 0.04, fc="#111827"),
        patches.Rectangle((0.03, -0.02), 0.01, 0.04, fc="#111827"),
    ]
    transform = (
        plt.matplotlib.transforms.Affine2D().rotate(heading)
        + plt.matplotlib.transforms.Affine2D().translate(x, y)
        + ax.transData
    )
    for part in parts:
        part.set_transform(transform)
        ax.add_patch(part)


def create_animation(output: Path, frames_per_segment: int = 10, fps: int = 20) -> None:
    grid_path = plan_path(MAZE, START, GOAL)
    positions = interpolate_path(grid_path, MAZE, frames_per_segment)
    goal_x, goal_y = to_world(GOAL, MAZE)

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame: int) -> None:
        ax.clear()
        draw_maze(ax, MAZE)
        travelled = positions[: frame + 1]
        ax.plot([p[0] for p in travelled], [p[1] for p in travelled], color="#2563eb", lw=2)
        x, y, heading = positions[frame]
        draw_robot(ax, x, y, heading)
        ax.plot(goal_x, goal_y, "go", ms=10, label="Goal")
        ax.set(xlim=(0, MAZE.shape[1] * CELL_SIZE), ylim=(0, MAZE.shape[0] * CELL_SIZE))
        ax.set_aspect("equal")
        ax.set_title("Micromouse Maze Navigation")
        ax.legend(loc="upper right")

    movie = animation.FuncAnimation(fig, update, frames=len(positions), interval=1000 / fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved animation to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/maze-navigation.gif"))
    parser.add_argument("--frames-per-segment", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_animation(args.output, args.frames_per_segment, args.fps)
