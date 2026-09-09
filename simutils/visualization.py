# -*- coding: utf-8 -*-
"""Plotting / walk helpers.

- ``plot_centers``           : trajectory + per-tool PNG rendering.
- ``run_env_walk`` /
  ``plot_single_trajectory`` : replay a move list / draw one grid state.

Everything the model sees is in (row, column) coordinates (both 0-based; (0, 0)
top-left; row increases downward, column rightward). matplotlib / image code
needs an (x, y) frame, so **for plotting only** these helpers use
``x = column`` and ``y = row`` and invert the y-axis (or place the image origin
at the top) so row 0 stays at the top. That swap is local to the drawing code;
callers always pass and receive (row, column).

``plot_centers`` takes the tools directly as a dict ``{name: (n_steps, 2) array
of (drow, dcol)}`` and writes one ``filled_cluster_<name>.png`` per tool.
"""

import random
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from .config import GRID_SIZE, GOAL_POS, GEODESIC_LEN, VECTOR_TO_DELTA
from .env_utils import StepResult


def plot_centers(tools):
    """tools: dict {name -> (n_steps, 2) array of (drow, dcol) moves}.
    Plots every tool's trajectory and saves one filled_cluster_<name>.png each."""
    # cumsum of the (drow, dcol) moves -> (row, col) offset from each tool's start.
    plt.figure('all tools', figsize=(3, 3))
    for name, actions in tools.items():
        traj = np.cumsum(np.asarray(actions), axis=0)
        row_cum = traj[:, 0]   # cumulative drow
        col_cum = traj[:, 1]   # cumulative dcol

        # plotting only: column on the x-axis, row on the y-axis (y-axis is
        # inverted below so row 0 stays at the top, matching the grid).
        plt.plot([0] + list(col_cum + np.random.randn(col_cum.shape[0]) * 0.05),
                 [0] + list(row_cum + np.random.randn(row_cum.shape[0]) * 0.05),
                 '-o', label=name)

    plt.axis('equal')
    plt.grid(True)
    plt.legend(fontsize=6)
    plt.xlabel('column')
    plt.ylabel('row')
    plt.gca().invert_yaxis()   # plotting only: row increases downward
    plt.title('Tool trajectories')
    plt.tight_layout()
    plt.show()

    imgs = []
    for name, actions in tools.items():
      traj = np.cumsum(np.asarray(actions), axis=0).reshape(len(actions), 2)
      cell_px = 10
      border_px = 2
      n = 2*GEODESIC_LEN + 1

      img_px = n*(cell_px + border_px) + border_px

      # black background = borders
      img = np.zeros((img_px, img_px, 3), dtype=np.uint8)

      # white squares
      for ii in range(n):
          for jj in range(n):
              x0 = int(border_px) + ii*(int(cell_px) + int(border_px))
              y0 = int(border_px) + jj*(int(cell_px) + int(border_px))

              img[int(y0):int(y0+cell_px), int(x0):int(x0+cell_px)] = [255,255,255]
      positions = [(0, 0)] + [tuple(p) for p in traj]   # each p = (row_offset, col_offset)

      for row_off, col_off in positions:
          # plotting only: image column <- column offset, image row <- row
          # offset, both centred; no y-flip, so the picture reads like the grid
          # (row grows downward).
          img_col = int(col_off + GEODESIC_LEN)
          img_row = int(row_off + GEODESIC_LEN)

          x0 = int(border_px) + img_col*(int(cell_px) + int(border_px))
          y0 = int(border_px) + img_row*(int(cell_px) + int(border_px))

          img[int(y0):int(y0+cell_px), int(x0):int(x0+cell_px)] = [0,0,255]
      Image.fromarray(img).save(f'filled_cluster_{name}.png')
      imgs.append(img)
    row = np.concatenate(imgs, axis=1)  # side by side
    plt.figure(figsize=(1*len(tools),1))
    plt.imshow(row)
    plt.axis('off')
    plt.show()


def run_env_walk(env: "MovingWorldEnv", num_steps: int, random_walk: bool = True, action_list: Optional[List[Tuple[int, int]]] = None) -> List["StepResult"]:
    """
    If random_walk=True runs a random walk in the given environment for a
    specified number of steps, returning a list of StepResult objects.

    If not use the actions in the action_list.
    """
    results: List["StepResult"] = []
    possible_actions = list(VECTOR_TO_DELTA.keys())

    # Capture initial state before any steps
    initial_car_positions = [car.position for car in env.cars]
    results.append(StepResult(
        agent_pos=env.agent_pos,
        requested_action=(0,0), # Placeholder for initial state
        moved=False,
        car_positions=initial_car_positions,
        collision=False,
        collided_with_car_id=None,
        goal_reached=False,
        terminal=False,
        outcome=None,
        timestep_index=0
    ))

    for i in range(num_steps):
        if random_walk:
            action = random.choice(possible_actions)
        else:
            if action_list is None or i >= len(action_list):
                print("Warning: action_list exhausted or not provided for non-random walk. Stopping.")
                break
            action = tuple(action_list[i])

        step_result = env.advance_one_timestep(action)
        results.append(step_result)
        if step_result.terminal:
            print(f"Random walk terminated early due to: {step_result.outcome}")
            break
    return results

def plot_single_trajectory(
    agent_path: List[Tuple[int, int]],
    final_car_positions: List[Tuple[int, int]],
    grid_size: int,
    title: str = "Agent Walk",
    window_bounds: Optional[object] = None, # WindowBounds object from environment cell
    save_only: Optional[str] = None, # If a string path is provided, saves to file instead of showing
    crop_rows: Optional[Tuple[int, int]] = None, # inclusive (lo, hi) row bounds; None -> full grid
    crop_cols: Optional[Tuple[int, int]] = None, # inclusive (lo, hi) col bounds; None -> full grid
    show_relative_ticks: bool = False, # label each cell's axis position as its (drow, dcol) offset from the agent's current cell, instead of leaving axes unlabeled
):
    """
    Plots a single agent trajectory with specific markers and car positions, optionally showing the observation window.
    If `save_only` is a string (filepath), the plot is saved to that file and not displayed.

    `agent_path` and `final_car_positions` are (row, column). Plotting only: this
    function draws with x = column and y = row, and inverts the y-axis so
    row 0 is at the top -- i.e. the picture matches the (row, column) grid.

    If `crop_rows` and `crop_cols` are given (inclusive cell-index bounds), the
    view is cropped to that rectangle at larger cell size and the figure aspect
    follows the crop. A requested bound may extend past the true grid (e.g. a
    hazard close-up near a wall); it is clamped to `[0, grid_size - 1]` first,
    so the rendered rectangle only ever shows real cells -- it shrinks instead
    of showing nonexistent cells, which would otherwise look identical to
    legitimately-unobserved (grey) ones.

    `show_relative_ticks` is independent of `crop_rows`/`crop_cols`: when True,
    every cell's row/column axis tick is labeled with its (drow, dcol) offset
    from the agent's current cell (`agent_path[-1]`) -- e.g. a row one above the
    agent reads "-1", the agent's own row reads "0". Grid lines are unaffected;
    only whether/how the axes are labeled changes.
    """
    fig_size_inches = 5 # As defined in plt.figure(figsize=(5,5))
    figure_dpi=plt.rcParams['figure.dpi'] #choose figure dpi
    figure_dpi=300 #choose figure dpi

    # View extent: full grid, or the requested rectangular crop.
    if crop_rows is not None and crop_cols is not None:
        lo_r, hi_r = crop_rows
        lo_c, hi_c = crop_cols
        # Clamp to the true grid extent: a requested crop may reach past an
        # edge (e.g. a hazard close-up near a wall), but no image should ever
        # show a cell that does not exist -- those cells are indistinguishable
        # from legitimately-unobserved (grey) ones otherwise, which misleads
        # the LLM into thinking it can still move there. Clamping shrinks the
        # rendered rectangle instead, which is itself a visible cue of
        # closeness to a wall.
        lo_r, hi_r = max(lo_r, 0), min(hi_r, grid_size - 1)
        lo_c, hi_c = max(lo_c, 0), min(hi_c, grid_size - 1)
    else:
        lo_r, lo_c = 0, 0
        hi_r, hi_c = grid_size - 1, grid_size - 1
    n_rows = hi_r - lo_r + 1
    n_cols = hi_c - lo_c + 1
    n_span = max(n_rows, n_cols)

    # Adjust marker size to be slightly smaller than a full cell to fit within borders
    # The 's' parameter in scatter is for area (points^2), so scale factor applies to linear dimension
    #cell_marker_area_points = (fig_size_inches * figure_dpi / grid_size)**2 * 0.22 # Reduced to 22%
    cell_marker_area_points = (fig_size_inches * 72 / n_span) ** 2

    _scale = fig_size_inches / n_span   # inches per cell; figure aspect follows the crop
    fig = plt.figure(figsize=(n_cols * _scale, n_rows * _scale))
    ax = fig.add_axes([0, 0, 1, 1])
    # Set plot limits to the view extent
    plt.xlim(lo_c - 0.5, hi_c + 0.5) # Center grid cells, from cell_center-0.5 to cell_center+0.5
    plt.ylim(lo_r - 0.5, hi_r + 0.5)

    # Invert y-axis to have (0,0) at the top-left
    ax.invert_yaxis()

    if show_relative_ticks:
        agent_row, agent_col = agent_path[-1]
        # Minor ticks at cell boundaries: grid lines only, never labeled.
        ax.set_xticks(np.arange(lo_c - 0.5, hi_c + 1.0, 1), minor=True)
        ax.set_yticks(np.arange(lo_r - 0.5, hi_r + 1.0, 1), minor=True)
        ax.grid(True, which='minor', linestyle='-', color='black', alpha=1.0, linewidth=4.0, zorder=1000)
        ax.tick_params(which='minor', length=0)

        # Major ticks at cell centers: labeled with (drow, dcol) offset from
        # the agent's current cell, e.g. a row just above the agent reads "-1".
        ax.set_xticks(np.arange(lo_c, hi_c + 1, 1))
        ax.set_xticklabels([str(c - agent_col) for c in range(lo_c, hi_c + 1)],
                            fontsize=16, fontweight='bold')
        ax.set_yticks(np.arange(lo_r, hi_r + 1, 1))
        ax.set_yticklabels([str(r - agent_row) for r in range(lo_r, hi_r + 1)],
                            fontsize=16, fontweight='bold')
        ax.tick_params(which='major', length=0)
    else:
        # Set ticks at cell boundaries to create grid lines between cells
        ax.set_xticks(np.arange(lo_c - 0.5, hi_c + 1.0, 1))
        ax.set_yticks(np.arange(lo_r - 0.5, hi_r + 1.0, 1))

        # Turn on the grid with distinct black lines
        ax.grid(True, linestyle='-', color='black', alpha=1.0, linewidth=4.0,zorder=1000)

        # Remove tick labels but keep the grid lines
        ax.set_xticklabels([])
        ax.set_yticklabels([])


    # Plot trajectory (yellow squares) - all but the last position
    if len(agent_path) > 1:
        trajectory_rows = [p[0] for p in agent_path[:-1]]
        trajectory_cols = [p[1] for p in agent_path[:-1]]
        ax.scatter(trajectory_cols, trajectory_rows, color='yellow', marker='s', s=cell_marker_area_points, label='Agent Trajectory', zorder=2)

    # Plot current walker position (blue square) - the last position
    current_agent_pos = agent_path[-1]
    ax.scatter(current_agent_pos[1], current_agent_pos[0], color='blue',
               marker='s', s=cell_marker_area_points, label='Agent Current', zorder=3)

    # Plot cars (red squares) - at their final positions
    car_rows = [p[0] for p in final_car_positions]
    car_cols = [p[1] for p in final_car_positions]
    ax.scatter(car_cols, car_rows, color='red', marker='s',
               s=cell_marker_area_points, label='Cars', zorder=1)

    # Gray out cells that are outside the observation window OR outside the grid
    # (the latter only matters for a zoom crop near an edge -- it makes the grid
    # boundary visible).
    if window_bounds: # window_bounds will be an instance of WindowBounds
        for r in range(lo_r, hi_r + 1):
            for c in range(lo_c, hi_c + 1):
                in_grid = (0 <= r < grid_size) and (0 <= c < grid_size)
                in_window = (
                    window_bounds.row_min <= r <= window_bounds.row_max and
                    window_bounds.col_min <= c <= window_bounds.col_max
                )
                is_goal = (r, c) == GOAL_POS # Use the global GOAL_POS

                if (not in_grid or not in_window) and not is_goal:
                    # Draw a gray rectangle
                    rect = Rectangle(
                        (c - 0.5, r - 0.5), # Lower-left corner for cell (r,c)
                        1, 1, # width, height
                        facecolor='gray',
                        alpha=1.0, # Completely opaque gray
                        zorder=99 # only this is seen, all other elements are behind
                    )
                    ax.add_patch(rect)

    # Plot goal position (green square)
    # Assuming GOAL_POS is available from the constants cell
    ax.scatter(GOAL_POS[1], GOAL_POS[0], color='green', marker='s',
               s=cell_marker_area_points, label='Goal', zorder=999)

    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
#    plt.legend()

    if save_only:
        plt.savefig(save_only, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()
