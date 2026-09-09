# -*- coding: utf-8 -*-
"""Env utils : cars, etc

Window-bounds computation, the rigorous mid-timestep collision math, and the
result dataclasses (`StepResult`, `TurnResult`, ...) used by `MovingWorldEnv`.

We work in (row, column) coordinates. Row and column both start at 0, so (0, 0)
is the top-left cell and, on a GRID_SIZE x GRID_SIZE grid,
(GRID_SIZE-1, GRID_SIZE-1) is the bottom-right cell. Every position and every
delta in this file is (row, column) / (drow, dcol); there is no cartesian x/y.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict


@dataclass
class WindowBounds:
    row_min: int
    row_max: int
    col_min: int
    col_max: int


def compute_window_bounds(agent_pos: Tuple[int, int], half_size: int, size: int) -> WindowBounds:
    """
    The one place window bounds are computed from an agent position -- used
    by both MovingWorldEnv.cars_in_window and (via that same call) the
    rendering cell, so bounds are never independently recomputed. Not
    clamped to the grid and not torus-aware: the agent's own frame of
    reference does not wrap, only cars do.
    """
    ar, ac = agent_pos
    return WindowBounds(
        row_min=ar - half_size,
        row_max=ar + half_size,
        col_min=ac - half_size,
        col_max=ac + half_size,
    )


def _t_in_half_open_unit_interval(numerator: int, denominator: int) -> bool:
    """True iff 0 <= numerator/denominator < 1, using exact integer arithmetic
    (no floats, no division) -- denominator must be nonzero."""
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    return 0 <= numerator < denominator


def _swept_positions_collide(
    a_old: Tuple[int, int], a_new: Tuple[int, int],
    b_old: Tuple[int, int], b_new_unwrapped: Tuple[int, int],
) -> bool:
    """
    Rigorous mid-timestep collision check: treats each actor's position over
    the timestep as linear in t -- a(t) = a_old + t*(a_new-a_old), b(t) =
    b_old + t*(b_new_unwrapped-b_old) -- and returns True iff a(t) == b(t)
    for some t in [0, 1). This is deliberately NOT a special-cased "did they
    swap cells" pattern match: it solves the actual line-intersection
    problem with exact integer cross-multiplication, so it's correct for
    any pair of constant-velocity actors over the step (unit speed, faster
    speeds, one of them holding still, etc.), not just the specific swap
    scenario that motivated adding it.

    t=1 is deliberately EXCLUDED here -- the endpoint case ("do they end up
    on the same cell") is handled separately, by comparing actual (wrapped)
    final positions, since b_new_unwrapped is the PRE-modulo continuous
    endpoint (needed for correct interior-of-interval math), not the car's
    real stored position after a same-timestep wraparound -- comparing
    endpoints via this function would be wrong whenever b wraps this step.
    """
    r0 = a_old[0] - b_old[0]
    c0 = a_old[1] - b_old[1]
    vr = (a_new[0] - a_old[0]) - (b_new_unwrapped[0] - b_old[0])
    vc = (a_new[1] - a_old[1]) - (b_new_unwrapped[1] - b_old[1])

    if vr == 0 and vc == 0:
        return r0 == 0 and c0 == 0  # degenerate: relative position constant
    if vr == 0:
        return r0 == 0 and _t_in_half_open_unit_interval(-c0, vc)
    if vc == 0:
        return c0 == 0 and _t_in_half_open_unit_interval(-r0, vr)
    # both nonzero: a single t must satisfy both equations simultaneously --
    # cross-multiply to check consistency without dividing
    if r0 * vc != c0 * vr:
        return False
    return _t_in_half_open_unit_interval(-r0, vr)


@dataclass
class StepResult:
    agent_pos: Tuple[int, int]
    requested_action: Tuple[int, int]      # the actual queued unit action for this timestep
                                            # (may be (0,0) if it's an explicit hold)
    moved: bool                            # whether agent_pos actually changed -- False for an
                                            # explicit (0,0) hold AND for an attempted move that
                                            # failed at the grid edge; requested_action still shows
                                            # what was tried
    car_positions: List[Tuple[int, int]]   # post-step snapshot, for logging fidelity
    collision: bool
    collided_with_car_id: Optional[int]
    goal_reached: bool
    terminal: bool
    outcome: Optional[str]                 # "goal" | "collision" | None
    timestep_index: int                    # env.timestep_count after this step


@dataclass
class TurnResult:
    requested_action_tuples: List[Tuple[int, int, int]]
    num_action_entries: int                # len(requested_action_tuples) -- number of RUNS,
                                            # distinct from the number of timesteps
    expanded_actions: List[Tuple[int, int]]  # flattened per-timestep unit actions; length ==
                                              # fixed_turn_length exactly, guaranteed by validation
    step_results: List[StepResult]
    timesteps_executed: int                # len(step_results), <= fixed_turn_length (early stop)
    final_position: Tuple[int, int]
    terminal: bool
    outcome: Optional[str]                 # "goal" | "collision" | None ("failed" is env-level only)
    reason: str
    turn_index: int                        # env.turn_count after this turn (1-indexed)
    total_timesteps_elapsed: int


@dataclass
class GoalOffset:
    drow: int           # goal_row - agent_row  (positive = goal is further down)
    dcol: int           # goal_col - agent_col  (positive = goal is further right)
    distance: float
    direction: str      # "down-right", "up", "here", ... (row/column words)


class CarWindowEntry(TypedDict):
    position: Tuple[int, int]            # (row, column)
    offset_from_agent: Tuple[int, int]   # (row - agent_row, col - agent_col) = (drow, dcol)
    # No velocity/speed/color: every car is identical (CAR_DIRECTION/CAR_SPEED),
    # so repeating those per car here would be pure noise -- the movement rule is
    # stated once, in the agent prompt.


class InitialStateJSON(TypedDict):
    size: int
    start_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    agent_pos: Tuple[int, int]
    cars: List[dict]
    fixed_turn_length: int
    seed: Optional[int]
    # no call-budget field here -- MAX_TURNS is a simulation-loop concept, not
    # something the environment itself knows about.


def _direction_word(drow: int, dcol: int) -> str:
    """(drow, dcol) -> a plain direction string in row/column terms;
    (0, 0) -> 'here'. Row increases downward, column increases rightward."""
    if drow == 0 and dcol == 0:
        return "here"
    vertical = "down" if drow > 0 else ("up" if drow < 0 else "")
    horizontal = "right" if dcol > 0 else ("left" if dcol < 0 else "")
    return "-".join(w for w in (vertical, horizontal) if w) or "here"
