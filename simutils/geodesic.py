# -*- coding: utf-8 -*-
"""Closed-form car positions + the geodesic search.

``find_geodesic`` searches the time-expanded state graph
``(agent_pos, t mod size)`` for the path from start to goal that is minimal in
the lexicographic key ``(number of timesteps, number of non-wait moves)`` --
i.e. the fastest path, and among the equally-fast ones the one that moves the
least. The second key removes a degeneracy of plain BFS: when the optimal path
must pause, plain BFS would spell the pause as an up-then-down bounce (or any
cancelling wiggle) purely because of action-enumeration order; minimising moves
spells it as waits, which is what an optimal agent actually does.

We work in (row, column) coordinates (both 0-based; (0, 0) top-left). Positions
are (row, column); each returned move is a (drow, dcol) delta:
  up = (-1, 0)   down = (1, 0)   left = (0, -1)   right = (0, 1)   wait = (0, 0)
"""

from typing import List, Optional, Tuple

from .config import CAR_DIRECTION, CAR_SPEED, VECTOR_TO_DELTA
from .env_utils import _swept_positions_collide


def car_position_at_time(
    spawn_position: Tuple[int, int],
    t: int,
    size: int,
    direction: Tuple[int, int] = CAR_DIRECTION,
    speed: int = CAR_SPEED,
) -> Tuple[int, int]:
    """
    Closed-form car position at timestep t -- every car moves identically
    (direction/speed fixed, wraparound), so this needs no incremental
    simulation: row(t) = (spawn_row + dr*speed*t) % size, same for col. This
    is also exactly periodic in t with period size (adding size to t doesn't
    change the result mod size), which is what makes find_geodesic's
    state space finite.
    """
    drow, dcol = direction
    r, c = spawn_position
    return ((r + drow * speed * t) % size, (c + dcol * speed * t) % size)


def find_geodesic(env: "MovingWorldEnv") -> Optional[List[Tuple[int, int]]]:
    """
    Search the time-expanded state graph (agent_pos, t mod env.size) for the
    ordered list of unit (drow, dcol) moves from env.start_pos to env.goal_pos
    that is minimal in the lexicographic key (timesteps, non-wait moves), or
    None if the goal is unreachable.

    Why (agent_pos, t mod size) is a valid, complete state for this search:
    car configuration at any t depends only on t (not on any agent
    decision), and every car's position is exactly periodic with period
    env.size (see car_position_at_time) -- so the SAME car configuration
    recurs every env.size timesteps regardless of which timestep t actually
    is. That makes the reachable state space finite (size^3 states: agent
    row x agent col x t mod size).

    Implementation: this is Dijkstra on the (timesteps, moves) key, done as a
    LAYERED BFS -- one timestep per layer (timesteps is the primary key and
    every edge adds exactly 1), and within a layer each newly reached state
    keeps the predecessor that gives it the fewest moves. That is O(edges),
    same cost as plain BFS. Timesteps still increase strictly layer by layer,
    so the first layer that contains a state at env.goal_pos gives the fewest
    possible timesteps; among the goal states in that layer the fewest-moves
    one is returned -- the true geodesic, not an approximation, no depth cap.

    Edge validity reuses _swept_positions_collide (from the environment
    cell) directly against car positions computed via car_position_at_time
    -- no need to clone/step a live env per neighbor, which would be far
    slower than this closed-form check.
    """
    size = env.size
    start = env.start_pos
    goal = env.goal_pos
    car_spawns = [car.position for car in env.cars]
    car_drow, car_dcol = CAR_DIRECTION

    if start == goal:
        return []

    def cars_at(t: int) -> List[Tuple[int, int]]:
        return [car_position_at_time(pos, t, size) for pos in car_spawns]

    def edge_collides(agent_old, agent_new, t_mod) -> bool:
        """
        _swept_positions_collide's own docstring requires its 4th argument
        to be the PRE-modulo continuous endpoint, not a car's real (wrapped)
        stored position -- "comparing endpoints via this function would be
        wrong whenever b wraps this step." So the swept check below must use
        a one-step-unwrapped extension computed directly from each car's
        already-correctly-wrapped OLD position (cars_at(t_mod)), never
        car_position_at_time(..., t_mod + 1, ...) again, which would
        silently re-wrap it -- confirmed to matter in practice by a
        differential test against real MovingWorldEnv.advance_one_timestep:
        using the wrapped value there produces false-positive collisions
        specifically on the timestep a car wraps around the grid edge (row
        0 -> row size-1), which happens once per car every `size`
        timesteps -- not a rare case.
        """
        cars_old = cars_at(t_mod)
        for old_c in cars_old:
            new_c_unwrapped = (old_c[0] + car_drow * CAR_SPEED, old_c[1] + car_dcol * CAR_SPEED)
            new_c_wrapped = (new_c_unwrapped[0] % size, new_c_unwrapped[1] % size)
            if agent_new == new_c_wrapped:
                return True
            if _swept_positions_collide(agent_old, agent_new, old_c, new_c_unwrapped):
                return True
        return False

    start_state = (start, 0)
    settled = {start_state}          # states already reached at their minimal timestep
    parent = {}                      # state -> (prev_state, action)
    frontier = {start_state: 0}      # states reachable at the current timestep -> moves so far

    while frontier:
        # A goal-position state in this layer means the fewest-timestep path is
        # found; among such states take the one reached with the fewest moves.
        goal_states = [s for s in frontier if s[0] == goal]
        if goal_states:
            cur = min(goal_states, key=lambda s: frontier[s])
            actions = []
            while cur != start_state:
                prev, a = parent[cur]
                actions.append(a)
                cur = prev
            actions.reverse()
            return actions

        next_frontier = {}           # state -> (moves, (prev_state, action))
        for (pos, t_mod), moves in frontier.items():
            for action, (drow, dcol) in VECTOR_TO_DELTA.items():  # identity map
                new_pos = (pos[0] + drow, pos[1] + dcol)
                if not (0 <= new_pos[0] < size and 0 <= new_pos[1] < size):
                    continue  # out-of-bounds attempt -- equivalent to the (0,0)
                              # hold edge already generated separately, so this
                              # is simply skipped rather than duplicated
                if edge_collides(pos, new_pos, t_mod):
                    continue
                new_state = (new_pos, (t_mod + 1) % size)
                if new_state in settled:
                    continue  # already reached at an earlier (strictly better) timestep
                new_moves = moves + (0 if action == (0, 0) else 1)
                best = next_frontier.get(new_state)
                if best is None or new_moves < best[0]:
                    next_frontier[new_state] = (new_moves, ((pos, t_mod), action))

        for new_state, (new_moves, par) in next_frontier.items():
            settled.add(new_state)
            parent[new_state] = par
        frontier = {s: mv for s, (mv, _) in next_frontier.items()}

    return None
