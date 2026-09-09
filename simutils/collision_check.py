# -*- coding: utf-8 -*-
"""Collision check against currently-visible cars only.

``collision_with_visible_cars(env, actions)`` answers one yes/no question:
would the agent, executing this fixed action sequence from its current cell,
hit a car that is *right now* inside the standard observation window
(``WINDOW_HALF_SIZE`` -- the same membership test ``MovingWorldEnv.cars_in_window``
uses)?

Cars outside the observation window are ignored -- the caller does not care
about them. Implementation: deep-copy the env, drop every out-of-window car,
replay the sequence on the copy, report whether a collision occurred. The real
env is never touched.
"""

import copy
from typing import Iterable, Sequence

from .config import WINDOW_HALF_SIZE
from .env_utils import compute_window_bounds


def collision_with_visible_cars(env, actions: Iterable[Sequence[int]],
                                half_size: int = WINDOW_HALF_SIZE) -> bool:
    """True iff executing ``actions`` (an iterable of ``(drow, dcol)`` steps)
    from the agent's current cell collides with a car currently inside the
    observation window. Cars outside the window are removed before the replay.
    """
    bounds = compute_window_bounds(env.agent_pos, half_size, env.size)

    env_copy = copy.deepcopy(env)
    env_copy.cars = [
        car for car in env_copy.cars
        if bounds.row_min <= car.position[0] <= bounds.row_max
        and bounds.col_min <= car.position[1] <= bounds.col_max
    ]

    for step in actions:
        result = env_copy.advance_one_timestep((int(step[0]), int(step[1])))
        if result.collision:
            return True
        if result.terminal:          # reached the goal without a visible-car hit
            break
    return False
