# -*- coding: utf-8 -*-
"""``MovingWorldEnv`` — the moving-cars grid environment.

World constants come from ``config``; the collision math and result dataclasses
from ``env_utils``.

We work in (row, column) coordinates. Row and column both start at 0, so (0, 0)
is the top-left cell and (GRID_SIZE-1, GRID_SIZE-1) is the bottom-right cell.
``agent_pos``, ``start_pos``, ``goal_pos`` and every car position are
(row, column); an action is a (drow, dcol) delta added to the agent's
(row, column). No cartesian x/y.
"""

import math
import random
from typing import List, Optional, Tuple

from .config import (
    GRID_SIZE, NUM_CARS, START_POS, GOAL_POS, FIXED_TURN_LENGTH, SEED,
    CAR_DIRECTION, CAR_SPEED, VECTOR_TO_DELTA, WINDOW_HALF_SIZE, Car,
)
from .env_utils import (
    WindowBounds, compute_window_bounds, _swept_positions_collide,
    StepResult, TurnResult, GoalOffset, CarWindowEntry, InitialStateJSON,
    _direction_word,
)


class MovingWorldEnv:
    def __init__(
        self,
        size: int = GRID_SIZE,
        num_cars: int = NUM_CARS,
        start_pos: Tuple[int, int] = START_POS,
        goal_pos: Tuple[int, int] = GOAL_POS,
        fixed_turn_length: int = FIXED_TURN_LENGTH,
        seed: Optional[int] = SEED,
    ) -> None:
        self.size = size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.fixed_turn_length = fixed_turn_length
        self.seed = seed

        # Owns its own RNG instance (not the global `random` module) --
        # deterministic, testable, no hidden global mutable state.
        self.rng = random.Random(seed)

        self.agent_pos = start_pos
        self.turn_count = 0
        self.timestep_count = 0
        self.outcome: Optional[str] = None

        self.cars = self._spawn_cars(num_cars)

    def _spawn_cars(self, num_cars: int) -> List["Car"]:
        """
        Every car's direction (CAR_DIRECTION) and speed (CAR_SPEED) are fixed,
        global constants -- identical for every car -- so only a random free
        starting position (excluding the exact start/goal cells) is spawned
        per car.
        """
        cars = []
        for car_id in range(num_cars):
            while True:
                pos = (self.rng.randrange(self.size), self.rng.randrange(self.size))
                if pos != self.start_pos and pos != self.goal_pos:
                    break
            cars.append(Car(id=car_id, position=pos))
        return cars

    def advance_one_timestep(self, action: Tuple[int, int]) -> StepResult:
        """
        The single lowest-level state-transition primitive. Applies one
        agent action (or hold), advances every car by the fixed
        CAR_DIRECTION/CAR_SPEED via modulo wraparound, and checks
        collision/goal. The only function that ever mutates agent_pos or
        car positions. Takes no visualization-related parameter -- frame
        capture happens at turn boundaries in the simulation loop, not here.

        Collision detection is rigorous, not just a final-position check:
        for each car it's true if EITHER (a) same-cell -- agent and the car
        share a cell after this step (using their real, wrapped positions),
        OR (b) their continuous paths crossed sometime strictly during the
        step (via _swept_positions_collide, parametrizing both as linear
        motion over t in [0,1)) -- this is what catches the agent and a car
        swapping places (each moving into the cell the other is leaving) in
        a single timestep, which never shows up as a shared cell at any
        single snapshot but is still a real collision.
        """
        drow, dcol = VECTOR_TO_DELTA[action]   # identity map: the action IS the (row, col) delta
        agent_old_pos = self.agent_pos
        ar, ac = self.agent_pos
        candidate = (ar + drow, ac + dcol)

        moved = False
        if 0 <= candidate[0] < self.size and 0 <= candidate[1] < self.size:
            self.agent_pos = candidate
            moved = True
        # else: out of bounds -- agent holds in place (no wrap; only cars wrap)

        car_drow, car_dcol = CAR_DIRECTION
        car_old_positions = {}
        car_new_unwrapped = {}
        for car in self.cars:
            car_old_positions[car.id] = car.position
            old_r, old_c = car.position
            new_r_unwrapped = old_r + car_drow * CAR_SPEED
            new_c_unwrapped = old_c + car_dcol * CAR_SPEED
            car_new_unwrapped[car.id] = (new_r_unwrapped, new_c_unwrapped)
            car.position = (new_r_unwrapped % self.size, new_c_unwrapped % self.size)

        self.timestep_count += 1

        collision = False
        collided_with_car_id = None
        for car in self.cars:
            same_cell = car.position == self.agent_pos
            crossed_mid_step = _swept_positions_collide(
                agent_old_pos, self.agent_pos,
                car_old_positions[car.id], car_new_unwrapped[car.id],
            )
            if same_cell or crossed_mid_step:
                collision = True
                collided_with_car_id = car.id
                break

        goal_reached = (not collision) and (self.agent_pos == self.goal_pos)
        terminal = collision or goal_reached
        outcome = "collision" if collision else ("goal" if goal_reached else None)
        if terminal and self.outcome is None:
            self.outcome = outcome

        return StepResult(
            agent_pos=self.agent_pos,
            requested_action=action,
            moved=moved,
            car_positions=[car.position for car in self.cars],
            collision=collision,
            collided_with_car_id=collided_with_car_id,
            goal_reached=goal_reached,
            terminal=terminal,
            outcome=outcome,
            timestep_index=self.timestep_count,
        )

    def execute_turn(self, action_tuples: List[Tuple[int, int, int]]) -> TurnResult:
        """
        The execute_turn function is designed to process a sequence of actions
        that constitute a 'turn' in the environment.

        Assumes action_tuples was already validated (sum(counts) ==
        fixed_turn_length, guaranteed) by the caller -- the assert below is a
        defensive internal-invariant check, not user-facing error handling.
        Expands each (drow, dcol, count) into
        `count` repetitions of (drow, dcol), loops advance_one_timestep over
        the flattened sequence, stops immediately at the first terminal
        StepResult. Holds no simulation state of its own beyond
        looping/counting -- the environment doesn't know about "turns."
        """
        total_counts = sum(count for _, _, count in action_tuples)
        assert total_counts == self.fixed_turn_length, (
            f"execute_turn called with unvalidated action_tuples: counts sum to "
            f"{total_counts}, expected {self.fixed_turn_length}"
        )

        expanded_actions: List[Tuple[int, int]] = []
        for drow, dcol, count in action_tuples:
            expanded_actions.extend([(drow, dcol)] * count)

        step_results: List[StepResult] = []
        for action in expanded_actions:
            result = self.advance_one_timestep(action)
            step_results.append(result)
            if result.terminal:
                break

        self.turn_count += 1

        last = step_results[-1]
        if last.outcome == "collision":
            reason = (
                f"collided with car #{last.collided_with_car_id} at {last.agent_pos} "
                f"on timestep {len(step_results)} of this turn"
            )
        elif last.outcome == "goal":
            reason = f"reached the goal at {last.agent_pos} on timestep {len(step_results)} of this turn"
        else:
            reason = f"completed all {len(step_results)} timesteps without reaching a terminal state"

        return TurnResult(
            requested_action_tuples=action_tuples,
            num_action_entries=len(action_tuples),
            expanded_actions=expanded_actions,
            step_results=step_results,
            timesteps_executed=len(step_results),
            final_position=last.agent_pos,
            terminal=last.terminal,
            outcome=last.outcome,
            reason=reason,
            turn_index=self.turn_count,
            total_timesteps_elapsed=self.timestep_count,
        )

    def is_terminal(self) -> bool:
        return self.outcome is not None

    def mark_failed(self) -> None:
        """Called by run_episode when MAX_TURNS is exhausted before
        the agent reaches the goal or collides."""
        if self.outcome is None:
            self.outcome = "failed"

    def goal_offset(self) -> GoalOffset:
        # (row, column) deltas from the agent to the goal.
        drow = self.goal_pos[0] - self.agent_pos[0]   # positive -> goal is further down
        dcol = self.goal_pos[1] - self.agent_pos[1]   # positive -> goal is further right
        distance = math.hypot(drow, dcol)
        return GoalOffset(drow=drow, dcol=dcol, distance=distance,
                          direction=_direction_word(drow, dcol))

    def cars_in_window(self, half_size: int = WINDOW_HALF_SIZE) -> Tuple[WindowBounds, List[CarWindowEntry]]:
        """ function to use if only a partial view of the env is used by the agent
        """
        bounds = compute_window_bounds(self.agent_pos, half_size, self.size)
        entries: List[CarWindowEntry] = []
        for car in self.cars:
            r, c = car.position
            if bounds.row_min <= r <= bounds.row_max and bounds.col_min <= c <= bounds.col_max:
                entries.append(CarWindowEntry(
                    position=car.position,
                    offset_from_agent=(r - self.agent_pos[0], c - self.agent_pos[1]),
                ))
        return bounds, entries

    def serialize_initial_state(self) -> InitialStateJSON:
        return InitialStateJSON(
            size=self.size,
            start_pos=self.start_pos,
            goal_pos=self.goal_pos,
            agent_pos=self.agent_pos,
            cars=[
                {"id": car.id, "spawn_position": car.position}
                for car in self.cars
            ],
            fixed_turn_length=self.fixed_turn_length,
            seed=self.seed,
        )
