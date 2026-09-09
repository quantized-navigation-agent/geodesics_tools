# -*- coding: utf-8 -*-
"""Non-LLM baseline: greedy 1-turn planner over the FREE action space.

No tools, no LLM, no learning. Each turn:

  1. enumerate every {FIXED_TURN_LENGTH}-step sequence of primitive actions
     (5 ** {FIXED_TURN_LENGTH} = 3125 candidates);
  2. rank them by (Manhattan distance from the sequence's end cell to the goal,
     then number of wait steps) -- best progress first;
  3. walk that ranking and pick the first sequence that does NOT hit a car
     currently inside the observation window (``simutils.collision_check``);
  4. if none is collision-free, hold (five waits).

This is the "planner with full action freedom, one-turn horizon, visible-car
collision filter" reference point for the tool-based / LLM runs.

Run:  python run_sim_notools_heuristic_v1.py [output_dir]
"""

# ============================================================
# Global parameters
# ============================================================
import sys
MAX_TURNS       = 25
NUM_CARS_SIM    = None     # None -> config.NUM_CARS
SIM_PLOTS_DIR   = sys.argv[1] if len(sys.argv) > 1 else "simulation_plots_notools_heuristic"
SIM_LOG_PKL     = "turn_execution_log.pkl"
TXT_LOG_FILE    = "chat.txt"
MAKE_ZIP        = False
NR_RUNS = 40   # number of times to run main()

import simutils.log_stderr_stdout   # tee stdout/stderr to output.txt

import pickle
import random
import shutil
import zipfile
from datetime import datetime
from itertools import product
from pathlib import Path

from simutils.config import (
    GRID_SIZE, START_POS, GOAL_POS, WINDOW_HALF_SIZE, FIXED_TURN_LENGTH, NUM_CARS,
)
from simutils.env import MovingWorldEnv
from simutils.env_utils import compute_window_bounds
from simutils.visualization import plot_single_trajectory
from simutils.collision_check import collision_with_visible_cars
from simutils.tool_names import tool_name

_PRIMS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]   # U D L R W


def _endpoint(agent_pos, seq, size):
    """Agent cell after `seq`, with the env's wall rule (a step that would exit
    the grid on either axis is absorbed)."""
    r, c = agent_pos
    for dr, dc in seq:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            r, c = nr, nc
    return r, c


def _choose_sequence(env):
    """Greedy pick: best-progress collision-free sequence, else all-wait."""
    gr, gc = GOAL_POS
    ar, ac = env.agent_pos
    all_seqs = product(_PRIMS, repeat=FIXED_TURN_LENGTH)
    ranked = sorted(
        all_seqs,
        key=lambda s: (
            (lambda e: abs(e[0] - gr) + abs(e[1] - gc))(_endpoint((ar, ac), s, env.size)),
            sum(1 for a in s if a == (0, 0)),
        ),
    )
    for s in ranked:
        if not collision_with_visible_cars(env, s):
            return list(s), True
    return [(0, 0)] * FIXED_TURN_LENGTH, False


def main():
    num_cars = NUM_CARS_SIM if NUM_CARS_SIM is not None else NUM_CARS

    simulation_plots_dir = Path(SIM_PLOTS_DIR)
    if simulation_plots_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(f"{SIM_PLOTS_DIR}_{timestamp}")
        shutil.move(str(simulation_plots_dir), str(backup_dir))
        print(f"Previous simulation plots in {simulation_plots_dir} moved to {backup_dir}")
    simulation_plots_dir.mkdir(exist_ok=True)
    print(f"Simulation plots will be saved to: {simulation_plots_dir}")

    simulation_log = []

    env_seed = random.randint(0, 100000)
    env = MovingWorldEnv(
        size=GRID_SIZE, num_cars=num_cars,
        start_pos=START_POS, goal_pos=GOAL_POS, seed=env_seed,
    )
    print(f"\n--- Starting new simulation for env_seed={env_seed} ---")
    print(f"Initial environment: agent_pos={env.agent_pos}, goal_pos={env.goal_pos}")

    simulation_agent_path = [env.agent_pos]
    simulation_car_positions_history = [[car.position for car in env.cars]]

    plot_single_trajectory(
        agent_path=simulation_agent_path,
        final_car_positions=simulation_car_positions_history[-1],
        grid_size=GRID_SIZE, title="Turn 0: Initial State (Full View)",
        save_only=str(simulation_plots_dir / "turn_0_full_view.png"),
    )
    initial_window_bounds = compute_window_bounds(env.agent_pos, WINDOW_HALF_SIZE, GRID_SIZE)
    plot_single_trajectory(
        agent_path=simulation_agent_path,
        final_car_positions=simulation_car_positions_history[-1],
        grid_size=GRID_SIZE, title="Turn 0: Observation Window",
        window_bounds=initial_window_bounds,
        save_only=str(simulation_plots_dir / "turn_0_obs_view.png"),
    )

    simulation_log.append({
        'turn': 0, 'chosen_tool': None, 'planned_actions': None, 'any_safe': None,
        'agent_path_steps': [env.agent_pos],
        'car_positions_pre_turn': [car.position for car in env.cars],
        'car_positions_post_turn': [car.position for car in env.cars],
        'outcome': None,
    })

    for turn_idx in range(MAX_TURNS):
        if env.is_terminal():
            print(f"Simulation terminated early at turn {turn_idx} with outcome: {env.outcome}")
            break
        print(f"\n--- Turn {turn_idx + 1} ---")

        car_positions_pre_turn = [car.position for car in env.cars]

        plot_single_trajectory(
            agent_path=simulation_agent_path,
            final_car_positions=simulation_car_positions_history[-1],
            grid_size=GRID_SIZE,
            title=f"Turn {turn_idx + 1}: Current State (Full View, Pre-decision)",
            save_only=str(simulation_plots_dir / f"turn_{turn_idx + 1}_full_view_pre.png"),
        )
        current_window_bounds = compute_window_bounds(env.agent_pos, WINDOW_HALF_SIZE, GRID_SIZE)
        plot_single_trajectory(
            agent_path=simulation_agent_path,
            final_car_positions=simulation_car_positions_history[-1],
            grid_size=GRID_SIZE, title=f"Turn {turn_idx + 1}: Observation Window",
            window_bounds=current_window_bounds,
            save_only=str(simulation_plots_dir / f"turn_{turn_idx + 1}_obs_view.png"),
        )

        planned_actions, any_safe = _choose_sequence(env)
        chosen_name = tool_name(planned_actions)
        print(f"heuristic pick: {chosen_name}  (a collision-free sequence found: {any_safe})")

        turn_result = env.execute_turn(
            action_tuples=[(int(a[0]), int(a[1]), 1) for a in planned_actions]
        )
        simulation_agent_path.append(turn_result.final_position)
        simulation_car_positions_history.append([car.position for car in env.cars])
        print(f"Agent moved to: {env.agent_pos}   turn outcome: {turn_result.outcome}")

        plot_single_trajectory(
            agent_path=simulation_agent_path,
            final_car_positions=simulation_car_positions_history[-1],
            grid_size=GRID_SIZE,
            title=f"Turn {turn_idx + 1}: State After {chosen_name} Execution (Full View)",
            save_only=str(simulation_plots_dir / f"turn_{turn_idx + 1}_full_view_post.png"),
        )

        simulation_log.append({
            'turn': turn_idx + 1,
            'chosen_tool': chosen_name,
            'planned_actions': planned_actions,
            'any_safe': any_safe,
            'agent_path_steps': [step.agent_pos for step in turn_result.step_results],
            'car_positions_pre_turn': car_positions_pre_turn,
            'car_positions_post_turn': [car.position for car in env.cars],
            'outcome': turn_result.outcome,
        })

    print("\n--- Simulation Finished ---")
    logfile = str(simulation_plots_dir / SIM_LOG_PKL)
    with open(logfile, "wb") as f:
        pickle.dump(simulation_log, f)
    print(f"Simulation log saved to {logfile}")

    lines = []
    for entry in simulation_log:
        lines.append(f"=== Turn {entry.get('turn')} ===")
        for k, v in entry.items():
            lines.append(f"{k}: {v}")
        lines.append("")
    (simulation_plots_dir / TXT_LOG_FILE).write_text("\n".join(lines), encoding="utf-8")

    if env.outcome is None:
        env.mark_failed()
    print(f"Final agent position: {env.agent_pos}")
    print(f"Overall simulation outcome: {env.outcome}")

    with open(str(simulation_plots_dir / env.outcome), "w") as f:
        pass

    plot_single_trajectory(
        agent_path=simulation_agent_path,
        final_car_positions=simulation_car_positions_history[-1],
        grid_size=GRID_SIZE, title=f"Final State (Outcome: {env.outcome})",
    )

    if MAKE_ZIP:
        zip_name = f"backup_results_{simulation_plots_dir}.zip"
        exts = {".txt", ".png", ".npz", ".pkl"}
        with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
            for f in simulation_plots_dir.rglob("*"):
                if f.is_file():
                    z.write(f, arcname=f.relative_to(simulation_plots_dir.parent))
            for f in Path(".").glob("*"):
                if f.is_file() and f.suffix.lower() in exts:
                    z.write(f, arcname=f.name)
        print("Created:", zip_name)


if __name__ == "__main__":
    for jj in range(NR_RUNS):
        print(f"start overall run number {jj}")
        main()
    print('end of all runs')
