# -*- coding: utf-8 -*-
"""Agent simulation -- LLM plans a whole turn of raw actions (no tools).

Derived from ``run_agent_simulation.py``. The ONLY thing that changes is the
decision step: there are no quantized tools and no tool-description file. Each
turn the vision LLM looks at the same observation image and directly returns
the next ``TURN_LENGTH`` unit actions (default: FIXED_TURN_LENGTH from config).
Those actions are executed with ``env.execute_turn`` exactly like before, the
agent re-observes, and the loop repeats for ``MAX_TURNS``.

We work in (row, column) coordinates (both 0-based; (0, 0) top-left; row
increases downward, column rightward). An action is a (drow, dcol) delta added
to the agent's (row, column):
    up = (-1, 0)   down = (1, 0)   left = (0, -1)   right = (0, 1)   wait = (0, 0)

Environment + plotting are reused from the ``simutils`` package
(``MovingWorldEnv``, ``compute_window_bounds``, ``plot_single_trajectory``,
``env_description_prompt``).

Run:  python run_agent_simulation_notools_v1.py
"""

# ============================================================
# Global parameters
# ============================================================
MODEL     = None     # if None use simutils.vision_qwen.MY_MODEL
MODEL_URL = None     # if None use simutils.vision_qwen.MY_URL

import sys

TURN_LENGTH     = None     # None -> FIXED_TURN_LENGTH  (number of actions per turn)
MAX_TURNS       = 30
NUM_CARS_SIM    = None     # None -> GRID_SIZE cars
SIM_PLOTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "simulation_plots_notools"
SIM_LOG_PKL     = "turn_execution_log.pkl"
TXT_LOG_FILE = "chat.txt"
MAKE_ZIP        = False    # local zip of the plots dir + *.txt/*.png/*.npz/*.pkl when done
NR_RUNS = 40 #number of times to run the main() function


import json
import pickle
import random
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from simutils.config import (
    GRID_SIZE, START_POS, GOAL_POS, WINDOW_HALF_SIZE, FIXED_TURN_LENGTH, VECTOR_TO_DELTA,USE_HAZARD_CLOSEUP,NUM_CARS
)
from simutils.env import MovingWorldEnv
from simutils.env_utils import compute_window_bounds
from simutils.visualization import plot_single_trajectory
from simutils.tool_names import tool_name
from simutils.prompts import env_description_prompt
from simutils.vision_qwen import vision_query, MY_MODEL, MY_URL

def _parse_actions(text: str, turn_length: int):
    """Pull the planned turn (list of ``[drow, dcol]``) out of the LLM's final
    answer -- strict ``json.loads`` first, then a forgiving ``{...}`` search.
    Always returns exactly ``turn_length`` tuples, each a valid action vector
    from VECTOR_TO_DELTA; anything missing / invalid becomes a ``(0, 0)`` hold,
    so a bad answer degrades to "wait" instead of crashing. Prints a WARNING
    whenever it had to repair the response (nothing parseable, wrong length, or
    a malformed / out-of-range entry)."""
    problem = None
    raw = None
    try:
        raw = json.loads(text)["actions"]
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                raw = json.loads(m.group(0))["actions"]
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                raw = None
    if not isinstance(raw, list):
        problem = "no valid 'actions' list in the response"
        raw = []
    elif len(raw) != turn_length:
        problem = f"'actions' had {len(raw)} entries, expected {turn_length}"

    actions = []
    for item in raw:
        try:
            a = (int(item[0]), int(item[1]))
        except (TypeError, ValueError, IndexError):
            a = (0, 0)
            problem = problem or "a malformed or out-of-range action entry"
        if a not in VECTOR_TO_DELTA:
            a = (0, 0)
            problem = problem or "a malformed or out-of-range action entry"
        actions.append(a)

    actions = actions[:turn_length]
    actions += [(0, 0)] * (turn_length - len(actions))
    if problem is not None:
        print(f"WARNING: LLM turn plan problem - {problem}; defaulting to {actions}")
    return actions


def main():
    model       = MODEL     or MY_MODEL
    model_url   = MODEL_URL  or MY_URL
    num_cars    = NUM_CARS_SIM if NUM_CARS_SIM is not None else NUM_CARS
    turn_length = TURN_LENGTH if TURN_LENGTH is not None else FIXED_TURN_LENGTH
    # the LLM may propose any turn_length moves, so its column reach is +/- turn_length
    DCOL_MIN, DCOL_MAX = -turn_length, turn_length

    # ---- the action-planning prompt (same info as the tool version: env
    #      description + observation image; only the output contract changes) ----
    example = ", ".join(["[drow, dcol]"] * turn_length)

    if USE_HAZARD_CLOSEUP:
        _images_intro = "You are given TWO images of the current situation:"
        _closeup_block = f"""
- a HAZARD CLOSE-UP: only the columns you can reach in {turn_length} moves
  (columns {DCOL_MIN}..{DCOL_MAX} relative to you) and the rows from just above
  you down to the bottom of the visible area. Every car that can possibly
  collide with your {turn_length}-move plan is in this crop; a car above you, or
  further left/right than this, cannot hit you. Its row and column axis ticks
  are labeled directly with the (drow, dcol) offset from you -- e.g. the row
  just above you reads "-1", your own row reads "0" -- so read a car's exact
  offset straight off these labels instead of counting cells. This crop is
  also clipped to the real grid: if it looks narrower or shorter than the
  column/row range stated above, that missing part is not hidden -- the world
  itself ends there, so no move in your plan can reach past it.

COMPULSORY, two separate uses for these images:
1. To decide whether a VISIBLE car makes any part of your plan unsafe, use
   only the HAZARD CLOSE-UP and its axis labels -- they are authoritative.
   Do not estimate or guess a car's offset by eye from the WIDE view; read it
   from the close-up's labeled axes.
2. To judge how close you are to a wall (especially the bottom re-entry zone),
   use the WIDE view's gray margin instead -- the close-up does not extend far
   enough to show this. This only justifies extra caution about gray cells in
   that direction, not a specific car location."""
        _closeup_reminder = ("\n- Reminder: collision judgment comes from the HAZARD CLOSE-UP's axis "
                             "labels, not from eyeballing the WIDE view.")
    else:
        _images_intro = "You are given ONE image of the current situation:"
        _closeup_block = ""
        _closeup_reminder = ""

    prompt_plan_turn = f""" {env_description_prompt}
{_images_intro}
- a WIDE view (the observation window centered on you) -- use it for context and
  for cars further away. This image always shows the ENTIRE world: cells
  outside your local window are grayed, but the image's own border IS the true
  edge of the grid. If the ungrayed area around you reaches close to that
  border, you are close to a wall in that direction -- in particular the
  bottom edge, where cars keep reappearing (see traffic model above).{_closeup_block}

The agent must decide its next {turn_length} actions (one "turn"), using only
the image(s) shown. Each action is a (drow, dcol) pair added to the
agent's current (row, column):

  [-1, 0] = up    (row - 1)        [ 1, 0] = down  (row + 1)
  [ 0, -1] = left (column - 1)     [ 0, 1] = right (column + 1)
  [ 0, 0] = wait  (hold position)

The {turn_length} actions are executed in order; execution stops early if the
agent collides with a car or reaches the goal. Then the agent re-observes and
plans the next turn.

Reasoning:
- Cars move up (row index decreasing) by one row per step, so a cell just below
  the visible region (a higher row index) may hold a car that moves into view.
- Account for uncertainty in unobserved cells.
- Estimate the risk of the planned actions using both observed and hidden traffic.
- Maintain progress toward the goal.
- Do not use wait actions unless justified by risks.{_closeup_reminder}

Output ONLY a JSON object of this form, with exactly {turn_length} actions:
{{
  "actions": [{example}],
  "risk_assessment": "...",
  "goal_progress": "...",
  "information_gain": "...",
  "justification": "..."
}}"""
    # ---- plots directory (archive an existing one) ----
    simulation_plots_dir = Path(SIM_PLOTS_DIR)
    if simulation_plots_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(f"{SIM_PLOTS_DIR}_{timestamp}")
        shutil.move(str(simulation_plots_dir), str(backup_dir))
        print(f"Previous simulation plots in {simulation_plots_dir} moved to {backup_dir}")
    simulation_plots_dir.mkdir(exist_ok=True)
    print(f"Simulation plots will be saved to: {simulation_plots_dir}")

    simulation_log = []

    # --- Simulation Loop ---

    # 1. Sample a new environment for the simulation
    env_seed = random.randint(0, 100000)
    env = MovingWorldEnv(
        size=GRID_SIZE,
        num_cars=num_cars,
        start_pos=START_POS,
        goal_pos=GOAL_POS,
        fixed_turn_length=turn_length,   # keep env's turn length == planned turn length
        seed=env_seed
    )
    print(f"\n--- Starting new simulation for env_seed={env_seed} ---")
    print(f"Initial environment: agent_pos={env.agent_pos}, goal_pos={env.goal_pos}, "
          f"turn_length={turn_length}, hazard_closeup={'on' if USE_HAZARD_CLOSEUP else 'off'}")

    simulation_agent_path = [env.agent_pos]
    simulation_car_positions_history = [[car.position for car in env.cars]]

    # Initial plotting for the first state (turn 0)
    # a. Plot the current full situation
    initial_full_view_path = simulation_plots_dir / f"turn_0_full_view.png"
    plot_single_trajectory(
        agent_path=simulation_agent_path,
        final_car_positions=simulation_car_positions_history[-1],
        grid_size=GRID_SIZE,
        title=f"Turn 0: Initial State (Full View)",
        save_only=str(initial_full_view_path)
    )
    print(f"Saved initial full view plot: {initial_full_view_path}")

    # b. Plot the observation window for the LLM
    initial_window_bounds = compute_window_bounds(env.agent_pos, WINDOW_HALF_SIZE, GRID_SIZE)
    initial_obs_view_path = simulation_plots_dir / f"turn_0_obs_view_for_llm.png"
    plot_single_trajectory(
        agent_path=simulation_agent_path,
        final_car_positions=simulation_car_positions_history[-1],
        grid_size=GRID_SIZE,
        title=f"Turn 0: Observation Window (for LLM)",
        window_bounds=initial_window_bounds,
        save_only=str(initial_obs_view_path)
    )
    print(f"Saved initial observation view plot for LLM: {initial_obs_view_path}")

    # Log initial state
    simulation_log.append({
        'turn': 0,
        'planned_actions': None,
        'agent_path_steps': [env.agent_pos],
        'car_positions_pre_turn': [car.position for car in env.cars],
        'car_positions_post_turn': [car.position for car in env.cars],
        'outcome': None
    })

    for turn_idx in range(MAX_TURNS):
        if env.is_terminal():
            print(f"Simulation terminated early at turn {turn_idx} with outcome: {env.outcome}")
            break

        print(f"\n--- Turn {turn_idx + 1} ---")

        # Store car positions before the turn for logging
        car_positions_pre_turn = [car.position for car in env.cars]

        # 1. Plot the current full situation (before LLM decision)
        current_full_view_path = simulation_plots_dir / f"turn_{turn_idx + 1}_full_view_pre_llm.png"
        plot_single_trajectory(
            agent_path=simulation_agent_path,
            final_car_positions=simulation_car_positions_history[-1],
            grid_size=GRID_SIZE,
            title=f"Turn {turn_idx + 1}: Current State (Full View, Pre-LLM)",
            save_only=str(current_full_view_path)
        )
        print(f"Saved full view plot: {current_full_view_path}")

        # 2. Generate the images for the LLM: a wide observation window and an
        #    agent-centered close-up (for reading exact offsets of nearby cars).
        current_window_bounds = compute_window_bounds(env.agent_pos, WINDOW_HALF_SIZE, GRID_SIZE)
        current_obs_view_path = simulation_plots_dir / f"turn_{turn_idx + 1}_obs_view_for_llm.png"
        plot_single_trajectory(
            agent_path=simulation_agent_path,
            final_car_positions=simulation_car_positions_history[-1],
            grid_size=GRID_SIZE,
            title=f"Turn {turn_idx + 1}: Observation Window (for LLM)",
            window_bounds=current_window_bounds,
            save_only=str(current_obs_view_path)
        )
        # hazard close-up: columns reachable in turn_length moves, rows from just
        # above the agent down to the bottom of the observation window.
        current_zoom_view_path = None
        if USE_HAZARD_CLOSEUP:
            _ar, _ac = env.agent_pos
            current_zoom_view_path = simulation_plots_dir / f"turn_{turn_idx + 1}_zoom_view_for_llm.png"
            plot_single_trajectory(
                agent_path=simulation_agent_path,
                final_car_positions=simulation_car_positions_history[-1],
                grid_size=GRID_SIZE,
                title=f"Turn {turn_idx + 1}: Hazard close-up (collision-relevant zone)",
                window_bounds=current_window_bounds,
                crop_rows=(_ar - 1, current_window_bounds.row_max),
                crop_cols=(_ac + DCOL_MIN, _ac + DCOL_MAX),
                show_relative_ticks=True,
                save_only=str(current_zoom_view_path)
            )
        print(f"Saved observation view for LLM: {current_obs_view_path}"
              + (f" + hazard close-up: {current_zoom_view_path}" if USE_HAZARD_CLOSEUP else ""))

        # 3. Ask the vision-Qwen agent to PLAN the next `turn_length` actions
        llm_image_paths = [str(current_obs_view_path)]
        if USE_HAZARD_CLOSEUP:
            llm_image_paths.append(str(current_zoom_view_path))
        final_response, thinking_response, all_chunks = vision_query(
            filenames_list=llm_image_paths,
            message=prompt_plan_turn,
            model=model,
            model_url=model_url,
        )
        print(f"LLM thinking ({len(thinking_response)} chars): {thinking_response[:500]}")
        print(f"LLM Raw Response: {final_response}")

        planned_actions = _parse_actions(final_response, turn_length)
        print(f"LLM planned turn: {planned_actions}")

        # 4. Execute the planned turn (stops early on collision / goal)
        turn_result = env.execute_turn(action_tuples=[(drow, dcol, 1) for (drow, dcol) in planned_actions])

        # Update agent path and car positions history
        # simulation_agent_path is cumulative from env.agent_pos, append only the final position
        simulation_agent_path.append(turn_result.final_position)
        # The environment's car positions are already updated after execute_turn
        simulation_car_positions_history.append([car.position for car in env.cars])

        print(f"Agent moved to: {env.agent_pos}")
        print(f"Turn outcome: {turn_result.outcome}")

        # Plot the full situation after the turn for visualization
        current_full_view_post_turn_path = simulation_plots_dir / f"turn_{turn_idx + 1}_full_view_post_llm.png"
        plot_single_trajectory(
            agent_path=simulation_agent_path,
            final_car_positions=simulation_car_positions_history[-1],
            grid_size=GRID_SIZE,
            title=f"Turn {turn_idx + 1}: State After LLM Turn {tool_name(planned_actions)} Execution (Full View)",
            save_only=str(current_full_view_post_turn_path)
        )
        print(f"Saved post-turn full view plot: {current_full_view_post_turn_path}")

        # Log turn details
        simulation_log.append({
            'turn': turn_idx + 1,
            'planned_actions': planned_actions,
            'llm_final_response': final_response,
            'llm_thinking_response': thinking_response,
            'agent_path_steps': [step.agent_pos for step in turn_result.step_results],
            'car_positions_pre_turn': car_positions_pre_turn,
            'car_positions_post_turn': [car.position for car in env.cars],
            'outcome': turn_result.outcome
        })

    print(f"\n--- Simulation Finished ---")
    # Save the simulation log to a pickle file
    with open(SIM_LOG_PKL, "wb") as f:
        pickle.dump(simulation_log, f)
    print(f"Simulation log saved to {SIM_LOG_PKL}")

    lines = []
    for entry in simulation_log:
        lines.append(f"=== Turn {entry.get('turn')} ===")
        for k, v in entry.items():
            lines.append(f"{k}: {v}")
        lines.append("")
    (simulation_plots_dir / TXT_LOG_FILE).write_text("\n".join(lines), encoding="utf-8")
    print(f"Text simulation log saved to {TXT_LOG_FILE}")

    
    print(f"Final agent position: {env.agent_pos}")
    print(f"Overall simulation outcome: {env.outcome}")

    #create a file with name of outcome
    with open(str( simulation_plots_dir / env.outcome) , "w") as f:
        pass
    
    # Display the final full view plot
    plot_single_trajectory(
        agent_path=simulation_agent_path,
        final_car_positions=simulation_car_positions_history[-1],
        grid_size=GRID_SIZE,
        title=f"Final State (Outcome: {env.outcome})"
    )

    # ---- optional local zip of the results ----
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
