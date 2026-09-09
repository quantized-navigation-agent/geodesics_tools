# -*- coding: utf-8 -*-
"""Run the tool-selection agent simulation with a vision-Qwen LLM.

  * the tools come from a name-keyed .npz file (5 elementary + up to 5 quantized);
    each tool is identified by its name, e.g. "RRWDD" (one letter per step:
    U/D/L/R/W),
  * the tool descriptions come from a text file (keyed by the same names),
  * the LLM call goes through the new streaming ``vision_query`` (Qwen), whose
    signature is
        vision_query(filenames_list=[], message="", model=MY_MODEL, model_url=MY_URL)
        -> (final_response, thinking_response, all_chunks)

Everything about the environment and the plots is reused as-is from the
``simutils`` package (``MovingWorldEnv``, ``compute_window_bounds``,
``plot_single_trajectory``, ``env_description_prompt``).

Coordinates are (row, column) throughout (both 0-based; (0, 0) top-left; row
increases downward, column rightward). A tool is a list of (drow, dcol) moves.
The tool-description .txt must be regenerated (generate_tool_description.py)
whenever quantized_actions.npz is rebuilt.

Run:  python run_agent_simulation_v2.py
"""

# ============================================================
# Global parameters
# ============================================================
TOOL_DESCRIPTION_FILE = "qwen3_6_35b_a3b_list_of_tool_descriptions.txt"   # text file with the tool descriptions
TOOL_ACTIONS_NPZ      = "quantized_actions.npz"     # name-keyed .npz: key = tool name ("RRWDD"),
                                                   # value = (GEODESIC_LEN, 2) (drow, dcol) array

MODEL     = None     # if None -> use simutils.vision_qwen.MY_MODEL
MODEL_URL = None     # if None -> use simutils.vision_qwen.MY_URL

import sys
MAX_TURNS       = 30
NUM_CARS_SIM    = None     # None -> config.NUM_CARS
SIM_PLOTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "simulation_plots_tools"
SIM_LOG_PKL     = "tool_execution_log.pkl"
MAKE_ZIP        = False    # local zip of the plots dir + *.txt/*.png/*.npz/*.pkl when done
LOG_FILE = "output.log"
TXT_LOG_FILE = "chat.txt"
NR_RUNS = 40 #number of times to run the main() function

import simutils.log_stderr_stdout
import json
import pickle
import random
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from simutils.config import (
    GRID_SIZE, START_POS, GOAL_POS, WINDOW_HALF_SIZE, USE_HAZARD_CLOSEUP,NUM_CARS
)
from simutils.env import MovingWorldEnv
from simutils.env_utils import compute_window_bounds
from simutils.visualization import plot_single_trajectory
from simutils.tool_names import tool_column_reach
from simutils.prompts import env_description_prompt
from simutils.vision_qwen import vision_query, encode_image, MY_MODEL, MY_URL

def _parse_tool_name(text: str, valid_names) -> str:
    """Pull the chosen tool NAME (a U/D/L/R/W string like 'RRWDD') out of the
    LLM's final answer -- strict json.loads first, then a braced JSON blob, then
    a regex on "chosen_tool". Returns a name from `valid_names` ('WWWWW' if
    present, else the first) when nothing usable is found."""
    valid = list(valid_names)
    fallback = "WWWWW" if "WWWWW" in valid else valid[0]

    blobs = [text]
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        blobs.append(m.group(0))
    for blob in blobs:
        try:
            v = str(json.loads(blob)["chosen_tool"]).strip().upper()
            if v in valid:
                return v
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

    m = re.search(r'"chosen_tool"\s*:\s*"?([UDLRWudlrw]+)"?', text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    print(f"WARNING: LLM tool choice unintelligible; defaulting to '{fallback}'.")
    return fallback
    

def main():
    model     = MODEL     or MY_MODEL
    model_url = MODEL_URL  or MY_URL
    num_cars  = NUM_CARS_SIM if NUM_CARS_SIM is not None else NUM_CARS

    # ---- load the tools (name-keyed .npz) and their descriptions ----
    _npz = np.load(TOOL_ACTIONS_NPZ)
    tools = {name: _npz[name] for name in _npz.files}   # {name -> (T, 2) (drow, dcol)}
    tool_names = list(tools)
    # column band any tool can reach -> width of the collision-hazard close-up
    DCOL_MIN, DCOL_MAX = tool_column_reach(tools)
    print(f"Loaded {len(tools)} tools from {TOOL_ACTIONS_NPZ}: {tool_names} "
          f"(column reach {DCOL_MIN}..{DCOL_MAX})")

    list_of_tool_descriptions = Path(TOOL_DESCRIPTION_FILE).read_text(encoding="utf-8")
    print(f"Loaded tool descriptions from {TOOL_DESCRIPTION_FILE} "
          f"({len(list_of_tool_descriptions)} chars)")


    if USE_HAZARD_CLOSEUP:
        _images_intro = "You are given TWO images of the current situation:"
        _closeup_block = f"""
- a HAZARD CLOSE-UP: only the columns any tool can reach (columns
  {DCOL_MIN}..{DCOL_MAX} relative to you) and the rows from just above you down
  to the bottom of the visible area. Every car that can possibly collide with a
  tool is in this crop; a car above you, or further left/right than this, cannot
  hit any tool. Its row and column axis ticks are labeled directly with the
  (drow, dcol) offset from you -- e.g. the row just above you reads "-1", your
  own row reads "0" -- so read a car's exact offset straight off these labels
  instead of counting cells. This crop is also clipped to the real grid: if it
  looks narrower or shorter than the column/row range stated above, that
  missing part is not hidden -- the world itself ends there, so no tool step
  can reach past it.

COMPULSORY, two separate uses for these images:
1. To decide whether a VISIBLE car makes a tool unsafe, use only the HAZARD
   CLOSE-UP and its axis labels -- they are authoritative. Do not rely on a
   tool's own forbidden_car_offsets / bad_when text for this: that text was
   generated offline, without seeing this exact situation, and can be wrong,
   especially near the edges of a tool's path.
2. To judge how close you are to a wall (especially the bottom re-entry zone),
   use the WIDE view's gray margin instead -- the close-up does not extend far
   enough to show this. This only justifies extra caution about gray cells in
   that direction, not a specific car location."""
        _closeup_reminder = ("\n- Reminder: collision judgment comes from the HAZARD CLOSE-UP's axis "
                             "labels,\n  not from a tool's own forbidden_car_offsets/bad_when text.")
    else:
        _images_intro = "You are given ONE image of the current situation:"
        _closeup_block = ""
        _closeup_reminder = ""

    prompt_choose_tool = f""" {env_description_prompt}
{_images_intro}
- a WIDE view (the observation window centered on you) -- use it for context and
  for cars further away. This image always shows the ENTIRE world: cells
  outside your local window are grayed, but the image's own border IS the true
  edge of the grid. If the ungrayed area around you reaches close to that
  border, you are close to a wall in that direction -- in particular the
  bottom edge, where cars keep reappearing (see traffic model above).{_closeup_block}

The agent has to choose its next move using only the image(s) shown. The move
consists in choosing one tool; that tool then implements its list of
(drow, dcol) actions until collision, reaching the goal, or exhaustion of the
list.

Not all tools are useful, the list is given in advance.

Each tool is named by its actions, one letter per step:
  U = up (-1, 0)   D = down (1, 0)   L = left (0, -1)   R = right (0, 1)   W = wait (0, 0)
e.g. the tool "RRWDD" is right, right, wait, down, down. Its actions are
(drow, dcol) pairs added to the current (row, column).

{list_of_tool_descriptions}
Reasoning:
- Cars move up (row index decreasing) by one row per step; observed traffic at a
  higher row index than the agent (i.e. visually below it) must be treated with
  extreme caution, since those cars move toward the agent.
- Account for uncertainty in unobserved cells.
- Estimate the risk of each tool using both observed and hidden traffic.
- Prefer tools that balance safety and goal progress.
- Do not choose tools that contain many wait actions unless justified by risks.{_closeup_reminder}

Output:
{{
  "chosen_tool": "the tool name, e.g. RRWDD",
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
        seed=env_seed
    )
    print(f"\n--- Starting new simulation for env_seed={env_seed} ---")
    print(f"Initial environment: agent_pos={env.agent_pos}, goal_pos={env.goal_pos}, "
          f"hazard_closeup={'on' if USE_HAZARD_CLOSEUP else 'off'}")

    simulation_agent_path = [env.agent_pos]
    simulation_car_positions_history = [[car.position for car in env.cars]]

    # Initial plotting for the first state (turn 0)
    # a. Plot the current full situation
    initial_full_view_path = simulation_plots_dir / "turn_0_full_view.png"
    plot_single_trajectory(
        agent_path=simulation_agent_path,
        final_car_positions=simulation_car_positions_history[-1],
        grid_size=GRID_SIZE,
        title="Turn 0: Initial State (Full View)",
        save_only=str(initial_full_view_path)
    )
    print(f"Saved initial full view plot: {initial_full_view_path}")

    # b. Plot the observation window for the LLM
    initial_window_bounds = compute_window_bounds(env.agent_pos, WINDOW_HALF_SIZE, GRID_SIZE)
    initial_obs_view_path = simulation_plots_dir / "turn_0_obs_view_for_llm.png"
    plot_single_trajectory(
        agent_path=simulation_agent_path,
        final_car_positions=simulation_car_positions_history[-1],
        grid_size=GRID_SIZE,
        title="Turn 0: Observation Window (for LLM)",
        window_bounds=initial_window_bounds,
        save_only=str(initial_obs_view_path)
    )
    print(f"Saved initial observation view plot for LLM: {initial_obs_view_path}")

    # Log initial state
    simulation_log.append({
        'turn': 0,
        'chosen_tool': None,
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
        # hazard close-up: columns any tool can reach, rows from just above the
        # agent down to the bottom of the observation window (only cars there can
        # collide with a tool).
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

        # 3. Query the vision-Qwen agent to determine which tool to use
        llm_image_paths = [str(current_obs_view_path)]
        if USE_HAZARD_CLOSEUP:
            llm_image_paths.append(str(current_zoom_view_path))
        final_response, thinking_response, all_chunks = vision_query(
            filenames_list=llm_image_paths,
            message=prompt_choose_tool,
            model=model,
            model_url=model_url,
        )
        print(f"LLM thinking ({len(thinking_response)} chars): {thinking_response[:500]}")
        print(f"LLM Raw Response: {final_response}")

        chosen_tool = _parse_tool_name(final_response, tool_names)
        print(f"LLM chose tool: {chosen_tool}")

        # _parse_tool_name already guarantees a known name; keep a guard anyway
        if chosen_tool not in tools:
            print(f"Warning: LLM chose unknown tool '{chosen_tool}'. Falling back to {tool_names[0]}.")
            chosen_tool = tool_names[0]

        # 4. Execute the actions corresponding to the chosen tool
        selected_actions = tools[chosen_tool]
        print(f"Executing actions from tool {chosen_tool}: {selected_actions}")

        turn_result = env.execute_turn(action_tuples=[(int(a[0]), int(a[1]), 1) for a in selected_actions])

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
            title=f"Turn {turn_idx + 1}: State After Tool {chosen_tool} Execution (Full View)",
            save_only=str(current_full_view_post_turn_path)
        )
        print(f"Saved post-turn full view plot: {current_full_view_post_turn_path}")

        # Log turn details
        simulation_log.append({
            'turn': turn_idx + 1,
            'chosen_tool': chosen_tool,
            'llm_final_response': final_response,
            'llm_thinking_response': thinking_response,
            'agent_path_steps': [step.agent_pos for step in turn_result.step_results],
            'car_positions_pre_turn': car_positions_pre_turn,
            'car_positions_post_turn': [car.position for car in env.cars],
            'outcome': turn_result.outcome
        })

    print("\n--- Simulation Finished ---")
    # Save the simulation log to a pickle file
    logfile=str( simulation_plots_dir / SIM_LOG_PKL)
    with open(logfile , "wb") as f:
        pickle.dump(simulation_log, f)
    print(f"Simulation log saved to {logfile}")

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

#    import os
#    os.rename("output.log", str( simulation_plots_dir / "output.log"))

if __name__ == "__main__":
    for jj in range(NR_RUNS):
        print(f"start overall run number {jj}")
        main()
    print('end of all runs')

