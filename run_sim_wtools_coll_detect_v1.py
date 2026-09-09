# -*- coding: utf-8 -*-
"""Tool-selection agent simulation with a CODE collision check.

Difference from ``run_agent_simulation_v2.py``: collision avoidance is no longer
the LLM's job. Each turn:

  1. a deterministic code check (``simutils.collision_check``) is run for every
     tool -- it replays the tool against only the cars currently inside the
     observation window and reports whether the agent would hit one;
  2. tools that would hit a visible car are dropped;
  3. if 0 tools remain           -> force "WWWWW", no LLM call at all;
     if exactly 1 tool remains   -> pick it, no LLM call;
     otherwise                   -> a short LLM prompt picks the tool that best
                                    advances toward the green target (no
                                    kinematics, no collision reasoning), with
                                    the answer constrained to the safe-tool set.

Tool descriptions come from ``tool_description.pkl`` (a ``{name: description}``
dict written by ``generate_tool_description.py``); only the safe tools' entries
are sent in the prompt.

Images sent to the LLM are the same as the no-tools script: the wide
observation window, plus -- when ``USE_HAZARD_CLOSEUP`` is set in config -- the
general agent-centred close-up (NOT a tool-specific one).

Run:  python run_sim_wtools_coll_detect_v1.py [output_dir]
"""

# ============================================================
# Global parameters
# ============================================================
TOOL_ACTIONS_NPZ     = "quantized_actions.npz"     # name-keyed .npz: key = tool name ("RRWDD")
TOOL_DESCRIPTION_PKL = "tool_description.pkl"       # {tool name -> short description}

MODEL     = None     # if None -> use simutils.vision_qwen.MY_MODEL
MODEL_URL = None     # if None -> use simutils.vision_qwen.MY_URL
LLM_THINK = False    # thinking OFF by default here (the tool pick is trivial once collisions are filtered)
LLM_MAX_TOKENS = 4096   # num_predict cap for the tool-pick call (small: the answer is one short JSON object)

import sys
MAX_TURNS       = 30
NUM_CARS_SIM    = None     # None -> config.NUM_CARS
SIM_PLOTS_DIR   = sys.argv[1] if len(sys.argv) > 1 else "simulation_plots_wtools_cd"
SIM_LOG_PKL     = "tool_execution_log.pkl"
TXT_LOG_FILE    = "chat.txt"
MAKE_ZIP        = False
NR_RUNS = 30   # number of times to run main()

import simutils.log_stderr_stdout   # tee stdout/stderr to output.txt

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
    GRID_SIZE, START_POS, GOAL_POS, WINDOW_HALF_SIZE, FIXED_TURN_LENGTH,
    USE_HAZARD_CLOSEUP, NUM_CARS,
)
from simutils.env import MovingWorldEnv
from simutils.env_utils import compute_window_bounds
from simutils.visualization import plot_single_trajectory
from simutils.collision_check import collision_with_visible_cars
from simutils.vision_qwen import vision_query, MY_MODEL, MY_URL


def _parse_choice(text: str, valid) -> str:
    """Pull "chosen_tool" out of the LLM answer, restricted to `valid`.
    With the response_format schema below the answer is already valid JSON with
    an in-set value; this is only a safety net."""
    valid = list(valid)
    blobs = []
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        blobs.append(m.group(0))
    blobs.append(text)
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
    print("WARNING: could not parse a safe-tool choice; picking a random safe tool")
    return random.choice(valid)


def main():
    model     = MODEL     or MY_MODEL
    model_url = MODEL_URL  or MY_URL
    num_cars  = NUM_CARS_SIM if NUM_CARS_SIM is not None else NUM_CARS

    # ---- load tools + their short descriptions ----
    _npz = np.load(TOOL_ACTIONS_NPZ)
    tools = {name: _npz[name] for name in _npz.files}   # {name -> (T, 2) (drow, dcol)}
    tool_names = list(tools)
    with open(TOOL_DESCRIPTION_PKL, "rb") as f:
        tool_desc = pickle.load(f)                      # {name -> short description}
    print(f"Loaded {len(tools)} tools: {tool_names}")

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

    # ---- sample a new environment ----
    env_seed = random.randint(0, 100000)
    env = MovingWorldEnv(
        size=GRID_SIZE, num_cars=num_cars,
        start_pos=START_POS, goal_pos=GOAL_POS, seed=env_seed,
    )
    print(f"\n--- Starting new simulation for env_seed={env_seed} ---")
    print(f"Initial environment: agent_pos={env.agent_pos}, goal_pos={env.goal_pos}, "
          f"hazard_closeup={'on' if USE_HAZARD_CLOSEUP else 'off'}")

    simulation_agent_path = [env.agent_pos]
    simulation_car_positions_history = [[car.position for car in env.cars]]

    # turn 0 renders
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
        grid_size=GRID_SIZE, title="Turn 0: Observation Window (for LLM)",
        window_bounds=initial_window_bounds,
        save_only=str(simulation_plots_dir / "turn_0_obs_view_for_llm.png"),
    )

    simulation_log.append({
        'turn': 0, 'chosen_tool': None, 'safe_tools': None, 'filtered_out': None,
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

        # 1. full view (log only)
        plot_single_trajectory(
            agent_path=simulation_agent_path,
            final_car_positions=simulation_car_positions_history[-1],
            grid_size=GRID_SIZE,
            title=f"Turn {turn_idx + 1}: Current State (Full View, Pre-LLM)",
            save_only=str(simulation_plots_dir / f"turn_{turn_idx + 1}_full_view_pre_llm.png"),
        )

        # 2. images for the LLM: wide observation window (+ optional general close-up)
        current_window_bounds = compute_window_bounds(env.agent_pos, WINDOW_HALF_SIZE, GRID_SIZE)
        current_obs_view_path = simulation_plots_dir / f"turn_{turn_idx + 1}_obs_view_for_llm.png"
        plot_single_trajectory(
            agent_path=simulation_agent_path,
            final_car_positions=simulation_car_positions_history[-1],
            grid_size=GRID_SIZE,
            title=f"Turn {turn_idx + 1}: Observation Window (for LLM)",
            window_bounds=current_window_bounds,
            save_only=str(current_obs_view_path),
        )
        current_zoom_view_path = None
        if USE_HAZARD_CLOSEUP:
            _ar, _ac = env.agent_pos
            current_zoom_view_path = simulation_plots_dir / f"turn_{turn_idx + 1}_zoom_view_for_llm.png"
            plot_single_trajectory(
                agent_path=simulation_agent_path,
                final_car_positions=simulation_car_positions_history[-1],
                grid_size=GRID_SIZE,
                title=f"Turn {turn_idx + 1}: Close-up",
                window_bounds=current_window_bounds,
                crop_rows=(_ar - 1, current_window_bounds.row_max),
                crop_cols=(_ac - FIXED_TURN_LENGTH, _ac + FIXED_TURN_LENGTH),
                show_relative_ticks=True,
                save_only=str(current_zoom_view_path),
            )
        llm_image_paths = [str(current_obs_view_path)]
        if USE_HAZARD_CLOSEUP:
            llm_image_paths.append(str(current_zoom_view_path))

        # 3. code collision filter: keep tools that do NOT hit a visible car
        safe_tools = [n for n in tool_names
                      if not collision_with_visible_cars(env, tools[n])]
        filtered_out = [n for n in tool_names if n not in safe_tools]
        print(f"visible-car collision filter: {len(safe_tools)}/{len(tool_names)} tools pass -> {safe_tools}")

        # 4. choose a tool
        if len(safe_tools) == 0:
            chosen_tool = "WWWWW" if "WWWWW" in tools else tool_names[0]
            llm_final_response = "(no LLM call: no tool avoids a visible car -> forced WWWWW)"
            llm_thinking_response = ""
            print(f"No safe tool -> forced {chosen_tool}, no LLM call.")
        elif len(safe_tools) == 1:
            chosen_tool = safe_tools[0]
            llm_final_response = f"(no LLM call: only one safe tool -> {chosen_tool})"
            llm_thinking_response = ""
            print(f"Only one safe tool -> {chosen_tool}, no LLM call.")
        else:
            tool_lines = "\n".join(
                f"- {n}: {tool_desc.get(n, '(no description)')}" for n in safe_tools
            )
            tool_lines = "\n".join(
    f"<tool {n} description begin>: " + " ".join(tool_desc.get(n, "(no description)").split()) + f" :<tool {n} description end>"
    for n in safe_tools
)
            prompt = f"""
You are given two images of a grid. The agent is the BLUE cell, the target is the GREEN cell, cars in RED cells. WHITE cells are free. GRAY cells are not observed and can contain cars or be free.

In the OBSERVATION image you see all grid, the unobserved parts are in GRAY. In the CLOSE-UP image you see a part of the grid close to the agent (if this is cropped this means the agent is near the border). Whenever possible use the CLOSE-UP image, it is more precise.

A code check has already removed every tool that would hit a currently visible car.
The tools below are the SAFE ones -- pick the single one that best advances from the BLUE cell toward the GREEN target. 
Avoid being close to grid edges when possible because edges are hard walls and actions are distorted there : any action that would hit a wall does nothing and the net displacement becomes unreliable. 
Reaching target at any time or intermediary time is considered success and everything stops.
The cars, agent, collisions, kinematics or timing is already handled, just state in what direction the BLUE agent has to go to reach the GREEN target cell, for instance  "below" or "to the left" or "to the right" and compare which tool matches best this expected progression. Choose the one. 

Answer immediately after stating the direction of the target and the adequation of each tool.

Safe tools (name: net displacement + strategy):
{tool_lines}
Output ONLY: {{
"target_relative_to_agent_from_observation": "<one sentence: describe, as seen from the OBSERVATION image only, in what direction the BLUE agent has to go to reach the target>",
"target_relative_to_agent_from_closeup": "<one sentence, from the CLOSE-UP image only: in what direction must the BLUE agent go to reach the GREEN target. If the GREEN target is not inside the CLOSE-UP, write exactly: not visible in close-up>",
"target_relative_to_agent_summary": "<one sentence: reconcile previous two answers to summarize, as seen from both images, in what direction the BLUE agent has to go to reach the GREEN target.>",
"justification": "<one sentence: what tool in the tool name list helps best the BLUE agent to approach fast or even reach the GREEN target>", 
"chosen_tool": "<the tool named above that best helps BLUE agent to approach fast or even reach the GREEN target>"}}"""
            schema = {
                "type": "object",
                "properties": {
                    "target_relative_to_agent_from_observation": {"type": "string"},
                    "target_relative_to_agent_from_closeup": {"type": "string"},
                    "target_relative_to_agent_summary": {"type": "string"},
                    "justification": {"type": "string"},
                    "chosen_tool": {"type": "string", "enum": list(safe_tools)},
                },
                "required": ["target_relative_to_agent_from_observation", "target_relative_to_agent_from_closeup", "target_relative_to_agent_summary", "justification", "chosen_tool"],
            }
            llm_final_response, llm_thinking_response, _ = vision_query(
                filenames_list=llm_image_paths, message=prompt,
                model=model, model_url=model_url,
                response_format=schema, think=LLM_THINK,
                num_predict=LLM_MAX_TOKENS,
            )
            print(f"LLM thinking ({len(llm_thinking_response)} chars): {llm_thinking_response[:300]}")
            print(f"LLM raw response: {llm_final_response}")
            chosen_tool = _parse_choice(llm_final_response, safe_tools)
            print(f"LLM chose tool: {chosen_tool}")

        # 5. execute
        selected_actions = tools.get(chosen_tool, np.zeros((FIXED_TURN_LENGTH, 2), dtype=int))
        print(f"Executing tool {chosen_tool}: {selected_actions.tolist()}")
        turn_result = env.execute_turn(
            action_tuples=[(int(a[0]), int(a[1]), 1) for a in selected_actions]
        )
        simulation_agent_path.append(turn_result.final_position)
        simulation_car_positions_history.append([car.position for car in env.cars])
        print(f"Agent moved to: {env.agent_pos}   turn outcome: {turn_result.outcome}")

        plot_single_trajectory(
            agent_path=simulation_agent_path,
            final_car_positions=simulation_car_positions_history[-1],
            grid_size=GRID_SIZE,
            title=f"Turn {turn_idx + 1}: State After Tool {chosen_tool} Execution (Full View)",
            save_only=str(simulation_plots_dir / f"turn_{turn_idx + 1}_full_view_post_llm.png"),
        )

        simulation_log.append({
            'turn': turn_idx + 1,
            'chosen_tool': chosen_tool,
            'safe_tools': safe_tools,
            'filtered_out': filtered_out,
            'llm_final_response': llm_final_response,
            'llm_thinking_response': llm_thinking_response,
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

    if env.outcome is None:          # MAX_TURNS exhausted without goal/collision
        env.mark_failed()
    print(f"Final agent position: {env.agent_pos}")
    print(f"Overall simulation outcome: {env.outcome}")

    # marker file named after the outcome
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
