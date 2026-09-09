# -*- coding: utf-8 -*-
"""simutils -- shared environment, geodesic, quantization and vision code.

## Coordinate convention

We work in (row, column) coordinates. Row and column both start at 0, so (0, 0)
is the top-left cell and, on a GRID_SIZE x GRID_SIZE grid,
(GRID_SIZE-1, GRID_SIZE-1) is the bottom-right cell. Row increases downward,
column increases rightward. An action is a (drow, dcol) delta added directly to
(row, column):  up=(-1,0)  down=(1,0)  left=(0,-1)  right=(0,1)  wait=(0,0).
There is no cartesian x/y; plotting code that needs an (x, y) frame does the
``x = column, y = row`` swap locally and says so.

## Module layout

- ``config``          : all parameters / world constants + the ``Car`` dataclass
- ``env_utils``       : window bounds, swept-collision math, result dataclasses
- ``env``             : ``MovingWorldEnv``
- ``geodesic``        : ``car_position_at_time`` + ``find_geodesic`` (shortest-path search)
- ``dataset``         : env sampling, geodesic sample generation, pickling, torch ``Dataset``
- ``clustering``      : the trajectory-clustering routines
- ``kmedoids_clara``  : CLARA / PAM k-medoids (default clustering)
- ``tool_names``      : name a tool by its action sequence (U/D/L/R/W), e.g. "RRWDD"
- ``collision_check`` : replay a tool against the currently observed cars, report collisions
- ``visualization``   : ``plot_centers`` (name-keyed tools dict), ``run_env_walk``, ``plot_single_trajectory``
- ``prompts``         : the environment / tool-description / decision prompt strings
- ``vision``          : single-image ``vision_query`` wrapper (delegates to ``vision_qwen``)
- ``vision_qwen``     : Ollama ``/api/generate`` vision query over ``requests`` (no ``openai``)
- ``log_stderr_stdout`` : stdout/stderr capture helper

Pipeline scripts next to this package (run in order):
  geodesic_dataset_generation.py -> geodesic_quantization.py -> generate_tool_description.py

Agent simulations (one episode per main() call, NR_RUNS of them per invocation):
  run_agent_simulation_v2.py            LLM picks a tool from the library
  run_agent_simulation_notools_v1.py    LLM proposes the next 5 raw actions
  run_sim_wtools_coll_detect_v1.py      LLM picks a tool, but a code collision check
                                        removes unsafe tools first
  run_sim_notools_heuristic_v1.py       no LLM: greedy 1-turn planner over the full
                                        5-action space, visible-car collision filter
"""
