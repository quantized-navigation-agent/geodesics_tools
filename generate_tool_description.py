# -*- coding: utf-8 -*-
"""Step 3 -- generate a natural-language description for each quantized tool.

For every tool in ``quantized_actions.npz`` (a name-keyed .npz written by
``geodesic_quantization.py``) it sends the tool image
(``filled_cluster_<name>.png``) plus the explicit action list to the vision LLM
(``simutils.vision.vision_query`` -> Ollama) and collects the description.

Tools are identified by name, not by index: a name is one letter per step
(U/D/L/R/W), e.g. "RRWDD".

Coordinates / actions are (row, column): each tool action is a (drow, dcol)
pair (up=(-1,0), down=(1,0), left=(0,-1), right=(0,1), wait=(0,0)). The prompt
(simutils.prompts.TOOL_DESCRIPTION_PROMPT) states this to the model.

Outputs
  tool_description.pkl                              dict {tool name -> description}
  <safe_model_name>_list_of_tool_descriptions.txt   e.g. qwen3_6_35b_a3b_list_of_tool_descriptions.txt
                                                    (this is what run_agent_simulation_v2.py reads)

Run after:  geodesic_quantization.py
Run:        python generate_tool_description.py
"""

import os
import pickle

import numpy as np

from simutils.prompts import *
from simutils.vision import *
from simutils.vision_qwen import safe_model_name

_LETTER_WORD = {"U": "up", "D": "down", "L": "left", "R": "right", "W": "wait"}

_npz = np.load("quantized_actions.npz")
tools = {name: _npz[name] for name in _npz.files}   # {name -> (GEODESIC_LEN, 2) (drow, dcol)}
print(f"Loaded {len(tools)} tools: {list(tools)}")

OUT_FILE = f"{safe_model_name}_list_of_tool_descriptions.txt"
PKL_FILE = "tool_description.pkl"   # {tool name -> short description}; what the simulations read

if os.path.isfile(PKL_FILE):  # load the description dict, do not regenerate
    print(f'WARNING : {PKL_FILE} exists -- loaded, no new description generated')
    with open(PKL_FILE, "rb") as f:
        tool_descriptions_dict = pickle.load(f)
else:  # generate descriptions
    tool_descriptions_dict = {}
    tool_chat_response_dict = {}
    for name, actions in tools.items():
        print(f"""tool description for tool {name}:{'*' * 10}\n""")
        img_filename = f"filled_cluster_{name}.png"

        spelled = ", ".join(f"{c}={_LETTER_WORD[c]}" for c in name)  # e.g. "R=right, R=right, W=wait, D=down, D=down"

        # Ollama `requests` path via simutils.vision.vision_query
        desc = vision_query(
            filename=img_filename,
            message=TOOL_DESCRIPTION_PROMPT_SHORT +
                    f"\nThis tool is named {name}. Its {len(name)} actions in order: {spelled}.",
        )
        print(desc)
        tool_descriptions_dict[name] = desc
        tool_chat_response_dict[name] = vision_query.response

    with open(PKL_FILE, "wb") as f:
        pickle.dump(tool_descriptions_dict, f)
    print(f"{PKL_FILE} saved successfully.")

# ---- human-readable .txt, rebuilt from the dict every run ----
list_of_tool_descriptions = """
  Each tool is a list of actions that affect the position of the agent. When a
  tool is selected all its actions are implemented until collision, reaching the
  goal, or exhaustion of the list. Each tool is named by its actions, one letter
  per step: U=up (-1,0), D=down (1,0), L=left (0,-1), R=right (0,1), W=wait (0,0).

  Available tools are:""" + f"""
  {'\n'.join(
      f'\n {'='*20}tool {k} {'='*20}:\n {v} \n {'='*20}end description tool {k} {'='*20}:\n'
      for k, v in tool_descriptions_dict.items()
  )}
  """
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(list_of_tool_descriptions)
print(f"{OUT_FILE} saved successfully.")

print(list_of_tool_descriptions)
