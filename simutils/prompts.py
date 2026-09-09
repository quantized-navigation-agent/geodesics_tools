# -*- coding: utf-8 -*-
"""Prompt strings for the agent / tool-description steps.

Everything is (row, column), 0-based, and EGOCENTRIC: positions are expressed as
a (drow, dcol) OFFSET from the agent's own cell, which is (0, 0). drow > 0 is
below the agent, dcol > 0 is to its right. Row increases downward, column
increases rightward. The agent is never handed an absolute position or a map
reference frame -- it reasons only about what it sees, relative to itself.
"""

from .config import GRID_SIZE, GEODESIC_LEN

_TOOL_IMG_SIZE = 2 * GEODESIC_LEN + 1   # the tool-sketch image is this many cells wide/high


# Coordinate + action convention (egocentric).
coordinate_convention = """
Coordinate convention (egocentric):
- The agent's own cell is the origin (0, 0). Every other cell is a (drow, dcol)
  offset from it: drow > 0 is BELOW the agent, dcol > 0 is to its RIGHT.
- Row increases downward (up = toward smaller row). Column increases rightward.
- Each action adds a (drow, dcol) delta to the current cell:
    up = (-1, 0)   down = (1, 0)   left = (0, -1)   right = (0, 1)   wait = (0, 0)
"""

# The traffic dynamics, stated once and precisely. The collision reasoning in the
# tool descriptions AND in the live decisions must use exactly this model.
kinematic_model = f"""
Traffic model (use exactly this):
- Every car moves UP by exactly 1 row each step (drow -1), keeping its column.
- Cars wrap: a car leaving row 0 reappears at the bottom row of the
  {GRID_SIZE} x {GRID_SIZE} grid. The bottom rows are therefore a busy re-entry
  zone -- cars keep appearing there -- so moving into / staying near the bottom
  edge is extra dangerous, but when the target is at the bottom it has to be done at some point.
- The agent does NOT wrap: the grid's edges are hard walls, only cars wrap.
  An action that would take the agent past row 0, row {GRID_SIZE - 1}, column 0,
  or column {GRID_SIZE - 1} has no effect -- the agent stays in its current
  cell for that one step while time still elapses and every car still moves.
  However it is not forbidden to use such an action, it will just not have any effect.
  A tool's stated net displacement only holds in open space, away from any
  edge; near a wall, the steps that would exit are simply absorbed and do not
  happen.
- A collision at step k = a car occupies the agent's cell at step k, OR the
  agent and a car swap cells that step (each entering the cell the other leaves).
- Relative closing speed, per step:
    * agent moving DOWN  -> the agent and an oncoming (upward) car close at 2 cells/step;
    * agent WAITING or moving SIDEWAYS -> a car below closes at 1 cell/step;
    * agent moving UP    -> moves with traffic (recedes, or swaps with the car just above).
- Consequence: a tool segment that spends k steps moving DOWN a column is
  threatened by a car anywhere up to about 2*k rows below where that segment
  starts, in that column. A k-step wait / sideways stay in a cell is threatened
  by a car up to k rows below it.
"""

env_description_prompt = f"""
You are a navigation agent on a partially observable {GRID_SIZE} x {GRID_SIZE} grid.
""" + coordinate_convention + kinematic_model + """
What you see:
- White cells: free space. Red cells: cars (moving up -- see the traffic model).
- Gray cells: not observed; they may hide cars.
- Blue cell: the agent (you). Yellow cells, if any: your past positions.
- Green cell: the goal (usually toward the bottom-right).
- The view is a window centered on you. You have NO map and NO absolute
  coordinates -- reason only about offsets from your own (blue) cell.

Objective: reach the goal with the fewest expected moves and least time.

Decision criteria, in order:
1. Critical: avoid collision with visible AND possible hidden cars.
2. Make progress toward the goal.
3. Gain information about unknown regions.
4. Avoid detours and future corrections.
"""


TOOL_DESCRIPTION_PROMPT = f"""{env_description_prompt}

You are describing ONE tool: a fixed sequence of {GEODESIC_LEN} actions applied
from the agent's current cell. When chosen, its actions run in order until a
collision, the goal, or the end of the list.

The image is a {_TOOL_IMG_SIZE} x {_TOOL_IMG_SIZE} SKETCH of the tool's shape,
centered on the start cell. It is NOT the world: it shows no cars, its size is
not the world size, and it cannot show where wait actions happen. The action
letters given to you are authoritative; use the picture only as a rough aid.

Do ALL of your reasoning in your thinking. Then output ONLY the filled template
below -- at most about 12 lines, no other prose -- and refer to actions by
letter (U/D/L/R/W), never as numeric vectors.

Method for the collision field, worked on the example tool "DDDDD" (five downs):
  At step t the agent is at offset (t, 0). A car starting at offset (a, 0) is at
  (a - t, 0) at step t.
  Same-cell collision: a - t = t  =>  a = 2, 4, 6, 8, 10.
  Swap collision at step t: the car started at (t, 0)  =>  a = 1, 3, 5, 7, 9.
  So "DDDDD" is unsafe if ANY car sits in the agent's column at drow +1..+10.
Apply the same method to this tool: for every column the path enters, and the
step it enters, work out which car offsets collide; keep the 2-cells/step
down-closing speed and the bottom-row wrap in mind; when unsure, over-state the
danger rather than under-state it.

TEMPLATE (fill every line; all offsets are (drow, dcol) from the agent's current cell):
tool: <name>
path: <step1 (dr,dc)> -> <step2 (dr,dc)> -> ... -> <step{GEODESIC_LEN} (dr,dc)>
net_offset: <(dr, dc) after the whole tool>
forbidden_car_offsets: <compact per-column ranges of car offsets that make this tool collide, e.g. "col 0: drow +1..+10 ; col +1: drow +2..+8">
bad_when: <one sentence: the car configuration in which NOT to pick this tool, stated so it stays safe even if car positions are only approximately known>
progress: <net direction and distance toward a bottom-right goal; note explicitly if it wastes steps waiting>
"""

# Short tool description: no kinematics, no collision reasoning (that is handled
# by a separate code check). Three clearly separated fields: the total
# displacement first (most important), then the compressed action list, then a
# one-line strategic idea.
TOOL_DESCRIPTION_PROMPT_SHORT = f"""You are describing ONE navigation tool: a fixed sequence of {GEODESIC_LEN}
actions applied from the agent's current cell on a {GRID_SIZE} x {GRID_SIZE}
grid whose goal is toward the bottom-right. Row increases downward, column
increases rightward. Actions, one letter per step:
  U = up (-1, 0)   D = down (1, 0)   L = left (0, -1)   R = right (0, 1)   W = wait (0, 0)

The image is a rough sketch of the tool's shape (blue = start cell); the action
letters given to you are authoritative.

Do NOT mention cars, collisions, safety, timing, danger or risk -- those are
handled separately by a code check. Output ONLY these three lines and nothing else:

total displacement: <where the agent ends up relative to where it started, in plain words, no coordinates. e.g. "one row down, no sideways change" or "one row down and four columns to the right" or "one row up, no sideways change" or "no movement">
action list: <the {GEODESIC_LEN} moves in order, merging identical consecutive moves, words: up / down / left / right / hold. e.g. "down once, then hold four steps" or "down once, then right four times" or "hold five steps">
strategy: <ONE line: the overall shape of the path e.g. "does nothing", " a single step up followed by waiting", " a staircase pattern moving steadily toward the bottom-right", "a straight run towards the right side", "a zig-zag that toward the bottom-right side", "commits sideways first, then descends", "a roughly diagonal up then left twice">
"""
