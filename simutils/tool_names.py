# -*- coding: utf-8 -*-
"""Tool names.

A tool is named by its action sequence, one capital letter per step (one letter
per elementary action):

    U = up    (-1, 0)      D = down  ( 1, 0)
    L = left  ( 0, -1)     R = right ( 0, 1)
    W = wait  ( 0, 0)

e.g. right, right, wait, down, down  ->  "RRWDD".

We work in (row, column) coordinates; an action is a (drow, dcol) delta.
"""

import numpy as np

ACTION_TO_CHAR = {
    (-1, 0): "U",
    (1, 0):  "D",
    (0, -1): "L",
    (0, 1):  "R",
    (0, 0):  "W",
}
CHAR_TO_ACTION = {c: a for a, c in ACTION_TO_CHAR.items()}


def tool_name(actions) -> str:
    """(n_steps, 2) array / list of (drow, dcol) steps -> name string like 'RRWDD'."""
    chars = []
    for step in actions:
        key = (int(round(step[0])), int(round(step[1])))
        if key not in ACTION_TO_CHAR:
            raise ValueError(
                f"tool_name: step {key} is not an elementary action "
                f"{sorted(ACTION_TO_CHAR)} -- the cluster centres are not valid "
                f"action sequences (use cluster_type='kmedoids')."
            )
        chars.append(ACTION_TO_CHAR[key])
    return "".join(chars)


def tool_actions(name: str) -> np.ndarray:
    """'RRWDD' -> (len(name), 2) int array of (drow, dcol) steps."""
    return np.array([CHAR_TO_ACTION[c] for c in name], dtype=int)


def tool_column_reach(tools) -> tuple:
    """Smallest / largest cumulative column offset (relative to the start cell,
    column 0 included) over every tool.

    `tools` is a name -> (n_steps, 2) mapping (or an iterable of such arrays).
    A car in a column outside this band can never reach any tool's path, so it
    is the natural column span for the collision-hazard crop.
    """
    seqs = tools.values() if hasattr(tools, "values") else tools
    lo, hi = 0, 0
    for actions in seqs:
        cc = np.cumsum(np.asarray(actions)[:, 1])
        lo = min(lo, int(cc.min()))
        hi = max(hi, int(cc.max()))
    return lo, hi
