# -*- coding: utf-8 -*-
"""Parameters — world constants, action table, dataset / clustering settings.

Other modules do ``from .config import *``, so these names are available as
module-level globals.
"""

import numpy as np
from pathlib import Path
from datetime import datetime

from dataclasses import dataclass
from typing import Tuple

# ---------------------------------------------------------------------------
# COORDINATE CONVENTION
# We work in (row, column) coordinates. Row and column both start at 0, so
# (0, 0) is the top-left cell and, on a GRID_SIZE x GRID_SIZE grid,
# (GRID_SIZE-1, GRID_SIZE-1) is the bottom-right cell. Row increases downward
# (row 0 = top), column increases rightward (column 0 = left). An action is a
# (drow, dcol) delta added directly to (row, column):
#   up = (-1, 0)   down = (1, 0)   left = (0, -1)   right = (0, 1)   wait = (0, 0)
# There is no cartesian (x, y) anywhere in the model; plotting code that needs
# x/y does the (row, col) -> (x=col, y=row) swap locally and says so.
# ---------------------------------------------------------------------------

# ============================================================
# World
# ============================================================
GRID_SIZE = 25
NUM_CARS = GRID_SIZE#was 100 
CAR_SPEED = 1               # cells/timestep -- fixed, identical for every car
CAR_DIRECTION = (-1, 0)     # (drow, dcol) every car moves by each timestep -- "up" (row - 1)

# Default start / goal / turn-length / observation-window parameters. The
# geodesic-dataset scripts pass explicit start_pos/goal_pos/seed to
# MovingWorldEnv; the agent simulations use START_POS/GOAL_POS/FIXED_TURN_LENGTH/
# WINDOW_HALF_SIZE directly.
START_POS = (0, 0)
GOAL_POS = (GRID_SIZE - 1, GRID_SIZE - 1)
SEED = None
FIXED_TURN_LENGTH = 5
WINDOW_HALF_SIZE = 2*FIXED_TURN_LENGTH # this ensures no collision with a car
# outside the observation window

# Whether the per-turn LLM decision also gets the agent-centred hazard close-up
# image (True) or only the wide observation view (False). Read by both
# simulation scripts so the tools / no-tools arms stay matched.
USE_HAZARD_CLOSEUP = True


# ============================================================
# Actions -- a (drow, dcol) delta added directly to (row, column).
# VECTOR_TO_DELTA is the identity map: an action IS its (row, column) delta.
# It is kept (rather than removed) only so code that iterates the set of legal
# moves / looks a move up still works unchanged.
# ============================================================
VECTOR_TO_DELTA = {
    (0, 0):  (0, 0),   # wait (hold position); this has to be first in order for the BFS search to prefer Wait-Wait to Up-Down for instance
    (-1, 0): (-1, 0),  # up:    row - 1
    (1, 0):  (1, 0),   # down:  row + 1
    (0, -1): (0, -1),  # left:  column - 1
    (0, 1):  (0, 1),   # right: column + 1
}

# ============================================================
# Dataset generation
# ============================================================
MOVEMENT_GEODESIC_DATASET_ROOT = Path("movement_geodesic_dataset")
NUM_DATASET_SAMPLES = 250    # small default for fast iteration -- set to 10000 for
                            # the real dataset run
GEODESIC_LEN= FIXED_TURN_LENGTH
GEODESIC_TARGET=['random','lower right'][1]

# ---- geodesic windowing for dataset construction ----
# 'random' : one random full-inside GEODESIC_LEN-step window per geodesic;
#            geodesics shorter than GEODESIC_LEN have no full-inside window and
#            are dropped (the generator just resamples another environment).
# 'first'  : the first GEODESIC_LEN steps (legacy); short geodesics are kept and
#            padded with waits.
GEODESIC_WINDOW_MODE = 'random'
GEODESIC_WINDOW_SEED = 0   # seeds the window-offset RNG. The start/goal sampler
                          # is deliberately unseeded, so this only makes the
                          # windows reproducible given the same set of geodesics.

RUN_TIMESTAMP = datetime.now()
EXPERIMENT_ID = RUN_TIMESTAMP.strftime("%Y%m%d_%H%M%S")   # one run's output filename

# ==============
# Geodesic clustering
# =============
# default 'kmedoids' -> true PAM k-medoids via CLARA (see simutils/clustering.py
# and simutils/kmedoids_clara.py)
cluster_type=['kmeans', 'kmedoids','kmedoids_pam','diverse','diverse_then_medoid'][1]
NB_CLUSTERS=5


@dataclass
class Car:
    id: int
    position: Tuple[int, int]     # (row, col), mutated in place by the env each timestep


# No per-car direction / per-car speed: every car moves identically
# (CAR_DIRECTION, CAR_SPEED). A closing/crossing/receding distinction never
# carries any information once every car shares the same direction and speed.
