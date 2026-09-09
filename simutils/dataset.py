# -*- coding: utf-8 -*-
"""Sample envs and generate geodesics.

Random start/goal sampling, one-sample and bulk geodesic generation, pickling,
and the PyTorch ``Dataset`` wrapper.

All coordinates are (row, column) (both 0-based; (0, 0) top-left). Each sample's
``start_pos`` / ``goal_pos`` are (row, column); ``actions_full`` is the whole
geodesic and ``actions_truncated`` is the GEODESIC_LEN-step window fed to
quantization (see GEODESIC_WINDOW_MODE) -- both lists of (drow, dcol) moves.
"""

import pickle
import random as _random
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from .config import (
    GRID_SIZE, NUM_CARS, CAR_DIRECTION, CAR_SPEED, GEODESIC_LEN, GEODESIC_TARGET,
    NUM_DATASET_SAMPLES, MOVEMENT_GEODESIC_DATASET_ROOT, EXPERIMENT_ID,
    GEODESIC_WINDOW_MODE, GEODESIC_WINDOW_SEED,
)
from .env import MovingWorldEnv
from .geodesic import find_geodesic

# Seeded RNG for the window offset, separate from the (unseeded) start/goal RNG.
_window_rng = _random.Random(GEODESIC_WINDOW_SEED)


def random_distinct_start_goal(size: int):
    """Uses the plain (unseeded) random module -- distinct from
    MovingWorldEnv's own seeded self.rng, which only controls car spawn."""
    while True:
        start = (_random.randrange(size), _random.randrange(size))
        if(GEODESIC_TARGET=='random'):
          goal = (_random.randrange(size), _random.randrange(size))
        elif (GEODESIC_TARGET=='lower right'):
          goal = (size-1, size-1)

        if start != goal:
            return start, goal


def collate_geodesics(batch: List[dict]) -> List[dict]:
    """
    Returns the batch unchanged (a plain list of per-sample dicts) rather
    than stacking into tensors -- car count and actions_truncated length are
    both ragged across samples, so there's no single fixed shape to stack
    into without a padding decision this dataset deliberately isn't making
    (training is handled elsewhere; this just needs to load correctly).
    """
    return list(batch)


def generate_geodesic_sample() -> Optional[dict]:
    """
    Samples one random environment, finds its geodesic, and returns the info
    needed to place it in the quantization dataset: grid_size,
    car_spawn_positions, car_direction, car_speed, start_pos, goal_pos,
    actions_full (the whole path), actions_truncated (the GEODESIC_LEN-step
    window fed to quantization) and window_offset.

    With GEODESIC_WINDOW_MODE == 'random' the window is one random full-inside
    slice of the geodesic; a geodesic shorter than GEODESIC_LEN has no
    full-inside window and returns None (the caller resamples). With 'first' the
    window is the first GEODESIC_LEN steps and short geodesics are kept (the
    caller pads them). Also returns None if the goal turned out unreachable.
    """
    start_pos, goal_pos = random_distinct_start_goal(GRID_SIZE)
    env = MovingWorldEnv(
        size=GRID_SIZE, num_cars=NUM_CARS,
        start_pos=start_pos, goal_pos=goal_pos, seed=None,
    )

    actions = find_geodesic(env)
    if actions is None:
        return None

    car_spawn_positions = [car.position for car in env.cars]

    if len(actions) < GEODESIC_LEN:
        if GEODESIC_WINDOW_MODE == 'random':
            return None                       # no full-inside window -> drop / resample
        window, offset = list(actions), 0     # 'first' mode: keep, caller pads
    elif GEODESIC_WINDOW_MODE == 'random':
        offset = _window_rng.randint(0, len(actions) - GEODESIC_LEN)
        window = list(actions[offset:offset + GEODESIC_LEN])
    else:  # 'first'
        offset = 0
        window = list(actions[:GEODESIC_LEN])

    # Cheap correctness sanity check: replay the geodesic from the fresh env up
    # to the end of the window and confirm it stays collision-free (find_geodesic
    # guarantees this; this also covers windows that do not start at t=0).
    for action in actions[:offset + len(window)]:
        result = env.advance_one_timestep(action)
        assert not result.collision, "find_geodesic returned a colliding action sequence"

    return {
        "grid_size": GRID_SIZE,
        "car_spawn_positions": car_spawn_positions,
        "car_direction": CAR_DIRECTION,
        "car_speed": CAR_SPEED,
        "start_pos": start_pos,
        "goal_pos": goal_pos,
        "actions_full": list(actions),
        "actions_truncated": window,
        "window_offset": offset,
    }


def generate_geodesic_samples(num_samples: int = NUM_DATASET_SAMPLES) -> List[dict]:
    """Resamples a fresh random environment whenever generate_geodesic_sample
    reports None (goal unreachable, or -- in 'random' window mode -- a geodesic
    too short for a full-inside window), until exactly num_samples are collected.
    In 'first' window mode geodesics shorter than GEODESIC_LEN are padded with
    (0,0) here."""
    samples: List[dict] = []
    skipped = 0
    with tqdm(total=num_samples, desc="Generating geodesics") as pbar:
      while len(samples) < num_samples:
          sample = generate_geodesic_sample()
          if sample is None:
              skipped += 1
              continue
          if len(sample["actions_truncated"]) < GEODESIC_LEN:
            #pad with zero = wait action
            sample["actions_truncated"] = sample["actions_truncated"] + [(0, 0)] * (GEODESIC_LEN - len(sample["actions_truncated"]))
#              skipped += 1
#              continue

          samples.append(sample)
          pbar.update(1)
          pbar.set_postfix(skipped=skipped,attempts=len(samples) + skipped)
    print(f"\n generated {len(samples)} samples ({skipped} unreachable environments skipped)")
    return samples


def save_geodesic_samples(samples: List[dict], root: Path = MOVEMENT_GEODESIC_DATASET_ROOT) -> Path:
    """Pickles the flat sample list as a single file -- one pickle per run,
    named by EXPERIMENT_ID so re-running never overwrites a previous dataset."""
    root.mkdir(parents=True, exist_ok=True)
    pickle_path = root / f"{EXPERIMENT_ID}.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(samples, f)
    return pickle_path


def save_full_geodesics(samples: List[dict], path: str = "actions_full.pkl") -> str:
    """Pickle just the variable-length full geodesic action lists
    (``[s["actions_full"] for s in samples]``), for later analysis. The run
    pickle from save_geodesic_samples already contains them too, alongside the
    per-sample env metadata; this is the lightweight standalone copy."""
    with open(path, "wb") as f:
        pickle.dump([s["actions_full"] for s in samples], f)
    return path


class GeodesicPickleDataset(Dataset):
    """
    Loads the flat list of sample dicts save_geodesic_samples wrote. Each
    sample carries the full geodesic (``actions_full``), the GEODESIC_LEN-step
    window used for quantization (``actions_truncated``, at ``window_offset``),
    and the env metadata needed to reconstruct it.
    """

    def __init__(self, pickle_path: Path):
        with open(pickle_path, "rb") as f:
            self.samples: List[dict] = pickle.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
