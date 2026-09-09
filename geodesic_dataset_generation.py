# -*- coding: utf-8 -*-
"""Step 1 -- generate the geodesic dataset.

Samples random ``MovingWorldEnv`` environments, finds each geodesic
(``simutils.geodesic.find_geodesic``), pickles the flat list of samples, and
extracts the per-geodesic quantization window as an array.

The window is one random GEODESIC_LEN-step slice of each geodesic
(GEODESIC_WINDOW_MODE == 'random'); geodesics too short for a full-inside window
are dropped and resampled. This makes the extracted set a fair sample of
optimal behaviour anywhere along a path, not just at the spawn.

We work in (row, column) coordinates (both 0-based; (0, 0) top-left). Each
sample's start/goal are (row, column) and its actions are (drow, dcol) moves
(up=(-1,0), down=(1,0), left=(0,-1), right=(0,1), wait=(0,0)).

Outputs
  movement_geodesic_dataset/<EXPERIMENT_ID>.pkl   flat list of sample dicts
                                                   (each with actions_full, actions_truncated, window_offset)
  actions_full.pkl                                 list of the variable-length full geodesics
  actions_truncated.npz                            (n_samples, GEODESIC_LEN, 2) of (drow, dcol) -- the windows

Run:   python geodesic_dataset_generation.py
Next:  geodesic_quantization.py
"""

from simutils.config import *
from simutils.dataset import *

print(f"This run's dataset will be saved under "
      f"{MOVEMENT_GEODESIC_DATASET_ROOT / (EXPERIMENT_ID + '.pkl')}")


# ============================================================
# Generate
# ============================================================
_geodesic_samples = generate_geodesic_samples(NUM_DATASET_SAMPLES)
geodesic_dataset_path = save_geodesic_samples(_geodesic_samples)
print(f"Pickled dataset written to: {geodesic_dataset_path}")

full_geodesics_path = save_full_geodesics(_geodesic_samples)
print(f"Full geodesic action lists written to: {full_geodesics_path}")

geodesic_dataset = GeodesicPickleDataset(geodesic_dataset_path)
geodesic_dataloader = DataLoader(
    geodesic_dataset, batch_size=32, shuffle=True, collate_fn=collate_geodesics,
)

# ---- quick look ----
print(f"len(geodesic_dataset) = {len(geodesic_dataset)}")
print("First sample:")
print(geodesic_dataset[0])

first_batch = next(iter(geodesic_dataloader))
print(f"\nFirst batch from geodesic_dataloader: {len(first_batch)} samples")
print("First sample of that batch:")
print(first_batch[0])

print((len(_geodesic_samples), _geodesic_samples[31]['actions_truncated']))


# ============================================================
# Extract the quantization windows as (n_samples, GEODESIC_LEN, 2)
# (in 'random' mode every window already has length GEODESIC_LEN; in 'first'
# mode short geodesics were padded by generate_geodesic_samples)
# ============================================================
all_actions_truncated = []
for i in range(len(_geodesic_samples)):
  sample = _geodesic_samples[i]['actions_truncated']
  if( len(sample)==GEODESIC_LEN):
      all_actions_truncated.append(sample)
  else:
      print(f"pb at idx={i}, sample={sample}")

actions_ndarray = np.array(all_actions_truncated)
print(f"Shape of the extracted actions array: {actions_ndarray.shape}")

np.savez('actions_truncated.npz', actions_truncated=actions_ndarray)
print("actions_truncated.npz saved successfully.")
