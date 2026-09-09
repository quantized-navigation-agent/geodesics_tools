# -*- coding: utf-8 -*-
"""Step 2 -- quantize the geodesics into a small set of action "tools".

Reads ``actions_truncated.npz``, builds cumulative trajectories, clusters them
(default: true PAM k-medoids via CLARA -- see ``simutils.clustering`` /
``simutils.kmedoids_clara``), turns the ``NB_CLUSTERS`` cluster centres into
action sequences, prepends the 5 elementary one-step tools, then NAMES each tool
by its action sequence (``simutils.tool_names.tool_name``: one letter per step,
U/D/L/R/W) and saves the tools as a name-keyed dict.
``plot_centers`` also writes one PNG per tool for the description step.

Outputs
  quantized_actions.npz        an .npz whose keys are the tool names ("RRWDD", ...)
                               and whose values are (GEODESIC_LEN, 2) (drow, dcol) arrays
  filled_cluster_<name>.png    one image per tool

We work in (row, column) coordinates (both 0-based; (0, 0) top-left). Trajectories
are cumulative (drow, dcol) offsets from each geodesic's start; the plots put
column on the x-axis and row on the y-axis (y inverted) -- that swap is for
plotting only.

Run after:  geodesic_dataset_generation.py
Run:        python geodesic_quantization.py
Next:       generate_tool_description.py
"""

import numpy as np
import matplotlib.pyplot as plt

from simutils.config import *
from simutils.clustering import *
from simutils.tool_names import tool_name

# ============================================================
# Load the truncated action array and build trajectories
# ============================================================
data = np.load('actions_truncated.npz')
print(data.files)
actions_ndarray = data['actions_truncated']
print(actions_ndarray.shape, actions_ndarray.dtype)

geodesic_trajectories = np.cumsum(actions_ndarray, axis=1)
print(f"Shape of geodesic_trajectories: {geodesic_trajectories.shape}")

# ---- visual check: all geodesic trajectories (as (row, col) offsets from the
#      start, which is exactly what gets clustered) ----
# plotting only: column on the x-axis, row on the y-axis, y-axis inverted so
# row 0 is at the top.
plt.figure(figsize=(3, 3))
ax = plt.gca()
for i in range(len(geodesic_trajectories)):
    row_cum = geodesic_trajectories[i, :, 0]   # cumulative drow
    col_cum = geodesic_trajectories[i, :, 1]   # cumulative dcol
    ax.plot(col_cum, row_cum, marker='o', markersize=3, linestyle='-', alpha=0.6)
plt.xlabel('column')
plt.ylabel('row')
plt.title('All Geodesic Trajectories (offset from start)')
plt.grid(True, linestyle='--', alpha=0.7)
ax.invert_yaxis()                              # plotting only
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# same, with the (0, 0) start point prepended
start_points = np.zeros((geodesic_trajectories.shape[0], 1, 2), dtype=int)
from_origin_geodesic_trajectories = np.concatenate((start_points, geodesic_trajectories), axis=1)
plt.figure(figsize=(3, 3))
ax = plt.gca()
for i in range(len(from_origin_geodesic_trajectories)):
    row_cum = from_origin_geodesic_trajectories[i, :, 0]
    col_cum = from_origin_geodesic_trajectories[i, :, 1]
    ax.plot(col_cum, row_cum, marker='o', markersize=3, linestyle='-', alpha=0.6)  # plotting only: x=col, y=row
plt.xlabel('column')
plt.ylabel('row')
plt.title('Geodesic Trajectories start=(0,0)')
plt.grid(True, linestyle='--', alpha=0.7)
ax.invert_yaxis()                              # plotting only
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


# ============================================================
# Quantization (clustering)
# ============================================================
NB_CLUSTERS = 5   # also defined in simutils.config

# default 'kmedoids' -> true PAM k-medoids via CLARA (falls back to a KMeans-based
# medoid picker if scikit-learn-extra is missing).
# other options: 'kmeans', 'kmedoids_pam', 'diverse', 'diverse_then_medoid'
cluster_type = 'kmedoids'
print(f'cluster_type={cluster_type}  (kmedoids backend: {KMEDOIDS_BACKEND})')

if(cluster_type=='kmeans'):
  centers,labels=cluster_trajectories_kmeans(geodesic_trajectories, k=NB_CLUSTERS, random_state=0)
elif(cluster_type=='kmedoids'):
  centers,labels=cluster_trajectories_medoid(geodesic_trajectories, k=NB_CLUSTERS)
elif(cluster_type=='kmedoids_pam'):
  centers,labels,medoids=kmedoids_pam(geodesic_trajectories, k=NB_CLUSTERS)
elif(cluster_type=='diverse'):
  centers,labels=cluster_trajectories_diverse(geodesic_trajectories, k=NB_CLUSTERS)
elif(cluster_type=='diverse_then_medoid'):
  centers_div, _ = cluster_trajectories_diverse(geodesic_trajectories, k=NB_CLUSTERS*2)
  centers,labels=cluster_trajectories_medoid(centers_div, k=NB_CLUSTERS)

cluster_sizes = np.bincount(labels, minlength=NB_CLUSTERS)
print(cluster_sizes)

quantized_actions = np.diff(centers, axis=1, prepend=np.zeros((centers.shape[0], 1, 2)))

# ---- prepend the 5 elementary one-step tools (one (drow, dcol) step then waits) ----
# tool 0 = all waits; tools 1..4 = one step up / down / left / right.
n_moves = quantized_actions.shape[1]
elementary_tools = np.zeros((5, n_moves, 2), dtype=quantized_actions.dtype)
elementary_tools[1, 0] = [-1, 0]   # up    then wait   (row - 1)
elementary_tools[2, 0] = [1, 0]    # down  then wait   (row + 1)
elementary_tools[3, 0] = [0, -1]   # left  then wait   (column - 1)
elementary_tools[4, 0] = [0, 1]    # right then wait   (column + 1)

quantized_actions = np.concatenate([elementary_tools, quantized_actions], axis=0)
print(quantized_actions.shape)

# ---- name each tool by its action sequence (U/D/L/R/W, one letter per step) ----
tools = {}
for actions in quantized_actions:
    name = tool_name(actions)                 # e.g. "RRWDD"
    if name in tools:
        print(f"  duplicate tool '{name}' -- keeping the first occurrence")
        continue
    tools[name] = np.asarray(actions, dtype=int)
print(f"{len(tools)} distinct tools: {list(tools)}")

# store as a name-keyed .npz: each key is a tool name, each value its (GEODESIC_LEN, 2) moves
np.savez('quantized_actions.npz', **tools)
print("quantized_actions.npz saved successfully.")


# ============================================================
# Cluster plot -- also writes filled_cluster_<name>.png
# (consumed by generate_tool_description.py)
# ============================================================
from simutils.visualization import plot_centers

plot_centers(tools)


# ============================================================
# Investigate the natural number of clusters (off by default)
# ============================================================
investigate_natural_number_of_clusters = False

# ---- Elbow method ----
if(investigate_natural_number_of_clusters):
  Ks = range(1, 21)
  costs = []
  X=geodesic_trajectories
  for k in Ks:
      centers, labels = cluster_trajectories_medoid(X, k)

      cost = 0
      Xf = X.reshape(len(X), -1)
      Cf = centers.reshape(len(centers), -1)

      for i, label in enumerate(labels):
          cost += np.linalg.norm(Xf[i] - Cf[label])

      costs.append(cost)

if(investigate_natural_number_of_clusters):
  plt.plot(Ks, costs, '-o')
  plt.xlabel("k")
  plt.ylim([0,np.max(costs)])
  plt.ylabel("Within-cluster distance")
  plt.grid()
  plt.show()

# ---- Silhouette ----
if(investigate_natural_number_of_clusters):
  from sklearn.metrics import silhouette_score

  scores = []

  Xf = X.reshape(len(X), -1)

  for k in range(2, 21):
      centers, labels = cluster_trajectories_medoid(X, k)
      scores.append(silhouette_score(Xf, labels))

  best_k = np.argmax(scores) + 2

if(investigate_natural_number_of_clusters):
  plt.plot(range(2,21), scores, '-o')
  plt.xlabel("k")
  plt.ylabel("Silhouette score")
  plt.show()
