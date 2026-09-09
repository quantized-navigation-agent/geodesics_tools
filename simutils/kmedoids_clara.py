"""kmedoids utility : implements CLARA PAM medoid 
"""

#requires : sklearn_extra 
#install with : !pip install -q scikit-learn-extra

# ideally :
import numpy as np
import time
from sklearn_extra.cluster import CLARA

def cluster_trajectories_medoid(X, k):
    Xf = X.reshape(len(X), -1)

    clara = CLARA(
        n_clusters=k,
        init="build",
        n_sampling=min(1_000, len(X)),
        n_sampling_iter=20,
        max_iter=100,
        random_state=42,
    ).fit(Xf)

    medoid_idx = clara.medoid_indices_
    medoids = X[medoid_idx]

    return medoids, clara.labels_

if(False):
  rng = np.random.default_rng(0)

  for n in (100, 1_000,10_000):
      X = rng.normal(size=(n, 5, 2))

      t0 = time.perf_counter()
      medoids, labels = cluster_trajectories_medoid(X, k=5)
      dt = time.perf_counter() - t0

      print(f"n={n:>4}: {dt:.3f}s | medoids={medoids.shape} | labels={labels.shape}")


""" Usage example 
dataset_actions= np.load("actions_truncated.npz")
vector_from_npz=dataset_actions[dataset_actions.files[0]]
print(vector_from_npz.shape)

trajectories=np.cumsum(vector_from_npz,axis=1)
medoids, labels = cluster_trajectories_medoid(trajectories, k=5)
print(medoids.shape, labels.shape)

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(3,3))
for ii in range(medoids.shape[0]):
  # Prepend a zero to the x-coordinates and y-coordinates
  x_coords = np.insert(medoids[ii,:,0], 0, 0)
  y_coords = np.insert(medoids[ii,:,1], 0, 0)
  plt.plot(x_coords, y_coords,"-o",alpha=1.,linewidth=4)
plt.xticks([])
plt.yticks([])
plt.scatter([0],[0],s=200,color='black',marker='*')
plt.box(False)
plt.savefig("medoids.pdf")
plt.show()

"""