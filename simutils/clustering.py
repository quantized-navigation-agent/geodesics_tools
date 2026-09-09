# -*- coding: utf-8 -*-
"""Routines to cluster trajectories.

We use NB_CLUSTERS clusters, defined in ``config``.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from scipy.spatial.distance import cdist


def cluster_trajectories_kmeans(X, k, random_state=0):
    """
    X: numpy array of shape (n_samples, 5, 2)
    k: number of clusters

    Returns
    -------
    centers : array of shape (k, 5, 2)
    labels  : array of shape (n_samples,)
    """

    n_samples, T, D = X.shape

    # Flatten trajectories to vectors of length 10
    X_flat = X.reshape(n_samples, T * D)

    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=20
    )
    labels = kmeans.fit_predict(X_flat)

    # Reshape centers back to trajectory format
    centers = kmeans.cluster_centers_.reshape(k, T, D)

    return centers, labels

def kmedoids_pam(X, k, max_iter=100):
    n = len(X)
    Xf = X.reshape(n, -1)

    D = pairwise_distances(Xf)

    medoids = np.random.choice(n, k, replace=False)

    for _ in range(max_iter):

        labels = np.argmin(D[:, medoids], axis=1)

        new_medoids = medoids.copy()

        for c in range(k):
            cluster = np.where(labels == c)[0]

            if len(cluster) == 0:
                continue

            intra = D[np.ix_(cluster, cluster)]
            new_medoids[c] = cluster[np.argmin(intra.sum(axis=1))]

        if np.array_equal(new_medoids, medoids):
            break

        medoids = new_medoids

    labels = np.argmin(D[:, medoids], axis=1)

    return X[medoids], labels, medoids

#Note  : Kmedoids from scikit-learn-extra is not working
#pip install
#if(False):#old version not working
#  #from sklearn_extra.cluster import KMedoids
#  def cluster_trajectories_medoids(X, k):
#      X_flat = X.reshape(len(X), -1)
#
#      model = KMedoids(          n_clusters=k,          metric="euclidean",          random_state=0      )
#
#      labels = model.fit_predict(X_flat)
#      centers = X[model.medoid_indices_]
#
#      return centers, labels

def cluster_trajectories_medoid(X, k):
    Xf = X.reshape(len(X), -1)

    km = KMeans(n_clusters=k, n_init=20, random_state=0)
    labels = km.fit_predict(Xf)

    centers = []

    for c in range(k):
        idx = np.where(labels == c)[0]

        D = pairwise_distances(Xf[idx])
        medoid = idx[np.argmin(D.sum(axis=1))]

        centers.append(X[medoid])

    return np.array(centers), labels


def cluster_trajectories_diverse(X, k, random_state=0):
    """
    X: numpy array of shape (n_samples, 5, 2)
    k: number of prototypes

    Returns
    -------
    centers : array of shape (k, 5, 2)
    labels  : array of shape (n_samples,)
    """

    rng = np.random.default_rng(random_state)

    n_samples, T, D = X.shape
    X_flat = X.reshape(n_samples, T * D)

    # first center
    centers_idx = [rng.integers(n_samples)]

    # distance matrix
    Dmat = cdist(X_flat, X_flat)

    # farthest-point sampling
    while len(centers_idx) < k:
        dmin = Dmat[:, centers_idx].min(axis=1)
        dmin[centers_idx] = -1  # avoid reselection
        centers_idx.append(np.argmax(dmin))

    centers_idx = np.array(centers_idx)

    # centers are actual trajectories
    centers = X[centers_idx]

    # assign each sample to nearest center
    labels = cdist(X_flat, X_flat[centers_idx]).argmin(axis=1)

    return centers, labels

#tests with clustering
#centers2, labels2=cluster_trajectories_diverse(geodesic_trajectories, k=NB_CLUSTERS)
#plot_centers(centers2)
#
#centers1,_=cluster_trajectories_medoid(geodesic_trajectories, k=NB_CLUSTERS*4)
##centers2, labels = cluster_trajectories_medoid(centers1, k=NB_CLUSTERS)
#centers2, labels = cluster_trajectories_diverse(centers1, k=NB_CLUSTERS)
#
#centers11, _ = cluster_trajectories_diverse(geodesic_trajectories, k=NB_CLUSTERS*2)
#centers21,labels=cluster_trajectories_medoid(centers11, k=NB_CLUSTERS+1)
#plot_centers(centers21)


# ---------------------------------------------------------------------------
# Default k-medoids backend.
#
# `cluster_type='kmedoids'` (the default in geodesic_quantization.py) should
# use true PAM k-medoids (CLARA), not the KMeans-then-pick-medoid
# `cluster_trajectories_medoid` defined above. When scikit-learn-extra is
# installed we import CLARA's implementation from simutils/kmedoids_clara.py --
# same `(X, k) -> (centers, labels)` signature -- and it REPLACES the name
# `cluster_trajectories_medoid`. If the package is missing we keep the
# KMeans-based version above as a fallback.
# ---------------------------------------------------------------------------
_cluster_trajectories_medoid_kmeans = cluster_trajectories_medoid  # fallback, kept reachable

try:
    from .kmedoids_clara import cluster_trajectories_medoid  # noqa: F811  (CLARA / PAM)
    KMEDOIDS_BACKEND = "clara"
except Exception as _clara_err:  # scikit-learn-extra not installed / incompatible
    KMEDOIDS_BACKEND = f"kmeans-medoid-fallback ({type(_clara_err).__name__}: {_clara_err})"
