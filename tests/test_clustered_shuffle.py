import itertools
import numbers
from builtins import zip

import numpy as np


def make_random_clusters(centers, n_samples_per_center, n_features=2, cluster_std=1.0,
                         center_ids=None, center_box=(-10.0, 10.0), random_state=None):
    """Generate isotropic Gaussian blobs for clustering.

    Parameters
    ----------
    centers : int or array of shape [n_centers, n_features]
        The number of centers to generate, or the fixed center locations.
    n_samples_per_center : int array
        Number of points for each cluster.
    n_features : int, optional (default=2)
        The number of features for each sample.
    cluster_std : float or sequence of floats, optional (default=1.0)
        The standard deviation of the clusters.
    center_ids : array of integer center ids, if None then centers will be numbered 0 .. n_centers-1
    center_box : pair of floats (min, max), optional (default=(-10.0, 10.0))
        The bounding box for each cluster center when centers are
        generated at random.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """
    rng = np.random.RandomState(random_state)

    if isinstance(centers, numbers.Integral):
        centers = np.sort(rng.uniform(center_box[0], center_box[1], \
                                      size=(centers, n_features)), axis=0)
    else:
        assert(isinstance(centers, np.ndarray))
        n_features = centers.shape[1]

    if center_ids is None:
        center_ids = np.arange(0, centers.shape[0])
        
    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.ones(len(centers)) * cluster_std

    X = []
    y = []

    n_centers = centers.shape[0]

    for i, (cid, n, std) in enumerate(zip(center_ids, n_samples_per_center, cluster_std)):
        X.append(centers[i] + rng.normal(scale=std, size=(n, n_features)))
        y += [cid] * n

    X = np.concatenate(X)
    y = np.array(y)

    return X, y


def random_clustered_shuffle(centers, n_samples_per_center, center_ids=None,cluster_std=1.0):

    if isinstance(centers, numbers.Integral):
        n_centers = centers
    else:
        assert(isinstance(centers, np.ndarray))
        n_centers = len(centers)
    
    center_box=(-float(n_centers), float(n_centers))
    X, y = make_random_clusters (centers, n_samples_per_center, n_features=1, \
                                 center_ids=center_ids, cluster_std=cluster_std, center_box=center_box)
    s = np.argsort(X,axis=0).ravel()
    return y[s].ravel()


x = random_clustered_shuffle(centers=6,center_ids=np.array([10,13,21,25,27,29]).reshape(-1,1), \
                             n_samples_per_center=np.array([1,3,10,15,7,9]))
