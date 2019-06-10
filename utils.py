from __future__ import division, absolute_import
from builtins import map, range, object, zip, input, str, next
from past.utils import old_div
from collections import defaultdict, Iterable, namedtuple
import sys, os.path, string, time, gc, math, datetime, numbers, itertools
import copy, pprint, logging
import yaml
import numpy as np
import scipy
from scipy import sparse


class Struct(object):
    def __init__(self, **items):
        self.__dict__.update(items)

    def update(self, items):
        self.__dict__.update(items)

    def __call__(self):
        return self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]


class Context(object):
    """
    A container replacement for global variables to be shared and modified by any function in a module.
    """

    def __init__(self, namespace_dict=None, **kwargs):
        self.update(namespace_dict, **kwargs)

    def update(self, namespace_dict=None, **kwargs):
        """
        Converts items in a dictionary (such as globals() or locals()) into context object internals.
        :param namespace_dict: dict
        """
        if namespace_dict is not None:
            self.__dict__.update(namespace_dict)
        self.__dict__.update(kwargs)

    def __call__(self):
        return self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]


class IncludeLoader(yaml.Loader):
    """
    YAML loader with `!include` handler.
    """

    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        yaml.Loader.__init__(self, stream)

    def include(self, node):
        """

        :param node:
        :return:
        """
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, IncludeLoader)


IncludeLoader.add_constructor('!include', IncludeLoader.include)


def config_logging(verbose):
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARN)


def get_root_logger():
    logger = logging.getLogger('dentate')
    return logger


def get_module_logger(name):
    logger = logging.getLogger('%s' % name)
    return logger


def get_script_logger(name):
    logger = logging.getLogger('dentate.%s' % name)
    return logger


# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)


def write_to_yaml(file_path, data, convert_scalars=False):
    """

    :param file_path: str (should end in '.yaml')
    :param data: dict
    :param convert_scalars: bool
    :return:
    """
    with open(file_path, 'w') as outfile:
        if convert_scalars:
            data = nested_convert_scalars(data)
        yaml.dump(data, outfile, default_flow_style=False)


def read_from_yaml(file_path, include_loader=None):
    """

    :param file_path: str (should end in '.yaml')
    :return:
    """
    if os.path.isfile(file_path):
        with open(file_path, 'r') as stream:
            if include_loader is None:
                Loader = yaml.FullLoader
            else:
                Loader = include_loader
            data = yaml.load(stream, Loader=Loader)
        return data
    else:
        raise IOError('read_from_yaml: invalid file_path: %s' % file_path)


def print_param_dict_like_yaml(param_dict, digits=6):
    """
    Assumes a flat dict with int or float values.
    :param param_dict: dict
    :param digits: int
    """
    for param_name, param_val in viewitems(param_dict):
        if isinstance(param_val, int):
            print('%s: %s' % (param_name, param_val))
        else:
            print('%s: %.*E' % (param_name, digits, param_val))


def nested_convert_scalars(data):
    """
    Crawls a nested dictionary, and converts any scalar objects from numpy types to python types.
    :param data: dict
    :return: dict
    """
    if isinstance(data, dict):
        for key in data:
            data[key] = nested_convert_scalars(data[key])
    elif isinstance(data, Iterable) and not isinstance(data, (str, tuple)):
        data = list(data)
        for i in range(len(data)):
            data[i] = nested_convert_scalars(data[i])
    elif hasattr(data, 'item'):
        data = data.item()
    return data


def list_index(element, lst):
    """

    :param element:
    :param lst:
    :return:
    """
    try:
        index_element = lst.index(element)
        return index_element
    except ValueError:
        return None


def list_find(f, lst):
    """

    :param f:
    :param lst:
    :return:
    """
    i = 0
    for x in lst:
        if f(x):
            return i
        else:
            i = i + 1
    return None


def list_find_all(f, lst):
    """

    :param f:
    :param lst:
    :return:
    """
    i = 0
    res = []
    for i, x in enumerate(lst):
        if f(x):
            res.append(i)
    return res


def list_argsort(f, seq):
    """
    http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3383106#3383106
    lambda version by Tony Veijalainen
    :param f:
    :param seq:
    :return:
    """
    return [i for i, x in sorted(enumerate(seq), key=lambda x: f(x[1]))]


def viewitems(obj, **kwargs):
    """
    Function for iterating over dictionary items with the same set-like
    behaviour on Py2.7 as on Py3.

    Passes kwargs to method."""
    func = getattr(obj, "viewitems", None)
    if not func:
        func = obj.items
    return func(**kwargs)


def viewkeys(obj, **kwargs):
    """
    Function for iterating over dictionary keys with the same set-like
    behaviour on Py2.7 as on Py3.

    Passes kwargs to method."""
    func = getattr(obj, "viewkeys", None)
    if not func:
        func = obj.keys
    return func(**kwargs)


def viewvalues(obj, **kwargs):
    """
    Function for iterating over dictionary values with the same set-like
    behaviour on Py2.7 as on Py3.

    Passes kwargs to method."""
    func = getattr(obj, "viewvalues", None)
    if func is not None:
        func = obj.values
    return func(**kwargs)


def zip_longest(*args, **kwds):
    if hasattr(itertools, 'izip_longest'):
        return itertools.izip_longest(*args, **kwds)
    else:
        return itertools.zip_longest(*args, **kwds)


def make_geometric_graph(x, y, z, edges):
    """ Builds a NetworkX graph with xyz node coordinates and the node indices
        of the end nodes.

        Parameters
        -----------
        x: ndarray
            x coordinates of the points
        y: ndarray
            y coordinates of the points
        z: ndarray
            z coordinates of the points
        edges: the (2, N) array returned by compute_delaunay_edges()
            containing node indices of the end nodes. Weights are applied to
            the edges based on their euclidean length for use by the MST
            algorithm.

        Returns
        ---------
        g: A NetworkX undirected graph

        Notes
        ------
        We don't bother putting the coordinates into the NX graph.
        Instead the graph node is an index to the column.
    """
    import networkx as nx
    xyz = np.array((x, y, z))

    def euclidean_dist(i, j):
        d = xyz[:, i] - xyz[:, j]
        return np.sqrt(np.dot(d, d))

    g = nx.Graph()
    for i, j in edges:
        g.add_edge(i, j, weight=euclidean_dist(i, j))
    return g


def random_choice_w_replacement(ranstream, n, p):
    """

    :param ranstream:
    :param n:
    :param p:
    :return:
    """
    return ranstream.multinomial(n, p.ravel())


def make_random_clusters(centers, n_samples_per_center, n_features=2, cluster_std=1.0, center_ids=None,
                         center_box=(-10.0, 10.0), random_seed=None):
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
    random_seed : int or None, optional (default=None)
        If int, random_seed is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    Examples
    --------
    >>> X, y = make_random_clusters (centers=6, n_samples_per_center=np.array([1,3,10,15,7,9]), n_features=1, \
                                     center_ids=np.array([10,13,21,25,27,29]).reshape(-1,1), cluster_std=1.0, \
                                     center_box=(-10.0, 10.0))
    >>> print(X.shape)
    (45, 1)
    >>> y
    array([10, 13, 13, 13, ..., 29, 29, 29])
    """
    rng = np.random.RandomState(random_seed)

    if isinstance(centers, numbers.Integral):
        centers = np.sort(rng.uniform(center_box[0], center_box[1], \
                                      size=(centers, n_features)), axis=0)
    else:
        assert (isinstance(centers, np.ndarray))
        n_features = centers.shape[1]

    if center_ids is None:
        center_ids = np.arange(0, centers.shape[0])

    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.ones(len(centers)) * cluster_std

    X = []
    y = []

    n_centers = centers.shape[0]

    for i, (cid, n, std) in enumerate(zip(center_ids, n_samples_per_center, cluster_std)):
        if n > 0:
            X.append(centers[i] + rng.normal(scale=std, size=(n, n_features)))
            y += [cid] * n

    X = np.concatenate(X)
    y = np.array(y)

    return X, y


def random_clustered_shuffle(centers, n_samples_per_center, center_ids=None, cluster_std=1.0, center_box=(-1.0, 1.0),
                             random_seed=None):
    """Generates a Gaussian random clustering given a number of cluster
    centers, samples per each center, optional integer center ids, and
    cluster standard deviation.

    Parameters
    ----------
    centers : int or array of shape [n_centers]
        The number of centers to generate, or the fixed center locations.
    n_samples_per_center : int array
        Number of points for each cluster.
    cluster_std : float or sequence of floats, optional (default=1.0)
        The standard deviation of the clusters.
    center_ids : array of integer center ids, if None then centers will be numbered 0 .. n_centers-1
    random_seed : int or None, optional (default=None)
        If int, random_seed is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    >>> x = random_clustered_shuffle(centers=6,center_ids=np.array([10,13,21,25,27,29]).reshape(-1,1), \
                                     n_samples_per_center=np.array([1,3,10,15,7,9]))
    >>> array([10, 13, 13, 25, 13, 29, 21, 25, 27, 21, 27, 29, 25, 25, 25, 21, 29,
               27, 25, 21, 29, 25, 25, 25, 25, 29, 21, 25, 21, 29, 29, 29, 21, 25,
               29, 21, 27, 27, 21, 27, 25, 21, 25, 27, 25])
    """

    if isinstance(centers, numbers.Integral):
        n_centers = centers
    else:
        assert (isinstance(centers, np.ndarray))
        n_centers = len(centers)

    X, y = make_random_clusters(centers, n_samples_per_center, n_features=1, \
                                center_ids=center_ids, cluster_std=cluster_std, center_box=center_box, \
                                random_seed=random_seed)
    s = np.argsort(X, axis=0).ravel()
    return y[s].ravel()


def NamedTupleWithDocstring(docstring, *ntargs):
    """
    A convenience wrapper to add docstrings to named tuples. This is only needed in
    python 2, where __doc__ is not writeable.
    https://stackoverflow.com/questions/1606436/adding-docstrings-to-namedtuples
    """
    nt = namedtuple(*ntargs)

    class NT(nt):
        __doc__ = docstring
        __slots__ = ()  ## disallow mutable slots in order to keep performance advantage of tuples

    return NT


def partitionn(items, predicate=int, n=2):
    """
    Filter an iterator into N parts lazily
    http://paddy3118.blogspot.com/2013/06/filtering-iterator-into-n-parts-lazily.html
    """
    tees = itertools.tee(((predicate(item), item)
                          for item in items), n)
    return ((lambda i: (item for pred, item in tees[i] if pred == i))(x)
            for x in range(n))


def generator_peek(iterable):
    """
    If the iterable is empty, return None, otherwise return a tuple with the
    first element and the iterable with the first element attached back.
    """
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first, itertools.chain([first], iterable)


def generator_ifempty(iterable):
    """
    If the iterable is empty, return None, otherwise return the
    iterable with the first element attached back.
    """
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return itertools.chain([first], iterable)


def compose_iter(f, iters):
    """
    Given a function and a tuple of iterators, apply the function to
    the first iterator in the tuple, and returns the next element from
    the second iterator in the tuple.
    """
    x = next(iters[0])
    f(x)
    yield next(iters[1])


def profile_memory(logger):
    from guppy import hpy
    hprof = hpy()
    logger.info(hprof.heap())


def update_bins(bins, binsize, *xs):
    idxs = tuple(math.floor(old_div(x, binsize)) for x in xs)
    if idxs in bins:
        bins[idxs] += 1
    else:
        bins[idxs] = 1


def finalize_bins(bins, binsize):
    bin_keys = zip_longest(*viewkeys(bins))
    bin_ranges = [(int(min(ks)), int(max(ks))) for ks in bin_keys]
    dims = tuple((imax - imin + 1) for imin, imax in bin_ranges)
    if len(dims) > 1:
        grid = sparse.dok_matrix(dims, dtype=np.int)
    else:
        grid = np.zeros(dims)
    bin_edges = [[binsize * k for k in range(imin, imax + 1)] for imin, imax in bin_ranges]
    for i in bins:
        idx = tuple([int(ii - imin) for ii, (imin, imax) in zip(i, bin_ranges)])
        grid[idx] = bins[i]
    result = tuple([grid] + [np.asarray(edges) for edges in bin_edges])
    return result


def merge_bins(bins1, bins2, datatype):
    for i, count in viewitems(bins2):
        if i in bins1:
            bins1[i] += count
        else:
            bins1[i] = count
    return bins1


def add_bins(bins1, bins2, datatype):
    for item in bins2:
        if item in bins1:
            bins1[item] += bins2[item]
        else:
            bins1[item] = bins2[item]
    return bins1


def baks(spktimes, time, a=1.5, b=None):
    """
    Bayesian Adaptive Kernel Smoother (BAKS)
    BAKS is a method for estimating firing rate from spike train data that uses kernel smoothing technique 
    with adaptive bandwidth determined using a Bayesian approach
    ---------------INPUT---------------
    - spktimes : spike event times [s]
    - time : time points at which the firing rate is estimated [s]
    - a : shape parameter (alpha) 
    - b : scale parameter (beta)
    ---------------OUTPUT---------------
    - rate : estimated firing rate [nTime x 1] (Hz)
    - h : adaptive bandwidth [nTime x 1]

    Based on "Estimation of neuronal firing rate using Bayesian adaptive kernel smoother (BAKS)"
    https://github.com/nurahmadi/BAKS
    """
    from scipy.special import gamma

    n = len(spktimes)
    sumnum = 0
    sumdenom = 0

    if b is None:
        b = 0.8
    b = float(n) ** b

    for i in range(n):
        numerator = (((time - spktimes[i]) ** 2) / 2. + 1. / b) ** (-a)
        denominator = (((time - spktimes[i]) ** 2) / 2. + 1. / b) ** (-a - 0.5)
        sumnum = sumnum + numerator
        sumdenom = sumdenom + denominator

    h = (old_div(gamma(a), gamma(a + 0.5))) * (old_div(sumnum, sumdenom))

    rate = np.zeros((len(time),))
    for j in range(n):
        K = (1. / (np.sqrt(2. * np.pi) * h)) * np.exp(old_div(-((time - spktimes[j]) ** 2), (2. * h ** 2)))
        rate = rate + K

    return rate, h


def kde_scipy(x, y, bin_size, **kwargs):
    """Kernel Density Estimation with Scipy"""
    from scipy.stats import gaussian_kde

    data = np.vstack([x, y])
    kde = gaussian_kde(data, **kwargs)

    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    dx = int(old_div((x_max - x_min), bin_size))
    dy = int(old_div((y_max - x_min), bin_size))

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, dx), \
                         np.linspace(y_min, y_max, dy))

    data_grid = np.vstack([xx.ravel(), yy.ravel()])
    z = kde.evaluate(data_grid)

    return xx, yy, np.reshape(z, xx.shape)


def kde_sklearn(x, y, binSize, bandwidth=1.0, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    from sklearn.neighbors import KernelDensity

    # create grid of sample locations
    xx, yy = np.mgrid[x.min():x.max():binSize,
             y.min():y.max():binSize]

    data_grid = np.vstack([xx.ravel(), yy.ravel()]).T
    data = np.vstack([x, y]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(data)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(data_grid))
    return xx, yy, np.reshape(z, xx.shape)


def akde(X, grid=None, gam=None, errtol=10 ** -5, maxiter=100, seed=0, verbose=False):
    """
    Adaptive Kernel Density Estimate (AKDE)
    Provides optimal accuracy/speed tradeoff, controlled with parameter gam
    ---------------INPUT---------------
    - x : event points
    - grid : points at which the density rate is estimated 
    - gam : scale parameter 
    - errtol : convergence tolerance (default: 10^-5)
    - maxiter : maximum iterations (default: 200)
    - seed : random number seed 
    ---------------OUTPUT---------------
    - pdf : estimated density 
    - grid : grid points at which density is estimated

    Usage:

    import numpy as np
    mu, sigma = 3., 1.
    X = np.random.lognormal(mu, sigma, 1000)
    pdf, grid = akde(X)

    Reference:
    Kernel density estimation via diffusion
    Z. I. Botev, J. F. Grotowski, and D. P. Kroese (2010)
    Annals of Statistics, Volume 38, Number 5, pages 2916-2957.
    """
    np.seterr(divide='raise')

    n = X.shape[0]
    assert (n > 1)

    if len(X.shape) > 1:
        d = X.shape[1]
    else:
        d = 1

    X = X.reshape((n, d))
    zd = np.where(np.diff(X[:, 0]) >= 1e-3)[0] + 1
    X = np.vstack([np.asarray([X[0, :]]).reshape((-1, d)), X[zd, :]])
    n = X.shape[0]

    if n < 2:
        if grid is None:
            return (np.zeros((1, 1)), X[0, 0])
        else:
            return (np.zeros(grid.shape), grid)

    xmax = np.max(X, axis=0)
    xmin = np.min(X, axis=0)
    r = xmax - xmin

    smax = xmax + r / 10.
    smin = xmin - r / 10.

    scaling = smax - smin

    X = X - smin

    X = np.divide(X, scaling)

    if gam is None:
        gam = int(math.ceil(math.sqrt(n)))

    if grid is None:
        step = old_div(scaling, (2 ** 12 - 1))
        npts = int(math.ceil(old_div(scaling, step))) + 1
        grid = np.linspace(smin, smax, npts)

    grid = grid.reshape((-1, 1))

    mesh = np.subtract(grid, smin)
    mesh = np.divide(mesh, scaling)

    ## algorithm initialization
    local_random = np.random.RandomState(seed=seed)
    bw = 0.2 / (n ** (old_div(d, (d + 4.))))
    perm = local_random.permutation(n)
    ##perm = list(xrange(0, n))
    mus = X[perm[0:gam], :]

    w = local_random.rand(gam)
    # w = np.linspace(0., 1., gam) + 0.001
    w = np.divide(w, np.sum(w))
    sigmas = (bw ** 2.) * local_random.rand(gam, d)
    ##sigmas = (bw ** 2.) * (np.linspace(0., 1., gam).reshape((-1,1)) + 0.01)
    ent = float("-inf")

    for i in range(maxiter):
        Eold = ent
        (w, mus, sigmas, bw, ent) = akde_reg_EM(w, mus, sigmas, bw, X)
        err = abs(old_div((ent - Eold), ent))
        if verbose:
            print('Iter.    Err.      Bandwidth \n')
            print('%4i    %8.2e   %8.2e\n' % (i, err, bw))

        assert (not math.isnan(bw))
        assert (not math.isnan(err))

        if (err < errtol):
            break

    pdf = old_div(akde_probfun(mesh, w, mus, sigmas), scaling)

    return pdf, grid


def akde_reg_EM(w, mus, sigmas, bw, X):
    gam, d = mus.shape
    n, d = X.shape

    log_lh = np.zeros((n, gam))
    log_sig = np.array(log_lh, copy=True)

    eps = np.finfo(float).eps
    for i in range(gam):
        s = sigmas[i, :]

        Xcentered = np.subtract(X, mus[i, :])
        xRinv = np.divide(Xcentered ** 2., s)
        xSig = np.sum(np.divide(xRinv, s), axis=1) + eps
        log_lh[:, i] = -0.5 * np.sum(xRinv, axis=1) - 0.5 * math.log(s) + math.log(w[i]) - d * math.log(
            2. * math.pi) / 2. - 0.5 * (bw ** 2.) * np.sum(1. / s)
        log_sig[:, i] = log_lh[:, i] + np.log(xSig)

    maxll = np.max(log_lh, axis=1).reshape((-1, 1))
    maxlsig = np.max(log_sig, axis=1).reshape((-1, 1))
    p = np.exp(np.subtract(log_lh, maxll))
    psig = np.exp(np.subtract(log_sig, maxlsig))
    density = np.sum(p, axis=1).reshape((-1, 1))
    psigd = np.sum(psig, axis=1).reshape((-1, 1))
    logpdf = np.log(density) + maxll
    logpsigd = np.log(psigd) + maxlsig
    p = np.divide(p, density)
    ent = np.sum(logpdf)
    w = np.sum(p, axis=0)

    for i in (np.where(w > 0.))[0]:
        mus[i, :] = np.dot(p[:, i].reshape((-1, 1)).T, np.divide(X, w[i]))
        Xcentered = np.subtract(X, mus[i, :])
        sigmas[i, :] = np.dot(p[:, i].reshape((-1, 1)).T, old_div((Xcentered ** 2.), w[i])) + bw ** 2.

    w = old_div(w, np.sum(w))
    curv = np.mean(np.exp(logpsigd - logpdf))
    bw = 1. / ((4. * n * (4. * math.pi) ** (d / 2.) * curv) ** (1. / (d + 2.)))
    return (w, mus, sigmas, bw, ent)


def akde_probfun(x, w, mus, sigmas):
    gam, d = mus.shape
    pdf = np.zeros(x.shape)

    for k in range(gam):
        s = sigmas[k, :]
        xx = np.subtract(x, mus[k, :]).reshape((-1, 1))
        xx = np.divide(xx ** 2., s)
        pdf = np.add(pdf, np.exp(
            -0.5 * np.sum(xx, axis=1).reshape((-1, 1)) + math.log(w[k]) - 0.5 * math.log(s) - d * math.log(
                2. * math.pi) / 2.))

    return pdf
