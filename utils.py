from collections import defaultdict, Iterable, namedtuple
import sys, os.path, string, time, gc, math, datetime, numbers, itertools
import copy, pprint, logging
import yaml
import numpy as np


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


def read_from_yaml(file_path):
    """

    :param file_path: str (should end in '.yaml')
    :return:
    """
    if os.path.isfile(file_path):
        with open(file_path, 'r') as stream:
            data = yaml.load(stream)
        return data
    else:
        raise IOError('read_from_yaml: invalid file_path: %s' % file_path)


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
        for i in range(len(data)):
            data[i] = nested_convert_scalars(data[i])
    elif hasattr(data, 'item'):
        try:
            data = np.asscalar(data)
        except TypeError:
            pass
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
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None


def list_find_all(f, lst):
    """

    :param f:
    :param lst:
    :return:
    """
    i=0
    res=[]
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
    return [i for i,x in sorted(enumerate(seq), key = lambda x: f(x[1]))]


def viewitems(obj, **kwargs):
    """
    Function for iterating over dictionary items with the same set-like
    behaviour on Py2.7 as on Py3.

    Passes kwargs to method."""
    func = getattr(obj, "viewitems", None)
    if not func:
        func = obj.items
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
        d = xyz[:,i] - xyz[:,j]
        return np.sqrt(np.dot(d, d))

    g = nx.Graph()
    for i, j in edges:
        g.add_edge(i, j, weight=euclidean_dist(i, j))
    return g


def random_choice_w_replacement(ranstream,n,p):
    """

    :param ranstream:
    :param n:
    :param p:
    :return:
    """
    return ranstream.multinomial(n,p.ravel())


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
        assert(isinstance(centers, np.ndarray))
        n_centers = len(centers)
    
    X, y = make_random_clusters (centers, n_samples_per_center, n_features=1, \
                                 center_ids=center_ids, cluster_std=cluster_std, center_box=center_box, \
                                 random_seed=random_seed)
    s = np.argsort(X,axis=0).ravel()
    return y[s].ravel()


def kde_sklearn(x, y, binSize, bandwidth=1.0, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    from sklearn.neighbors import KernelDensity

    # create grid of sample locations
    xx, yy = np.mgrid[x.min():x.max():binSize, 
                      y.min():y.max():binSize]

    data_grid = np.vstack([xx.ravel(), yy.ravel()]).T
    data  = np.vstack([x, y]).T
    
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(data)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(data_grid))
    return xx, yy, np.reshape(z, xx.shape)


def kde_scipy(x, y, binSize, **kwargs):
    """Kernel Density Estimation with Scipy"""
    from scipy.stats import gaussian_kde

    data  = np.vstack([x, y])
    kde   = gaussian_kde(data, **kwargs)

    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    dx = int((x_max - x_min) / binSize)
    dy = int((y_max - x_min) / binSize)

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, dx), \
                         np.linspace(y_min, y_max, dy))

    data_grid = np.vstack([xx.ravel(), yy.ravel()])
    z    = kde.evaluate(data_grid)
    
    return xx, yy, np.reshape(z, xx.shape)

def NamedTupleWithDocstring(docstring, *ntargs):
    """
    A convenience wrapper to add docstrings to named tuples. This is only needed in
    python 2, where __doc__ is not writeable.
    https://stackoverflow.com/questions/1606436/adding-docstrings-to-namedtuples
    """
    nt = namedtuple(*ntargs)
    class NT(nt):
        __doc__ = docstring
        __slots__ = () ## disallow mutable slots in order to keep performance advantage of tuples
    return NT

def partitionn(items, predicate=int, n=2):
    """
    Filter an iterator into N parts lazily
    http://paddy3118.blogspot.com/2013/06/filtering-iterator-into-n-parts-lazily.html
    """
    tees = itertools.tee( ((predicate(item), item)
                               for item in items), n )
    return ( (lambda i:(item for pred, item in tees[i] if pred==i))(x)
                 for x in range(n) )

def generator_peek(iterable):
    """
    If the iterable is empty, return None, otherwise return the
    iterable with the first element attached back.
    """
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first, itertools.chain([first], iterable)
