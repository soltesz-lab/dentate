import itertools
from collections import defaultdict, Iterable, namedtuple
import sys, os.path, string, time, gc, math, datetime
import copy
import numpy as np
import yaml
import pprint


class IncludeLoader(yaml.Loader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        yaml.Loader.__init__(self, stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, IncludeLoader)

IncludeLoader.add_constructor('!include', IncludeLoader.include)


def write_to_yaml(file_path, data, convert_scalars=False):
    """

    :param file_path: str (should end in '.yaml')
    :param data: dict
    :param convert_scalars: bool
    :return:
    """
    import yaml
    with open(file_path, 'w') as outfile:
        if convert_scalars:
            data = nested_convert_scalars(data)
        yaml.dump(data, outfile, default_flow_style=False)


def read_from_yaml(file_path):
    """

    :param file_path: str (should end in '.yaml')
    :return:
    """
    import yaml
    if os.path.isfile(file_path):
        with open(file_path, 'r') as stream:
            data = yaml.load(stream)
        return data
    else:
        raise Exception('File: {} does not exist.'.format(file_path))


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
        for i in xrange(len(data)):
            data[i] = nested_convert_scalars(data[i])
    elif hasattr(data, 'item'):
        try:
            data = np.asscalar(data)
        except TypeError:
            pass
    return data


def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None


def list_argsort(f, seq):
    # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3383106#3383106
    #lambda version by Tony Veijalainen
    return [i for i,x in sorted(enumerate(seq), key = lambda x: f(x[1]))]
