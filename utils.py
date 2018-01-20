import itertools
from collections import defaultdict
import sys, os.path, string, time, gc
import numpy as np
import yaml
import os.path


class IncludeLoader(yaml.Loader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        yaml.Loader.__init__(self, stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, yaml.Loader)

IncludeLoader.add_constructor('!include', IncludeLoader.include)


def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None


