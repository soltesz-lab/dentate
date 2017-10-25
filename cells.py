
import itertools
from neuron import h
import numpy as np

def get_node_attribute (name, content, sec, secnodes, x=None):
    if content.has_key(name):
        if x is None:
            return content[name]
        elif sec.n3d() == 0:
            return content[name][0]
        else:
            for i in xrange(sec.n3d()):
                if sec.arc3d(i)/sec.L >= x:
                    return content[name][secnodes[i]]
    else:
        return None

def make_neurotree_cell (template_class, local_id=0, gid=0, dataset_path="", neurotree_dict={}):
    vx       = neurotree_dict['x']
    vy       = neurotree_dict['y']
    vz       = neurotree_dict['z']
    vradius  = neurotree_dict['radius']
    vlayer   = neurotree_dict['layer']
    vsection = neurotree_dict['section']
    secnodes = neurotree_dict['section_topology']['nodes']
    vsrc     = neurotree_dict['section_topology']['src']
    vdst     = neurotree_dict['section_topology']['dst']
    swc_type = neurotree_dict['swc_type']
    cell     = template_class (local_id, gid, dataset_path, vlayer, vsrc, vdst, secnodes, vx, vy, vz, vradius, swc_type)
    return cell


def make_cell (template_class, local_id=0, gid=0, dataset_path=""):
    cell = template_class (local_id, gid, dataset_path)
    return cell

