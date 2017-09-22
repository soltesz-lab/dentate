
import itertools
from neuron import h
import numpy as np

def get_node_attribute (name, content, sec, x=None):
    if content.has_key(name):
        if x is None:
            return content[name]
        elif sec.n3d() == 0:
            return content[name][0]
        else:
            for i in xrange(sec.n3d()):
                if sec.arc3d(i)/sec.L >= x:
                    return content[name][i]
    else:
        return None

def make_neurotree_cell (template_name, local_id=0, gid=0, dataset_path="", neurotree_dict={}):
    h('objref cell, vx, vy, vz, vradius, vlayer, vsection, secnodes, vsrc, vdst, swc_type')
    h.vx       = neurotree_dict['x']
    h.vy       = neurotree_dict['y']
    h.vz       = neurotree_dict['z']
    h.vradius  = neurotree_dict['radius']
    h.vlayer   = neurotree_dict['layer']
    h.vsection = neurotree_dict['section']
    h.secnodes = neurotree_dict['section_topology']['nodes']
    h.vsrc     = neurotree_dict['section_topology']['src']
    h.vdst     = neurotree_dict['section_topology']['dst']
    h.swc_type = neurotree_dict['swc_type']
    hstmt      = 'cell = new %s(%d, %d, "%s", vlayer, vsrc, vdst, secnodes, vx, vy, vz, vradius, swc_type)' % (template_name, local_id, gid, dataset_path)
    h(hstmt)
    return h.cell

