import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import gc
import copy
import pprint
import pickle
import math
import random


"""
Structure of Mechanism Dictionary: dict of dicts

keys:               description:
'mechanism name':   Value is dictionary specifying how to set parameters at the mechanism level.
'cable':            Value is dictionary specifying how to set basic cable parameters at the section level. Includes
                        'Ra', 'cm', and the special parameter 'spatial_res', which scales the number of segments per
                        section for the specified sec_type by a factor of an exponent of 3.
'ions':             Value is dictionary specifying how to set parameters for ions at the section or segment level.
                    These parameters must be specified **after** all other mechanisms have been inserted.
values:
None:               Use default values for all parameters within this mechanism.
dict:
    keys:
    'parameter name':
    values:     dict:
                        keys:        value:
                        'origin':   'self':     Use 'value' as a baseline value.
                                    sec_type:   Inherit value from last seg of the closest node with sec of
                                                sec_type along the path to root.
                        'value':    float:      If 'origin' is 'self', contains the baseline value.
                        'slope':    float:      If exists, contains slope in units per um. If not, use
                                                constant 'value' for the full length of sec.
                        'max':      float:      If 'slope' exists, 'max' is an upper limit for the value
                        'min':      float:      If 'slope' exists, min is a lower limit for the value

"""

default_mech_dict = {'ais': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                             'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}},
                     'apical': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                                'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}},
                     'axon': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                              'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}},
                     'axon_hill': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                              'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}},
                     'basal': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                               'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}},
                     'soma': {'cable': {'Ra': {'value': 150.}, 'cm': {'value': 1.}},
                              'pas': {'e': {'value': -67.}, 'g': {'value': 2.5e-05}}},
                     'trunk': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                               'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}},
                     'tuft': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                              'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}},
                     'spine_neck': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                              'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}},
                     'spine_head': {'cable': {'Ra': {'origin': 'soma'}, 'cm': {'origin': 'soma'}},
                              'pas': {'e': {'origin': 'soma'}, 'g': {'origin': 'soma'}}}}


def clean_axes(axes):
    """
    Remove top and right axes from pyplot axes object.
    :param axes:
    """
    if not type(axes) in [np.ndarray, list]:
        axes = [axes]
    for axis in axes:
        axis.tick_params(direction='out')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.get_xaxis().tick_bottom()
        axis.get_yaxis().tick_left()


def split_array(a, size):
    start = 0
    chunk_size = int(math.ceil(float(len(a)) / size))
    sub_arrays = []
    for i in range(size):
        if i == size - 1:
            sub_arrays.append(a[start:])
        else:
            sub_arrays.append(a[start:start+chunk_size])
            start += chunk_size
    return sub_arrays


def serial_neuroh5_get_attr(file_path, population, namespace, gid, header='Populations'):
    """
    DEPRECATED
    :param file_path: str
    :param population: str
    :param namespace: str
    :param gid: int
    :param header: str
    :return: dict
    """
    attr_dict = {}
    if not os.path.isfile(file_path):
        raise IOError('serial_neuroh5_get_attr: invalid file_path: %s' % file_path)
    with h5py.File(file_path, 'r') as f:
        if header not in f:
            raise AttributeError('serial_neuroh5_get_attr: invalid header: %s' % header)
        elif population not in f[header]:
            raise AttributeError('serial_neuroh5_get_attr: invalid population: %s' % population)
        elif namespace not in f[header][population]:
            raise AttributeError('serial_neuroh5_get_attr: invalid namespace: %s' % namespace)
        group = f[header][population][namespace]
        gid_offset = None
        pop_size = None
        attr_dict[namespace] = {}
        for attr in group:
            if not ('gid' in group[attr] and 'ptr' in group[attr] and 'value' in group[attr]):
                print('serial_neuroh5_get_attr: excluding namespace: %s; attribute: %s; format not recognized' % \
                      (namespace, attr))
            else:
                if gid_offset is None:
                    gid_offset = group[attr]['gid'][0]
                    pop_size = len(group[attr]['gid'])
                gid_index = gid - gid_offset
                if gid_index > pop_size:
                    raise ValueError('serial_neuroh5_get_attr: gid: %i out of range for population: %s' %
                                     (gid, population))
                else:
                    start = group[attr]['ptr'][gid_index]
                    stop = group[attr]['ptr'][gid_index+1]
                    attr_dict[namespace][attr] = group[attr]['value'][start:stop]
    return attr_dict