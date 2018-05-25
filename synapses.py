import itertools
from collections import defaultdict
import sys, os.path, string, math
from neuron import h
import numpy as np
from dentate import utils, neuron_utils
import logging

## This logger will inherit its setting from its root logger, dentate,
## which is created in module env
logger = logging.getLogger('dentate.%s' % __name__)


class SynapseAttributes(object):
    """
    As a network is constructed, this class provides an interface to store, retrieve, and modify attributes of synaptic
    mechanisms. Handles instantiation of complex subcellular gradients of synaptic mechanism attributes.
    """
    def __init__(self, syn_mech_names, syn_param_rules):
        """
        An Env object containing imported network configuration metadata uses an instance of SynapseAttributes to track
        all metadata related to the identity, location, and configuration of all synaptic connections in the network.
        :param syn_mech_names: dict
        :param syn_param_rules: dict
        """
        self.syn_mech_names = syn_mech_names
        self.syn_param_rules = syn_param_rules
        # TODO: these two dicts need to also be indexed by the namespace
        self.select_cell_attr_index_map = {}  # population name (str): gid (int): index in file (int)
        # dest population name (str): source population name (str): gid (int): index in file (int)
        self.select_edge_attr_index_map = defaultdict(dict)

        self.syn_id_attr_dict = {}  # gid (int): attr_name (str): array
        # gid (int): syn_id (int): dict
        self.syn_mech_attr_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.syn_id_attr_index_map = {}  # gid (int): syn_id (int): index in syn_id_attr_dict (int)
        # gid (int): sec_id (int): list of indexes in syn_id_attr_dict (int)
        self.sec_index_map = defaultdict(lambda: defaultdict(list))

    def load_syn_id_attrs(self, gid, syn_id_attr_dict):
        """

        :param gid: int
        :param syn_id_attr_dict: dict
        """
        self.syn_id_attr_dict[gid] = syn_id_attr_dict
        # value of -1 used to indicate not yet assigned; all source populations are associated with positive integers
        self.syn_id_attr_dict[gid]['syn_sources'] = \
            np.full(self.syn_id_attr_dict[gid]['syn_ids'].shape, -1, dtype='int8')
        self.syn_id_attr_index_map[gid] = {syn_id: i for i, syn_id in enumerate(syn_id_attr_dict['syn_ids'])}
        for i, sec_id in enumerate(syn_id_attr_dict['syn_secs']):
            self.sec_index_map[gid][sec_id].append(i)
        for sec_id in self.sec_index_map[gid]:
            self.sec_index_map[gid][sec_id] = np.array(self.sec_index_map[gid][sec_id], dtype='uint32')

    def load_syn_weights(self, gid, syn_name, syn_ids, weights):
        """

        :param gid: int
        :param syn_name: str
        :param syn_ids: array of int
        :param weights: array of float
        """
        for i, syn_id in enumerate(syn_ids):
            params = {'weight': float(weights[i])}
            self.set_mech_attrs(gid, syn_id, syn_name, params)

    def load_edge_attrs(self, gid, source_name, syn_ids, env, delays=None):
        """

        :param gid: int
        :param source_name: str; name of source population
        :param syn_ids: array of int
        :param env: :class:'Env'
        :param delays: array of float; axon conduction (netcon) delays
        """
        source = int(env.pop_dict[source_name])
        indexes = [self.syn_id_attr_index_map[gid][syn_id] for syn_id in syn_ids]
        self.syn_id_attr_dict[gid]['syn_sources'][indexes] = source
        if delays is not None:
            if not self.syn_id_attr_dict[gid].has_key('delays'):
                self.syn_id_attr_dict[gid]['delays'] = \
                    np.full(self.syn_id_attr_dict[gid]['syn_ids'].shape, 0., dtype='float32')
            self.syn_id_attr_dict[gid]['delays'][indexes] = delays

    def append_netcon(self, gid, syn_id, syn_name, nc):
        """

        :param gid: int
        :param syn_id: int
        :param syn_name: str
        :param nc: :class:'h.NetCon'
        """
        self.syn_mech_attr_dict[gid][syn_id][syn_name]['netcon'] = nc

    def has_netcon(self, gid, syn_id, syn_name):
        """

        :param gid: int
        :param syn_id: int
        :param syn_name: str
        :return: bool
        """
        if (self.syn_mech_attr_dict[gid][syn_id].has_key(syn_name) and
                self.syn_mech_attr_dict[gid][syn_id][syn_name].has_key('netcon')):
            return True
        else:
            return False

    def get_netcon(self, gid, syn_id, syn_name):
        """

        :param gid: int
        :param syn_id: int
        :param syn_name: str
        :return: :class:'h.NetCon'
        """
        if self.has_netcon(gid, syn_id, syn_name):
            return self.syn_mech_attr_dict[gid][syn_id][syn_name]['netcon']
        else:
            return None

    def append_vecstim(self, gid, syn_id, syn_name, vs):
        """

        :param gid: int
        :param syn_id: int
        :param syn_name: str
        :param :class:'h.VecStim'
        """
        self.syn_mech_attr_dict[gid][syn_id][syn_name]['vecstim'] = vs

    def has_mech_attrs(self, gid, syn_id, syn_name):
        """

        :param gid: int
        :param syn_id: int
        :param syn_name: str
        :return: bool
        """
        if self.syn_mech_attr_dict[gid][syn_id][syn_name].has_key('attrs') and \
                len(self.syn_mech_attr_dict[gid][syn_id][syn_name]['attrs']) > 0:
            return True
        else:
            return False

    def get_mech_attrs(self, gid, syn_id, syn_name):
        """

        :param gid: int
        :param syn_id: int
        :param syn_name: str
        :return: :class:'h.NetCon'
        """
        if self.has_mech_attrs(gid, syn_id, syn_name):
            return self.syn_mech_attr_dict[gid][syn_id][syn_name]['attrs']
        else:
            return None

    def set_mech_attrs(self, gid, syn_id, syn_name, params):
        """

        :param gid: int
        :param syn_id: int
        :param syn_name: str
        :param params: dict
        """
        if not self.has_mech_attrs(gid, syn_id, syn_name):
            self.syn_mech_attr_dict[gid][syn_id][syn_name]['attrs'] = params
        else:
            for param, val in params.iteritems():
                self.syn_mech_attr_dict[gid][syn_id][syn_name]['attrs'][param] = val

    def cleanup(self, gid):
        """

        :param gid: int
        """
        if gid in self.syn_id_attr_dict:
            del self.syn_id_attr_index_map[gid]
            del self.syn_id_attr_dict[gid]
            del self.sec_index_map[gid]
        for syn_id in self.syn_mech_attr_dict[gid]:
            for syn_name in self.syn_mech_attr_dict[gid][syn_id]:
                if 'attrs' in self.syn_mech_attr_dict[gid][syn_id][syn_name]:
                    del self.syn_mech_attr_dict[gid][syn_id][syn_name]['attrs']


def get_filtered_syn_indexes(syn_id_attr_dict, syn_indexes=None, syn_types=None, layers=None, sources=None,
                             swc_types=None):
    """

    :param syn_id_attr_dict: dict (already indexed by gid)
    :param syn_indexes: array of int
    :param syn_types: list of enumerated type: synapse category
    :param layers: list of enumerated type: layer
    :param sources: list of enumerated type: population names of source projections
    :param swc_types: list of enumerated type: swc_type
    :return: array of int
    """
    matches = np.vectorize(lambda query, item: (query is None) or (item in query), excluded={0})
    if syn_indexes is None:
        syn_indexes = np.arange(len(syn_id_attr_dict['syn_ids']), dtype='uint32')
    else:
        syn_indexes = np.array(syn_indexes, dtype='uint32')
    filtered_indexes = np.where(matches(syn_types, syn_id_attr_dict['syn_types'][syn_indexes]) &
                                matches(layers, syn_id_attr_dict['syn_layers'][syn_indexes]) &
                                matches(sources, syn_id_attr_dict['syn_sources'][syn_indexes]) &
                                matches(swc_types, syn_id_attr_dict['swc_types'][syn_indexes]))[0]
    return syn_indexes[filtered_indexes]


def organize_syn_ids_by_source(gid, env, syn_ids=None):
    """

    :param gid: int
    :param env: :class:'Env'
    :param syn_ids: array of int
    """
    source_names = {id: name for name, id in env.pop_dict.iteritems()}
    source_syn_ids = defaultdict(list)
    syn_attrs = env.synapse_attributes
    syn_id_attr_dict = syn_attrs.syn_id_attr_dict[gid]
    syn_id_attr_index_map = syn_attrs.syn_id_attr_index_map[gid]

    for syn_id in syn_ids:
        source_id = syn_id_attr_dict['syn_sources'][syn_id_attr_index_map[syn_id]]
        source_name = source_names[source_id]
        source_syn_ids[source_name].append(syn_id)
    return source_syn_ids


def insert_syns_from_mech_attrs(gid, env, postsyn_name, presyn_name, syn_ids, unique=False):
    """
    TODO: add a check for 'delays' in syn_id_attr_dict to initialize netcon delays
    1) make syns (if not unique, keep track of syn_in_seg for shared synapses)
    2) initialize syns with syn_mech_params from config_file
    3) make netcons
    4) initialize netcons with syn_mech_params from config_file

    :param gid: int
    :param env: :class:'Env'
    :param postsyn_name: str
    :param presyn_name: str
    :param syn_ids: array of int
    :param unique: bool; whether to insert synapses if none exist at syn_id
    """
    rank = int(env.pc.id())
    if not env.biophys_cells[postsyn_name].has_key(gid):
        raise KeyError('insert_syns_from_mech_attrs: problem locating BiophysCell with gid: %i' % gid)
    cell = env.biophys_cells[postsyn_name][gid]
    syn_attrs = env.synapse_attributes
    syn_params = env.connection_generator[postsyn_name][presyn_name].synapse_parameters

    syn_id_attr_dict = syn_attrs.syn_id_attr_dict[gid]
    syn_id_attr_index_map = syn_attrs.syn_id_attr_index_map[gid]

    shared_syns_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syn_obj_dict = defaultdict(dict)

    add_synapse = add_unique_synapse if unique else add_shared_synapse

    syn_count = 0
    syns_set = set()
    for syn_id in syn_ids:
        syn_index = syn_id_attr_index_map[syn_id]
        sec_id = syn_id_attr_dict['syn_secs'][syn_index]
        sec = cell.tree.get_node_with_index(sec_id).sec
        syn_loc = syn_id_attr_dict['syn_locs'][syn_index]
        for syn_name, mech_params in syn_params.iteritems():
            syn = add_synapse(syn_name=syn_name, seg=sec(syn_loc), syns_dict=shared_syns_dict,
                              mech_names=syn_attrs.syn_mech_names)
            syn_obj_dict[syn_id][syn_name] = syn
            if syn not in syns_set:
                config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules, mech_names=syn_attrs.syn_mech_names,
                           syn=syn, **mech_params)
                cell.hoc_cell.syns.append(syn)
                syns_set.add(syn)
                env.syns_set[gid].add(syn)
                syn_count += 1

    nc_count = 0
    for syn_id in syn_ids:
        for syn_name, syn in syn_obj_dict[syn_id].iteritems():
            this_nc, this_vecstim = mknetcon_vecstim(syn)
            syn_attrs.append_netcon(gid, syn_id, syn_name, this_nc)
            syn_attrs.append_vecstim(gid, syn_id, syn_name, this_vecstim)
            config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules, mech_names=syn_attrs.syn_mech_names,
                       nc=this_nc, **syn_params[syn_name])
            nc_count += 1

    if env.verbose and rank == 0:
        print 'insert_syns_from_mech_attrs: source: %s; target: %s cell %i: ' \
              'created %i syns and %i netcons for %i syn_ids' % \
              (presyn_name, postsyn_name, gid, syn_count, nc_count, len(syn_ids))


def config_syns_from_mech_attrs(gid, env, postsyn_name, syn_ids=None, insert=False, unique=None, verbose=None):
    """
    1) organize syn_ids by source population
    2) if insert, collate syn_ids without netcons, iterate over sources and call insert_syns_from_mech_attrs
       (requires a BiophysCell with the specified gid to be present in the Env).
    3) iterate over all syn_ids, and call config_syn with params from syn_mech_attr_dict (which may be empty)
    :param gid: int
    :param env: :class:'Env'
    :param postsyn_name: str
    :param syn_ids: array of int
    :param insert: bool; whether to insert synapses if none exist at syn_id
    :param unique: bool; whether newly inserted synapses should be unique or shared per segment
    :param verbose: bool
    """
    rank = int(env.pc.id())
    syn_attrs = env.synapse_attributes
    syn_id_attr_dict = syn_attrs.syn_id_attr_dict[gid]

    synapse_config = env.celltypes[postsyn_name]['synapses']
    if unique is None:
        if synapse_config.has_key('unique'):
            unique = synapse_config['unique']
        else:
            unique = False

    if syn_ids is None:
        syn_ids = syn_id_attr_dict['syn_ids']

    source_syn_ids = organize_syn_ids_by_source(gid, env, syn_ids)

    if insert:
        if not env.biophys_cells[postsyn_name].has_key(gid):
            raise KeyError('config_syns_from_mech_attrs: insert: problem locating BiophysCell with gid: %i' % gid)
        insert_syn_ids = defaultdict(list)
        for presyn_name in source_syn_ids:
            syn_names = env.connection_generator[postsyn_name][presyn_name].synapse_parameters.keys()
            for syn_id in source_syn_ids[presyn_name]:
                for syn_name in syn_names:
                    if not syn_attrs.has_netcon(gid, syn_id, syn_name):
                        insert_syn_ids[presyn_name].append(syn_id)
                        break
        for presyn_name in insert_syn_ids:
            insert_syns_from_mech_attrs(gid, env, postsyn_name, presyn_name, insert_syn_ids[presyn_name], unique=unique)

    nc_count = 0
    syn_count = 0
    syns_set = set()
    for presyn_name in source_syn_ids:
        syn_names = env.connection_generator[postsyn_name][presyn_name].synapse_parameters.keys()
        for syn_id in source_syn_ids[presyn_name]:
            for syn_name in syn_names:
                mech_params = syn_attrs.get_mech_attrs(gid, syn_id, syn_name)
                if mech_params is not None:
                    this_netcon = syn_attrs.get_netcon(gid, syn_id, syn_name)
                    if this_netcon is not None:
                        syn = this_netcon.syn()
                        if syn not in syns_set:
                            syns_set.add(syn)
                            syn_count += 1
                        else:
                            syn = None
                        config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules,
                                   mech_names=syn_attrs.syn_mech_names, syn=syn, nc=this_netcon, **mech_params)
                        nc_count += 1

    if verbose is None:
        verbose = env.verbose
    if verbose and rank == 0:
        print 'config_syns_from_mech_attrs: population: %s; cell %i: ' \
              'updated mech_params for %i syns and %i netcons for %i syn_ids' % \
              (postsyn_name, gid, syn_count, nc_count, len(syn_ids))


def syn_in_seg(syn_name, seg, syns_dict):
    """
    If a synaptic mechanism of the specified type already exists in the specified segment, it is returned.
    Otherwise, it returns None.
    :param syn_name: str
    :param seg: hoc segment
    :param syns_dict: nested defaultdict
    :return: hoc point process or None
    """
    sec = seg.sec
    for x in syns_dict[sec]:
        if sec(x) == seg:
            if syn_name in syns_dict[sec][x]:
                syn = syns_dict[sec][x][syn_name]
                return syn
    return None


def make_syn_mech(mech_name, seg):
    """
    :param mech_name: str (name of the point_process, specified by Env.synapse_attributes.syn_mech_names)
    :param seg: hoc segment
    :return: hoc point process
    """
    if hasattr(h, mech_name):
        syn = getattr(h, mech_name)(seg)
    else:
        raise ValueError('make_syn_mech: unrecognized synaptic mechanism name %s' % mech_name)
    return syn

def add_shared_synapse(syn_name, seg, syns_dict, mech_names=None):
    """
    If a synaptic mechanism of the specified type already exists in the specified segment, it is returned.
    Otherwise, this method creates one in the provided segment and adds it to the provided syns_dict before it is
    returned.
    :param syn_name: str
    :param seg: hoc segment
    :param syns_dict: nested defaultdict
    :param mech_names: dict to convert syn_name to hoc mechanism name
    :return: hoc point process
    """

    if not syn_in_seg(seg, syns_dict):
        if PP_dict is not None:
            syn = make_specialized_syn_mech(mech_name, seg, PP_dict)
        else:
            syn = make_syn_mech(mech_name, seg)
        syns_dict[seg.sec][syn.get_segment().x] = syn
        return syn
    else:
        for x, syn in syns_dict[seg.sec].iteritems():
            if seg.sec(x) == seg:
               return syn


def add_unique_synapse(syn_name, seg, syns_dict=None, mech_names=None):
    """
    Creates a new synapse in the provided segment, and returns it.
    :param syn_name: str
    :param seg: hoc segment
    :param syns_dict: nested defaultdict
    :param mech_names: dict to convert syn_name to hoc mechanism name
    :return: hoc point process
    """
    if mech_names is not None:
        mech_name = mech_names[syn_name]
    else:
        mech_name = syn_name
    syn = make_syn_mech(mech_name, seg)
    return syn


def config_syn(syn_name, rules, mech_names=None, syn=None, nc=None, **params):
    """

    :param syn_name: str
    :param rules: dict to correctly parse params for specified hoc mechanism
    :param mech_names: dict to convert syn_name to hoc mechanism name
    :param syn: synaptic mechanism object
    :param nc: :class:'h.NetCon'
    :param params: dict
    """
    if mech_names is not None:
        mech_name = mech_names[syn_name]
    else:
        mech_name = syn_name
    for param, val in params.iteritems():
        failed = True
        if param in rules[mech_name]['mech_params']:
            if syn is None:
                failed = False
            elif hasattr(syn, param):
                setattr(syn, param, val)
                failed = False
        elif param in rules[mech_name]['netcon_params']:
            if nc is None:
                failed = False
            else:
                i = rules[mech_name]['netcon_params'][param]
                if nc.wcnt() >= i:
                    nc.weight[i] = val
                    failed = False
        if failed:
            raise AttributeError('config_syn: problem setting attribute: %s for synaptic mechanism: %s' %
                                 (param, mech_name))


def get_syn_mech_param(syn_name, rules, param_name, mech_names=None, nc=None):
    """

    :param syn_name: str
    :param rules: dict to correctly parse params for specified hoc mechanism
    :param param_name: str
    :param mech_names: dict to convert syn_name to hoc mechanism name
    :param nc: :class:'h.NetCon'
    """
    if mech_names is not None:
        mech_name = mech_names[syn_name]
    else:
        mech_name = syn_name
    if nc is not None:
        syn = nc.syn()
        if param_name in rules[mech_name]['mech_params']:
            if syn is not None and hasattr(syn, param_name):
                return getattr(syn, param_name)
        elif param_name in rules[mech_name]['netcon_params']:
            i = rules[mech_name]['netcon_params'][param_name]
            if nc.wcnt() >= i:
                return nc.weight[i]
    raise AttributeError('get_syn_mech_param: problem setting attribute: %s for synaptic mechanism: %s' %
                         (param_name, mech_name))


def mksyns(gid, cell, syn_ids, syn_params, env, edge_count, add_synapse=add_shared_synapse):
    """

    :param gid: int
    :param cell: hoc cell object created from template
    :param syn_ids: array of int
    :param syn_params: dict
    :param env: :class:'Env'
    :param add_synapse: callable
    :return: nested dict of hoc point processes
    """
    rank = int(env.pc.id())

    syns_dict_dend = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_axon = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_ais  = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_hill = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_soma = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    py_sections = [sec for sec in cell.sections]

    # syn_type_excitatory = env.Synapse_Types['excitatory']
    # syn_type_inhibitory = env.Synapse_Types['inhibitory']

    swc_type_apical = env.SWC_Types['apical']
    swc_type_basal  = env.SWC_Types['basal']
    swc_type_soma   = env.SWC_Types['soma']
    swc_type_axon   = env.SWC_Types['axon']
    swc_type_ais   = env.SWC_Types['ais']
    swc_type_hill  = env.SWC_Types['hillock']
    
    syn_attrs = env.synapse_attributes
    syn_attr_id_dict = syn_attrs.syn_id_attr_dict[gid]
    syn_indexes = [syn_attrs.syn_id_attr_index_map[gid][syn_id] for syn_id in syn_ids]

    syn_obj_dict = defaultdict(dict)

    for syn_id, syn_index in itertools.izip(syn_ids, syn_indexes):
        swc_type = syn_attr_id_dict['swc_types'][syn_index]
        syn_loc = syn_attr_id_dict['syn_locs'][syn_index]
        syn_section = syn_attr_id_dict['syn_secs'][syn_index]

        sec = py_sections[syn_section]
        if swc_type == swc_type_apical:
            syns_dict = syns_dict_dend
        elif swc_type == swc_type_basal:
            syns_dict = syns_dict_dend
        elif swc_type == swc_type_axon:
            syns_dict = syns_dict_axon
        elif swc_type == swc_type_ais:
            syns_dict = syns_dict_ais
        elif swc_type == swc_type_hill:
            syns_dict = syns_dict_hill
        elif swc_type == swc_type_soma:
            syns_dict = syns_dict_soma
        else:
            raise RuntimeError("Unsupported synapse SWC type %d" % swc_type)

        for syn_name, params in syn_params.iteritems():
            syn = add_synapse(syn_name=syn_name, seg=sec(syn_loc), syns_dict=syns_dict,
                              mech_names=syn_attrs.syn_mech_names)
            if syn not in env.syns_set[gid]:
                config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules, mech_names=syn_attrs.syn_mech_names,
                           syn=syn, **params)
                env.syns_set[gid].add(syn)
            # cell.syns.append(syn)
            # cell.syntypes.o(syn_type).append(syn)
            syn_obj_dict[syn_id][syn_name] = syn

    if env.verbose and rank == 0 and edge_count == 0:
        sec = syns_dict.iterkeys().next()
        print 'syns_dict[%s]:' % sec.hname()
        pprint.pprint(syns_dict[sec])

    return syn_obj_dict


def mknetcon(pc, srcgid, dstgid, syn, delay=0.1, weight=1):
    """
    Creates a network connection from the provided source to the provided synaptic point process.
    :param pc: :class:'h.ParallelContext'
    :param srcgid: int; source gid
    :param dstgid: int; destination gid
    :param syn: synapse point process
    :param delay: float
    :param weight: float
    :return: :class:'h.NetCon'
    """
    assert pc.gid_exists(dstgid)
    nc = pc.gid_connect(srcgid, syn)
    nc.weight[0] = weight
    nc.delay = delay
    return nc


def mknetcon_vecstim(syn, delay=0.1, weight=1):
    """
    Creates a VecStim object to drive the provided synaptic point process, and a network connection from the VecStim
    source to the synapse target.
    :param syn: synapse point process
    :param delay: float
    :param weight: float
    :return: :class:'h.NetCon', :class:'h.VecStim'
    """
    vs = h.VecStim()
    nc = h.NetCon(vs, syn)
    nc.weight[0] = weight
    nc.delay = delay
    return nc, vs


# ------------------------- Methods to distribute synapse locations -------------------------------------------------- #


def get_node_attribute(name, content, sec, secnodes, x=None):
    """

    :param name:
    :param content:
    :param sec:
    :param secnodes:
    :param x:
    :return:
    """
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


def synapse_seg_density(syn_type_dict, layer_dict, layer_density_dicts, sec_index_dict, seglist, seed,
                        neurotree_dict=None):
    """
    Computes per-segment density of synapse placement.
    :param syn_type_dict:
    :param layer_dict:
    :param layer_density_dicts:
    :param sec_index_dict:
    :param seglist:
    :param seed:
    :param neurotree_dict:
    :return:
    """
    segdensity_dict = {}
    layers_dict = {}

    if neurotree_dict is not None:
        secnodes_dict = neurotree_dict['section_topology']['nodes']
    else:
        secnodes_dict = None
    for (syn_type_label, layer_density_dict) in layer_density_dicts.iteritems():
        syn_type = syn_type_dict[syn_type_label]
        rans = {}
        for (layer_label, density_dict) in layer_density_dict.iteritems():
            if layer_label == 'default':
                layer = layer_label
            else:
                layer = layer_dict[layer_label]
            ran = h.Random(seed)
            ran.normal(density_dict['mean'], density_dict['variance'])
            rans[layer] = ran
        segdensity = []
        layers = []
        for seg in seglist:
            L = seg.sec.L
            nseg = seg.sec.nseg
            if neurotree_dict is not None:
                secindex = sec_index_dict[seg.sec]
                secnodes = secnodes_dict[secindex]
                layer = get_node_attribute('layer', neurotree_dict, seg.sec, secnodes, seg.x)
            else:
                layer = -1
            layers.append(layer)

            ran = None

            if layer > -1:
                if rans.has_key(layer):
                    ran = rans[layer]
                elif rans.has_key('default'):
                    ran = rans['default']
                else:
                    ran = None
            elif rans.has_key('default'):
                ran = rans['default']
            else:
                ran = None
            if ran is not None:
                dens = ran.repick()
                segdensity.append(dens)
            else:
                segdensity.append(0)

        segdensity_dict[syn_type] = segdensity
        layers_dict[syn_type] = layers
    return (segdensity_dict, layers_dict)


def synapse_seg_counts(syn_type_dict, layer_dict, layer_density_dicts, sec_index_dict, seglist, seed,
                       neurotree_dict=None):
    """
    Computes per-segment relative counts of synapse placement.
    :param syn_type_dict:
    :param layer_dict:
    :param layer_density_dicts:
    :param sec_index_dict:
    :param seglist:
    :param seed:
    :param neurotree_dict:
    :return:
    """
    segcounts_dict = {}
    layers_dict = {}
    segcount_total = 0
    if neurotree_dict is not None:
        secnodes_dict = neurotree_dict['section_topology']['nodes']
    else:
        secnodes_dict = None
    for (syn_type_label, layer_density_dict) in layer_density_dicts.iteritems():
        syn_type = syn_type_dict[syn_type_label]
        rans = {}
        for (layer_label, density_dict) in layer_density_dict.iteritems():
            if layer_label == 'default':
                layer = layer_label
            else:
                layer = layer_dict[layer_label]
            ran = h.Random(seed)
            ran.normal(density_dict['mean'], density_dict['variance'])
            rans[layer] = ran
        segcounts = []
        layers = []
        for seg in seglist:
            L = seg.sec.L
            nseg = seg.sec.nseg
            if neurotree_dict is not None:
                secindex = sec_index_dict[seg.sec]
                secnodes = secnodes_dict[secindex]
                layer = get_node_attribute('layer', neurotree_dict, seg.sec, secnodes, seg.x)
            else:
                layer = -1
            layers.append(layer)

            ran = None

            if layer > -1:
                if rans.has_key(layer):
                    ran = rans[layer]
                elif rans.has_key('default'):
                    ran = rans['default']
                else:
                    ran = None
            elif rans.has_key('default'):
                ran = rans['default']
            else:
                ran = None
            if ran is not None:
                l = L / nseg
                dens = ran.repick()
                rc = dens * l
                segcount_total += rc
                segcounts.append(rc)
            else:
                segcounts.append(0)

        segcounts_dict[syn_type] = segcounts
        layers_dict[syn_type] = layers
    return (segcounts_dict, segcount_total, layers_dict)


def distribute_uniform_synapses(seed, syn_type_dict, swc_type_dict, layer_dict, sec_layer_density_dict, neurotree_dict,
                                sec_dict, secidx_dict):
    """
    Computes uniformly-spaced synapse locations.
    :param seed:
    :param syn_type_dict:
    :param swc_type_dict:
    :param layer_dict:
    :param sec_layer_density_dict:
    :param neurotree_dict:
    :param sec_dict:
    :param secidx_dict:
    :return:
    """
    syn_ids = []
    syn_locs = []
    syn_secs = []
    syn_layers = []
    syn_types = []
    swc_types = []
    syn_index = 0

    for (sec_name, layer_density_dict) in sec_layer_density_dict.iteritems():
        sec_index_dict = secidx_dict[sec_name]
        swc_type = swc_type_dict[sec_name]
        seg_list = []
        sec_obj_index_dict = {}
        L_total = 0
        (seclst, maxdist) = sec_dict[sec_name]
        secidxlst = secidx_dict[sec_name]
        for (sec, secindex) in itertools.izip(seclst, secidxlst):
            sec_obj_index_dict[sec] = int(secindex)
            if maxdist is None:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0:
                        seg_list.append(seg)
            else:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0 and ((L_total + sec.L * seg.x) <= maxdist):
                        seg_list.append(seg)
            L_total += sec.L
        segcounts_dict, total, layers_dict = \
            synapse_seg_counts(syn_type_dict, layer_dict, layer_density_dict, sec_obj_index_dict, seg_list, seed,
                               neurotree_dict=neurotree_dict)

        sample_size = total
        cumcount = 0
        for (syn_type_label, _) in layer_density_dict.iteritems():
            syn_type = syn_type_dict[syn_type_label]
            segcounts = segcounts_dict[syn_type]
            layers = layers_dict[syn_type]
            for i in xrange(0, len(seg_list)):
                seg = seg_list[i]
                seg_start = seg.x - (0.5 / seg.sec.nseg)
                seg_end = seg.x + (0.5 / seg.sec.nseg)
                seg_range = seg_end - seg_start
                seg_count = segcounts[i]
                int_seg_count = math.floor(seg_count)
                layer = layers[i]
                syn_count = 0
                while syn_count < int_seg_count:
                    syn_loc = seg_start + seg_range * ((syn_count + 1) / math.ceil(seg_count))
                    assert ((syn_loc <= 1) & (syn_loc >= 0))
                    if syn_loc < 1.0:
                        syn_locs.append(syn_loc)
                        syn_ids.append(syn_index)
                        syn_secs.append(sec_obj_index_dict[seg.sec])
                        syn_layers.append(layer)
                        syn_types.append(syn_type)
                        swc_types.append(swc_type)
                        syn_index += 1
                        syn_count += 1
                cumcount += syn_count

    assert (len(syn_ids) > 0)
    syn_dict = {'syn_ids': np.asarray(syn_ids, dtype='uint32'),
                'syn_locs': np.asarray(syn_locs, dtype='float32'),
                'syn_secs': np.asarray(syn_secs, dtype='uint32'),
                'syn_layers': np.asarray(syn_layers, dtype='int8'),
                'syn_types': np.asarray(syn_types, dtype='uint8'),
                'swc_types': np.asarray(swc_types, dtype='uint8')}

    return syn_dict


def distribute_poisson_synapses(seed, syn_type_dict, swc_type_dict, layer_dict, sec_layer_density_dict, neurotree_dict,
                                sec_dict, secidx_dict, verbose):
    """
    Computes synapse locations according to a Poisson distribution.
    :param seed:
    :param syn_type_dict:
    :param swc_type_dict:
    :param layer_dict:
    :param sec_layer_density_dict:
    :param neurotree_dict:
    :param sec_dict:
    :param secidx_dict:
    :param verbose:
    :return:
    """
    syn_ids = []
    syn_locs = []
    syn_secs = []
    syn_layers = []
    syn_types = []
    swc_types = []
    syn_index = 0

    r = np.random.RandomState()

    for (sec_name, layer_density_dict) in sec_layer_density_dict.iteritems():

        sec_index_dict = secidx_dict[sec_name]
        swc_type = swc_type_dict[sec_name]
        seg_list = []
        sec_obj_index_dict = {}
        L_total = 0
        (seclst, maxdist) = sec_dict[sec_name]
        secidxlst = secidx_dict[sec_name]
        for (sec, secindex) in itertools.izip(seclst, secidxlst):
            sec_obj_index_dict[sec] = int(secindex)
            if maxdist is None:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0:
                        seg_list.append(seg)
            else:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0 and ((L_total + sec.L * seg.x) <= maxdist):
                        seg_list.append(seg)
            L_total += sec.L
        segdensity_dict, layers_dict = \
            synapse_seg_density(syn_type_dict, layer_dict, layer_density_dict, sec_obj_index_dict, seg_list, seed,
                                neurotree_dict=neurotree_dict)

        cumcount = 0
        for (syn_type_label, _) in layer_density_dict.iteritems():
            syn_type = syn_type_dict[syn_type_label]
            segdensity = segdensity_dict[syn_type]
            layers = layers_dict[syn_type]
            for i in xrange(len(seg_list)):
                seg = seg_list[i]
                seg_start = seg.x - (0.5 / seg.sec.nseg)
                seg_end = seg.x + (0.5 / seg.sec.nseg)
                seg_range = seg_end - seg_start
                L = seg.sec.L
                layer = layers[i]
                density = segdensity[i]
                syn_count = 0
                if density > 0.:
                    beta = 1. / density
                    interval = r.exponential(beta)
                    while interval < seg_range * L:
                        syn_loc = seg_start + interval / L
                        assert ((syn_loc <= 1) & (syn_loc >= 0))
                        if syn_loc < 1.0:
                            syn_locs.append(syn_loc)
                            syn_ids.append(syn_index)
                            syn_secs.append(sec_obj_index_dict[seg.sec])
                            syn_layers.append(layer)
                            syn_types.append(syn_type)
                            swc_types.append(swc_type)
                            syn_index += 1
                            syn_count += 1
                        interval += r.exponential(beta)
                cumcount += syn_count

    assert (len(syn_ids) > 0)
    syn_dict = {'syn_ids': np.asarray(syn_ids, dtype='uint32'),
                'syn_locs': np.asarray(syn_locs, dtype='float32'),
                'syn_secs': np.asarray(syn_secs, dtype='uint32'),
                'syn_layers': np.asarray(syn_layers, dtype='int8'),
                'syn_types': np.asarray(syn_types, dtype='uint8'),
                'swc_types': np.asarray(swc_types, dtype='uint8')}

    return syn_dict



def add_unique_synapse(mech_name, seg, syns_dict):
    """Creates a synapse in the given segment."""
    syn = make_syn_mech(mech_name, seg)
    return syn
    
def mksyns(gid,cell,syn_ids,syn_types,swc_types,syn_locs,syn_sections,syn_kinetic_params,env,add_synapse=add_shared_synapse,
           spines=False):

    syns_dict_dend = defaultdict(lambda: defaultdict(lambda: {}))
    syns_dict_axon = defaultdict(lambda: defaultdict(lambda: {}))
    syns_dict_ais  = defaultdict(lambda: defaultdict(lambda: {}))
    syns_dict_hill = defaultdict(lambda: defaultdict(lambda: {}))
    syns_dict_soma = defaultdict(lambda: defaultdict(lambda: {}))
    py_sections    = [sec for sec in cell.sections]

    syn_type_excitatory = env.Synapse_Types['excitatory']
    syn_type_inhibitory = env.Synapse_Types['inhibitory']

    swc_type_apical = env.SWC_Types['apical']
    swc_type_basal  = env.SWC_Types['basal']
    swc_type_soma   = env.SWC_Types['soma']
    swc_type_axon   = env.SWC_Types['axon']
    swc_type_ais    = env.SWC_Types['ais']
    swc_type_hill   = env.SWC_Types['hillock']
    
    syn_obj_dict = {}

    for i in xrange(0, syn_ids.size):

      syn_id      = syn_ids[i]
      if not (syn_id < syn_types.size):
          logger.error('mksyns syn_ids for gid %i: ' % gid, syn_ids)
          raise ValueError('mksyns: cell %i received invalid syn_id %d' % (gid, syn_id))
      
      syn_type    = syn_types[syn_id]
      swc_type    = swc_types[syn_id]
      syn_loc     = syn_locs[syn_id]
      syn_section = syn_sections[syn_id]
      
      sref = None
      sec = py_sections[syn_section]
      if swc_type == swc_type_apical:
        syns_dict = syns_dict_dend
        if syn_type == syn_type_excitatory: 
            if spines and h.ismembrane('spines',sec=sec):
                sec(syn_loc).count_spines += 1
      elif swc_type == swc_type_basal:
        syns_dict = syns_dict_dend
        if syn_type == syn_type_excitatory: 
            if spines and h.ismembrane('spines',sec=sec):
                sec(syn_loc).count_spines += 1
      elif swc_type == swc_type_axon:
        syns_dict = syns_dict_axon
      elif swc_type == swc_type_ais:
        syns_dict = syns_dict_ais
      elif swc_type == swc_type_hill:
        syns_dict = syns_dict_hill
      elif swc_type == swc_type_soma:
        syns_dict = syns_dict_soma
      else: 
        raise RuntimeError ("Unsupported synapse SWC type %d" % swc_type)
      syn_mech_dict = {}
      for (syn_mech, params) in syn_kinetic_params.iteritems():
        syn = add_synapse(syn_mech, sec(syn_loc), syns_dict)
        syn.tau1 = params['t_rise']
        syn.tau2 = params['t_decay']
        syn.e    = params['e_rev']
        cell.syns.append(syn)
        cell.syntypes.o(syn_type).append(syn)
        syn_mech_dict[syn_mech] = syn
      syn_obj_dict[syn_id] = syn_mech_dict
        
    if spines:
        cell.correct_for_spines()

    return syn_obj_dict

#New version of function, for use with dentate.cells
def mk_syns(gid, cell, syn_ids, syn_types, swc_types, syn_locs, syn_sections, syn_kinetic_params, env,
            add_synapse=add_shared_synapse, spines=False):
    syns_dict_dend = defaultdict(lambda: defaultdict(lambda: {}))
    syns_dict_axon = defaultdict(lambda: defaultdict(lambda: {}))
    syns_dict_ais = defaultdict(lambda: defaultdict(lambda: {}))
    syns_dict_hill = defaultdict(lambda: defaultdict(lambda: {}))
    syns_dict_soma = defaultdict(lambda: defaultdict(lambda: {}))
    py_sections = [sec for sec in cell.sections]

    syn_type_excitatory = env.Synapse_Types['excitatory']
    syn_type_inhibitory = env.Synapse_Types['inhibitory']

    swc_type_apical = env.SWC_Types['apical']
    swc_type_basal = env.SWC_Types['basal']
    swc_type_soma = env.SWC_Types['soma']
    swc_type_axon = env.SWC_Types['axon']
    swc_type_ais = env.SWC_Types['ais']
    swc_type_hill = env.SWC_Types['hillock']

    syn_obj_dict = {}

    for i in xrange(0, syn_ids.size):

        syn_id = syn_ids[i]
        if not (syn_id < syn_types.size):
            logger.error('mksyns syn_ids for gid %i: ' % gid, syn_ids)
            raise ValueError('mksyns: cell %i received invalid syn_id %d' % (gid, syn_id))

        syn_type = syn_types[syn_id]
        swc_type = swc_types[syn_id]
        syn_loc = syn_locs[syn_id]
        syn_section = syn_sections[syn_id]

        sref = None
        sec = py_sections[syn_section]
        if swc_type == swc_type_apical:
            syns_dict = syns_dict_dend
            if syn_type == syn_type_excitatory:
                if spines and h.ismembrane('spines', sec=sec):
                    sec(syn_loc).count_spines += 1
        elif swc_type == swc_type_basal:
            syns_dict = syns_dict_dend
            if syn_type == syn_type_excitatory:
                if spines and h.ismembrane('spines', sec=sec):
                    sec(syn_loc).count_spines += 1
        elif swc_type == swc_type_axon:
            syns_dict = syns_dict_axon
        elif swc_type == swc_type_ais:
            syns_dict = syns_dict_ais
        elif swc_type == swc_type_hill:
            syns_dict = syns_dict_hill
        elif swc_type == swc_type_soma:
            syns_dict = syns_dict_soma
        else:
            raise RuntimeError("Unsupported synapse SWC type %d" % swc_type)
        syn_mech_dict = {}
        for (syn_mech, params) in syn_kinetic_params.iteritems():
            syn = add_synapse(syn_mech, sec(syn_loc), syns_dict, env.synapse_mech_name_dict)
            syn.tau1 = params['t_rise']
            syn.tau2 = params['t_decay']
            syn.e = params['e_rev']
            cell.syns.append(syn)
            cell.syntypes.o(syn_type).append(syn)
            syn_mech_dict[syn_mech] = syn
        syn_obj_dict[syn_id] = syn_mech_dict

    return syn_obj_dict
        

