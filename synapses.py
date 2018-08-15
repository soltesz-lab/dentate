
from dentate.neuron_utils import *
from dentate.cells import get_mech_rules_dict, get_donor, get_distance_to_node, get_param_val_by_distance, import_mech_dict_from_file, custom_filter_by_branch_order, custom_filter_by_terminal, make_neurotree_graph
import networkx as nx


# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)


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
        if syn_id in self.syn_mech_attr_dict[gid] and syn_name in self.syn_mech_attr_dict[gid][syn_id] and \
                'netcon' in self.syn_mech_attr_dict[gid][syn_id][syn_name]:
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
        if syn_id in self.syn_mech_attr_dict[gid] and syn_name in self.syn_mech_attr_dict[gid][syn_id] and \
                'attrs' in self.syn_mech_attr_dict[gid][syn_id][syn_name] and \
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

    def get_filtered_syn_indexes(self, gid, syn_indexes=None, syn_types=None, layers=None, sources=None,
                                 swc_types=None):
        """

        :param gid: int
        :param syn_indexes: array of int
        :param syn_types: list of enumerated type: synapse category
        :param layers: list of enumerated type: layer
        :param sources: list of enumerated type: population names of source projections
        :param swc_types: list of enumerated type: swc_type
        :return: array of int
        """
        syn_id_attr_dict = self.syn_id_attr_dict[gid]
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
        if source_names.has_key(source_id):
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
    syn_params = env.connection_config[postsyn_name][presyn_name].mechanisms

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

    if rank == 0:
        logger.info('insert_syns_from_mech_attrs: source: %s; target: %s cell %i: created %i syns and %i netcons for '
                    '%i syn_ids' % (presyn_name, postsyn_name, gid, syn_count, nc_count, len(syn_ids)))


def config_syns_from_mech_attrs(gid, env, postsyn_name, syn_ids=None, insert=False, unique=None, verbose=False):
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
            syn_names = env.connection_config[postsyn_name][presyn_name].mechanisms.keys()
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
        syn_names = env.connection_config[postsyn_name][presyn_name].mechanisms.keys()
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

    if verbose and rank == 0:
        logger.info('config_syns_from_mech_attrs: population: %s; cell %i: updated mech_params for %i syns and %i '
                    'netcons for %i syn_ids' % (postsyn_name, gid, syn_count, nc_count, len(syn_ids)))


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
    syn = syn_in_seg(syn_name, seg, syns_dict)
    if syn is None:
        if mech_names is not None:
            mech_name = mech_names[syn_name]
        else:
            mech_name = syn_name
        syn = make_syn_mech(mech_name, seg)
        syns_dict[seg.sec][seg.x][syn_name] = syn
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
                if int(nc.wcnt()) >= i:
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
            syn_obj_dict[syn_id][syn_name] = syn

    #if rank == 0 and edge_count == 0:
    #    sec = syns_dict.iterkeys().next()
    #    logger.info('syns_dict[%s]:' % sec.hname())
    #    logger.info('%s' % pprint.pformat(syns_dict[sec]))

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


# ------------------------------- Methods to specify synaptic mechanisms  -------------------------------------------- #


def get_syn_filter_dict(env, rules, convert=False):
    """
    Used by modify_syn_mech_param. Takes in a series of arguments and constructs a validated rules dictionary that
    specifies to which sets of synapses a rule applies. Values of filter queries are validated by the provided Env.
    :param env: :class:'Env'
    :param rules: dict
    :param convert: bool; whether to convert string values to enumerated type
    :return: dict
    """
    valid_filter_names = ['syn_types', 'layers', 'sources']
    for name in rules:
        if name not in valid_filter_names:
            raise ValueError('get_syn_filter_dict: unrecognized filter category: %s' % name)
    rules_dict = copy.deepcopy(rules)
    if 'syn_types' in rules_dict:
        for i, syn_type in enumerate(rules_dict['syn_types']):
            if syn_type not in env.Synapse_Types:
                raise ValueError('get_syn_filter_dict: syn_type: %s not recognized by network configuration' %
                                 syn_type)
            if convert:
                rules_dict['syn_types'][i] = env.Synapse_Types[syn_type]
    if 'layers' in rules_dict:
        for i, layer in enumerate(rules_dict['layers']):
            if layer not in env.layers:
                raise ValueError('get_syn_filter_dict: layer: %s not recognized by network configuration' % layer)
            if convert:
                rules_dict['layers'][i] = env.layers[layer]
    if 'sources' in rules_dict:
        for i, source in enumerate(rules_dict['sources']):
            if source not in env.pop_dict:
                raise ValueError('get_syn_filter_dict: presynaptic population: %s not recognized by network '
                                 'configuration' % source)
            if convert:
                rules_dict['sources'][i] = env.pop_dict[source]
    return rules_dict


def validate_syn_mech_param(env, syn_name, param_name):
    """

    :param env: :class:'Env'
    :param syn_name: str
    :param param_name: str
    :return: bool
    """
    syn_mech_names = env.synapse_attributes.syn_mech_names
    if syn_name not in syn_mech_names:
        return False
    syn_param_rules = env.synapse_attributes.syn_param_rules
    mech_name = syn_mech_names[syn_name]
    if mech_name not in syn_param_rules:
        return False
    if 'mech_params' in syn_param_rules[mech_name] and param_name in syn_param_rules[mech_name]['mech_params']:
        return True
    if 'netcon_params' in syn_param_rules[mech_name] and param_name in syn_param_rules[mech_name]['netcon_params']:
        return True
    return False


def modify_syn_mech_param(cell, env, sec_type, syn_name, param_name=None, value=None, origin=None, slope=None, tau=None,
                          xhalf=None, min=None, max=None, min_loc=None, max_loc=None, outside=None, custom=None,
                          append=False, filters=None, origin_filters=None, update_targets=False):
    """
    Modifies a cell's mechanism dictionary to specify attributes of a synaptic mechanism by sec_type. This method is
    meant to be called manually during initial model specification, or during parameter optimization. For modifications
    to persist across simulations, the mechanism dictionary must be saved to a file using export_mech_dict() and
    re-imported during BiophysCell initialization.
    Calls update_syn_mech_by_sec_type to set placeholder values in the syn_mech_attrs_dict of a SynapseAttributes
    object. If update_targets flag is True, the attributes of any target synaptic point_process and netcon objects that
    have been inserted will also be updated. Otherwise, they can be updated separately by calling
    config_syns_from_mech_attrs.
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param sec_type: str
    :param syn_name: str
    :param param_name: str
    :param value: float
    :param origin: str (sec_type)
    :param slope: float
    :param tau: float
    :param xhalf: float
    :param min: float
    :param max: float
    :param min_loc: float
    :param max_loc: float
    :param outside: float
    :param custom: dict
    :param append: bool
    :param filters: dict
    :param origin_filters: dict
    :param update_targets: bool
    """
    if sec_type not in cell.nodes:
        raise ValueError('modify_syn_mech_param: sec_type: %s not in cell' % sec_type)
    if param_name is None:
        raise ValueError('modify_syn_mech_param: missing required parameter to modify synaptic mechanism: %s '
                         'in sec_type: %s' % (syn_name, sec_type))
    if not validate_syn_mech_param(env, syn_name, param_name):
        raise ValueError('modify_syn_mech_param: synaptic mechanism: %s or parameter: %s not recognized by network '
                         'configuration' % (syn_name, param_name))
    if value is None:
        if origin is None:
            raise ValueError('modify_syn_mech_param: mechanism: %s; parameter: %s; missing origin or value for '
                             'sec_type: %s' % (syn_name, param_name, sec_type))
        elif origin_filters is None:
            raise ValueError('modify_syn_mech_param: mechanism: %s; parameter: %s; sec_type: %s cannot inherit from '
                             'origin: %s without origin_filters' % (syn_name, param_name, sec_type, origin))
    rules = get_mech_rules_dict(cell, value=value, origin=origin, slope=slope, tau=tau, xhalf=xhalf, min=min,
                                      max=max, min_loc=min_loc, max_loc=max_loc, outside=outside, custom=custom)
    if filters is not None:
        syn_filters = get_syn_filter_dict(env, filters)
        rules['filters'] = syn_filters

    if origin_filters is not None:
        origin_syn_filters = get_syn_filter_dict(env, origin_filters)
        rules['origin_filters'] = origin_syn_filters

    backup_mech_dict = copy.deepcopy(cell.mech_dict)

    mech_content = {param_name: rules}

    # No mechanisms have been specified in this type of section yet
    if sec_type not in cell.mech_dict:
        cell.mech_dict[sec_type] = {'synapses': {syn_name: mech_content}}
    # No synaptic mechanisms have been specified in this type of section yet
    elif 'synapses' not in cell.mech_dict[sec_type]:
        cell.mech_dict[sec_type]['synapses'] = {syn_name: mech_content}
    # Synaptic mechanisms have been specified in this type of section, but not this syn_name
    elif syn_name not in cell.mech_dict[sec_type]['synapses']:
        cell.mech_dict[sec_type]['synapses'][syn_name] = mech_content
    # This parameter of this syn_name has already been specified in this type of section, and the user wants to append
    # a new rule set
    elif param_name in cell.mech_dict[sec_type]['synapses'][syn_name] and append:
        if isinstance(cell.mech_dict[sec_type]['synapses'][syn_name][param_name], dict):
            cell.mech_dict[sec_type]['synapses'][syn_name][param_name] = \
                [cell.mech_dict[sec_type]['synapses'][syn_name][param_name], rules]
        elif isinstance(cell.mech_dict[sec_type]['synapses'][syn_name][param_name], list):
            cell.mech_dict[sec_type]['synapses'][syn_name][param_name].append(rules)
    # This syn_name has been specified, but not this parameter, or the user wants to replace an existing rule set
    else:
       cell.mech_dict[sec_type]['synapses'][syn_name][param_name] = rules

    try:
        update_syn_mech_by_sec_type(cell, env, sec_type, syn_name, mech_content, update_targets)
    except (KeyError, ValueError, AttributeError, NameError, RuntimeError, IOError) as e:
        cell.mech_dict = copy.deepcopy(backup_mech_dict)
        raise RuntimeError, 'modify_syn_mech_param: problem updating mechanism: %s; parameter: %s; in sec_type: %s\n' \
                            '%s' % (syn_name, param_name, sec_type, e), sys.exc_info()[2]


def update_syn_mech_by_sec_type(cell, env, sec_type, syn_name, mech_content, update_targets=False):
    """
    For the provided sec_type and synaptic mechanism, this method loops through the parameters specified in the
    mechanism dictionary, interprets the rules, and sets placeholder values in the syn_mech_attr_dict of a
    SynapseAttributes object.
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param sec_type: str
    :param syn_name: str
    :param mech_content: dict
    :param update_targets: bool
    """
    for param_name in mech_content:
        # accommodate either a dict, or a list of dicts specifying rules for a single parameter
        if isinstance(mech_content[param_name], dict):
            update_syn_mech_param_by_sec_type(cell, env, sec_type, syn_name, param_name, mech_content[param_name],
                                              update_targets)
        elif isinstance(mech_content[param_name], list):
            for mech_content_entry in mech_content[param_name]:
                # print mech_content_entry
                update_syn_mech_param_by_sec_type(cell, env, sec_type, syn_name, param_name, mech_content_entry,
                                                  update_targets)


def update_syn_mech_param_by_sec_type(cell, env, sec_type, syn_name, param_name, rules, update_targets=False):
    """
    For the provided synaptic mechanism and parameter, this method loops through nodes of the provided sec_type,
    interprets the provided rules, and sets placeholder values in the syn_mech_attr_dict of a SynapseAttributes object.
    If filter queries are provided, their values are converted to enumerated types.
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param sec_type: str
    :param syn_name: str
    :param param_name: str
    :param rules: dict
    :param update_targets: bool
    """
    new_rules = copy.deepcopy(rules)
    if 'filters' in new_rules:
        filters = get_syn_filter_dict(env, new_rules['filters'], convert=True)
        del new_rules['filters']
    else:
        filters = None
    if 'origin_filters' in new_rules:
        origin_filters = get_syn_filter_dict(env, new_rules['origin_filters'], convert=True)
        del new_rules['origin_filters']
    else:
        origin_filters = None
    if sec_type in cell.nodes:
        for node in cell.nodes[sec_type]:
            update_syn_mech_param_by_node(cell, env, node, syn_name, param_name, new_rules, filters, origin_filters,
                                          update_targets)


def update_syn_mech_param_by_node(cell, env, node, syn_name, param_name, rules, filters=None, origin_filters=None,
                                  update_targets=False):
    """
    For the provided synaptic mechanism and parameter, this method first determines the set of placeholder synapses in
    the provided node that match any provided filters. Then calls parse_syn_mech_rules to interpret the provided rules,
    and set placeholder values in the syn_mech_attr_dict of a SynapseAttributes object.
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param node: :class:'SHocNode'
    :param syn_name: str
    :param param_name: str
    :param rules: dict
    :param filters: dict: {category: list of int}
    :param origin_filters: dict: {category: list of int}
    :param update_targets: bool
    """
    gid = cell.gid
    syn_attrs = env.synapse_attributes
    syn_id_attr_dict = syn_attrs.syn_id_attr_dict[gid]
    sec_index_map = syn_attrs.sec_index_map[gid]
    if filters is None:
        filtered_syn_indexes = sec_index_map[node.index]
    else:
        filtered_syn_indexes = syn_attrs.get_filtered_syn_indexes(gid, sec_index_map[node.index], **filters)
    if len(filtered_syn_indexes) > 0:
        syn_ids = syn_id_attr_dict['syn_ids'][filtered_syn_indexes]
        parse_syn_mech_rules(cell, env, node, syn_ids, syn_name, param_name, rules, origin_filters,
                             update_targets=update_targets)


def parse_syn_mech_rules(cell, env, node, syn_ids, syn_name, param_name, rules, origin_filters=None, donor=None,
                         update_targets=False):
    """
    Provided a synaptic mechanism, a parameter, a node, a list of syn_ids, and a dict of rules. Interprets the provided
    rules, including complex gradient and inheritance rules. Gradients can be specified as linear, exponential, or
    sigmoidal. Custom functions can also be provided to specify more complex distributions. Calls inherit_syn_mech_param
    to retrieve a value from a donor node, if necessary. Calls set_syn_mech_params to sets placeholder values in the
    syn_mech_attr_dict of a SynapseAttributes object.
    1) A 'value' with no 'origin' requires no further processing
    2) An 'origin' with no 'value' requires a donor node to inherit a baseline value
    3) An 'origin' with a 'value' requires a donor node to use as a reference point for applying a distance-dependent
    gradient
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param node: :class:'SHocNode'
    :param syn_ids: array of int
    :param syn_name: str
    :param param_name: str
    :param rules: dict
    :param origin_filters: dict: {category: list of int}
    :param donor: :class:'SHocNode'
    :param update_targets: bool
    """
    if 'origin' in rules and donor is None:
        donor = get_donor(cell, node, rules['origin'])
        if donor is None:
            raise RuntimeError('parse_syn_mech_rules: problem identifying donor of origin_type: %s for synaptic '
                               'mechanism: %s parameter: %s in sec_type: %s' %
                               (rules['origin'], syn_name, param_name, node.type))
    if 'value' in rules:
        baseline = rules['value']
    elif donor is None:
        raise RuntimeError('parse_syn_mech_rules: cannot set value of synaptic mechanism: %s parameter: %s in '
                           'sec_type: %s without a provided origin or value' % (syn_name, param_name, node.type))
    else:
        baseline = inherit_syn_mech_param(cell, env, donor, syn_name, param_name, origin_filters)
    if 'custom' in rules:
        parse_custom_syn_mech_rules(cell, env, node, syn_ids, syn_name, param_name, baseline, rules, donor,
                                    update_targets)
    else:
        set_syn_mech_param(cell, env, node, syn_ids, syn_name, param_name, baseline, rules, donor, update_targets)


def inherit_syn_mech_param(cell, env, donor, syn_name, param_name, origin_filters=None):
    """
    Follows path from the provided donor node to root until synapses are located that match the provided filter. Returns
    the requested parameter value from the synapse closest to the end of the section.
    for the requested parameter.
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param donor: :class:'SHocNode'
    :param syn_name: str
    :param param_name: str
    :param origin_filters: dict: {category: list of int}
    :return: float
    """
    gid = cell.gid
    syn_attrs = env.synapse_attributes
    syn_id_attr_dict = syn_attrs.syn_id_attr_dict[gid]
    sec_index_map = syn_attrs.sec_index_map[gid]
    if origin_filters is None:
        filtered_syn_indexes = sec_index_map[donor.index]
    else:
        filtered_syn_indexes = syn_attrs.get_filtered_syn_indexes(gid, sec_index_map[donor.index], **origin_filters)
    if len(filtered_syn_indexes) > 0:
        valid_syn_indexes = []
        for syn_index in filtered_syn_indexes:
            syn_id = syn_id_attr_dict['syn_ids'][syn_index]
            if syn_attrs.has_mech_attrs(gid, syn_id, syn_name):
                valid_syn_indexes.append(syn_index)
        if len(valid_syn_indexes) > 0:
            valid_syn_indexes.sort(key=lambda x: syn_id_attr_dict['syn_locs'][x])
            syn_id = syn_id_attr_dict['syn_ids'][valid_syn_indexes[-1]]
            return syn_attrs.get_mech_attrs(gid, syn_id, syn_name)[param_name]
    if donor is cell.tree.root:
        return
    else:
        return inherit_syn_mech_param(cell, env, donor.parent, syn_name, param_name, origin_filters)


def set_syn_mech_param(cell, env, node, syn_ids, syn_name, param_name, baseline, rules, donor=None,
                       update_targets=False):
    """
    Provided a synaptic mechanism, a parameter, a node, a list of syn_ids, and a dict of rules. Sets placeholder values
    for each provided syn_id in the syn_mech_attr_dict of a SynapseAttributes object. If the provided rules specify a
    distance-dependent gradient, a baseline value and a donor node are required as reference points.
    If update_targets flag is True, the attributes of any target synaptic point_process and netcon objects that have
    been inserted will also be updated. Otherwise, they can be updated separately by calling
    config_syns_from_mech_attrs.
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param node: :class:'SHocNode'
    :param syn_ids: array of int
    :param syn_name: str
    :param param_name: str
    :param baseline: float
    :param rules: dict
    :param donor: :class:'SHocNode'
    :param update_targets: bool
    """
    if not ('min_loc' in rules or 'max_loc' in rules or 'slope' in rules):
        for syn_id in syn_ids:
            syn_attrs = env.synapse_attributes
            syn_attrs.set_mech_attrs(cell.gid, syn_id, syn_name, {param_name: baseline})
    elif donor is None:
        raise RuntimeError('set_syn_mech_param: cannot set value of synaptic mechanism: %s parameter: %s in '
                           'sec_type: %s without a provided donor node' % (syn_name, param_name, node.type))
    else:
        min_distance = rules['min_loc'] if 'min_loc' in rules else 0.
        max_distance = rules['max_loc'] if 'max_loc' in rules else None
        min_val = rules['min'] if 'min' in rules else None
        max_val = rules['max'] if 'max' in rules else None
        slope = rules['slope'] if 'slope' in rules else None
        tau = rules['tau'] if 'tau' in rules else None
        xhalf = rules['xhalf'] if 'xhalf' in rules else None
        outside = rules['outside'] if 'outside' in rules else None

        gid = cell.gid
        syn_attrs = env.synapse_attributes
        syn_id_attr_dict = syn_attrs.syn_id_attr_dict[gid]
        syn_id_attr_index_map = syn_attrs.syn_id_attr_index_map[gid]

        for syn_id in syn_ids:
            syn_index = syn_id_attr_index_map[syn_id]
            syn_loc = syn_id_attr_dict['syn_locs'][syn_index]
            distance = get_distance_to_node(cell, donor, node, syn_loc)
            value = get_param_val_by_distance(distance, baseline, slope, min_distance, max_distance, min_val, max_val,
                                              tau, xhalf, outside)
            if value is not None:
                syn_attrs.set_mech_attrs(cell.gid, syn_id, syn_name, {param_name: value})
    if update_targets:
        config_syns_from_mech_attrs(cell.gid, env, cell.pop_name, syn_ids=syn_ids, insert=False)


def parse_custom_syn_mech_rules(cell, env, node, syn_ids, syn_name, param_name, baseline, rules, donor,
                                update_targets=False):
    """
    If the provided node meets custom criteria, rules are modified and passed back to parse_mech_rules with the
    'custom' item removed. Avoids having to determine baseline and donor over again.
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param node: :class:'SHocNode'
    :param syn_ids: array of int
    :param syn_name: str
    :param param_name: str
    :param baseline: float
    :param rules: dict
    :param origin_filters: dict: {category: list of int}
    :param donor: :class:'SHocNode' or None
    :param update_targets: bool
    """
    if 'func' not in rules['custom'] or rules['custom']['func'] is None:
        raise RuntimeError('parse_custom_syn_mech_rules: no custom function provided for synaptic mechanism: %s '
                           'parameter: %s in sec_type: %s' % (syn_name, param_name, node.type))
    if rules['custom']['func'] in globals() and callable(globals()[rules['custom']['func']]):
        func = globals()[rules['custom']['func']]
    else:
        raise RuntimeError('parse_custom_syn_mech_rules: problem locating custom function: %s for synaptic '
                           'mechanism: %s parameter: %s in sec_type: %s' %
                           (rules['custom']['func'], syn_name, param_name, node.type))
    custom = copy.deepcopy(rules['custom'])
    del custom['func']
    new_rules = copy.deepcopy(rules)
    del new_rules['custom']
    new_rules['value'] = baseline
    new_rules = func(cell, node, baseline, new_rules, donor, **custom)
    if new_rules:
        parse_syn_mech_rules(cell, env, node, syn_ids, syn_name, param_name, new_rules, donor=donor,
                             update_targets=update_targets)


def init_syn_mech_attrs(cell, env=None, mech_file_path=None, from_file=False, update_targets=False):
    """
    Consults a dictionary specifying parameters of NEURON synaptic mechanisms (point processes) for each type of section
    in a BiophysCell. Traverses through the tree of SHocNode nodes following order of inheritance. Calls
    update_syn_mech_by_sec_type to set placeholder values in the syn_mech_attrs_dict of a SynapseAttributes object. If
    update_targets flag is True, the attributes of any target synaptic point_process and netcon objects that have been
    inserted will also be updated. Otherwise, they can be updated separately by calling config_syns_from_mech_attrs.
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param mech_file_path: str (path)
    :param from_file: bool
    :param update_targets: bool
    """
    if from_file:
        import_mech_dict_from_file(cell, mech_file_path)
    for sec_type in default_ordered_sec_types:
        if sec_type in cell.mech_dict and sec_type in cell.nodes:
            if cell.nodes[sec_type] and 'synapses' in cell.mech_dict[sec_type]:
                for syn_name in cell.mech_dict[sec_type]['synapses']:
                    update_syn_mech_by_sec_type(cell, env, sec_type, syn_name,
                                                cell.mech_dict[sec_type]['synapses'][syn_name],
                                                update_targets=update_targets)


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


def make_synapse_graph(syn_dict, neurotree_dict, return_root=False):
    """
    Creates a graph of synapses that follows the topological organization of the given neuron.
    :param syn_dict:
    :param neurotree_dict:
    :return: NetworkX.DiGraph
    """
    sec_graph = make_neurotree_graph(neurotree_dict)

    syn_ids  = syn_dict['syn_ids']
    syn_locs = syn_dict['syn_locs']
    syn_secs = syn_dict['syn_secs']

    sec_syn_dict = defaultdict(list)
    for syn_id, sec_id, syn_loc in itertools.izip(syn_ids, syn_secs, syn_locs):
        sec_syn_dict[sec_id].append(syn_id, syn_loc)
        
    syn_graph = nx.DiGraph()

    for sec_id, syn_id_locs in sec_syn_dict.iteritems():

        sec_parents = sec_graph.ancestors(sec_id)
        sec_children = sec_graph.descendants(sec_id)

        assert(len(sec_parents) <= 1)
        if len(sec_parents) > 0:
            sec_parent = sec_parents[0]
        else:
            sec_parent = None

        syn_id_locs.sort(key=lambda x: x[1])
            
        if sec_parent:
            syn_graph.add_edge(sec_syn_dict[sec_parent][-1], syn_id_locs[0][0])

        for sec_child in sec_children:
            syn_graph.add_edge(syn_id_locs[-1][0], sec_syn_dict[sec_child][0])
            
        for i, j in itertools.izip(syn_id_locs[:-1], syn_id_locs[1:]):
            syn_graph.add_edge(i[0], j[0])

    if return_root:
        order = ns.topological_sort(syn_graph)
        root = order[0]
    else:
        root = None
        
    if root:
        return (syn_graph, root)
    else:
        return syn_graph
    
    

    
def synapse_seg_density(syn_type_dict, layer_dict, layer_density_dicts, seg_dict, seed, neurotree_dict=None):
    """
    Computes per-segment density of synapse placement.
    :param syn_type_dict:
    :param layer_dict:
    :param layer_density_dicts:
    :param seg_dict:
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
        segdensity = defaultdict(list)
        layers = defaultdict(list)
        for sec_index, seg_list in seg_dict.iteritems():
            for seg in seg_list:
                L = seg.sec.L
                nseg = seg.sec.nseg
                if neurotree_dict is not None:
                    secnodes = secnodes_dict[sec_index]
                    layer = get_node_attribute('layer', neurotree_dict, seg.sec, secnodes, seg.x)
                else:
                    layer = -1
                layers[sec_index].append(layer)

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
                else:
                    dens = 0
                segdensity[sec_index].append(dens)

        segdensity_dict[syn_type] = segdensity
        layers_dict[syn_type] = layers
    return (segdensity_dict, layers_dict)


def synapse_seg_counts(syn_type_dict, layer_dict, layer_density_dicts, sec_index_dict, seg_dict, seed, neurotree_dict=None):
    """
    Computes per-segment relative counts of synapse placement.
    :param syn_type_dict:
    :param layer_dict:
    :param layer_density_dicts:
    :param sec_index_dict:
    :param seg_dict:
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
        for sec_index, seg_list in sec_dict.iteritems():
            for seg in seg_list:
                L = seg.sec.L
                nseg = seg.sec.nseg
                if neurotree_dict is not None:
                    secnodes = secnodes_dict[sec_index]
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



def distribute_uniform_synapses(density_seed, syn_type_dict, swc_type_dict, layer_dict, sec_layer_density_dict, neurotree_dict,
                                cell_sec_dict, cell_secidx_dict):
    """
    Computes uniformly-spaced synapse locations.
    :param density_seed:
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

    segcounts_per_sec = {}
    for (sec_name, layer_density_dict) in sec_layer_density_dict.iteritems():
        sec_index_dict = secidx_dict[sec_name]
        swc_type = swc_type_dict[sec_name]
        seg_list = []
        L_total = 0
        (seclst, maxdist) = cell_sec_dict[sec_name]
        secidxlst = cell_secidx_dict[sec_name]
        sec_dict = { int(idx): sec for sec, idx in itertools.izip(seclst, secidxlst) }
        seg_dict = {}
        for (sec_index, sec) in sec_dict.iteritems():
            seg_list = []
            if maxdist is None:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0:
                        seg_list.append(seg)
            else:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0 and ((L_total + sec.L * seg.x) <= maxdist):
                        seg_list.append(seg)
            L_total += sec.L
            seg_dict[sec_index] = seg_list
        segcounts_dict, total, layers_dict = \
            synapse_seg_counts(syn_type_dict, layer_dict, layer_density_dict, seg_dict, density_seed,
                               neurotree_dict=neurotree_dict)
        segcounts_per_sec[sec_name] = segcounts_dict
        sample_size = total
        for (syn_type_label, _) in layer_density_dict.iteritems():
            syn_type = syn_type_dict[syn_type_label]
            segcounts = segcounts_dict[syn_type]
            layers = layers_dict[syn_type]
            for sec_index, seg_list in seg_dict.iteritems():
                for seg, layer, seg_count in itertools.izip(seg_list, layers, segcounts):
                    seg_start = seg.x - (0.5 / seg.sec.nseg)
                    seg_end = seg.x + (0.5 / seg.sec.nseg)
                    seg_range = seg_end - seg_start
                    int_seg_count = math.floor(seg_count)
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

    assert (len(syn_ids) > 0)
    syn_dict = {'syn_ids': np.asarray(syn_ids, dtype='uint32'),
                'syn_locs': np.asarray(syn_locs, dtype='float32'),
                'syn_secs': np.asarray(syn_secs, dtype='uint32'),
                'syn_layers': np.asarray(syn_layers, dtype='int8'),
                'syn_types': np.asarray(syn_types, dtype='uint8'),
                'swc_types': np.asarray(swc_types, dtype='uint8')}

    return (syn_dict, segcounts_per_dict)

def distribute_poisson_synapses(density_seed, syn_type_dict, swc_type_dict, layer_dict, sec_layer_density_dict, neurotree_dict,
                                cell_sec_dict, cell_secidx_dict, traversal_order='bfs'):
    """
    Computes synapse locations according to a Poisson distribution.
    :param density_seed:
    :param syn_type_dict:
    :param swc_type_dict:
    :param layer_dict:
    :param sec_layer_density_dict:
    :param neurotree_dict:
    :param cell_sec_dict:
    :param cell_secidx_dict:
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

    sec_graph = make_neurotree_graph(neurotree_dict, return_root=False)

    seg_density_per_sec = {}
    for (sec_name, layer_density_dict) in sec_layer_density_dict.iteritems():

        swc_type = swc_type_dict[sec_name]
        seg_dict = {}
        L_total = 0
        (seclst, maxdist) = cell_sec_dict[sec_name]
        secidxlst = cell_secidx_dict[sec_name]
        sec_dict = { int(idx): sec for sec, idx in itertools.izip(seclst, secidxlst) }
        if len(sec_dict) > 1:
            sec_subgraph = sec_graph.subgraph(sec_dict.keys())
            if len(sec_subgraph.edges()) > 0:
                sec_roots = [ n for n,d in sec_subgraph.in_degree() if d==0 ] 
                sec_edges = []
                for sec_root in sec_roots:
                    if traversal_order == 'dfs':
                        sec_edges.append(list(nx.dfs_edges(sec_subgraph, sec_root)))
                    elif traversal_order == 'bfs':
                        sec_edges.append(list(nx.bfs_edges(sec_subgraph, sec_root)))
                    else:
                        raise ValueError('Unknown traversal order')
                    sec_edges.append([(None, sec_root)])
                sec_edges = [val for sublist in sec_edges for val in sublist]
            else:
                sec_edges = [(None, idx) for idx in sec_dict.keys() ]
        else:
            sec_edges = [(None, idx) for idx in sec_dict.keys() ]
        for sec_index, sec in sec_dict.iteritems():
            seg_list = []
            if maxdist is None:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0:
                        seg_list.append(seg)
            else:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0 and ((L_total + sec.L * seg.x) <= maxdist):
                        seg_list.append(seg)
            seg_dict[sec_index] = seg_list
            L_total += sec.L
        seg_density_dict, layers_dict = \
            synapse_seg_density(syn_type_dict, layer_dict, \
                                layer_density_dict, \
                                seg_dict, density_seed, \
                                neurotree_dict=neurotree_dict)
        seg_density_per_sec[sec_name] = seg_density_dict
        for (syn_type_label, _) in layer_density_dict.iteritems():
            syn_type = syn_type_dict[syn_type_label]
            seg_density = seg_density_dict[syn_type]
            layers = layers_dict[syn_type]
            end_distance = {}
            for sec_parent, sec_index in sec_edges:
                seg_list        = seg_dict[sec_index]
                sec_seg_layers  = layers[sec_index]
                sec_seg_density = seg_density[sec_index]
                start_seg       = seg_list[0]
                interval        = 0.
                syn_loc         = 0.
                for seg, layer, density in itertools.izip(seg_list,sec_seg_layers,sec_seg_density):
                    seg_start = seg.x - (0.5 / seg.sec.nseg)
                    seg_end   = seg.x + (0.5 / seg.sec.nseg)
                    L = seg.sec.L
                    L_seg_start = seg_start * L
                    L_seg_end   = seg_end * L
                    if density > 0.:
                        beta = 1. / density
                        interval += r.exponential(beta)
                        while interval < L_seg_end:
                            if interval >= L_seg_start:
                                syn_loc = interval / L
                                assert ((syn_loc <= 1) and (syn_loc >= seg_start))
                                if syn_loc < 1.0:
                                    syn_locs.append(syn_loc)
                                    syn_ids.append(syn_index)
                                    syn_secs.append(sec_index)
                                    syn_layers.append(layer)
                                    syn_types.append(syn_type)
                                    swc_types.append(swc_type)
                                    syn_index += 1
                            interval += r.exponential(beta)
                    else:
                        interval = seg_end * L
                end_distance[sec_index] = (1.0 - syn_loc) * L
            
    assert (len(syn_ids) > 0)
    syn_dict = {'syn_ids': np.asarray(syn_ids, dtype='uint32'),
                'syn_locs': np.asarray(syn_locs, dtype='float32'),
                'syn_secs': np.asarray(syn_secs, dtype='uint32'),
                'syn_layers': np.asarray(syn_layers, dtype='int8'),
                'syn_types': np.asarray(syn_types, dtype='uint8'),
                'swc_types': np.asarray(swc_types, dtype='uint8')}

    return (syn_dict, seg_density_per_sec)
