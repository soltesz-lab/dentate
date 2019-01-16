import time, collections, traceback
from dentate.neuron_utils import *
from dentate.utils import viewitems, zip_longest, partitionn, generator_ifempty, NamedTupleWithDocstring
from dentate.cells import get_mech_rules_dict, get_donor, get_distance_to_node, get_param_val_by_distance, \
    import_mech_dict_from_file, custom_filter_by_branch_order, custom_filter_modify_slope_if_terminal, \
    custom_filter_if_terminal, make_neurotree_graph
from neuroh5.io import write_cell_attributes
from collections import namedtuple, defaultdict

# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)


class SynapseSource(object):
    """This class provides information about the presynaptic (source) cell
    connected to a synapse.
      - gid - gid of source cell (int)
      - population - population index of source cell (int)
      - delay - connection delay (float)
    """
    __slots__ = 'gid', 'population', 'delay'
    def __init__(self):
        self.gid = None
        self.population = None
        self.delay = None

SynapsePointProcess = NamedTupleWithDocstring(
    """This class provides information about the point processes associated with a synapse.
      - mech - dictionary of synapse mechanisms
      - netcon - dictionary of netcons
      - vecstim - dictionary of vecstims
    """,
    "SynapsePointProcess",
    ['mech', 'netcon', 'vecstim'])


Synapse = NamedTupleWithDocstring(
    """A container for synapse configuration, synapse mechanism instantiation,
     and associated netcon/vecstim instances.
    - syn_type - enumerated synapse type (int)
    - swc_type - enumerated swc type (int)
    - syn_layer - enumerated synapse layer (int)
    - syn_loc - synapse location in segment (float)
    - syn_section - synapse section index (int)
    - source: instance of SynapseSource with the slots: 
       - gid - source cell gid (int)
       - population - enumerated source population index (int)
       - delay - connection delay (float)
    - attr_dict - dictionary of attributes per synapse mechanism
      (for cases when multiple mechanisms are associated with a
      synapse, e.g. GABA_A and GABA_B)
    """,
    'Synapse',
    ['syn_type',
     'swc_type',
     'syn_layer',
     'syn_loc',
     'syn_section',
     'source',
     'attr_dict'
    ])


class SynapseAttributes(object):
    """This class provides an interface to store, retrieve, and modify
    attributes of synaptic mechanisms. Handles instantiation of
    complex subcellular gradients of synaptic mechanism attributes.
    """
    def __init__(self, env, syn_mech_names, syn_param_rules):
        """An Env object containing imported network configuration metadata
        uses an instance of SynapseAttributes to track all metadata
        related to the identity, location, and configuration of all
        synaptic connections in the network.

        :param env: :class:'Env'
        :param syn_mech_names: dict of the form: { label: mechanism name }
        :param syn_param_rules: dict of the form:
               { mechanism name:
                    mech_file: path.mod
                    mech_params: list of parameter names
                    netcon_params: dictionary { parameter name: index }
                }

        """
        self.env = env
        self.syn_mech_names = syn_mech_names
        self.syn_param_rules = syn_param_rules
        self.syn_name_index_dict = { label: index for index, label in (enumerate(syn_mech_names.keys())) } # int : mech_name dict
        self.syn_id_attr_dict = defaultdict(lambda: defaultdict(lambda: None))
        self.sec_dict = defaultdict(lambda: defaultdict(lambda: []))
        self.pps_dict = defaultdict(lambda: defaultdict(lambda: SynapsePointProcess(mech={}, netcon={}, vecstim={})))
        self.filter_cache = {}
        
    def init_syn_id_attrs_from_iter(self, cell_iter):
        """
        Initializes synaptic attributes given an iterator that returns (gid, attr_dict).
        See `init_syn_id_attrs` for details on the format of the input dictionary.
        """
        for (gid, attr_dict) in cell_iter:
            self.init_syn_id_attrs(gid, attr_dict)

    def init_syn_id_attrs(self, gid, syn_id_attr_dict):
        """
        Initializes synaptic attributes for the given cell gid.
        Only the intrinsic properties of a synapse, such as type, layer, location are set.

        Connection edge attributes such as source gid, point process
        parameters, and netcon/vecstim objects are initialized to None
        or empty dictionaries.

        :param syn_id_attr_dict: a dictionary containing the following keys:
          - syn_ids: synapse ids
          - syn_layers: layer index for each synapse id
          - syn_types: synapse type for each synapse id
          - swc_types: swc type for each synapse id
          - syn_secs: section index for each synapse id
          - syn_locs: section location for each synapse id

        """
        if gid in self.syn_id_attr_dict:
            raise RuntimeError('Entry %i exists in synapse attribute dictionary' % gid)
        else:
            syn_ids = syn_id_attr_dict['syn_ids']
            syn_layers = syn_id_attr_dict['syn_layers']
            syn_types = syn_id_attr_dict['syn_types']
            swc_types = syn_id_attr_dict['swc_types']
            syn_secs = syn_id_attr_dict['syn_secs']
            syn_locs = syn_id_attr_dict['syn_locs']

            syn_dict = self.syn_id_attr_dict[gid]
            sec_dict = self.sec_dict[gid]
            for i, (syn_id, syn_layer, syn_type, swc_type, syn_sec, syn_loc) in \
              enumerate(zip_longest(syn_ids, syn_layers, syn_types, swc_types, syn_secs, syn_locs)):
                syn = Synapse(syn_type=syn_type, syn_layer=syn_layer, syn_section=syn_sec, syn_loc=syn_loc,
                              swc_type=swc_type, source=SynapseSource(), attr_dict=defaultdict(dict))
                syn_dict[syn_id] = syn
                sec_dict[syn_sec].append((syn_id,syn))

    def init_edge_attrs(self, gid, presyn_name, presyn_gids, edge_syn_ids, delays=None):
        """
        Sets connection edge attributes for the specified synapse ids.

        :param gid: gid for post-synaptic (target) cell (int)
        :param presyn_name: name of presynaptic (source) population (string)
        :param presyn_ids: gids for presynaptic (source) cells (array of int)
        :param edge_syn_ids: synapse ids on target cells to be used for connections (array of int)
        :param delays: axon conduction (netcon) delays (array of float)
        """

        presyn_index = int(self.env.Populations[presyn_name])
        connection_velocity = float(self.env.connection_velocity[presyn_name])

        if delays is None:
            delays = [h.dt] * len(edge_syn_ids)

        syn_id_dict = self.syn_id_attr_dict[gid]
              
        for edge_syn_id, presyn_gid, delay in zip_longest(edge_syn_ids, presyn_gids, delays):
            syn = syn_id_dict[edge_syn_id]
            if syn is None:
                raise RuntimeError('init_edge_attrs: gid %i: synapse id %i has not been initialized' %
                                   (gid, edge_syn_id))

            if syn.source.gid is not None:
                raise RuntimeError('init_edge_attrs: gid %i: synapse id %i has already been initialized with edge '
                                   'attributes' % (gid, edge_syn_id))

            syn.source.gid = presyn_gid
            syn.source.population = presyn_index
            syn.source.delay = delay

    def init_edge_attrs_from_iter(self, pop_name, presyn_name, attr_info, edge_iter, set_edge_delays=True):
        """
        Initializes edge attributes for all cell gids returned by iterator.

        :param pop_name: name of postsynaptic (target) population (string)
        :param source_name: name of presynaptic (source) population (string)
        :param attr_info: dictionary mapping attribute name to indices in iterator tuple
        :param edge_iter: edge attribute iterator
        :param set_edge_delays: bool
        """
        connection_velocity = float(self.env.connection_velocity[presyn_name])
        if pop_name in attr_info and presyn_name in attr_info[pop_name]:
            edge_attr_info = attr_info[pop_name][presyn_name]
        else:
            raise RuntimeError('init_edge_attrs_from_iter: missing edge attributes for projection %s -> %s' % \
                               (presyn_name, pop_name))

        if 'Synapses' in edge_attr_info and \
                'syn_id' in edge_attr_info['Synapses'] and \
                'Connections' in edge_attr_info and \
                'distance' in edge_attr_info['Connections']:
            syn_id_attr_index = edge_attr_info['Synapses']['syn_id']
            distance_attr_index = edge_attr_info['Connections']['distance']
        else:
            raise RuntimeError('init_edge_attrs_from_iter: missing edge attributes for projection %s -> %s' % \
                               (presyn_name, pop_name))

        for (postsyn_gid, edges) in edge_iter:
            presyn_gids, edge_attrs = edges
            edge_syn_ids = edge_attrs['Synapses'][syn_id_attr_index]
            edge_dists = edge_attrs['Connections'][distance_attr_index]

            if set_edge_delays:
                delays = [((distance / connection_velocity) + h.dt) for distance in edge_dists]
            else:
                delays = None

            self.init_edge_attrs(postsyn_gid, presyn_name, presyn_gids, edge_syn_ids, delays=delays)

    def add_pps(self, gid, syn_id, syn_name, pps):
        """
        Adds mechanism point process for the specified cell/synapse id/mechanism name.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param pps: hoc point process
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        if syn_index in pps_dict.mech:
            raise RuntimeError('add_pps: gid %i Synapse id %i already has mechanism %s' % (gid, syn_id, syn_name))
        else:
            pps_dict.mech[syn_index] = pps

    def has_pps(self, gid, syn_id, syn_name):
        """
        Returns True if the given synapse id already has the named mechanism, False otherwise.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: bool
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        return syn_index in pps_dict.mech

    def get_pps(self, gid, syn_id, syn_name, throw_error=True):
        """
        Returns the mechanism for the given synapse id on the given cell.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: mechanism name
        :return: hoc point process
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        if syn_index in pps_dict.mech:
            return pps_dict.mech[syn_index]
        else:
            if throw_error:
                raise RuntimeError('get_pps: gid %i synapse id %i has no point process for mechanism %s' %
                                   (gid, syn_id, syn_name))
            else:
                return None

    def add_netcon(self, gid, syn_id, syn_name, nc):
        """
        Adds a NetCon object for the specified cell/synapse id/mechanism name.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param nc: :class:'h.NetCon'
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        if syn_index in pps_dict.netcon:
            raise RuntimeError('add_netcon: gid %i Synapse id %i mechanism %s already has netcon' %
                               (gid, syn_id, syn_name))
        else:
            pps_dict.netcon[syn_index] = nc

    def has_netcon(self, gid, syn_id, syn_name):
        """
        Returns True if a netcon exists for the specified cell/synapse id/mechanism name, False otherwise.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: bool
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        return syn_index in pps_dict.netcon
    
    def get_netcon(self, gid, syn_id, syn_name, throw_error=True):
        """
        Returns the NetCon object associated with the specified cell/synapse id/mechanism name.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: :class:'h.NetCon'
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        if syn_index in pps_dict.netcon:
            return pps_dict.netcon[syn_index]
        else:
            if throw_error:
                raise RuntimeError('get_netcon: gid %i synapse id %i has no netcon for mechanism %s' %
                                   (gid, syn_id, syn_name))
            else:
                return None

    def add_vecstim(self, gid, syn_id, syn_name, vs):
        """
        Adds a VecStim object for the specified cell/synapse id/mechanism name.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param vs: :class:'h.VecStim'
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        if syn_index in pps_dict.vecstim:
            raise RuntimeError('add_vecstim: gid %i synapse id %i mechanism %s already has vecstim' %
                               (gid, syn_id, syn_name))
        else:
            pps_dict.vecstim[syn_index] = vs
            
    def has_vecstim(self, gid, syn_id, syn_name):
        """
        Returns True if a vecstim exists for the specified cell/synapse id/mechanism name, False otherwise.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: bool
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        return syn_index in pps_dict.vecstim
    
    def get_vecstim(self, gid, syn_id, syn_name, throw_error=True):
        """
        Returns the VecStim object associated with the specified cell/synapse id/mechanism name.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: :class:'h.VecStim'
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        if syn_index in pps_dict.vecstim:
            return pps_dict.vecstim[syn_index]
        else:
            if throw_error:
                raise RuntimeError('get_vecstim: gid %d synapse %d: vecstim for mechanism %s not found' %
                                   (gid, syn_id, syn_name))
            else:
                return None

    def has_mech_attrs(self, gid, syn_id, syn_name):
        """
        Returns True if mechanism attributes have been specified for the given cell id/synapse id/mechanism name, False otherwise.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: bool
        """
        syn_index = self.syn_name_index_dict[syn_name]
        syn_id_dict = self.syn_id_attr_dict[gid]
        syn = syn_id_dict[syn_id]
        return syn_index in syn.attr_dict

    def get_mech_attrs(self, gid, syn_id, syn_name, throw_error=True):
        """
        Returns mechanism attribute dictionary associated with the given cell id/synapse id/mechanism name, False otherwise.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: dict
        """
        syn_index = self.syn_name_index_dict[syn_name]
        syn_id_dict = self.syn_id_attr_dict[gid]
        syn = syn_id_dict[syn_id]
        if syn_index in syn.attr_dict:
            return syn.attr_dict[syn_index]
        else:
            if throw_error:
                raise RuntimeError('get_mech_attrs: gid %d synapse %d: attributes for mechanism %s not found' %
                                   (gid, syn_id, syn_name))
            else:
                return None

    def add_mech_attrs(self, gid, syn_id, syn_name, params):
        """
        Specifies mechanism attribute dictionary for the given cell id/synapse id/mechanism name. Assumes mechanism
        attributes have not been set yet for this synapse mechanism.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param params: dict
        """
        syn_index = self.syn_name_index_dict[syn_name]
        syn_id_dict = self.syn_id_attr_dict[gid]
        syn = syn_id_dict[syn_id]
        attr_dict = syn.attr_dict[syn_index]
        for k, v in viewitems(params):
            if k in attr_dict:
                raise RuntimeError('add_mech_attrs: gid %i synapse id %i mechanism %s already has parameter %s' %
                                   (gid, syn_id, syn_name, str(k)))
            else:
                attr_dict[k] = v

    def modify_mech_attrs(self, gid, syn_id, syn_name, params, update=lambda old, new: new):
        """
        Modifies mechanism attributes for the given cell id/synapse id/mechanism name. 

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param params: dict
        :param update: lambda (old, new)
        """
        rules = self.syn_param_rules
        syn_index = self.syn_name_index_dict[syn_name]
        syn_id_dict = self.syn_id_attr_dict[gid]
        mech_name = self.syn_mech_names[syn_name]
        syn = syn_id_dict[syn_id]

        attr_dict = syn.attr_dict[syn_index]
        for k, v in viewitems(params):
            if k in rules[mech_name]['mech_params']:
                if k in attr_dict:
                    attr_dict[k] = update(attr_dict[k], v)
                else:
                    attr_dict[k] = v
            elif k in rules[mech_name]['netcon_params']:
                if k in attr_dict:
                    attr_dict[k] = update(attr_dict[k], v)
                else:
                    attr_dict[k] = v
            else:
                  raise RuntimeError('modify_mech_attrs: unknown type of parameter %s' % k)

    def add_mech_attrs_from_iter(self, gid, syn_name, params_iter):
        """
        Adds mechanism attributes for the given cell id/synapse id/synapse mechanism.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param params: dict
        """
        syn_index = self.syn_name_index_dict[syn_name]
        syn_id_dict = self.syn_id_attr_dict[gid]
        for syn_id, params_dict in params_iter:
            syn = syn_id_dict[syn_id]
            if syn is None:
                raise RuntimeError('add_mech_attrs_from_iter: '
                                   'gid %i synapse id %i has not been created yet' % (gid, syn_id))
            if syn_index in syn.attr_dict:
                raise RuntimeError('add_mech_attrs_from_iter: '
                                   'gid %i Synapse id %i mechanism %s already has parameters' % (gid, syn_id, syn_name))
            attr_dict = syn.attr_dict[syn_index]
            for k, v in viewitems(params_dict):
                attr_dict[k] = v

    def filter_synapses(self, gid, syn_sections=None, syn_indexes=None, syn_types=None, layers=None, sources=None,
                        swc_types=None, cache=False):
        """
        Returns a subset of the synapses of the given cell according to the given criteria.

        :param gid: int
        :param syn_sections: array of int
        :param syn_indexes: array of int: syn_ids
        :param syn_types: list of enumerated type: synapse category
        :param layers: list of enumerated type: layer
        :param sources: list of enumerated type: population names of source projections
        :param swc_types: list of enumerated type: swc_type
        :param cache: bool
        :return: dictionary { syn_id: { attribute: value } }
        """
        matches = lambda items: all(itertools.imap (lambda (query, item): (query is None) or (item in query), items))

        if cache:
            cache_args = tuple(map(lambda x: tuple(x) if isinstance(x,list) else x,
                                   [gid, syn_sections, syn_indexes, syn_types, layers, sources, swc_types]))
            if cache_args in self.filter_cache:
                return self.filter_cache[cache_args]
        
        if sources is None:
            source_indexes = None
        else:
            source_indexes = set(sources)

        if syn_sections is not None:
            # Fast path
            sec_dict = self.sec_dict[gid]
            it = itertools.chain.from_iterable([ sec_dict[sec_index] for sec_index in syn_sections ])
            syn_dict = { k: v for (k,v) in it }
        else:
            syn_dict = self.syn_id_attr_dict[gid]

        result = {k: v for k, v in viewitems(syn_dict)
                  if matches([(syn_indexes, k),
                              (syn_types, v.syn_type),
                              (layers, v.syn_layer),
                              (source_indexes, v.source.population),
                              (swc_types, v.swc_type)])}
        if cache:
            self.filter_cache[cache_args] = result

        return result
  
    def partition_synapses_by_source(self, gid, syn_ids=None):
        """
        Partitions the synapse objects for the given cell based on the
        presynaptic (source) population index.
        
        :param gid: int
        :param syn_ids: array of int

        """
        source_names = {id: name for name, id in viewitems(self.env.Populations)}
        if syn_ids is None:
            syn_id_attr_dict = self.syn_id_attr_dict[gid]
        else:
            syn_id_attr_dict = {syn_id: self.syn_id_attr_dict[gid][syn_id] for syn_id in syn_ids}

        source_iter = partitionn(viewitems(syn_id_attr_dict), lambda((syn_id,syn)): syn.source.population,
                                 n=len(source_names))

        return dict(map(lambda(source_id,x): (source_names[source_id], generator_ifempty (x)), enumerate(source_iter)))

    def get_filtered_syn_ids(self, gid, syn_sections=None, syn_indexes=None, syn_types=None, layers=None,
                             sources=None, swc_types=None, cache=False):
        """
        Returns a subset of the synapse ids of the given cell according to the given criteria.
        :param gid:
        :param syn_sections:
        :param syn_indexes:
        :param syn_types:
        :param layers:
        :param sources:
        :param swc_types:
        :param cache:
        :return: sequence
        """
        return self.filter_synapses(gid, syn_sections=syn_sections, syn_indexes=syn_indexes, syn_types=syn_types,
                                    layers=layers, sources=sources, swc_types=swc_types, cache=cache).keys()

    def partition_syn_ids_by_source(self, gid, syn_ids=None):
        """
        Partitions the synapse ids for the given cell based on the
        presynaptic (source) population index.
        
        :param gid: int
        :param syn_ids: array of int

        """
        start_time = time.time()
        source_names = {id: name for name, id in viewitems(self.env.Populations)}
        syn_id_attr_dict = self.syn_id_attr_dict[gid]
        if syn_ids is None:
            syn_ids = syn_id_attr_dict.keys()

        source_iter = partitionn(syn_ids, lambda(syn_id): syn_id_attr_dict[syn_id].source.population,
                                 n=len(source_names))

        return dict(map(lambda(source_id,x): (source_names[source_id], generator_ifempty (x)), enumerate(source_iter)))

    def del_syn_id_attr_dict(self, gid):
        """
        Removes the synapse attributes associated with the given cell gid.
        """
        del self.syn_id_attr_dict[gid]
        del self.sec_dict[gid]

    def clear_filter_cache(self):
        self.filter_cache.clear()

    def gids(self):
        return self.syn_id_attr_dict.keys()
  
    def items(self):
        return viewitems(self.syn_id_attr_dict)

    def __getitem__(self, gid):
        return self.syn_id_attr_dict[gid]


def insert_hoc_cell_syns(env, syn_params, gid, cell, syn_ids, unique=False, insert_netcons=False,
                         insert_vecstims=False):
    """
    TOOD: Only config the point process object if it has not already been configured.

    Insert mechanisms into given cell according to the synapse objects created in env.synapse_attributes.
    Configures mechanisms according to parameter values specified in syn_params.
    
    :param env: :class:'Env'
    :param syn_params: dictionary of the form { mech_name: params }
    :param gid: cell id (int)
    :param cell: hoc cell object
    :param syn_ids: synapse ids (array of int)
    :param unique: True, if unique mechanisms are to be inserted for each synapse; False, if synapse mechanisms within
            a compartment will be shared.
    :param insert_netcons: bool; whether to build new netcons for newly constructed synapses
    :param insert_vecstims: bool; whether to build new vecstims for newly constructed netcons
    :param verbose: bool
    :return: number of inserted mechanisms

    """

    swc_type_apical = env.SWC_Types['apical']
    swc_type_basal = env.SWC_Types['basal']
    swc_type_soma = env.SWC_Types['soma']
    swc_type_axon = env.SWC_Types['axon']
    swc_type_ais = env.SWC_Types['ais']
    swc_type_hill = env.SWC_Types['hillock']

    syns_dict_dend = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_axon = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_ais = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_hill = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_soma = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))

    syns_dict_by_type = { swc_type_apical: syns_dict_dend,
                          swc_type_basal: syns_dict_dend,
                          swc_type_axon: syns_dict_axon,
                          swc_type_ais: syns_dict_ais,
                          swc_type_hill: syns_dict_hill,
                          swc_type_soma: syns_dict_soma }
    py_sections = [sec for sec in cell.sections]
    
    syn_attrs = env.synapse_attributes
    syn_id_attr_dict = syn_attrs.syn_id_attr_dict[gid]

    make_syn_mech = make_unique_synapse_mech if unique else make_shared_synapse_mech

    syn_count = 0
    nc_count = 0
    mech_count = 0
    for syn_id in syn_ids:

        syn = syn_id_attr_dict[syn_id]
        swc_type = syn.swc_type
        syn_loc = syn.syn_loc
        syn_section = syn.syn_section

        sec = py_sections[syn_section]
        if swc_type in syns_dict_by_type:
            syns_dict = syns_dict_by_type[swc_type]
        else:
            raise RuntimeError("insert_hoc_cell_syns: unsupported synapse SWC type %d for synapse %d" %
                               (swc_type, syn_id))

        if 'default' in syn_params:
            mech_params = syn_params['default']
        else:
            mech_params = syn_params[swc_type]

        for syn_name, params in viewitems(mech_params):
            
            syn_mech = make_syn_mech(syn_name=syn_name, seg=sec(syn_loc), syns_dict=syns_dict,
                                     mech_names=syn_attrs.syn_mech_names)

            syn_attrs.add_pps(gid, syn_id, syn_name, syn_mech)
                       
            mech_count += 1

            if insert_netcons or insert_vecstims:
                syn_pps = syn_attrs.get_pps(gid, syn_id, syn_name)
                if insert_netcons and insert_vecstims:
                    this_nc, this_vecstim = mknetcon_vecstim(syn_pps, delay=syn.source.delay)
                else:
                    this_vecstim = None
                    this_nc = mknetcon(env.pc, syn.source.gid, syn_pps, delay=syn.source.delay)
                if insert_netcons:
                    syn_attrs.add_netcon(gid, syn_id, syn_name, this_nc)
                if insert_vecstims:
                    syn_attrs.add_vecstim(gid, syn_id, syn_name, this_vecstim)
                config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules,
                           mech_names=syn_attrs.syn_mech_names,
                           syn=syn_mech, nc=this_nc, **params)
                nc_count += 1
            else:
                config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules,
                           mech_names=syn_attrs.syn_mech_names,
                           syn=syn_mech, **params)

        syn_count += 1
        
    return syn_count, mech_count, nc_count


def insert_biophys_cell_syns(env, gid, postsyn_name, presyn_name, syn_ids, unique=None, insert_netcons=True,
                             insert_vecstims=True, verbose=False):
    """
    
    1) make syns (if not unique, keep track of syn_in_seg for shared synapses)
    2) initialize syns with syn_mech_params from config_file
    3) make netcons
    4) initialize netcons with syn_mech_params from config_file

    :param env: :class:'Env'
    :param gid: int
    :param postsyn_name: str
    :param presyn_name: str
    :param syn_ids: array of int
    :param unique: bool; whether to insert synapses if none exist at syn_id
    :param insert_netcons: bool; whether to build new netcons for newly constructed synapses
    :param insert_vecstims: bool; whether to build new vecstims for newly constructed netcons
    :param verbose: bool
    """
    if not (gid in env.biophys_cells[postsyn_name]):
        raise KeyError('insert_biophys_cell_syns: BiophysCell with gid %i does not exist' % gid)

    cell = env.biophys_cells[postsyn_name][gid]
    syn_params = env.connection_config[postsyn_name][presyn_name].mechanisms

    synapse_config = env.celltypes[postsyn_name]['synapses']

    if unique is None:
        if 'unique' in synapse_config:
            unique = synapse_config['unique']
        else:
            unique = False

    syn_count, mech_count, nc_count = insert_hoc_cell_syns(env, syn_params, gid, cell.hoc_cell, syn_ids,
                                                           unique=unique, insert_netcons=insert_netcons,
                                                           insert_vecstims=insert_vecstims)

    if verbose:
        logger.info('insert_biophys_cell_syns: source: %s target: %s cell %i: created %i mechanisms and %i '
                    'netcons for %i syn_ids' % (presyn_name, postsyn_name, gid, mech_count, nc_count, syn_count))


def config_biophys_cell_syns(env, gid, postsyn_name, syn_ids=None, unique=None, insert=False, insert_netcons=False,
                             insert_vecstims=False, verbose=False, throw_error=False):
    """
    Configures the given syn_ids, and call config_syn with mechanism
    and netcon parameters (which must not be empty).  If syn_ids=None,
    configures all synapses for the cell with the given gid.  If
    insert=True, iterate over sources and call
    insert_biophys_cell_syns (requires a BiophysCell with the
    specified gid to be present in the Env).

    :param gid: int
    :param env: :class:'Env'
    :param postsyn_name: str
    :param syn_ids: array of int
    :param unique: bool; whether newly inserted synapses should be unique or shared per segment
    :param insert: bool; whether to insert a synaptic point process if none exists at syn_id
    :param insert_netcons: bool; whether to build new netcons for newly constructed synapses
    :param insert_vecstims: bool; whether to build new vecstims for newly constructed netcons
    :param verbose: bool
    :param throw_error: bool; whether to require that all encountered syn_ids have inserted synapse
    """
    syn_attrs = env.synapse_attributes
    syn_id_attr_dict = syn_attrs.syn_id_attr_dict[gid]

    if syn_ids is None:
        syn_ids = syn_id_attr_dict.keys()

    if insert:
        source_syn_ids_dict = syn_attrs.partition_syn_ids_by_source(gid, syn_ids)
        if not (gid in env.biophys_cells[postsyn_name]):
            raise KeyError('config_biophys_cell_syns: insert: BiophysCell with gid %i does not exist' % gid)

        for presyn_name, source_syn_ids in viewitems(source_syn_ids_dict):
            if source_syn_ids is not None:
                insert_biophys_cell_syns(env, gid, postsyn_name, presyn_name, source_syn_ids, unique=unique,
                                         insert_netcons=insert_netcons, insert_vecstims=insert_vecstims,
                                         verbose=verbose)

    cell = env.biophys_cells[postsyn_name][gid]
    syn_count, mech_count, nc_count = config_hoc_cell_syns(env, gid, postsyn_name, cell=cell.hoc_cell, syn_ids=syn_ids,
                                                           insert=False, verbose=False, throw_error=throw_error)

    if verbose:
        logger.info('config_biophys_cell_syns: target: %s; cell %i: set parameters for %i syns and %i '
                    'netcons for %i syn_ids' % (postsyn_name, gid, mech_count, nc_count, syn_count))

    return syn_count, nc_count


def config_hoc_cell_syns(env, gid, postsyn_name, cell=None, syn_ids=None, unique=None, insert=False,
                         insert_netcons=False, insert_vecstims=False, verbose=False, throw_error=False):
    """
    Configures the given syn_ids, and call config_syn with mechanism and netcon parameters (which must not be empty).
    If syn_ids=None, configures all synapses for the cell with the given gid.
    If insert=True, iterate over sources and call insert_hoc_cell_syns
       (requires the cell object is given or registered with h.ParallelContext on this rank).

    :param env: :class:'Env'
    :param gid: int
    :param postsyn_name: str
    :param syn_ids: array of int
    :param unique: bool; whether newly inserted synapses should be unique or shared per segment
    :param insert: bool; whether to insert a synaptic point process if none exists at syn_id
    :param insert_netcons: bool; whether to build new netcons for newly constructed synapses
    :param insert_vecstims: bool; whether to build new vecstims for newly constructed netcons
    :param verbose: bool
    :param throw_error: bool; whether to require that all encountered syn_ids have inserted synapse
    """
    rank = int(env.pc.id())
    syn_attrs = env.synapse_attributes
    syn_id_attr_dict = syn_attrs.syn_id_attr_dict[gid]

    synapse_config = env.celltypes[postsyn_name]['synapses']

    if unique is None:
        if 'unique' in synapse_config:
            unique = synapse_config['unique']
        else:
            unique = False

    if syn_ids is None:
        syn_ids = syn_id_attr_dict.keys()

    if insert:
        source_syn_dict = syn_attrs.partition_synapses_by_source(gid, syn_ids)
        last_time = time.time() 
        if (cell is None) and (not (env.pc.gid_exists(gid))):
            raise RuntimeError('config_hoc_cell_syns: insert: cell with gid %i does not exist on rank %i' % (gid, rank))
        if cell is None:
            cell = env.pc.gid2cell(gid)
        for presyn_name, source_syns in viewitems(source_syn_dict):
            if source_syns is not None:
                source_syn_ids = [x[0] for x in source_syns]
                syn_params = env.connection_config[postsyn_name][presyn_name].mechanisms
                syn_count, mech_count, nc_count = insert_hoc_cell_syns(env, syn_params, gid, cell, source_syn_ids, unique=unique,
                                     insert_netcons=insert_netcons, insert_vecstims=insert_vecstims)
                if verbose:
                    logger.info('config_hoc_cell_syns: population: %s; cell %i: inserted %i mechanisms for source %s' % (postsyn_name, gid, mech_count, presyn_name))
        if verbose:
              logger.info('config_hoc_cell_syns: population: %s; cell %i: inserted mechanisms in %f s' % \
                          (postsyn_name, gid, time.time() - last_time))

    source_syn_dict = syn_attrs.partition_synapses_by_source(gid, syn_ids)

    total_nc_count = 0
    total_mech_count = 0
    total_syn_id_count = 0
    config_time = time.time()
    for presyn_name, source_syns in viewitems(source_syn_dict):

        if source_syns is None:
            continue

        nc_count = 0
        mech_count = 0

        mech_config_dict = env.connection_config[postsyn_name][presyn_name].mechanisms
        sec_indexes = mech_config_dict.keys()
        syn_names = set(itertools.chain.from_iterable([ mech_config_dict[sec_index].keys() for sec_index in sec_indexes ]))
        syn_indexes = set([syn_attrs.syn_name_index_dict[syn_name] for syn_name in syn_names])
        for syn_id, syn in source_syns:
            total_syn_id_count += 1
            for syn_name, syn_index in zip_longest(syn_names, syn_indexes):
                if syn_index in syn.attr_dict:
                    this_pps = syn_attrs.get_pps(gid, syn_id, syn_name, throw_error=False)
                    if this_pps is None and throw_error:
                       raise RuntimeError('config_hoc_cell_syns: insert: cell gid %i synapse %i does not have a point '
                                          'process for mechanism %s' % (gid, syn_id, syn_name))

                    this_netcon = syn_attrs.get_netcon(gid, syn_id, syn_name, throw_error=False)
                    if this_netcon is None and throw_error:
                          raise RuntimeError('config_hoc_cell_syns: insert: cell gid %i synapse %i does not have a '
                                             'netcon for mechanism %s' % (gid, syn_id, syn_name))

                    params = syn.attr_dict[syn_index]
                    (mech_set, nc_set) = config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules,
                                                    mech_names=syn_attrs.syn_mech_names, syn=this_pps, nc=this_netcon,
                                                    **params)
                    if mech_set:
                        mech_count += 1
                    if nc_set:
                        nc_count += 1

        total_nc_count += nc_count
        total_mech_count += mech_count
        
    if verbose:
        logger.info('config_hoc_cell_syns: target: %s; cell %i: set parameters for %i syns and %i netcons for %i '
                    'syn_ids' % (postsyn_name, gid, total_mech_count, total_nc_count, total_syn_id_count))

    return total_syn_id_count, total_mech_count, total_nc_count


def config_syn(syn_name, rules, mech_names=None, syn=None, nc=None, **params):
    """
    TODO: Why was the hasattr(syn, param) check removed?
    Initializes synaptic and connection mechanisms with parameters specified in the synapse attribute dictionaries.

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
    mech_rules = rules[mech_name]

    nc_param = False
    mech_param = False

    for param, val in viewitems(params):
        failed = True
        if param in mech_rules['mech_params']:
            if syn is None:
                failed = False
            else:
                setattr(syn, param, val)
                mech_param = True
                failed = False

        elif param in mech_rules['netcon_params']:
            if nc is None:
                failed = False
            else:
                i = mech_rules['netcon_params'][param]

                if int(nc.wcnt()) >= i:
                    old = nc.weight[i]
                    nc.weight[i] = val
                    nc_param = True
                    failed = False
        if failed:
            raise AttributeError('config_syn: problem setting attribute: %s for synaptic mechanism: %s' %
                                 (param, mech_name))
    return (mech_param, nc_param)


def syn_in_seg(syn_name, seg, syns_dict):
    """
    If a synaptic mechanism of the specified type already exists in the specified segment, it is returned. Otherwise,
    it returns None.
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
    TODO: Why was the hasattr(h, mech_name) check removed?
    :param mech_name: str (name of the point_process, specified by Env.synapse_attributes.syn_mech_names)
    :param seg: hoc segment
    :return: hoc point process
    """
    syn = getattr(h, mech_name)(seg)
    return syn


def make_shared_synapse_mech(syn_name, seg, syns_dict, mech_names=None):
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
    syn_mech = syn_in_seg(syn_name, seg, syns_dict)
    if syn_mech is None:
        if mech_names is not None:
            mech_name = mech_names[syn_name]
        else:
            mech_name = syn_name
        syn_mech = make_syn_mech(mech_name, seg)
        syns_dict[seg.sec][seg.x][syn_name] = syn_mech
    return syn_mech


def make_unique_synapse_mech(syn_name, seg, syns_dict=None, mech_names=None):
    """
    Creates a new synapse in the provided segment, and returns it.

    :param syn_name: str
    :param seg: hoc segment
    :param syns_dict: nested defaultdict
    :param mech_names: map of synapse name to hoc mechanism name
    :return: hoc point process
    """
    if mech_names is not None:
        mech_name = mech_names[syn_name]
    else:
        mech_name = syn_name
    syn_mech = make_syn_mech(mech_name, seg)
    return syn_mech


# ------------------------------- Methods to specify synaptic mechanisms  -------------------------------------------- #

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


def get_syn_filter_dict(env, rules, convert=False):
    """Used by modify_syn_param. Takes in a series of arguments and
    constructs a validated rules dictionary that specifies to which
    sets of synapses a rule applies. Values of filter queries are
    validated by the provided Env.

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
            if source not in env.Populations:
                raise ValueError('get_syn_filter_dict: presynaptic population: %s not recognized by network '
                                 'configuration' % source)
            if convert:
                rules_dict['sources'][i] = env.Populations[source]
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


def modify_syn_param(cell, env, sec_type, syn_name, param_name=None, value=None, origin=None, slope=None, tau=None,
                     xhalf=None, min=None, max=None, min_loc=None, max_loc=None, outside=None, custom=None,
                     append=False, filters=None, origin_filters=None, update_targets=False, verbose=False):
    """Modifies a cell's mechanism dictionary to specify attributes of a
    synaptic mechanism by sec_type. This method is meant to be called
    manually during initial model specification, or during parameter
    optimization. For modifications to persist across simulations, the
    mechanism dictionary must be saved to a file using
    export_mech_dict() and re-imported during BiophysCell
    initialization.

    Calls update_syn_mech_by_sec_type to set placeholder values in the
    syn_mech_attrs_dict of a SynapseAttributes object. If
    update_targets flag is True, the attributes of any target synaptic
    point_process and netcon objects that have been inserted will also
    be updated. Otherwise, they can be updated separately by calling

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
    :param verbose: bool
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
    rules = get_mech_rules_dict(cell, value=value, origin=origin, slope=slope, tau=tau, xhalf=xhalf, min=min, max=max,
                                min_loc=min_loc, max_loc=max_loc, outside=outside, custom=custom)
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
        update_syn_mech_by_sec_type(cell, env, sec_type, syn_name, mech_content, update_targets, verbose)
    except Exception as e:
        cell.mech_dict = copy.deepcopy(backup_mech_dict)
        traceback.print_tb(sys.exc_info()[2])
        print 'modify_syn_mech_param: problem updating mechanism: %s; parameter: %s; in sec_type: %s' % \
              (syn_name, param_name, sec_type)
        raise e


def update_syn_mech_by_sec_type(cell, env, sec_type, syn_name, mech_content, update_targets=False, verbose=False):
    """For the provided sec_type and synaptic mechanism, this method
    loops through the parameters specified in the mechanism
    dictionary, interprets the rules, and sets placeholder values in
    the syn_mech_attr_dict of a SynapseAttributes object.

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param sec_type: str
    :param syn_name: str
    :param mech_content: dict
    :param update_targets: bool
    :param verbose: bool
    """
    for param_name, param_content in viewitems(mech_content):
        # accommodate either a dict, or a list of dicts specifying rules for a single parameter
        if isinstance(param_content, dict):
            mech_content = [param_content]
        elif isinstance(param_content, list):
            mech_content = param_content
        else:
            raise RuntimeError('update_syn_mech_by_sec_type: rule for synaptic mechanism: %s parameter: %s was not '
                               'specified properly' % (syn_name, param_name))

        for mech_content_entry in mech_content:
            # print mech_content_entry
            update_syn_mech_param_by_sec_type(cell, env, sec_type, syn_name, param_name, mech_content_entry,
                                              update_targets, verbose)


def update_syn_mech_param_by_sec_type(cell, env, sec_type, syn_name, param_name, rules, update_targets=False,
                                      verbose=False):
    """For the provided synaptic mechanism and parameter, this method
    loops through nodes of the provided sec_type, interprets the
    provided rules, and sets placeholder values in the
    syn_mech_attr_dict of a SynapseAttributes object.  If filter
    queries are provided, their values are converted to enumerated
    types.

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param sec_type: str
    :param syn_name: str
    :param param_name: str
    :param rules: dict
    :param update_targets: bool
    :param verbose: bool
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
                                          update_targets, verbose)


def update_syn_mech_param_by_node(cell, env, node, syn_name, param_name, rules, filters=None, origin_filters=None,
                                  update_targets=False, verbose=False):
    """For the provided synaptic mechanism and parameter, this method
    first determines the set of placeholder synapses in the provided
    node that match any provided filters. Then calls
    parse_syn_mech_rules to interpret the provided rules, and set
    placeholder values in the syn_mech_attr_dict of a
    SynapseAttributes object.


    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param node: :class:'SHocNode'
    :param syn_name: str
    :param param_name: str
    :param rules: dict
    :param filters: dict: {category: list of int}
    :param origin_filters: dict: {category: list of int}
    :param update_targets: bool
    :param verbose: bool
    """
    gid = cell.gid
    cache_queries = env.cache_queries
    syn_attrs = env.synapse_attributes
    if filters is None:
        filtered_syns = syn_attrs.filter_synapses(gid, syn_sections=[node.index], cache=cache_queries)
    else:
        filtered_syns = syn_attrs.filter_synapses(gid, syn_sections=[node.index], cache=cache_queries, **filters)

    if len(filtered_syns) > 0:
        syn_ids = filtered_syns.keys()
        parse_syn_mech_rules(cell, env, node, syn_ids, syn_name, param_name, rules, origin_filters,
                             update_targets=update_targets, verbose=verbose)


def parse_syn_mech_rules(cell, env, node, syn_ids, syn_name, param_name, rules, origin_filters=None, donor=None,
                         update_targets=False, verbose=False):
    """Provided a synaptic mechanism, a parameter, a node, a list of
    syn_ids, and a dict of rules. Interprets the provided rules,
    including complex gradient and inheritance rules. Gradients can be
    specified as linear, exponential, or sigmoidal. Custom functions
    can also be provided to specify more complex distributions. Calls
    inherit_syn_mech_param to retrieve a value from a donor node, if
    necessary. Calls set_syn_mech_param to sets placeholder values in
    the syn_mech_attr_dict of a SynapseAttributes object.

    1) A 'value' with no 'origin' requires no further processing
    2) An 'origin' with no 'value' requires a donor node to inherit a baseline value
    3) An 'origin' with a 'value' requires a donor node to use as a reference point for applying a distance-dependent
    gradient
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param node: :class:'SHocNode'
    :param syn_ids: sequence of int
    :param syn_name: str
    :param param_name: str
    :param rules: dict
    :param origin_filters: dict: {category: list of int}
    :param donor: :class:'SHocNode'
    :param update_targets: bool
    :param verbose: bool
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
                                    update_targets, verbose)
    else:
        set_syn_mech_param(cell, env, node, syn_ids, syn_name, param_name, baseline, rules, donor, update_targets,
                           verbose)


def inherit_syn_mech_param(cell, env, donor, syn_name, param_name, origin_filters=None):
    """Follows path from the provided donor node to root until synapses
    are located that match the provided filter. Returns the requested
    parameter value from the synapse closest to the end of the
    section.

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param donor: :class:'SHocNode'
    :param syn_name: str
    :param param_name: str
    :param origin_filters: dict: {category: list of int}
    :return: float

    """
    gid = cell.gid
    cache_queries = env.cache_queries
    syn_attrs = env.synapse_attributes
    if origin_filters is None:
        filtered_syns = syn_attrs.filter_synapses(gid, syn_sections=[donor.index], cache=cache_queries)
    else:
        filtered_syns = syn_attrs.filter_synapses(gid, syn_sections=[donor.index], cache=cache_queries,
                                                  **origin_filters)

    if len(filtered_syns) > 0:
        valid_syns = []
        for syn_id, syn in viewitems(filtered_syns):
            if syn_attrs.has_mech_attrs(gid, syn_id, syn_name):
                valid_syns.append((syn_id,syn))
        if len(valid_syns) > 0:
            valid_syns.sort(key=lambda x: x[1].syn_loc)
            syn_id = valid_syns[-1][0]
            mech_attrs = syn_attrs.get_mech_attrs(gid, syn_id, syn_name)
            if param_name not in mech_attrs:
                raise RuntimeError('inherit_syn_mech_param: synaptic mechanism: %s at provided donor: %s does not '
                                   'contain the specified parameter: %s' % (syn_name, donor.name, param_name))
            return mech_attrs[param_name]
    if donor is cell.tree.root:
        return
    else:
        return inherit_syn_mech_param(cell, env, donor.parent, syn_name, param_name, origin_filters)


def set_syn_mech_param(cell, env, node, syn_ids, syn_name, param_name, baseline, rules, donor=None,
                       update_targets=False, verbose=False):
    """Provided a synaptic mechanism, a parameter, a node, a list of
    syn_ids, and a dict of rules. Sets placeholder values for each
    provided syn_id in the syn_mech_attr_dict of a SynapseAttributes
    object. If the provided rules specify a distance-dependent
    gradient, a baseline value and a donor node are required as
    reference points.  If update_targets flag is True, the attributes
    of any target synaptic point_process and netcon objects that have
    been inserted will also be updated. Otherwise, they can be updated
    separately by calling config_syns.

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
    :param verbose: bool
    """
    syn_attrs = env.synapse_attributes
    if not ('min_loc' in rules or 'max_loc' in rules or 'slope' in rules):
        for syn_id in syn_ids:
            syn_attrs.modify_mech_attrs(cell.gid, syn_id, syn_name, {param_name: baseline})
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

        for syn_id in syn_ids:
            syn = syn_id_attr_dict[syn_id]
            syn_loc = syn.syn_loc
            distance = get_distance_to_node(cell, donor, node, syn_loc)
            value = get_param_val_by_distance(distance, baseline, slope, min_distance, max_distance, min_val, max_val,
                                              tau, xhalf, outside)

            if value is not None:
                syn_attrs.modify_mech_attrs(cell.gid, syn_id, syn_name, {param_name: value})
                
    if update_targets:
        config_biophys_cell_syns(env, cell.gid, cell.pop_name, syn_ids=syn_ids, insert=False, verbose=verbose)


def parse_custom_syn_mech_rules(cell, env, node, syn_ids, syn_name, param_name, baseline, rules, donor,
                                update_targets=False, verbose=False):
    """If the provided node meets custom criteria, rules are modified and
    passed back to parse_mech_rules with the 'custom' item
    removed. Avoids having to determine baseline and donor over again.

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
    :param verbose: bool
    """
    if 'func' not in rules['custom'] or rules['custom']['func'] is None:
        raise RuntimeError('parse_custom_syn_mech_rules: no custom function provided for synaptic mechanism: %s '
                           'parameter: %s in sec_type: %s' % (syn_name, param_name, node.type))
    if rules['custom']['func'] in globals() and isinstance(globals()[rules['custom']['func']], collections.Callable):
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
                             update_targets=update_targets, verbose=verbose)


def init_syn_mech_attrs(cell, env=None, mech_file_path=None, from_file=False, update_targets=False):
    """Consults a dictionary specifying parameters of NEURON synaptic mechanisms (point processes) for each type of
    section in a BiophysCell. Traverses through the tree of SHocNode nodes following order of inheritance. Calls
    update_syn_mech_by_sec_type to set placeholder values in the syn_mech_attrs_dict of a SynapseAttributes object. If
    update_targets flag is True, the attributes of any target synaptic point_process and netcon objects that have been
    inserted will also be updated. Otherwise, they can be updated separately by calling config_syns.

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


def write_syn_mech_attrs(env, pop_name, gids, output_path, filters=None, syn_names=None, write_kwds={}):
    """
    Write mechanism attributes for the given cell ids to a NeuroH5 file.
    Assumes that attributes have been set via config_syn.
    
    :param env: instance of env.Env
    :param pop_name: population name
    :param gids: cell ids
    :param output_path: path to NeuroH5 file
    :param filters: optional filter for synapses
    """

    syn_attrs = env.synapse_attributes
    rules=syn_attrs.syn_param_rules

    if syn_names is None:
        syn_names = syn_attrs.syn_name_index_dict.keys()

    output_dict = { syn_name: defaultdict(lambda: defaultdict(list)) for syn_name in syn_names }
    for gid in gids:
        if filters is None:
            syns_dict = syn_attrs.syn_id_attr_dict[gid]
        else:
            syns_dict = syn_attrs.filter_synapses(gid, **filters)
        for syn_id, syn in viewitems(syns_dict):
            syn_netcon_dict = syn_attrs.pps_dict[gid][syn_id].netcon
            syn_pps_dict = syn_attrs.pps_dict[gid][syn_id].mech
            for syn_name in syn_names:
                mech_name = syn_attrs.syn_mech_names[syn_name]
                syn_index = syn_attrs.syn_name_index_dict[syn_name]
                if syn_index in syn.attr_dict:
                    attr_keys = syn.attr_dict[syn_index].keys()
                    output_dict[syn_name][gid]['syn_id'].append(syn_id)
                    for k in attr_keys:
                        if (syn_index in syn_netcon_dict) and k in rules[mech_name]['netcon_params']:
                            i = rules[mech_name]['netcon_params'][k]
                            v = getattr(syn_netcon_dict[syn_index], 'weight')[i]
                        elif (syn_index in syn_pps_dict) and hasattr(syn_pps_dict[syn_index], k):
                            v = getattr(syn_pps_dict[syn_index], k)
                        elif (syn_index in syn_netcon_dict) and hasattr(syn_netcon_dict[syn_index], k):
                            v = getattr(syn_netcon_dict[syn_index], k)
                        else:
                            raise RuntimeError('write_syn_mech_attrs: gid %d syn id %d does not have attribute %s '
                                               'set in either %s point process or netcon' % (gid, syn_id, k, syn_name))
                        output_dict[syn_name][gid][k].append(v)

    for syn_name, syn_attrs_dict in viewitems(output_dict):

        attr_dict = {}
        for gid, gid_syn_attrs_dict in viewitems(syn_attrs_dict):
            for attr_name, attr_vals in viewitems(gid_syn_attrs_dict):
                if attr_name == 'syn_ids':
                    attr_dict[gid] = { 'syn_ids': np.asarray(attr_vals, dtype='uint32') }
                else:
                    attr_dict[gid] = { attr_name: np.asarray(attr_vals, dtype='float32') }
        write_cell_attributes(output_path, pop_name, attr_dict,
                              namespace='%s Attributes' % syn_name,
                              **write_kwds)


def sample_syn_mech_attrs(env, pop_name, gids, comm=None):
    """
    Writes mechanism attributes for the given cells to a NeuroH5 file.
    Assumes that attributes have been set via config_syn.
    
    :param env: instance of env.Env
    :param pop_name: population name
    :param gids: cell ids
    """
    if comm is None:
        comm = env.comm
        
    write_syn_mech_attrs(env, pop_name, gids, env.results_file_path, write_kwds={'comm': comm})

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
    if name in content:
        if x is None:
            return content[name]
        elif sec.n3d() == 0:
            return content[name][0]
        else:
            for i in range(sec.n3d()):
                if sec.arc3d(i)/sec.L >= x:
                    return content[name][secnodes[i]]
    else:
        return None


def make_synapse_graph(syn_dict, neurotree_dict):
    """
    Creates a graph of synapses that follows the topological organization of the given neuron.
    :param syn_dict:
    :param neurotree_dict:
    :return: NetworkX.DiGraph
    """
    import networkx as nx
    
    sec_graph = make_neurotree_graph(neurotree_dict)

    syn_ids  = syn_dict['syn_ids']
    syn_locs = syn_dict['syn_locs']
    syn_secs = syn_dict['syn_secs']

    sec_syn_dict = defaultdict(list)
    for syn_id, sec_id, syn_loc in zip(syn_ids, syn_secs, syn_locs):
        sec_syn_dict[sec_id].append(syn_id, syn_loc)
        
    syn_graph = nx.DiGraph()

    for sec_id, syn_id_locs in viewitems(sec_syn_dict):

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
            
        for i, j in zip(syn_id_locs[:-1], syn_id_locs[1:]):
            syn_graph.add_edge(i[0], j[0])

    return syn_graph

    
def synapse_seg_density(syn_type_dict, layer_dict, layer_density_dicts, seg_dict, ran, neurotree_dict=None):
    """
    Computes per-segment density of synapse placement.
    :param syn_type_dict:
    :param layer_dict:
    :param layer_density_dicts:
    :param seg_dict:
    :param ran:
    :param neurotree_dict:
    :return:
    """
    segdensity_dict = {}
    layers_dict = {}

    if neurotree_dict is not None:
        secnodes_dict = neurotree_dict['section_topology']['nodes']
    else:
        secnodes_dict = None
    for (syn_type_label, layer_density_dict) in viewitems(layer_density_dicts):
        syn_type = syn_type_dict[syn_type_label]
        rans = {}
        for (layer_label, density_dict) in viewitems(layer_density_dict):
            if layer_label == 'default':
                layer = layer_label
            else:
                layer = int(layer_dict[layer_label])
            rans[layer] = ran
        segdensity = defaultdict(list)
        layers = defaultdict(list)
        total_seg_density = 0.
        for sec_index, seg_list in viewitems(seg_dict):
            for seg in seg_list:
                L = seg.sec.L
                nseg = seg.sec.nseg
                if neurotree_dict is not None:
                    secnodes = secnodes_dict[sec_index]
                    layer = get_node_attribute('layer', neurotree_dict, seg.sec, secnodes, seg.x)
                else:
                    layer = -1
                layers[sec_index].append(layer)

                this_ran = None

                if layer > -1:
                    if layer in rans:
                        this_ran = rans[layer]
                    elif 'default' in rans:
                        this_ran = rans['default']
                    else:
                        this_ran = None
                elif 'default' in rans:
                    this_ran = rans['default']
                else:
                    this_ran = None
                if this_ran is not None:
                    while True:
                        dens = this_ran.normal(density_dict['mean'], density_dict['variance'])
                        if dens > 0.0:
                            break
                else:
                    dens = 0.
                total_seg_density += dens
                segdensity[sec_index].append(dens)

        if total_seg_density < 1e-6:
            logger.warning("sections with zero %s synapse density: %s; rans: %s; density_dict: %s; morphology: %s" % (syn_type_label, str(segdensity), str(rans), str(density_dict), str(neurotree_dict)))

        segdensity_dict[syn_type] = segdensity

        layers_dict[syn_type] = layers
    return (segdensity_dict, layers_dict)


def synapse_seg_counts(syn_type_dict, layer_dict, layer_density_dicts, sec_index_dict, seg_dict, ran, neurotree_dict=None):
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
    for (syn_type_label, layer_density_dict) in viewitems(layer_density_dicts):
        syn_type = syn_type_dict[syn_type_label]
        rans = {}
        for (layer_label, density_dict) in viewitems(layer_density_dict):
            if layer_label == 'default':
                layer = layer_label
            else:
                layer = layer_dict[layer_label]
            
            rans[layer] = ran
        segcounts = []
        layers = []
        for sec_index, seg_list in viewitems(seg_dict):
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
                    if layer in rans:
                        ran = rans[layer]
                    elif 'default' in rans:
                        ran = rans['default']
                    else:
                        ran = None
                elif 'default' in rans:
                    ran = rans['default']
                else:
                    ran = None
                if ran is not None:
                    l = L / nseg
                    dens = ran.normal(density_dict['mean'], density_dict['variance'])
                    rc = dens * l
                    segcount_total += rc
                    segcounts.append(rc)
                else:
                    segcounts.append(0)

            segcounts_dict[syn_type] = segcounts
            layers_dict[syn_type] = layers
    return (segcounts_dict, segcount_total, layers_dict)


def distribute_uniform_synapses(density_seed, syn_type_dict, swc_type_dict, layer_dict, sec_layer_density_dict,
                                neurotree_dict, cell_sec_dict, cell_secidx_dict):
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

    r = np.random.RandomState()
    local_random.seed(int(seed))

    segcounts_per_sec = {}
    for (sec_name, layer_density_dict) in viewitems(sec_layer_density_dict):
        sec_index_dict = cell_secidx_dict[sec_name]
        swc_type = swc_type_dict[sec_name]
        seg_list = []
        L_total = 0
        (seclst, maxdist) = cell_sec_dict[sec_name]
        secidxlst = cell_secidx_dict[sec_name]
        sec_dict = { int(idx): sec for sec, idx in zip(seclst, secidxlst) }
        seg_dict = {}
        for (sec_index, sec) in viewitems(sec_dict):
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
            synapse_seg_counts(syn_type_dict, layer_dict, layer_density_dict, \
                               sec_index_dict=sec_index_dict, seg_dict=seg_dict, ran=r, \
                               neurotree_dict=neurotree_dict)
        segcounts_per_sec[sec_name] = segcounts_dict
        sample_size = total
        for (syn_type_label, _) in viewitems(layer_density_dict):
            syn_type = syn_type_dict[syn_type_label]
            segcounts = segcounts_dict[syn_type]
            layers = layers_dict[syn_type]
            for sec_index, seg_list in viewitems(seg_dict):
                for seg, layer, seg_count in zip(seg_list, layers, segcounts):
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
                            syn_secs.append(sec_index_dict[seg.sec])
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

    return (syn_dict, segcounts_per_sec)


def distribute_poisson_synapses(density_seed, syn_type_dict, swc_type_dict, layer_dict, sec_layer_density_dict,
                                neurotree_dict, cell_sec_dict, cell_secidx_dict):
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
    import networkx as nx
    
    syn_ids = []
    syn_locs = []
    syn_secs = []
    syn_layers = []
    syn_types = []
    swc_types = []
    syn_index = 0


    sec_graph = make_neurotree_graph(neurotree_dict)

    debug_flag = False
    secnodes_dict = neurotree_dict['section_topology']['nodes']
    for sec, secnodes in viewitems(secnodes_dict):
        if len(secnodes) < 2:
            debug_flag = True

    if debug_flag:
        print 'sec_graph: ', list(sec_graph.edges)
        print 'neurotree_dict: ', neurotree_dict

    seg_density_per_sec = {}
    r = np.random.RandomState()
    r.seed(int(density_seed))
    for (sec_name, layer_density_dict) in viewitems(sec_layer_density_dict):

        swc_type = swc_type_dict[sec_name]
        seg_dict = {}
        L_total = 0

        (seclst, maxdist) = cell_sec_dict[sec_name]
        secidxlst = cell_secidx_dict[sec_name]
        sec_dict = { int(idx): sec for sec, idx in zip(seclst, secidxlst) }
        if len(sec_dict) > 1:
            sec_subgraph = sec_graph.subgraph(list(sec_dict.keys()))
            if len(sec_subgraph.edges()) > 0:
                sec_roots = [ n for n,d in sec_subgraph.in_degree() if d==0 ] 
                sec_edges = []
                for sec_root in sec_roots:
                    sec_edges.append(list(nx.dfs_edges(sec_subgraph, sec_root)))
                    sec_edges.append([(None, sec_root)])
                sec_edges = [val for sublist in sec_edges for val in sublist]
            else:
                sec_edges = [(None, idx) for idx in list(sec_dict.keys()) ]
        else:
            sec_edges = [(None, idx) for idx in list(sec_dict.keys()) ]
        for sec_index, sec in viewitems(sec_dict):
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
                                seg_dict, r, \
                                neurotree_dict=neurotree_dict)
        seg_density_per_sec[sec_name] = seg_density_dict
        for (syn_type_label, _) in viewitems(layer_density_dict):
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
                for seg, layer, density in zip(seg_list,sec_seg_layers,sec_seg_density):
                    seg_start = seg.x - (0.5 / seg.sec.nseg)
                    seg_end   = seg.x + (0.5 / seg.sec.nseg)
                    L = seg.sec.L
                    L_seg_start = seg_start * L
                    L_seg_end   = seg_end * L
                    if density > 0.:
                        beta = 1. / density
                        if interval > 0.:
                            sample = r.exponential(beta)
                        else:
                            while True:
                                sample = r.exponential(beta)
                                if (sample >= L_seg_start) and (sample < L_seg_end):
                                    break
                        interval += sample
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


def generate_log_normal_weights(weights_name, mu, sigma, seed, source_syn_dict):
    local_random = np.random.RandomState()
    local_random.seed(int(seed))
    source_weights = local_random.lognormal(mu, sigma, len(source_syn_dict))
    syn_weight_dict = {}
    # weights are synchronized across all inputs from the same source_gid
    for this_source_gid, this_weight in zip(source_syn_dict, source_weights):
        for this_syn_id in source_syn_dict[this_source_gid]:
            syn_weight_dict[this_syn_id] = this_weight
    weights = np.array(list(syn_weight_dict.values())).astype('float32', copy=False)
    normed_weights = weights 
    weights_dict = \
      { 'syn_id': np.array(list(syn_weight_dict.keys())).astype('uint32', copy=False),
        weights_name: normed_weights }
    return weights_dict
