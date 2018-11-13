import time
from dentate.neuron_utils import *
from dentate.utils import viewitems, zip_longest, partitionn, generator_peek, NamedTupleWithDocstring
from dentate.cells import get_mech_rules_dict, \
    get_donor, get_distance_to_node, get_param_val_by_distance, \
    import_mech_dict_from_file, custom_filter_by_branch_order, \
    custom_filter_by_terminal, make_neurotree_graph
import collections
from collections import namedtuple, defaultdict

# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)

"""This class provides information about the presynaptic (source) cell
connected to a synapse.
  - gid - gid of source cell (int)
  - population - population index of source cell (int)
  - delay - connection delay (float)
"""
class SynapseSource(object):
      __slots__ = 'gid', 'population', 'delay'
      def __init__(self):
            self.gid = None
            self.population = None
            self.delay = None

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
    - mech_attr_dict - dictionary of attributes per synapse mechanism
      (for cases when multiple mechanisms are associated with a
      synapse, e.g. GABA_A and GABA_B)
    - netcon_weights_dict - dictionary of network connection weights
    """,
    'Synapse',
    ['syn_type',
     'swc_type',
     'syn_layer',
     'syn_loc',
     'syn_section',
     'source',
     'mech_attr_dict',
     'netcon_weights_dict',
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
        # dictionary with attributes for each synapse:
        #    { gid: { synapse id: { attribute: value } } }
        self.syn_id_attr_dict = defaultdict(lambda: defaultdict(lambda: None))
        # dictionary of mechanism point processes for each synapse:
        #    { gid: { synapse id: { attribute: value } } }
        self.pps_dict = defaultdict(lambda: defaultdict(lambda: None))
        self.netcon_dict = defaultdict(lambda: defaultdict(lambda: None))
        self.vecstim_dict = defaultdict(lambda: defaultdict(lambda: None))
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

            for i, (syn_id, syn_layer, syn_type, swc_type, syn_sec, syn_loc) in \
              enumerate(zip_longest(syn_ids, syn_layers, syn_types, swc_types, syn_secs, syn_locs)):
                self.syn_id_attr_dict[gid][syn_id] = Synapse(syn_type=syn_type, \
                                                            syn_layer=syn_layer, \
                                                            syn_section=syn_sec, \
                                                            syn_loc=syn_loc, \
                                                            swc_type=swc_type, \
                                                            source=SynapseSource(), \
                                                            mech_attr_dict=defaultdict(dict), \
                                                            netcon_weights_dict={})
                self.pps_dict[gid][syn_id] = {}
                self.netcon_dict[gid][syn_id] = {}
                self.vecstim_dict[gid][syn_id] = {}

    def init_edge_attrs(self, gid, presyn_name, presyn_gids, edge_syn_ids, delays=None):
        """
        Sets connection edge attributes for the specified synapse ids.

        :param gid: gid for post-synaptic (target) cell (int)
        :param presyn_name: name of presynaptic (source) population (string)
        :param presyn_ids: gids for presynaptic (source) cells (array of int)
        :param edge_syn_ids: synapse ids on target cells to be used for connections (array of int)
        :param delays: axon conduction (netcon) delays (array of float)
        """

        presyn_index = int(self.env.pop_dict[presyn_name])
        connection_velocity = float(self.env.connection_velocity[presyn_name])

        if delays is None:
            delays = [h.dt] * len(edge_syn_ids)

        syn_id_dict = self.syn_id_attr_dict[gid]
              
        for edge_syn_id, presyn_gid, delay in zip_longest(edge_syn_ids, presyn_gids, delays):
            syn = syn_id_dict[edge_syn_id]
            if syn is None:
                raise RuntimeError('init_edge_attrs: gid %i: synapse id %i has not been initialized' % \
                                   (gid, edge_syn_id))

            if syn.source.gid is not None:
                raise RuntimeError('init_edge_attrs: gid %i: synapse id %i has already been initialized with edge attributes' % \
                                   (gid, edge_syn_id))

            syn.source.gid = presyn_gid
            syn.source.population = presyn_index
            syn.source.delay = delay


    def init_edge_attrs_from_iter(self, pop_name, presyn_name, attr_info, edge_iter):
        """
        Initializes edge attributes for all cell gids returned by iterator.

        :param pop_name: name of postsynaptic (target) population (string)
        :param source_name: name of presynaptic (source) population (string)
        :param attr_info: dictionary mapping attribute name to indices in iterator tuple
        :param edge_iter: edge attribute iterator
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

            delays = [((distance / connection_velocity) + h.dt) for distance in edge_dists]

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
        if syn_index in pps_dict:
            raise RuntimeError('add_pps: gid %i Synapse id %i already has mechanism %s' % (gid, syn_id, syn_name))
        else:
            pps_dict[syn_index] = pps

            
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
        return (syn_index in pps_dict)

    
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
        if syn_index in pps_dict:
            return pps_dict[syn_index]
        else:
            if throw_error:
                raise RuntimeError('get_pps: gid %i synapse id %i has no point process for mechanism %s' % (gid, syn_id, syn_name))
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
        gid_netcon_dict = self.netcon_dict[gid]
        netcon_dict = gid_netcon_dict[syn_id]
        if syn_index in netcon_dict:
            raise RuntimeError('add_netcon: gid %i Synapse id %i mechanism %s already has netcon' % (gid, syn_id, syn_name))
        else:
            netcon_dict[syn_index] = nc

            
    def has_netcon(self, gid, syn_id, syn_name):
        """
        Returns True if a netcon exists for the specified cell/synapse id/mechanism name, False otherwise.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: bool
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_netcon_dict = self.netcon_dict[gid]
        netcon_dict = gid_netcon_dict[syn_id]
        return (syn_index in netcon_dict)

    
    def get_netcon(self, gid, syn_id, syn_name, throw_error=True):
        """
        Returns the NetCon object associated with the specified cell/synapse id/mechanism name.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: :class:'h.NetCon'
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_netcon_dict = self.netcon_dict[gid]
        netcon_dict = gid_netcon_dict[syn_id]
        if syn_index in netcon_dict:
            return netcon_dict[syn_index]
        else:
            if throw_error:
                raise RuntimeError('get_netcon: gid %i synapse id %i has no netcon for mechanism %s' % (gid, syn_id, syn_name))
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
        gid_vecstim_dict = self.vecstim_dict[gid]
        vecstim_dict = gid_vecstim_dict[syn_id]
        if syn_index in vecstim_dict:
            raise RuntimeError('add_vecstim: gid %i synapse id %i mechanism %s already has vecstim' % (gid, syn_id, syn_name))
        else:
            self.vecstim_dict[syn_index] = vs
            
    def has_vecstim(self, gid, syn_id, syn_name):
        """
        Returns True if a vecstim exists for the specified cell/synapse id/mechanism name, False otherwise.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: bool
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_vecstim_dict = self.vecstim_dict[gid]
        vecstim_dict = gid_vecstim_dict[syn_id]
        return (syn_index in vecstim_dict)

    
    def get_vecstim(self, gid, syn_id, syn_name, throw_error=True):
        """
        Returns the VecStim object associated with the specified cell/synapse id/mechanism name.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: :class:'h.VecStim'
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_vecstim_dict = self.vecstim_dict[gid]
        vecstim_dict = gid_vecstim_dict[syn_id]
        if syn_index in vecstim_dict:
            return vecstim_dict[syn_index]
        else:
            if throw_error:
                raise RuntimeError('get_vecstim: gid %d synapse %d: vecstim for mechanism %s not found' % (gid, syn_id, syn_name))
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
        return syn_index in syn.mech_attr_dict

  
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
        if syn_index in syn.mech_attr_dict:
            return syn.mech_attr_dict[syn_index]
        else:
            if throw_error:
                raise RuntimeError('get_mech_attrs: gid %d synapse %d: attributes for mechanism %s not found' % (gid, syn_id, syn_name))
            else:
                return None
        
        
    def add_mech_attrs(self, gid, syn_id, syn_name, params):
        """
        Specifies mechanism attribute dictionary for the given cell id/synapse id/mechanism name. Assumes mechanism attributes have not been set yet for this synapse mechanism.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param params: dict
        """
        syn_index = self.syn_name_index_dict[syn_name]
        syn_id_dict = self.syn_id_attr_dict[gid]
        syn = syn_id_dict[syn_id]
        mech_attr_dict = syn.mech_attr_dict[syn_index]
        for k, v in viewitems(params):
            if k in mech_attr_dict:
                raise RuntimeError('add_mech_attrs: gid %i synapse id %i mechanism %s already has parameter %s' % (gid, syn_id, syn_name, str(k)))
            else:
                mech_attr_dict[k] = v

                
    def modify_mech_attrs(self, gid, syn_id, syn_name, params, update=lambda old, new: new):
        """
        Modifies mechanism attributes for the given cell id/synapse id/mechanism name. 

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param params: dict
        :param update: lambda (old, new)
        """
        syn_index = self.syn_name_index_dict[syn_name]
        syn_id_dict = self.syn_id_attr_dict[gid]
        syn = syn_id_dict[syn_id]

        mech_attr_dict = syn.mech_attr_dict[syn_index]
        for k, v in viewitems(params):
            if k in mech_attr_dict:
                mech_attr_dict[k] = update(mech_attr_dict[k], v)
            else:
                mech_attr_dict[k] = v
                

    def has_netcon_weights(self, gid, syn_id, syn_name):
        """
        Returns True if netcon weights have been specified for the given cell id/synapse id/synapse mechanism.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: bool
        """
        syn_index = self.syn_name_index_dict[syn_name]
        syn_id_dict = self.syn_id_attr_dict[gid]
        syn = syn_id_dict[syn_id]
        return (syn_index in syn.netcon_weights_dict)

  
    def get_netcon_weights(self, gid, syn_id, syn_name):
        """
        Returns the netcon weights associated with the given cell id/synapse id/synapse mechanism.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: dict
        """
        syn_index = self.syn_name_index_dict[syn_name]
        syn_id_dict = self.syn_id_attr_dict[gid]
        syn = syn_id_dict[syn_id]

        if syn_index in syn.netcon_weights_dict:
            return syn.netcon_weights_dict[syn_index]
        else:
            raise RuntimeError('get_netcon_weights: gid %d synapse %d: weights for mechanism %s not found' % (gid, syn_id, syn_name))
        
        
    def add_netcon_weights(self, gid, syn_name, params):
        """
        Adds netcon weights for the given cell id/synapse id/synapse mechanism.
        Assumes the weights have not been set yet for this synapse.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param params: dict
        """
        syn_index = self.syn_name_index_dict[syn_name]
        syn_id_dict = self.syn_id_attr_dict[gid]
        for syn_id, val in viewitems(params):
            syn = syn_id_dict[syn_id]
            if syn_index in syn.netcon_weights_dict:
                raise RuntimeError('add_netcon_weights: gid %i synapse id %i mechanism %s already has netcon weight' % (gid, syn_id, syn_name))
            else:
                syn.netcon_weights_dict[syn_index] = val
                
    def add_netcon_weights_from_iter(self, gid, syn_name, params_iter):
        """
        Adds netcon weights for the given cell id/synapse id/synapse mechanism.
        Assumes the weights have not been set yet for this synapse.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param params: dict
        """
        syn_index = self.syn_name_index_dict[syn_name]
        syn_id_dict = self.syn_id_attr_dict[gid]
        for syn_id, val in params_iter:
            syn = syn_id_dict[syn_id]
            if syn_index in syn.netcon_weights_dict:
                raise RuntimeError('add_netcon_weights_from_iter: gid %i Synapse id %i mechanism %s already has netcon weight' % (gid, syn_id, syn_name))
            else:
                syn.netcon_weights_dict[syn_index] = val

                
    def modify_netcon_weights(self, gid, syn_id, syn_name, params, update=lambda (old, new): new):
        """
        Modifies netcon weights for the given cell id/synapse id/synapse mechanism.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param params: dict
        """
        syn_index = self.syn_name_index_dict[syn_name]
        syn_id_dict = self.syn_id_attr_dict[gid]
        syn = syn_id_dict[syn_id]
        netcon_weights_dict = syn.netcon_weights_dict[syn_index]
        for k, v in viewitems(params):
            if k in netcon_weights_dict:
                netcon_weights_dict[k] = update(netcon_weights_dict[k], v)
            else:
                netcon_weights_dict[k] = v
                
                
        
    def filter_synapses(self, gid, syn_sections=None, syn_indexes=None, syn_types=None, layers=None, sources=None, swc_types=None, cache=False):
        """
        Returns a subset of the synapses of the given cell according to the given criteria.

        :param gid: int
        :param syn_sections: array of int
        :param syn_indexes: array of int
        :param syn_types: list of enumerated type: synapse category
        :param layers: list of enumerated type: layer
        :param sources: list of enumerated type: population names of source projections
        :param swc_types: list of enumerated type: swc_type
        :return: dicionary { syn_id: { attribute: value } }
        """
        syn_id_attr_dict = self.syn_id_attr_dict[gid]
        matches = lambda query, item: (query is None) or (item in query)

        if cache:
            args = [syn_sections, syn_indexes, syn_types, layers, sources, swc_types]
            if args in self.filter_cache:
                return self.filter_cache[args]
          
        if sources is None:
            source_indexes = None
        else:
            source_indexes = set([self.env.pop_dict(source) for source in sources])
        syn_dict = {k: v for k, v in viewitems(syn_id_attr_dict) if matches(syn_sections, v.syn_section) & \
                                                                    matches(syn_indexes, k) & \
                                                                    matches(syn_types, v.syn_type) & \
                                                                    matches(layers, v.syn_layer) & \
                                                                    matches(source_indexes, v.source.population) & \
                                                                    matches(swc_types, v.swc_type)}

        return syn_dict
  
    def partition_synapses_by_source(self, gid, syn_ids=None):
        """
        Partitions the synapse objects for the given cell based on the
        presynaptic (source) population index.
        
        :param gid: int
        :param syn_ids: array of int

        """
        source_names = {id: name for name, id in viewitems(self.env.pop_dict)}
        syn_id_attr_dict = self.syn_id_attr_dict[gid]

        source_iter = partitionn(viewitems(syn_id_attr_dict), \
                                 lambda((syn_id,syn)): syn.source.population, \
                                 n=len(source_names))

        return dict(map(lambda(source_id,x): (source_names[source_id], x), enumerate(source_iter)))

    def get_filtered_syn_indexes(self, gid, syn_sections=None, syn_indexes=None, syn_types=None, layers=None, sources=None, swc_types=None, cache=False):
        """
        Returns a subset of the synapse ids of the given cell according to the given criteria.
        """
        return self.filter_synapses(gid, syn_sections=syn_sections, syn_indexes=syn_indexes, \
                                    syn_types=syn_types, layers=layers, sources=source, swc_types=swc_types, \
                                    cache=cache).keys()

  
    def partition_syn_ids_by_source(self, gid, syn_ids=None):
        """
        Partitions the synapse ids for the given cell based on the
        presynaptic (source) population index.
        
        :param gid: int
        :param syn_ids: array of int

        """
        start_time = time.time()
        source_names = {id: name for name, id in viewitems(self.env.pop_dict)}
        syn_id_attr_dict = self.syn_id_attr_dict[gid]
        source_iter = partitionn(syn_id_attr_dict.keys(), lambda(syn_id): syn_id_attr_dict[syn_id].source.population, n=len(source_names))

        return dict(map(lambda(source_id,x): (source_names[source_id], x), enumerate(source_iter)))

    def del_syn_id_attr_dict(self, gid):
        """
        Removes the synapse attributes associated with the given cell gid.
        """
        del self.syn_id_attr_dict[gid]

    def clear_filter_cache(self):
        self.filter_cache.clear()

        
def insert_hoc_cell_syns(env, syn_params, gid, cell, syn_ids, unique=False, insert_netcons=False, insert_vecstims=False):
    """
    Insert mechanisms into given cell according to the synapse objects created in env.synapse_attributes.
    Configures mechanisms according to parameter values specified in syn_params.
    
    :param env: :class:'Env'
    :param syn_params: dictionary of the form { mech_name: params }
    :param gid: cell id (int)
    :param cell: hoc cell object
    :param syn_ids: synapse ids (array of int)
    :param unique: True, if unique mechanisms are to be inserted for
    each synapse; False, if synapse mechanisms within a compartment
    will be shared.

    :return: number of inserted mechanisms

    """
    rank = int(env.pc.id())

    syns_dict_dend = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_axon = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_ais = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_hill = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_soma = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    py_sections = [sec for sec in cell.sections]

    swc_type_apical = env.SWC_Types['apical']
    swc_type_basal = env.SWC_Types['basal']
    swc_type_soma = env.SWC_Types['soma']
    swc_type_axon = env.SWC_Types['axon']
    swc_type_ais = env.SWC_Types['ais']
    swc_type_hill = env.SWC_Types['hillock']
    
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
            raise RuntimeError("insert_hoc_cell_syns: unsupported synapse SWC type %d for synapse %d" % (swc_type, syn_id))

        for syn_name, params in viewitems(syn_params):
            syn_mech = make_syn_mech(syn_name=syn_name, seg=sec(syn_loc), syns_dict=syns_dict, \
                                     mech_names=syn_attrs.syn_mech_names)
            syn_attrs.add_pps(gid, syn_id, syn_name, syn_mech)
            config_syn(syn_name=syn_name, \
                       rules=syn_attrs.syn_param_rules, \
                       mech_names=syn_attrs.syn_mech_names, \
                       syn=syn_mech, **params)
                       
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
                config_netcon(syn_name=syn_name, rules=syn_attrs.syn_param_rules, \
                              mech_names=syn_attrs.syn_mech_names, \
                              nc=this_nc, **params)
                nc_count += 1

        syn_count += 1
        
    return syn_count, mech_count, nc_count


def insert_biophys_cell_syns(env, gid, postsyn_name, presyn_name, syn_ids, unique=None, insert_netcons=True, insert_vecstims=True):
    """
    
    1) make syns (if not unique, keep track of syn_in_seg for shared synapses)
    2) initialize syns with syn_mech_params from config_file
    3) make netcons
    4) initialize netcons with parameters from env.connection_config

    :param env: :class:'Env'
    :param gid: int
    :param postsyn_name: str
    :param presyn_name: str
    :param syn_ids: array of int
    :param unique: bool; whether to insert synapses if none exist at syn_id
    """
    rank = int(env.pc.id())
    if not (gid in env.biophys_cells[postsyn_name]):
        raise KeyError('insert_biophys_cell_syns: BiophysCell with gid %i does not exist' % gid)
    cell = env.biophys_cells[postsyn_name][gid]

    syn_attrs = env.synapse_attributes
    syn_params = env.connection_config[postsyn_name][presyn_name].mechanisms

    synapse_config = env.celltypes[postsyn_name]['synapses']
    if unique is None:
        if 'unique' in synapse_config:
            unique = synapse_config['unique']
        else:
            unique = False

    syn_id_attr_dict = syn_attrs.syn_id_attr_dict[gid]
    syn_count, mech_count, nc_count = insert_hoc_cell_syns(env, syn_params, gid, cell.hoc_cell, syn_ids, \
                                                           unique=unique, insert_netcons=insert_netcons, \
                                                           insert_vecstims=insert_vecstims)
    

    if rank == 0:
        logger.info('insert_biophys_cell_syns: source: %s target: %s cell %i: created %i mechanisms and %i netcons for '
                    '%i syn_ids' % (presyn_name, postsyn_name, gid, mech_count, nc_count, syn_count))


def config_biophys_cell_syns(env, gid, postsyn_name, syn_ids=None, insert=False, unique=None, insert_netcons=False, insert_vecstims=False):
    """
    Configures the given syn_ids, and call config_syn with mechanism and netcon parameters (which must not be empty).
    If syn_ids=None, configures all synapses for the cell with the given gid.
    If insert=True, iterate over sources and call insert_biophys_cell_syns
       (requires a BiophysCell with the specified gid to be present in the Env).

    :param env: :class:'Env'
    :param gid: int
    :param postsyn_name: str
    :param syn_ids: array of int
    :param insert: bool; whether to insert synapses if none exist at syn_id
    :param unique: bool; whether newly inserted synapses should be unique or shared per segment
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

    source_syn_ids_dict = syn_attrs.partition_syn_ids_by_source(gid, syn_ids)

    if insert:
        if not (gid in env.biophys_cells[postsyn_name]):
            raise KeyError('config_syns: insert: BiophysCell with gid %i does not exist' % gid)
        for presyn_name, source_syn_ids in viewitems(source_syn_ids_dict):
            p = generator_peek(source_syn_ids)
            if p is not None:
               first, seq = p
               insert_biophys_cell_syns(env, gid, postsyn_name, presyn_name, seq, unique=unique, \
                                        insert_netcons=insert_netcons, insert_vecstims=insert_vecstims)

    cell = env.biophys_cells[postsyn_name][gid]
    syn_count, mech_count, nc_count = config_hoc_cell_syns(env, gid, postsyn_name, \
                                                           cell=cell.hoc_cell, syn_ids=syn_ids, \
                                                           insert=False, unique=unique)

    if rank == 0:
        logger.info('config_biophys_cell_syns: target: %s; cell %i: set parameters for %i syns and %i '
                    'netcons for %i syn_ids' % (postsyn_name, gid, syn_count, nc_count, len(syn_ids)))

    return syn_count, nc_count


def config_hoc_cell_syns(env, gid, postsyn_name, cell=None, syn_ids=None, unique=None, insert=False, insert_netcons=False, insert_vecstims=False):
    """
    Configures the given syn_ids, and call config_syn with mechanism and netcon parameters (which must not be empty).
    If syn_ids=None, configures all synapses for the cell with the given gid.
    If insert=True, iterate over sources and call insert_hoc_cell_syns
       (requires the cell object is given or registered with h.ParallelContext on this rank).

    :param env: :class:'Env'
    :param gid: int
    :param postsyn_name: str
    :param syn_ids: array of int
    :param insert: bool; whether to insert synapses if none exist at syn_id
    :param unique: bool; whether newly inserted synapses should be unique or shared per segment
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
            p = generator_peek(source_syns)
            if p is not None:
                first, seq = p
                source_syn_ids = [x[0] for x in seq]
                syn_params = env.connection_config[postsyn_name][presyn_name].mechanisms
                insert_hoc_cell_syns(env, syn_params, gid, cell, source_syn_ids, unique=unique, \
                                     insert_netcons=insert_netcons, insert_vecstims=insert_vecstims)
        if rank == 0:
           logger.info('config_hoc_cell_syns: population: %s; cell %i: inserted mechanisms in %f s' % \
                       (postsyn_name, gid, time.time() - last_time))

    source_syn_dict = syn_attrs.partition_synapses_by_source(gid, syn_ids)

    total_nc_count = 0
    total_mech_count = 0
    config_time = time.time()
    for presyn_name, source_syns in viewitems(source_syn_dict):
        p = generator_peek(source_syns)
        if p is None:
            continue

        first, seq = p
        
        last_time = time.time() 
        nc_count = 0
        mech_count = 0

        syn_names = list(env.connection_config[postsyn_name][presyn_name].mechanisms.keys())
        syn_indexes = [syn_attrs.syn_name_index_dict[syn_name] for syn_name in syn_names]
        for syn_id, syn in seq:
            for syn_name, syn_index in zip_longest(syn_names, syn_indexes):
                this_pps = None
                this_netcon = None
                if syn_index in syn.mech_attr_dict:
                    this_pps = syn_attrs.get_pps(gid, syn_id, syn_name, throw_error=False)
                    if this_pps is None:
                       raise RuntimeError('config_hoc_cell_syns: insert: cell gid %i synapse %i does not have point process for mechanism %s' % (gid, syn_id, syn_name))
                    mech_params = syn.mech_attr_dict[syn_index]
                    config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules, \
                               mech_names=syn_attrs.syn_mech_names, syn=this_pps, \
                               **mech_params)
                    mech_count += 1
                if syn_index in syn.netcon_weights_dict:
                    this_netcon = syn_attrs.get_netcon(gid, syn_id, syn_name, throw_error=False)
                    if this_netcon is None:
                          raise RuntimeError('config_hoc_cell_syns: insert: cell gid %i synapse %i does not have netcon for mechanism %s' % (gid, syn_id, syn_name))
                    netcon_params = { 'weight': syn.netcon_weights_dict[syn_index] }
                    config_netcon(syn_name=syn_name, rules=syn_attrs.syn_param_rules, \
                                  mech_names=syn_attrs.syn_mech_names, nc=this_netcon, \
                                  **netcon_params)
                          
                    nc_count += 1

        total_nc_count += nc_count
        total_mech_count += mech_count

    if rank == 0:
          logger.info('config_hoc_cell_syns: population: %s; cell %i: configured %i synapses in %f s' % \
                      (postsyn_name, gid, len(syn_ids), time.time() - config_time))

    return len(syn_ids), total_mech_count, total_nc_count


def config_syn(syn_name, rules, syn, mech_names=None, **params):
    """
    Initializes synaptic mechanisms with parameters specified in the synapse attribute dictionaries.

    :param syn_name: synapse mechanism name
    :param rules: parameter rules for specified hoc mechanism
    :param mech_names: mapping from synapse mechanism name to hoc mechanism name
    :param syn: synaptic mechanism point process
    :param params: dict
    """
    if mech_names is not None:
        mech_name = mech_names[syn_name]
    else:
        mech_name = syn_name
    for param, val in viewitems(params):
        if param in rules[mech_name]['mech_params']:
            if hasattr(syn, param):
                setattr(syn, param, val)
            else:
                raise AttributeError('config_syn: point process has no attribute: %s for synaptic mechanism: %s' % \
                                     (param, syn_name))

            
def config_netcon(syn_name, rules, nc, mech_names=None, **params):
    """
    Initializes synaptic netcons with parameters specified in the synapse attribute dictionaries.

    :param syn_name: synapse mechanism name
    :param rules: parameter rules for specified hoc mechanism
    :param mech_names: mapping from synapse mechanism name to hoc mechanism name
    :param syn: synaptic mechanism point process
    :param nc: :class:'h.NetCon'
    :param params: dict
    """
    if mech_names is not None:
        mech_name = mech_names[syn_name]
    else:
        mech_name = syn_name
    for param, val in viewitems(params):
        if param in rules[mech_name]['netcon_params']:
            i = rules[mech_name]['netcon_params'][param]
            if int(nc.wcnt()) >= i:
                nc.weight[i] = val
            else:
                raise AttributeError('config_netcon: invalid parameter index for netcon parameter: %s for synaptic mechanism: %s' % \
                                         (param, syn_name))


def syn_in_seg(syn_name, seg, syns_dict):
    """
    If a synaptic mechanism of the specified type already exists in the specified segment, it is returned. Otherwise, it returns None.

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
    syn = syn_in_seg(syn_name, seg, syns_dict)
    if syn is None:
        if mech_names is not None:
            mech_name = mech_names[syn_name]
        else:
            mech_name = syn_name
        syn = make_syn_mech(mech_name, seg)
        syns_dict[seg.sec][seg.x][syn_name] = syn
    return syn


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
    syn = make_syn_mech(mech_name, seg)
    return syn





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
    """Used by modify_syn_mech_param. Takes in a series of arguments and
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
    config_syns.

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
    rules = get_mech_rules_dict(cell, value=value, origin=origin, slope=slope, tau=tau, xhalf=xhalf, min=min, \
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

#    try:
    update_syn_mech_by_sec_type(cell, env, sec_type, syn_name, mech_content, update_targets)
#    except (KeyError, ValueError, AttributeError, NameError, RuntimeError, IOError) as e:
#        cell.mech_dict = copy.deepcopy(backup_mech_dict)
#        raise RuntimeError('modify_syn_mech_param: problem updating mechanism: %s; parameter: %s; in sec_type: %s\n' \
#                           '%s\n%s' % (syn_name, param_name, sec_type, e, str(sys.exc_info()[2])))


def update_syn_mech_by_sec_type(cell, env, sec_type, syn_name, mech_content, update_targets=False):
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

    """
    gid = cell.gid
    syn_attrs = env.synapse_attributes
    syn_id_attr_dict = syn_attrs.syn_id_attr_dict[gid]
    if filters is None:
        filtered_syns = syn_attrs.filter_synapses(gid, syn_sections=[node.index])
    else:
        filtered_syns = syn_attrs.filter_synapses(gid, **filters)
    if len(filtered_syns) > 0:
        syn_ids = filtered_syns.keys()
        parse_syn_mech_rules(cell, env, node, syn_ids, syn_name, param_name, rules, origin_filters,
                             update_targets=update_targets)


def parse_syn_mech_rules(cell, env, node, syn_ids, syn_name, param_name, rules, origin_filters=None, donor=None,
                         update_targets=False):
    """Provided a synaptic mechanism, a parameter, a node, a list of
    syn_ids, and a dict of rules. Interprets the provided rules,
    including complex gradient and inheritance rules. Gradients can be
    specified as linear, exponential, or sigmoidal. Custom functions
    can also be provided to specify more complex distributions. Calls
    inherit_syn_mech_param to retrieve a value from a donor node, if
    necessary. Calls set_syn_mech_params to sets placeholder values in
    the syn_mech_attr_dict of a SynapseAttributes object.

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
    """Follows path from the provided donor node to root until synapses
    are located that match the provided filter. Returns the requested
    parameter value from the synapse closest to the end of the
    section.  for the requested parameter.

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
    if origin_filters is None:
        filtered_syns = syn_attrs.filter_synapses(gid, syn_sections=[donor.index])
    else:
        filtered_syns = syn_attrs.filter_synapses(gid, **origin_filters)

    if len(filtered_syns) > 0:
        valid_syns = []
        for syn_id, syn in viewitems(filtered_syns):
            if syn_attrs.has_mech_attrs(gid, syn_id, syn_name):
                valid_syns.append((syn_id,syn))
        if len(valid_syns) > 0:
            valid_syns.sort(key=lambda x: x[1].syn_loc)
            syn_id = valid_syns[-1][0]
            return syn_attrs.get_mech_attrs(gid, syn_id, syn_name)[param_name]
    if donor is cell.tree.root:
        return
    else:
        return inherit_syn_mech_param(cell, env, donor.parent, syn_name, param_name, origin_filters)


def set_syn_mech_param(cell, env, node, syn_ids, syn_name, param_name, baseline, rules, donor=None,
                       update_targets=False):
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

    """
    if not ('min_loc' in rules or 'max_loc' in rules or 'slope' in rules):
        for syn_id in syn_ids:
            syn_attrs = env.synapse_attributes
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

        syn_index = syn_attrs.syn_name_index_dict[syn_name]
        for syn_id in syn_ids:
            syn = syn_attrs.syn_id_attr_dict[gid][syn_id]
            syn_loc = syn.syn_loc
            distance = get_distance_to_node(cell, donor, node, syn_loc)
            value = get_param_val_by_distance(distance, baseline, slope, \
                                              min_distance, max_distance, min_val, max_val, \
                                              tau, xhalf, outside)

            if value is not None:
                syn_attrs.modify_mech_attrs(cell.gid, syn_id, syn_name, {param_name: value})
    if update_targets:
        config_biophys_cell_syns(env, cell.gid, cell.pop_name, syn_ids=syn_ids, insert=False)


def parse_custom_syn_mech_rules(cell, env, node, syn_ids, syn_name, param_name, baseline, rules, donor,
                                update_targets=False):
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
                             update_targets=update_targets)


def init_syn_mech_attrs(cell, env=None, mech_file_path=None, from_file=False, update_targets=False):
    """Consults a dictionary specifying parameters of NEURON synaptic
    mechanisms (point processes) for each type of section in a
    BiophysCell. Traverses through the tree of SHocNode nodes
    following order of inheritance. Calls update_syn_mech_by_sec_type
    to set placeholder values in the syn_mech_attrs_dict of a
    SynapseAttributes object. If update_targets flag is True, the
    attributes of any target synaptic point_process and netcon objects
    that have been inserted will also be updated. Otherwise, they can
    be updated separately by calling config_syns.

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
    for (syn_type_label, layer_density_dict) in viewitems(layer_density_dicts):
        syn_type = syn_type_dict[syn_type_label]
        rans = {}
        for (layer_label, density_dict) in viewitems(layer_density_dict):
            if layer_label == 'default':
                layer = layer_label
            else:
                layer = layer_dict[layer_label]
            ran = h.Random(seed)
            ran.normal(density_dict['mean'], density_dict['variance'])
            rans[layer] = ran
        segdensity = defaultdict(list)
        layers = defaultdict(list)
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
                    while True:
                        dens = ran.repick()
                        if dens > 0.0:
                            break
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
    for (syn_type_label, layer_density_dict) in viewitems(layer_density_dicts):
        syn_type = syn_type_dict[syn_type_label]
        rans = {}
        for (layer_label, density_dict) in viewitems(layer_density_dict):
            if layer_label == 'default':
                layer = layer_label
            else:
                layer = layer_dict[layer_label]
            ran = h.Random(seed)
            ran.normal(density_dict['mean'], density_dict['variance'])
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
                               sec_index_dict=sec_index_dict, seg_dict=seg_dict, seed=density_seed, \
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

def distribute_poisson_synapses(density_seed, syn_type_dict, swc_type_dict, layer_dict, sec_layer_density_dict, neurotree_dict,
                                cell_sec_dict, cell_secidx_dict):
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

    seg_density_per_sec = {}
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
                                seg_dict, density_seed, \
                                neurotree_dict=neurotree_dict)
        seg_density_per_sec[sec_name] = seg_density_dict
        for (syn_type_label, _) in viewitems(layer_density_dict):
            r = np.random.RandomState()
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
    normed_weights = weights / np.percentile(weights, 99)
    weights_dict = \
      { 'syn_id': np.array(list(syn_weight_dict.keys())).astype('uint32', copy=False),
        weights_name: normed_weights }
    return weights_dict
