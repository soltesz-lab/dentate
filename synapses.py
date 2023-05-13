import sys, collections, copy, itertools, math, pprint, uuid, time, traceback
from functools import reduce
from collections import defaultdict
import numpy as np
from scipy import signal, spatial
from neuroh5.io import write_cell_attributes
from dentate.nnls import nnls_gdal
from dentate.cells import get_distance_to_node, get_donor, get_mech_rules_dict, get_param_val_by_distance, \
    import_mech_dict_from_file, make_section_graph, custom_filter_if_terminal, \
    custom_filter_modify_slope_if_terminal, custom_filter_by_branch_order
from dentate.neuron_utils import h, default_ordered_sec_types, mknetcon, mknetcon_vecstim, interplocs, list_find
from dentate.utils import KDDict, ExprClosure, Promise, NamedTupleWithDocstring, get_module_logger, generator_ifempty, map, range, str, \
     viewitems, viewkeys, zip, zip_longest, partitionn, rejection_sampling

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
    def __repr__(self): 
       if self.delay is None:
           repr_delay = 'None'
       else:
           repr_delay = f'{self.delay:.02f}'
       return f'SynapseSource({self.gid}, {self.population}, {repr_delay})'
    def __str__(self): 
       if self.delay is None:
           str_delay = 'None'
       else:
           str_delay = f'{self.delay:.02f}'
       return f'SynapseSource({self.gid}, {self.population}, {str_delay})'

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
        self.syn_config = { k: v['synapses'] for k, v in viewitems(env.celltypes) if 'synapses' in v }
        self.syn_param_rules = syn_param_rules
        self.syn_name_index_dict = {label: index for index, label in enumerate(syn_mech_names)}  # int : mech_name dict
        self.syn_id_attr_dict = defaultdict(lambda: defaultdict(lambda: None))
        self.syn_id_attr_backup_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
        self.sec_dict = defaultdict(lambda: defaultdict(lambda: dict()))
        self.pps_dict = defaultdict(lambda: defaultdict(lambda: SynapsePointProcess(mech={}, netcon={}, vecstim={})))
        self.presyn_names = {id: name for name, id in viewitems(env.Populations)}
        self.filter_cache = {}

    def init_syn_id_attrs_from_iter(self, cell_iter, attr_type='dict', attr_tuple_index=None, debug=False):
        """
        Initializes synaptic attributes given an iterator that returns (gid, attr_dict).
        See `init_syn_id_attrs` for details on the format of the input dictionary.
        """
        
        first_gid = True
        if attr_type == 'dict':
            for (gid, attr_dict) in cell_iter:
                syn_ids = attr_dict['syn_ids']
                syn_layers = attr_dict['syn_layers']
                syn_types = attr_dict['syn_types']
                swc_types = attr_dict['swc_types']
                syn_secs = attr_dict['syn_secs']
                syn_locs = attr_dict['syn_locs']
                self.init_syn_id_attrs(gid, syn_ids, syn_layers, syn_types, swc_types, syn_secs, syn_locs)
        elif attr_type == 'tuple':
            syn_ids_ind = attr_tuple_index.get('syn_ids', None)
            syn_locs_ind = attr_tuple_index.get('syn_locs', None)
            syn_layers_ind = attr_tuple_index.get('syn_layers', None)
            syn_types_ind = attr_tuple_index.get('syn_types', None)
            swc_types_ind = attr_tuple_index.get('swc_types', None)
            syn_secs_ind = attr_tuple_index.get('syn_secs', None)
            syn_locs_ind = attr_tuple_index.get('syn_locs', None)
            for (gid, attr_tuple) in cell_iter:
                syn_ids = attr_tuple[syn_ids_ind]
                syn_layers = attr_tuple[syn_layers_ind]
                syn_types = attr_tuple[syn_types_ind]
                swc_types = attr_tuple[swc_types_ind]
                syn_secs = attr_tuple[syn_secs_ind]
                syn_locs = attr_tuple[syn_locs_ind]
                self.init_syn_id_attrs(gid, syn_ids, syn_layers, syn_types, swc_types, syn_secs, syn_locs)
        else:
            raise RuntimeError(f'init_syn_id_attrs_from_iter: unrecognized input attribute type {attr_type}')

    def init_syn_id_attrs(self, gid, syn_ids, syn_layers, syn_types, swc_types, syn_secs, syn_locs):

        """
        Initializes synaptic attributes for the given cell gid.
        Only the intrinsic properties of a synapse, such as type, layer, location are set.

        Connection edge attributes such as source gid, point process
        parameters, and netcon/vecstim objects are initialized to None
        or empty dictionaries.

          - syn_ids: synapse ids
          - syn_layers: layer index for each synapse id
          - syn_types: synapse type for each synapse id
          - swc_types: swc type for each synapse id
          - syn_secs: section index for each synapse id
          - syn_locs: section location for each synapse id

        """
        if gid in self.syn_id_attr_dict:
            raise RuntimeError(f'Entry {gid} exists in synapse attribute dictionary')
        else:
            syn_dict = self.syn_id_attr_dict[gid]
            sec_dict = self.sec_dict[gid]
            for i, (syn_id, syn_layer, syn_type, swc_type, syn_sec, syn_loc) in \
                    enumerate(zip_longest(syn_ids, syn_layers, syn_types, swc_types, syn_secs, syn_locs)):
                syn = Synapse(syn_type=syn_type, syn_layer=syn_layer, syn_section=syn_sec, syn_loc=syn_loc,
                              swc_type=swc_type, source=SynapseSource(), attr_dict=defaultdict(dict))
                syn_dict[syn_id] = syn
                sec_dict[syn_sec][syn_id] = syn


    def modify_syn_locs(self, gid, syn_ids, syn_secs, syn_locs):
        """
        Modifies synaptic section and location for existing synapses.
        """
        syn_dict = self.syn_id_attr_dict[gid]
        sec_dict = self.sec_dict[gid]
        for syn_id, syn_sec, syn_loc in zip(syn_ids, syn_secs, syn_locs):
            syn = syn_dict[syn_id]
            old_syn_sec = syn.syn_section
            del(sec_dict[old_syn_sec][syn_id])
            syn = syn._replace(syn_section=syn_sec, syn_loc=syn_loc)
            syn_dict[syn_id] = syn
            sec_dict[syn_sec][syn_id] = syn

        
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
            delays = [2.0*h.dt] * len(edge_syn_ids)

        syn_id_dict = self.syn_id_attr_dict[gid]

        for edge_syn_id, presyn_gid, delay in zip_longest(edge_syn_ids, presyn_gids, delays):
            syn = syn_id_dict[edge_syn_id]
            if syn is None:
                raise RuntimeError(f'init_edge_attrs: gid {gid}: synapse id {edge_syn_id} has not been initialized')

            if syn.source.gid is not None:
                raise RuntimeError('init_edge_attrs: gid {gid}: synapse id {edge_syn_id} has already been initialized with edge attributes')

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
            raise RuntimeError(f'init_edge_attrs_from_iter: missing edge attributes for projection {presyn_name} -> {pop_name}')

        if 'Synapses' in edge_attr_info and \
                'syn_id' in edge_attr_info['Synapses'] and \
                'Connections' in edge_attr_info and \
                'distance' in edge_attr_info['Connections']:
            syn_id_attr_index = edge_attr_info['Synapses']['syn_id']
            distance_attr_index = edge_attr_info['Connections']['distance']
        else:
            raise RuntimeError(f'init_edge_attrs_from_iter: missing edge attributes for projection {presyn_name} -> {pop_name}')

        for (postsyn_gid, edges) in edge_iter:
            presyn_gids, edge_attrs = edges
            edge_syn_ids = edge_attrs['Synapses'][syn_id_attr_index]
            edge_dists = edge_attrs['Connections'][distance_attr_index]

            if set_edge_delays:
                delays = [max((distance / connection_velocity), 2.0*h.dt) for distance in edge_dists]
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
            raise RuntimeError(f'add_pps: gid {gid} Synapse id {syn_id} already has mechanism {syn_name}')
        else:
            pps_dict.mech[syn_index] = pps
        return pps

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
                raise RuntimeError(f'get_pps: gid {gid} synapse id {syn_id} has no point process for mechanism {syn_name}')
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
            raise RuntimeError(f'add_netcon: gid {gid} Synapse id {syn_id} mechanism {syn_name} already has netcon')
        else:
            pps_dict.netcon[syn_index] = nc
        return nc

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
                raise RuntimeError(f'get_netcon: gid {gid} synapse id {syn_id} has no netcon for mechanism {syn_name}')
            else:
                return None
            
    def del_netcon(self, gid, syn_id, syn_name, throw_error=True):
        """
        Removes a NetCon object for the specified cell/synapse id/mechanism name.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        if syn_index in pps_dict.netcon:
            del pps_dict.netcon[syn_index]
        else:
            if throw_error:
                raise RuntimeError(f'del_netcon: gid {gid} synapse id {syn_id} has no netcon for mechanism {syn_name}')


    def add_vecstim(self, gid, syn_id, syn_name, vs, nc):
        """
        Adds a VecStim object and associated NetCon for the specified cell/synapse id/mechanism name.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param vs: :class:'h.VecStim'
        :param nc: :class:'h.NetCon'
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        if syn_index in pps_dict.vecstim:
            raise RuntimeError(f'add_vecstim: gid {gid} synapse id {syn_id} mechanism {syn_name} already has vecstim')
        else:
            pps_dict.vecstim[syn_index] = vs, nc
        return vs

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
        Returns the VecStim and NetCon objects associated with the specified cell/synapse id/mechanism name.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: tuple of :class:'h.VecStim' :class:'h.NetCon'
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        if syn_index in pps_dict.vecstim:
            return pps_dict.vecstim[syn_index]
        else:
            if throw_error:
                raise RuntimeError(f'get_vecstim: gid {gid} synapse {syn_id}: vecstim for mechanism {syn_name} not found')
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
                raise RuntimeError(f'get_mech_attrs: gid {gid} synapse {syn_id}: attributes for mechanism {syn_name} not found')
            else:
                return None

    def add_mech_attrs(self, gid, syn_id, syn_name, params, append=False):
        """
        Specifies mechanism attribute dictionary for the given cell id/synapse id/mechanism name. Assumes mechanism
        attributes have not been set yet for this synapse mechanism.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param params: dict
        :param append: whether to append attribute values with the same attribute name
        """
        self.add_mech_attrs_from_iter(gid, syn_name, iter({syn_id: params}), multiple='error', append=append)

    def stash_syn_attrs(self, pop_name, gid):
        """
        Preserves synaptic attributes for the given cell id. 

        :param pop_name: population name
        :param gid: cell id
        :param syn_id: synapse id
        """
        rules = self.syn_param_rules
        syn_id_dict = self.syn_id_attr_dict[gid]
        syn_id_backup_dict = self.syn_id_attr_backup_dict[gid]
        stash_id = uuid.uuid4()
        syn_id_backup_dict[stash_id] = copy.deepcopy(syn_id_dict)
        return stash_id

    def restore_syn_attrs(self, pop_name, gid, stash_id):
        """
        Restores synaptic attributes for the given cell id. 

        :param pop_name: population name
        :param gid: cell id
        :param syn_id: synapse id
        """
        rules = self.syn_param_rules
        syn_id_backup_dict = self.syn_id_attr_backup_dict[gid][stash_id]
        if syn_id_backup_dict is not None:
            self.syn_id_attr_dict[gid] = copy.deepcopy(syn_id_backup_dict)
            del(self.syn_id_attr_backup_dict[gid][stash_id])


        
    def modify_mech_attrs(self, pop_name, gid, syn_id, syn_name, params, expr_param_check='ignore'):
        """
        Modifies mechanism attributes for the given cell id/synapse id/mechanism name. 

        :param pop_name: population name
        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param params: dict
        """
        rules = self.syn_param_rules
        syn_index = self.syn_name_index_dict[syn_name]
        syn_id_dict = self.syn_id_attr_dict[gid]
        mech_name = self.syn_mech_names[syn_name]
        syn = syn_id_dict[syn_id]
        presyn_name = self.presyn_names.get(syn.source.population, None)
        if presyn_name:
            connection_syn_params = self.env.connection_config[pop_name][presyn_name].mechanisms
        else:
            connection_syn_params = None

        mech_params = {}
        if connection_syn_params is not None:
            if 'default' in connection_syn_params:
                section_syn_params = connection_syn_params['default']
            else:
                section_syn_params = connection_syn_params[syn.swc_type]
            mech_params = section_syn_params.get(syn_name, {})

        attr_dict = syn.attr_dict[syn_index]
        for k, v in viewitems(params):
            if k in rules[mech_name]['mech_params']:
                mech_param = mech_params.get(k, None)
                if isinstance(mech_param, ExprClosure):
                    if mech_param.parameters[0] == 'delay':
                        new_val = mech_param(syn.source.delay)
                    else:
                        raise RuntimeError(f'modify_mech_attrs: unknown dependent expression parameter {mech_param.parameters}')
                else:
                    new_val = v
                assert(new_val is not None)
                old_val = attr_dict.get(k, mech_param)
                if isinstance(new_val, ExprClosure):
                    if isinstance(old_val, Promise):
                        old_val.clos = new_val
                    else:
                        attr_dict[k] = Promise(new_val, old_val)
                elif isinstance(new_val, dict):
                    if isinstance(old_val, Promise):
                        for sk, sv in viewitems(new_val):
                            old_val.clos[sk] = sv
                    elif isinstance(old_val, ExprClosure):
                        for sk, sv in viewitems(new_val):
                            old_val[sk] = sv
                    else:
                        if expr_param_check == 'ignore':
                            pass
                        else:
                            raise RuntimeError(f'modify_mech_attrs: dictionary value provided to a non-expression parameter {k}')
                else:
                    attr_dict[k] = new_val
            elif k in rules[mech_name]['netcon_params']:

                mech_param = mech_params.get(k, None)
                if isinstance(mech_param, ExprClosure):
                    if mech_param.parameters[0] == 'delay':
                        new_val = mech_param(syn.source.delay)

                    else:
                        raise RuntimeError(f'modify_mech_attrs: unknown dependent expression parameter {mech_param.parameters}')
                else:
                    new_val = v
                assert(new_val is not None)
                old_val = attr_dict.get(k, mech_param)
                if isinstance(new_val, ExprClosure):
                    if isinstance(old_val, Promise):
                        old_val.clos = new_val
                    else:
                        attr_dict[k] = Promise(new_val, old_val)
                elif isinstance(new_val, dict):
                    if isinstance(old_val, Promise):
                        for sk, sv in viewitems(new_val):
                            old_val.clos[sk] = sv
                    elif isinstance(old_val, ExprClosure):
                        for sk, sv in viewitems(new_val):
                            old_val[sk] = sv
                    else:
                        if expr_param_check == 'ignore':
                            pass
                        else:
                            raise RuntimeError('modify_mech_attrs: dictionary value provided to a non-expression parameter '
                                               f'{k} mechanism: {mech_name} presynaptic: {presyn_name} old value: {old_val}')
                else:
                    attr_dict[k] = new_val
                
            else:
                raise RuntimeError(f'modify_mech_attrs: unknown type of parameter {k}')
        syn.attr_dict[syn_index] = attr_dict

    def add_mech_attrs_from_iter(self, gid, syn_name, params_iter, multiple='error', append=False):
        """
        Adds mechanism attributes for the given cell id/synapse id/synapse mechanism.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param params_iter: iterator
        :param multiple: behavior when an attribute value is provided for a synapse that already has attributes:
               - 'error' (default): raise an error
               - 'skip': do not update attribute value
               - 'overwrite': overwrite value
        :param append: whether to append attribute values with the same attribute name
        """
        syn_index = self.syn_name_index_dict[syn_name]
        syn_id_dict = self.syn_id_attr_dict[gid]
        for syn_id, params_dict in params_iter:
            syn = syn_id_dict[syn_id]
            if syn is None:
                raise RuntimeError('add_mech_attrs_from_iter: '
                                   f'gid {gid} synapse id {syn_id} has not been created yet')
            if syn_index in syn.attr_dict:
                if multiple == 'error':
                    raise RuntimeError('add_mech_attrs_from_iter: '
                                       f'gid {gid} synapse id {syn_id} mechanism {syn_name} already has parameters')
                elif multiple == 'skip':
                    continue
                elif multiple == 'overwrite':
                    pass
                else:
                    raise RuntimeError(f'add_mech_attrs_from_iter: unknown multiple value {multiple}')

            attr_dict = syn.attr_dict[syn_index]
            for k, v in viewitems(params_dict):
                if v is None:
                    raise RuntimeError('add_mech_attrs_from_iter: '
                                       f'gid {gid} synapse id {syn_id} mechanism {syn_name} parameter {k} has no value')
                if append:
                    k_val = attr_dict.get(k, [])
                    k_val.append(v)
                    attr_dict[k] = k_val
                else:
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
        matches = lambda items: all(
            map(lambda query_item: (query_item[0] is None) or (query_item[1] in query_item[0]), items))

        if cache:
            cache_args = tuple([tuple(x) if isinstance(x, list) else x for x in
                                [gid, syn_sections, syn_indexes, syn_types, layers, sources, swc_types]])
            if cache_args in self.filter_cache:
                return self.filter_cache[cache_args]

        if sources is None:
            source_indexes = None
        else:
            source_indexes = set(sources)

        sec_dict = self.sec_dict[gid]
        if syn_sections is not None:
            # Fast path
            it = itertools.chain.from_iterable([sec_dict[sec_index].items() for sec_index in syn_sections])
            syn_dict = {k: v for (k, v) in it}
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
        source_names[-1] = None

        if syn_ids is None:
            syn_id_attr_dict = self.syn_id_attr_dict[gid]
        else:
            syn_id_attr_dict = {syn_id: self.syn_id_attr_dict[gid][syn_id] for syn_id in syn_ids}

        source_iter = partitionn(viewitems(syn_id_attr_dict),
                                 lambda syn_id_syn: syn_id_syn[1].source.population+1
                                     if syn_id_syn[1].source.population is not None else 0,
                                 n=len(source_names))

        return dict([(source_names[source_id_x[0]-1], generator_ifempty(source_id_x[1])) for source_id_x in
                     enumerate(source_iter)])

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
        return list(self.filter_synapses(gid, syn_sections=syn_sections, syn_indexes=syn_indexes, syn_types=syn_types,
                                         layers=layers, sources=sources, swc_types=swc_types, cache=cache).keys())

    def partition_syn_ids_by_source(self, gid, syn_ids=None):
        """
        Partitions the synapse ids for the given cell based on the
        presynaptic (source) population index.
        
        :param gid: int
        :param syn_ids: array of int

        """
        start_time = time.time()
        source_names = {id: name for name, id in viewitems(self.env.Populations)}
        source_names[-1] = None
        syn_id_attr_dict = self.syn_id_attr_dict[gid]
        if syn_ids is None:
            syn_ids = list(syn_id_attr_dict.keys())

        def partition_pred(syn_id):
            syn = syn_id_attr_dict[syn_id]
            return syn.source.population+1 if syn.source.population is not None else 0

        source_iter = partitionn(syn_ids, partition_pred, n=len(source_names))

        return dict([(source_names[source_id_x[0]-1], generator_ifempty(source_id_x[1])) for source_id_x in
                     enumerate(source_iter)])

    def del_syn_id_attr_dict(self, gid):
        """
        Removes the synapse attributes associated with the given cell gid.
        """
        del self.syn_id_attr_dict[gid]
        del self.sec_dict[gid]

    def clear(self):
        self.syn_id_attr_dict = defaultdict(lambda: defaultdict(lambda: None))
        self.sec_dict = defaultdict(lambda: defaultdict(lambda: dict()))
        self.pps_dict = defaultdict(lambda: defaultdict(lambda: SynapsePointProcess(mech={}, netcon={}, vecstim={})))
        self.filter_cache = {}

    def clear_filter_cache(self):
        self.filter_cache.clear()

    def has_gid(self, gid):
        return (gid in self.syn_id_attr_dict)

    def gids(self):
        return viewkeys(self.syn_id_attr_dict)

    def items(self):
        return viewitems(self.syn_id_attr_dict)

    def __getitem__(self, gid):
        return self.syn_id_attr_dict[gid]



class PlasticityTransform:
    
    def __init__(self, X, w0, U=None, uw=None, logger=None):
            
        self.logger = logger
        self.y = None

        self.xin = X.copy()
        self.uin = None
        if U is not None:
            self.uin = U.copy()
        
        xlb = np.min(X, axis=1)
        xub = np.max(X, axis=1)
        xrng = np.where(np.isclose(xub - xlb, 0., rtol=1e-6, atol=1e-6), 1., xub - xlb) 

        self.xlb = xlb
        self.xub = xub
        self.xrng = xrng
        
        xdim = X.shape[0]
        N = X.shape[1]
        self.xn = np.zeros_like(X)
        for i in range(xdim):
            self.xn[i,:] = (X[i,:] - self.xlb[i]) / self.xrng[i]

        ulb = np.min(U, axis=1)
        uub = np.max(U, axis=1)
        urng = np.where(np.isclose(uub - ulb, 0., rtol=1e-6, atol=1e-6), 1., uub - ulb) 

        self.un = None
        self.ulb = ulb
        self.uub = uub
        self.urng = urng

        if U is not None:
            udim = U.shape[0]
            N = U.shape[1]
            self.un = np.zeros_like(U)
            for i in range(udim):
                self.un[i,:] = (U[i,:] - self.ulb[i]) / self.urng[i]

        self.uw = uw
        self.w0 = w0
        initial = np.dot(self.xin, w0)
        if self.un is not None:
            initial += np.dot(self.uin, uw)

        self.wnorm = np.mean(initial)

        self.scaled_y = None
        self.w = None
        
    def fit(self, y, max_amplitude=3, max_opt_iter=1000, optimize_tol=1e-8, verbose=False):
        
        scaled_initial = np.dot(self.xin, self.w0 / self.wnorm)
        if self.uw is not None:
            scaled_initial += np.dot(self.uin, self.uw / self.wnorm)
        scaled_initial -= 1.
        y_scaling_factor = max_amplitude / np.max(y)
        self.scaled_y = (y.flatten() * y_scaling_factor) + scaled_initial
        self.y_shape = y.shape
        self.lsqr_target = self.scaled_y
        if self.un is not None:
            self.lsqr_target -= np.dot(self.un, self.uw / self.wnorm)
        res = nnls_gdal(self.xn, self.lsqr_target.reshape((-1,1)),
                        max_n_iter=max_opt_iter, epsilon=optimize_tol, verbose=verbose)
        lsqr_weights = np.asarray(res, dtype=np.float32).reshape((res.shape[0],))
        self.w = lsqr_weights
        return self.w

    
def insert_hoc_cell_syns(env, gid, cell, syn_ids, syn_params, unique=False, insert_netcons=False,
                         insert_vecstims=False):
    """
    TODO: Only config the point process object if it has not already been configured.

    Insert mechanisms into given cell according to the synapse objects created in env.synapse_attributes.
    Configures mechanisms according to parameter values specified in syn_params.
    
    :param env: :class:'Env'
    :param gid: cell id (int)
    :param cell: hoc cell object
    :param syn_ids: synapse ids (array of int)
    :param syn_params: dictionary of the form { mech_name: params }
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

    syns_dict_by_type = {swc_type_apical: syns_dict_dend,
                         swc_type_basal: syns_dict_dend,
                         swc_type_axon: syns_dict_axon,
                         swc_type_ais: syns_dict_ais,
                         swc_type_hill: syns_dict_hill,
                         swc_type_soma: syns_dict_soma}
    py_sections = None
    if hasattr(cell, 'sections'):
        py_sections = [sec for sec in cell.sections]
    is_reduced = False
    if hasattr(cell, 'is_reduced'):
        is_reduced = cell.is_reduced

    cell_soma = None
    cell_dendrite = None
    if is_reduced:
        if hasattr(cell, 'soma'):
            cell_soma = cell.soma
            if isinstance(cell_soma, list):
                cell_soma = cell_soma[0]
            if isinstance(cell_soma, list):
                cell_soma = cell_soma[0]
        if hasattr(cell, 'dend'):
            cell_dendrite = cell.dend

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

        if is_reduced:
            if (swc_type == swc_type_soma) and (cell_soma is not None):
                sec = cell_soma
            elif (swc_type == swc_type_axon) and (cell_soma is not None):
                sec = cell_soma
            elif (swc_type == swc_type_ais) and (cell_soma is not None):
                sec = cell_soma
            elif (swc_type == swc_type_hill) and (cell_soma is not None):
                sec = cell_soma
            elif (swc_type == swc_type_apical) and (cell_dendrite is not None):
                sec = cell_dendrite
            elif (swc_type == swc_type_basal) and (cell_dendrite is not None):
                sec = cell_dendrite
            else:
                sec = py_sections[0]
        else:
            sec = py_sections[syn_section]


        if swc_type in syns_dict_by_type:
            syns_dict = syns_dict_by_type[swc_type]
        else:
            raise RuntimeError(f"insert_hoc_cell_syns: unsupported synapse SWC type {swc_type} for synapse {syn_id}")

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
                this_vecstim, this_vecstim_nc = None, None
                this_nc = None
                if insert_vecstims:
                    this_nc, this_vecstim = mknetcon_vecstim(syn_pps, delay=syn.source.delay)
                    syn_attrs.add_vecstim(gid, syn_id, syn_name, this_vecstim, this_nc)
                if insert_netcons:
                    if this_nc is None:
                        this_nc = mknetcon(env.pc, syn.source.gid, syn_pps, delay=syn.source.delay)
                    syn_attrs.add_netcon(gid, syn_id, syn_name, this_nc)
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
    4) initialize netcons with syn_mech_params environment configuration

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
        raise KeyError(f'insert_biophys_cell_syns: biophysical cell with gid {gid} does not exist')

    cell = env.biophys_cells[postsyn_name][gid]

    connection_syn_params = env.connection_config[postsyn_name][presyn_name].mechanisms
    
    synapse_config = env.celltypes[postsyn_name]['synapses']

    if unique is None:
        if 'unique' in synapse_config:
            unique = synapse_config['unique']
        else:
            unique = False

    syn_count, mech_count, nc_count = insert_hoc_cell_syns(env, gid, cell.hoc_cell, syn_ids, connection_syn_params,
                                                            unique=unique, insert_netcons=insert_netcons,
                                                            insert_vecstims=insert_vecstims)

    if verbose:
        logger.info(f'insert_biophys_cell_syns: source: {presyn_name} target: {postsyn_name} cell {gid}: created {mech_count} mechanisms and {nc_count} '
                    f'netcons for {syn_count} syn_ids')


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
        syn_ids = list(syn_id_attr_dict.keys())

    if insert:
        source_syn_ids_dict = syn_attrs.partition_syn_ids_by_source(gid, syn_ids)
        if not (gid in env.biophys_cells[postsyn_name]):
            raise KeyError(f'config_biophys_cell_syns: insert: biophysical cell with gid {gid} does not exist')

        for presyn_name, source_syn_ids in viewitems(source_syn_ids_dict):
            if (presyn_name is not None) and (source_syn_ids is not None):
                insert_biophys_cell_syns(env, gid, postsyn_name, presyn_name, source_syn_ids, unique=unique,
                                         insert_netcons=insert_netcons, insert_vecstims=insert_vecstims,
                                         verbose=verbose)

    cell = env.biophys_cells[postsyn_name][gid]
    syn_count, mech_count, nc_count = config_hoc_cell_syns(env, gid, postsyn_name, cell=cell.hoc_cell, syn_ids=syn_ids,
                                                           insert=False, verbose=False, throw_error=throw_error)

    if verbose:
        logger.info(f'config_biophys_cell_syns: target: {postsyn_name}; cell {gid}: set parameters for {mech_count} syns and {nc_count} '
                    f'netcons for {syn_count} syn_ids')

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
    weights_dict = synapse_config.get('weights', {})
    param_closure_dict = {}
    if 'closure' in weights_dict:
        param_closure_dict['weight'] = weights_dict['closure']
    
    if unique is None:
        if 'unique' in synapse_config:
            unique = synapse_config['unique']
        else:
            unique = False

    if syn_ids is None:
        syn_ids = list(syn_id_attr_dict.keys())

    if insert:
        source_syn_dict = syn_attrs.partition_synapses_by_source(gid, syn_ids)
        last_time = time.time()
        if (cell is None) and (not (env.pc.gid_exists(gid))):
            raise RuntimeError(f'config_hoc_cell_syns: insert: cell with gid {gid} does not exist on rank {rank}')
        if cell is None:
            cell = env.pc.gid2cell(gid)
        for presyn_name, source_syns in viewitems(source_syn_dict):
            if (presyn_name is not None) and (source_syns is not None):
                source_syn_ids = [x[0] for x in source_syns]
                connection_syn_params = env.connection_config[postsyn_name][presyn_name].mechanisms
                syn_count, mech_count, nc_count = insert_hoc_cell_syns(env, gid, cell, source_syn_ids, connection_syn_params, 
                                                                       unique=unique, insert_netcons=insert_netcons,
                                                                       insert_vecstims=insert_vecstims)
                if verbose:
                    logger.info(f'config_hoc_cell_syns: population: {postsyn_name}; cell {gid}: inserted {mech_count} mechanisms for source {presyn_name}')
        if verbose:
            logger.info(f'config_hoc_cell_syns: population: {postsyn_name}; cell {gid}: inserted mechanisms in {time.time() - last_time:.2f} s')

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

        syn_names = set(syn_attrs.syn_mech_names.keys())
            
        for syn_id, syn in source_syns:
            total_syn_id_count += 1
            for syn_name in syn_names:
                syn_index = syn_attrs.syn_name_index_dict[syn_name]
                if syn_index in syn.attr_dict:
                    this_pps = syn_attrs.get_pps(gid, syn_id, syn_name, throw_error=False)
                    if this_pps is None and throw_error:
                        raise RuntimeError(f'config_hoc_cell_syns: insert: cell gid {gid} synapse {syn_id} does not have a point '
                                           f'process for mechanism {syn_name}')

                    this_netcon = syn_attrs.get_netcon(gid, syn_id, syn_name, throw_error=False)
                    if this_netcon is None and throw_error:
                        raise RuntimeError(f'config_hoc_cell_syns: insert: cell gid {gid} synapse {syn_id} does not have a '
                                           f'netcon for mechanism {syn_name}')

                    params = syn.attr_dict[syn_index]
                    upd_params = {}
                    for param_name, param_val in viewitems(params):
                        if param_val is None:
                            raise RuntimeError(f'config_hoc_cell_syns: insert: cell gid {gid} synapse {syn_id} presyn source {presyn_name} does not have a '
                                               f'value set for parameter {param_name}')
                        if isinstance(param_val, Promise):
                            new_param_val = param_val.clos(*param_val.args)
                        elif param_name in param_closure_dict and isinstance(param_val, list):
                            new_param_val = param_closure_dict[param_name](*param_val)
                        else:
                            new_param_val = param_val
                        upd_params[param_name] = new_param_val

                    (mech_set, nc_set) = config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules,
                                                    mech_names=syn_attrs.syn_mech_names, syn=this_pps,
                                                    nc=this_netcon, **upd_params)
                    if mech_set:
                        mech_count += 1
                    if nc_set:
                        nc_count += 1

        total_nc_count += nc_count
        total_mech_count += mech_count

    if verbose:
        logger.info(f'config_hoc_cell_syns: target: {postsyn_name}; cell {gid}: '
                    f'set parameters for {total_mech_count} syns and {total_nc_count} netcons for {total_syn_id_count} syn_ids')

    return total_syn_id_count, total_mech_count, total_nc_count


def config_syn(syn_name, rules, mech_names=None, syn=None, nc=None, **params):
    """
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
                if isinstance(val, ExprClosure) and (nc is not None):
                    if val.parameters[0] == 'delay':
                        setattr(syn, param, val(nc.delay))
                        mech_param = True
                        failed = False
                    else:
                        failed = True
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
                    if isinstance(val, ExprClosure):
                        if val.parameters[0] == 'delay':
                            new = val(nc.delay)
                            nc.weight[i] = new
                            nc_param = True
                            failed = False
                        else:
                            failed = True
                    else:
                        if val is None:
                            raise AttributeError(f'config_syn: netcon attribute {param} is None for synaptic mechanism: {mech_name}')
                        if isinstance(val, list):
                            if len(val) > 1:
                                raise AttributeError('config_syn: netcon attribute {param} is list of length > 1 for synaptic mechanism: {mech_name}')
                            new = val[0]
                        else:
                            new = val
                        nc.weight[i] = new
                        nc_param = True
                        failed = False
        if failed:
            raise AttributeError(f'config_syn: problem setting attribute: {param} for synaptic mechanism: {mech_name}')
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
    raise AttributeError('get_syn_mech_param: problem setting attribute: {param_name} for synaptic mechanism: {mech_name}')


def get_syn_filter_dict(env, rules, convert=False, check_valid=True):
    """Used by modify_syn_param. Takes in a series of arguments and
    constructs a validated rules dictionary that specifies to which
    sets of synapses a rule applies. Values of filter queries are
    validated by the provided Env.

    :param env: :class:'Env'
    :param rules: dict
    :param convert: bool; whether to convert string values to enumerated type
    :return: dict

    """
    valid_filter_names = ['syn_types', 'layers', 'sources', 'swc_types']
    if check_valid:
        for name in rules:
            if name not in valid_filter_names:
                raise ValueError(f'get_syn_filter_dict: unrecognized filter category: {name}')
    rules_dict = copy.deepcopy(rules)
    syn_types = rules_dict.get('syn_types', None)
    swc_types = rules_dict.get('swc_types', None)
    layers = rules_dict.get('layers', None)
    sources = rules_dict.get('sources', None)
    if syn_types is not None:
        for i, syn_type in enumerate(syn_types):
            if syn_type not in env.Synapse_Types:
                raise ValueError(f'get_syn_filter_dict: syn_type: {syn_type} not recognized by network configuration')
            if convert:
                rules_dict['syn_types'][i] = env.Synapse_Types[syn_type]
    if swc_types is not None:
        for i, swc_type in enumerate(swc_types):
            if swc_type not in env.SWC_Types:
                raise ValueError(f'get_syn_filter_dict: swc_type: {swc_type} not recognized by network configuration')
            if convert:
                rules_dict['swc_types'][i] = env.SWC_Types[swc_type]
    if layers is not None:
        for i, layer in enumerate(layers):
            if layer not in env.layers:
                raise ValueError(f'get_syn_filter_dict: layer: {layer} not recognized by network configuration')
            if convert:
                rules_dict['layers'][i] = env.layers[layer]
    if sources is not None:
        source_idxs = []
        for i, source in enumerate(sources):
            if source not in env.Populations:
                raise ValueError(f'get_syn_filter_dict: presynaptic population: {source} not recognized by network '
                                 'configuration')
            source_idxs.append(env.Populations[source])
        if convert:
            rules_dict['sources'] = source_idxs
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
                     xhalf=None, min=None, max=None, min_loc=None, max_loc=None, outside=None, decay=None, custom=None,
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
        raise ValueError(f'modify_syn_mech_param: sec_type: {sec_type} not in cell')
    if param_name is None:
        raise ValueError(f'modify_syn_mech_param: missing required parameter to modify synaptic mechanism: {syn_name} '
                         f'in sec_type: {sec_type}')
    if not validate_syn_mech_param(env, syn_name, param_name):
        raise ValueError('modify_syn_mech_param: synaptic mechanism: '
                         f'{syn_name} or parameter: {param_name} not recognized by network configuration')
    if value is None:
        if origin is None:
            raise ValueError(f'modify_syn_mech_param: mechanism: {syn_name}; parameter: {param_name}; missing origin or value for '
                             'sec_type: {sec_type}')
        elif origin_filters is None:
            raise ValueError(f'modify_syn_mech_param: mechanism: {syn_name}; parameter: {param_name}; sec_type: {sec_type} cannot inherit from '
                             f'origin: {origin} without origin_filters')
    rules = get_mech_rules_dict(cell, value=value, origin=origin, slope=slope, tau=tau, xhalf=xhalf, min=min, max=max,
                                min_loc=min_loc, max_loc=max_loc, outside=outside, decay=decay, custom=custom)
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
        traceback.print_exc(file=sys.stdout)
        logger.error(f'modify_syn_mech_param: gid {cell.gid}: '
                     f'problem updating mechanism: {syn_name}; parameter: {param_name}; in sec_type: {sec_type}')
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
            mech_param_contents = [param_content]
        elif isinstance(param_content, list):
            mech_param_contents = param_content
        else:
            raise RuntimeError('update_syn_mech_by_sec_type: rule for synaptic mechanism: '
                               f'{syn_name} parameter: {param_name} was not specified properly')
        for param_content_entry in mech_param_contents:
            update_syn_mech_param_by_sec_type(cell, env, sec_type, syn_name, param_name, param_content_entry,
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
        synapse_filters = get_syn_filter_dict(env, new_rules['filters'], convert=True)
        del new_rules['filters']
    else:
        synapse_filters = None
    if 'origin_filters' in new_rules:
        origin_filters = get_syn_filter_dict(env, new_rules['origin_filters'], convert=True)
        del new_rules['origin_filters']
    else:
        origin_filters = None

    is_reduced = False
    if hasattr(cell, 'is_reduced'):
        is_reduced = cell.is_reduced
        
    if is_reduced:
        synapse_filters['swc_types'] = [env.SWC_Types[sec_type]]
        apply_syn_mech_rules(cell, env, syn_name, param_name, new_rules, 
                             synapse_filters=synapse_filters, origin_filters=origin_filters,
                             update_targets=update_targets, verbose=verbose)
    elif sec_type in cell.nodes:
        for node in cell.nodes[sec_type]:
            apply_syn_mech_rules(cell, env, syn_name, param_name, new_rules, node=node,
                                 synapse_filters=synapse_filters, origin_filters=origin_filters,
                                 update_targets=update_targets, verbose=verbose)


def apply_syn_mech_rules(cell, env, syn_name, param_name, rules, node=None, syn_ids=None, 
                         synapse_filters=None, origin_filters=None, donor=None, 
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
    if syn_ids is None:
        syn_attrs = env.synapse_attributes
        if synapse_filters is None:
            synapse_filters = {}
        if node is None:
            filtered_syns = syn_attrs.filter_synapses(cell.gid, cache=env.cache_queries, 
                                                      **synapse_filters)
        else:
            filtered_syns = syn_attrs.filter_synapses(cell.gid, syn_sections=[node.index], 
                                                      cache=env.cache_queries, **synapse_filters)
        if len(filtered_syns) == 0:
            return
        syn_distances = []
        for syn_id, syn in viewitems(filtered_syns):
            syn_distances.append(get_distance_to_node(cell, cell.tree.root, node, loc=syn.syn_loc))
        target_distance = min(syn_distances)
        syn_ids = list(filtered_syns.keys())

            
    if 'origin' in rules and donor is None:
        if node is None:
            donor = None
        else:
            donor = get_donor(cell, node, rules['origin'])
        if donor is None:
            raise RuntimeError('apply_syn_mech_rules: problem identifying donor of origin_type: '
                               f"{rules['origin']} for synaptic mechanism: {syn_name} parameter: "
                               f"{param_name} in sec_type: {node.type if node is not None else None}")

    if 'value' in rules:
        baseline = rules['value']
    elif donor is None:
        raise RuntimeError('apply_syn_mech_rules: cannot set value of synaptic mechanism: '
                           f'{syn_name} parameter: {param_name} in '
                           f'sec_type: {node.type if node is not None else None}')
    else:
        baseline = inherit_syn_mech_param(cell, env, donor, syn_name, param_name, origin_filters,
                                          target_distance=target_distance)
    if baseline is None and node is not None:
        baseline = inherit_syn_mech_param(cell, env, node, syn_name, param_name, origin_filters,
                                          target_distance=target_distance)
    if baseline is not None:
        if 'custom' in rules:
            apply_custom_syn_mech_rules(cell, env, node, syn_ids, syn_name, param_name, baseline, rules, donor,
                                        update_targets, verbose)
        else:
            set_syn_mech_param(cell, env, node, syn_ids, syn_name, param_name, baseline, rules, donor,
                               update_targets, verbose)


def inherit_syn_mech_param(cell, env, donor, syn_name, param_name, origin_filters=None, target_distance=None):
    """Follows path from the provided donor node to root until synapses
    are located that match the provided filter. Returns the requested
    parameter value from the synapse closest to the end of the
    section, or if provided, a target_distance from the root node.

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param donor: :class:'SHocNode'
    :param syn_name: str
    :param param_name: str
    :param origin_filters: dict: {category: list of int}
    :param target_distance: float
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
                valid_syns.append((syn_id, syn))
        if len(valid_syns) > 0:
            if target_distance is None:
                valid_syns.sort(key=lambda x: x[1].syn_loc)
                syn_id = valid_syns[-1][0]
            else:
                valid_syns.sort(
                    key=lambda x: abs(target_distance - get_distance_to_node(cell, cell.tree.root, donor,
                                                                             loc=x[1].syn_loc)))
                syn_id = valid_syns[0][0]
            mech_attrs = syn_attrs.get_mech_attrs(gid, syn_id, syn_name)
            if param_name not in mech_attrs:
                raise RuntimeError(f'inherit_syn_mech_param: synaptic mechanism: {syn_name} '
                                   f'at provided donor: {donor.name} does not contain the specified parameter: {param_name}')
            return mech_attrs[param_name]
    if donor is cell.tree.root:
        return
    else:
        return inherit_syn_mech_param(cell, env, donor.parent, syn_name, param_name, origin_filters,
                                      target_distance=target_distance)


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
    if not ('min_loc' in rules or 'max_loc' in rules or 'slope' in rules or 'decay' in rules):
        for syn_id in syn_ids:
            syn_attrs.modify_mech_attrs(cell.pop_name, cell.gid, syn_id, syn_name, 
                                        {param_name: baseline})
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
        decay = rules.get('decay', None)

        gid = cell.gid
        syn_attrs = env.synapse_attributes
        syn_id_attr_dict = syn_attrs.syn_id_attr_dict[gid]

        for syn_id in syn_ids:
            syn = syn_id_attr_dict[syn_id]
            syn_loc = syn.syn_loc
            distance = get_distance_to_node(cell, donor, node, syn_loc)
            value = get_param_val_by_distance(distance, baseline, slope, min_distance, max_distance,
                                              min_val, max_val, tau, xhalf, outside, decay)
                
            if value is not None:
                syn_attrs.modify_mech_attrs(cell.pop_name, cell.gid, syn_id, syn_name, 
                                            {param_name: value})

    if update_targets:
        config_biophys_cell_syns(env, cell.gid, cell.pop_name, syn_ids=syn_ids, insert=False, verbose=verbose)


def apply_custom_syn_mech_rules(cell, env, node, syn_ids, syn_name, param_name, baseline, rules, donor,
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
        raise RuntimeError('apply_custom_syn_mech_rules: no custom function provided for synaptic mechanism: %s '
                           'parameter: %s in sec_type: %s' % (syn_name, param_name, node.type))
    if rules['custom']['func'] in globals() and isinstance(globals()[rules['custom']['func']], collections.Callable):
        func = globals()[rules['custom']['func']]
    else:
        raise RuntimeError('apply_custom_syn_mech_rules: problem locating custom function: %s for synaptic '
                           'mechanism: %s parameter: %s in sec_type: %s' %
                           (rules['custom']['func'], syn_name, param_name, node.type))
    custom = copy.deepcopy(rules['custom'])
    del custom['func']
    new_rules = copy.deepcopy(rules)
    del new_rules['custom']
    new_rules['value'] = baseline
    new_rules = func(cell, node, baseline, new_rules, donor, env=env, syn_ids=syn_ids, syn_name=syn_name,
                     param_name=param_name, **custom)
    if new_rules:
        apply_syn_mech_rules(cell, env, syn_name, param_name, new_rules, 
                             node=node, syn_ids=syn_ids, donor=donor,
                             update_targets=update_targets, verbose=verbose)


def init_syn_mech_attrs(cell, env=None, reset_mech_dict=False, update_targets=False):
    """Consults a dictionary specifying parameters of NEURON synaptic mechanisms (point processes) for each type of
    section in a BiophysCell. Traverses through the tree of SHocNode nodes following order of inheritance. Calls
    update_syn_mech_by_sec_type to set placeholder values in the syn_mech_attrs_dict of a SynapseAttributes object. If
    update_targets flag is True, the attributes of any target synaptic point_process and netcon objects that have been
    inserted will also be updated. Otherwise, they can be updated separately by calling config_syns.

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param reset_mech_dict: bool
    :param update_targets: bool

    """
    if reset_mech_dict:
        cell.mech_dict = copy.deepcopy(cell.init_mech_dict)
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

    rank = int(env.pc.id())

    syn_attrs = env.synapse_attributes
    rules = syn_attrs.syn_param_rules

    if syn_names is None:
        syn_names = list(syn_attrs.syn_name_index_dict.keys())

    output_dict = {syn_name: defaultdict(lambda: defaultdict(list)) for syn_name in syn_names}
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
                    attr_keys = list(syn.attr_dict[syn_index].keys())
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
                            raise RuntimeError(f'write_syn_mech_attrs: gid {gid} syn id {syn_id} does not have attribute {k} '
                                               f'set in either {syn_name} point process or netcon')
                        output_dict[syn_name][gid][k].append(v)


    for syn_name in sorted(output_dict):

        syn_attrs_dict = output_dict[syn_name]
        attr_dict = {}
        

        for gid, gid_syn_attrs_dict in viewitems(syn_attrs_dict):
            for attr_name, attr_vals in viewitems(gid_syn_attrs_dict):
                if attr_name == 'syn_ids':
                    attr_dict[gid] = {'syn_ids': np.asarray(attr_vals, dtype='uint32')}
                else:
                    attr_dict[gid] = {attr_name: np.asarray(attr_vals, dtype='float32')}
                    
        logger.info(f"write_syn_mech_attrs: rank {rank}: population {pop_name}: "
                    f" writing mechanism {syn_name} attributes for {len(attr_dict)} gids")
        write_cell_attributes(output_path, pop_name, attr_dict,
                              namespace=f'{syn_name} Attributes',
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

    write_syn_mech_attrs(env, pop_name, gids, env.results_file_path, write_kwds={'comm': comm, 'io_size': env.io_size})



def write_syn_spike_count(env, pop_name, output_path, filters=None, syn_names=None, write_kwds={}):
    """
    Writes spike counts per presynaptic source for each cell in the given population to a NeuroH5 file.
    Assumes that attributes have been set via config_syn.
    
    :param env: instance of env.Env
    :param pop_name: population name
    :param output_path: path to NeuroH5 file
    :param filters: optional filter for synapses
    """

    rank = int(env.pc.id())

    syn_attrs = env.synapse_attributes
    rules = syn_attrs.syn_param_rules

    filters_dict = None
    if filters is not None:
        filters_dict = get_syn_filter_dict(env, filters, convert=True)
    
    if syn_names is None:
        syn_names = list(syn_attrs.syn_name_index_dict.keys())

    output_dict = {syn_name: defaultdict(lambda: defaultdict(int)) for syn_name in syn_names}

    gids = []
    if pop_name in env.biophys_cells:
       gids = list(env.biophys_cells[pop_name].keys())

    for gid in gids:
        if filters_dict is None:
            syns_dict = syn_attrs.syn_id_attr_dict[gid]
        else:
            syns_dict = syn_attrs.filter_synapses(gid, **filters_dict)
        logger.info(f"write_syn_mech_spike_counts: rank {rank}: population {pop_name}: gid {gid}: {len(syns_dict)} synapses")
        
        for syn_id, syn in viewitems(syns_dict):
            source_population = syn.source.population
            syn_netcon_dict = syn_attrs.pps_dict[gid][syn_id].netcon
            for syn_name in syn_names:
                mech_name = syn_attrs.syn_mech_names[syn_name]
                syn_index = syn_attrs.syn_name_index_dict[syn_name]
                if syn_index in syn_netcon_dict and 'count' in rules[mech_name]['netcon_state']:
                    count_index = rules[mech_name]['netcon_state']['count']
                    nc = syn_netcon_dict[syn_index]
                    spike_count = nc.weight[count_index]
                    output_dict[syn_name][gid][source_population] += spike_count

    for syn_name in sorted(output_dict):

        syn_attrs_dict = output_dict[syn_name]
        attr_dict = defaultdict(lambda: dict())

        for gid, gid_syn_spk_count_dict in viewitems(syn_attrs_dict):
            for source_index, source_count in viewitems(gid_syn_spk_count_dict):
                source_pop_name = syn_attrs.presyn_names[source_index]
                attr_dict[gid][source_pop_name] = np.asarray([source_count], dtype='uint32')
                    
        logger.info(f"write_syn_mech_spike_counts: rank {rank}: population {pop_name}: writing mechanism {syn_name} spike counts for {len(attr_dict)} gids")
        write_cell_attributes(output_path, pop_name, attr_dict,
                              namespace=f'{syn_name} Spike Counts',
                              **write_kwds)

    

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
            prev = None
            for i in range(sec.n3d()):
                pos = (sec.arc3d(i) / sec.L)
                if pos >= x:
                    if (prev is None) or (abs(pos - x) < abs(prev - x)):
                        return content[name][secnodes[i]]
                    else:
                        return content[name][secnodes[i - 1]]
                else:
                    prev = pos
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

    sec_graph = make_section_graph(neurotree_dict)

    syn_ids = syn_dict['syn_ids']
    syn_locs = syn_dict['syn_locs']
    syn_secs = syn_dict['syn_secs']

    sec_syn_dict = defaultdict(list)
    for syn_id, sec_id, syn_loc in zip(syn_ids, syn_secs, syn_locs):
        sec_syn_dict[sec_id].append(syn_id, syn_loc)

    syn_graph = nx.DiGraph()

    for sec_id, syn_id_locs in viewitems(sec_syn_dict):

        sec_parents = sec_graph.ancestors(sec_id)
        sec_children = sec_graph.descendants(sec_id)

        assert (len(sec_parents) <= 1)
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
        density_dict = {}
        syn_type = syn_type_dict[syn_type_label]
        rans = {}
        for (layer_label, density_params) in viewitems(layer_density_dict):
            if layer_label == 'default':
                layer = layer_label
            else:
                layer = int(layer_dict[layer_label])
            rans[layer] = ran
            density_dict[layer] = density_params
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
                density_params = None
                if layer > -1:
                    if layer in rans:
                        this_ran = rans[layer]
                        density_params = density_dict[layer]
                    elif 'default' in rans:
                        this_ran = rans['default']
                        density_params = density_dict['default']
                elif 'default' in rans:
                    this_ran = rans['default']
                    density_params = density_dict['default']
                    
                dens = 0.
                if this_ran is not None:
                    if density_params['mean'] > 1.0e-4:
                        while True:
                            dens = this_ran.normal(density_params['mean'], density_params['variance'])
                            if dens > 0.0:
                                break

                total_seg_density += dens
                segdensity[sec_index].append(dens)

        if total_seg_density < 1e-6:
            logger.warning(f"sections with zero {syn_type_label} synapse density: {segdensity}; "
                           f"seg_dict: {seg_dict}; "
                           f'rans: {rans}; density_dict: {density_dict}; morphology: {neurotree_dict}')

        segdensity_dict[syn_type] = segdensity

        layers_dict[syn_type] = layers
    return (segdensity_dict, layers_dict)


def synapse_seg_counts(syn_type_dict, layer_dict, layer_density_dicts, sec_index_dict, seg_dict, ran,
                       neurotree_dict=None):
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
                    l = (L / nseg)
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
    syn_cdists = []
    syn_secs = []
    syn_layers = []
    syn_types = []
    swc_types = []
    syn_index = 0

    r = np.random.RandomState()
    local_random.seed(int(seed))

    sec_interp_loc_dict = {}
    segcounts_per_sec = {}
    for (sec_name, layer_density_dict) in viewitems(sec_layer_density_dict):
        sec_index_dict = cell_secidx_dict[sec_name]
        swc_type = swc_type_dict[sec_name]
        seg_list = []
        L_total = 0
        (seclst, maxdist) = cell_sec_dict[sec_name]
        secidxlst = cell_secidx_dict[sec_name]
        for sec, idx in zip(seclst, secidxlst):
            npts_interp = max(int(round(sec.L)), 3)
            sec_interp_loc_dict[idx] = interplocs(sec, np.linspace(0, 1, npts_interp), return_interpolant=True)
        sec_dict = {int(idx): sec for sec, idx in zip(seclst, secidxlst)}
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
                interp_loc = sec_interp_loc_dict[sec_index]
                for seg, layer, seg_count in zip(seg_list, layers, segcounts):
                    seg_start = seg.x - (0.5 / seg.sec.nseg)
                    seg_end = seg.x + (0.5 / seg.sec.nseg)
                    seg_range = seg_end - seg_start
                    int_seg_count = math.floor(seg_count)
                    syn_count = 0
                    while syn_count < int_seg_count:
                        syn_loc = seg_start + seg_range * (syn_count + 1) / math.ceil(seg_count)
                        assert ((syn_loc <= 1) & (syn_loc >= 0))
                        if syn_loc < 1.0:
                            syn_cdist = math.sqrt(reduce(lambda a, b: a+b, ( interp_loc[i](syn_loc)**2 for i in range(3) )))
                            syn_cdists.append(syn_cdist)
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
                'syn_cdists': np.asarray(syn_cdists, dtype='float32'),
                'syn_locs': np.asarray(syn_locs, dtype='float32'),
                'syn_secs': np.asarray(syn_secs, dtype='uint32'),
                'syn_layers': np.asarray(syn_layers, dtype='int8'),
                'syn_types': np.asarray(syn_types, dtype='uint8'),
                'swc_types': np.asarray(swc_types, dtype='uint8')}

    return (syn_dict, segcounts_per_sec)


def distribute_poisson_synapses(density_seed, syn_type_dict, swc_type_dict, layer_dict, sec_layer_density_dict,
                                neurotree_dict, cell_sec_dict, cell_secidx_dict):
    """
    Computes synapse locations distributed according to a Poisson distribution.

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
    syn_cdists = []
    syn_secs = []
    syn_layers = []
    syn_types = []
    swc_types = []
    syn_index = 0

    sec_graph = make_section_graph(neurotree_dict)

    debug_flag = False
    secnodes_dict = neurotree_dict['section_topology']['nodes']
    for sec, secnodes in viewitems(secnodes_dict):
        if len(secnodes) < 2:
            debug_flag = True

    if debug_flag:
        logger.debug(f'sec_graph: {list(sec_graph.edges)}')
        logger.debug(f'neurotree_dict: {neurotree_dict}')

    sec_interp_loc_dict = {}
    seg_density_per_sec = {}
    r = np.random.RandomState()
    r.seed(int(density_seed))
    for (sec_name, layer_density_dict) in viewitems(sec_layer_density_dict):

        swc_type = swc_type_dict[sec_name]
        seg_dict = {}
        L_total = 0

        (seclst, maxdist) = cell_sec_dict[sec_name]
        secidxlst = cell_secidx_dict[sec_name]
        for sec, idx in zip(seclst, secidxlst):
            npts_interp = max(int(round(sec.L)), 3)
            sec_interp_loc_dict[idx] = interplocs(sec, np.linspace(0, 1, npts_interp), return_interpolant=True)
        sec_dict = {int(idx): sec for sec, idx in zip(seclst, secidxlst)}
        if len(sec_dict) > 1:
            sec_subgraph = sec_graph.subgraph(list(sec_dict.keys()))
            if len(sec_subgraph.edges()) > 0:
                sec_roots = [n for n, d in sec_subgraph.in_degree() if d == 0]
                sec_edges = []
                for sec_root in sec_roots:
                    sec_edges.append(list(nx.dfs_edges(sec_subgraph, sec_root)))
                    sec_edges.append([(None, sec_root)])
                sec_edges = [val for sublist in sec_edges for val in sublist]
            else:
                sec_edges = [(None, idx) for idx in list(sec_dict.keys())]
        else:
            sec_edges = [(None, idx) for idx in list(sec_dict.keys())]
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

            for sec_parent, sec_index in sec_edges:
                interp_loc = sec_interp_loc_dict[sec_index]
                seg_list = seg_dict[sec_index]
                sec_seg_layers = layers[sec_index]
                sec_seg_density = seg_density[sec_index]
                start_seg = seg_list[0]
                interval = 0.
                syn_loc = 0.
                for seg, layer, density in zip(seg_list, sec_seg_layers, sec_seg_density):
                    seg_start = seg.x - (0.5 / seg.sec.nseg)
                    seg_end = seg.x + (0.5 / seg.sec.nseg)
                    L = seg.sec.L
                    L_seg_start = seg_start * L
                    L_seg_end = seg_end * L
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
                                syn_loc = (interval / L)
                                assert ((syn_loc <= 1) and (syn_loc >= seg_start))
                                if syn_loc < 1.0:
                                    syn_cdist = math.sqrt(reduce(lambda a, b: a+b, ( interp_loc[i](syn_loc)**2 for i in range(3) )))
                                    syn_cdists.append(syn_cdist)
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

    assert (len(syn_ids) > 0)
    syn_dict = {'syn_ids': np.asarray(syn_ids, dtype='uint32'),
                'syn_locs': np.asarray(syn_locs, dtype='float32'),
                'syn_cdists': np.asarray(syn_cdists, dtype='float32'),
                'syn_secs': np.asarray(syn_secs, dtype='uint32'),
                'syn_layers': np.asarray(syn_layers, dtype='int8'),
                'syn_types': np.asarray(syn_types, dtype='uint8'),
                'swc_types': np.asarray(swc_types, dtype='uint8')}

    return (syn_dict, seg_density_per_sec)




def distribute_clustered_poisson_synapses(density_seed, syn_type_dict, swc_type_dict, layer_dict, sec_layer_density_dict,
                                          neurotree_dict, cell_sec_dict, cell_secidx_dict, syn_cluster_dict, cluster_syn_count_max=50):
    """
    Computes synapse locations distributed according to a given
    per-section clustering and Poisson distribution within the
    section.

    :param cluster_dict:
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
    syn_cdists = []
    syn_secs = []
    syn_layers = []
    syn_types = []
    swc_types = []
    syn_index = 0

    sec_graph = make_section_graph(neurotree_dict)

    debug_flag = False
    secnodes_dict = neurotree_dict['section_topology']['nodes']

    for sec, secnodes in viewitems(secnodes_dict):
        if len(secnodes) < 2:
            debug_flag = True

    if debug_flag:
        logger.debug(f'sec_graph: {list(sec_graph.edges)}')
        logger.debug(f'neurotree_dict: {neurotree_dict}')
    sec_interp_loc_dict = {}
    seg_density_per_sec = {}
    r = np.random.RandomState()
    r.seed(int(density_seed))

    cluster_syn_ids_count = 0
    for _, syn_clusters in syn_cluster_dict.items():
        for _, syn_cluster in syn_clusters.items():
            cluster_syn_ids_count += len(syn_cluster)

    syn_cluster_dict = copy.deepcopy(dict(syn_cluster_dict))
    sec_syn_count = defaultdict(int)
    
    while cluster_syn_ids_count > 0:

        for (sec_name, layer_density_dict) in viewitems(sec_layer_density_dict):

            swc_type = swc_type_dict[sec_name]
            seg_dict = {}
            seg_syn_count_dict = {}
            L_total = 0

            (seclst, maxdist) = cell_sec_dict[sec_name]
            secidxlst = cell_secidx_dict[sec_name]
            for sec, idx in zip(seclst, secidxlst):
                npts_interp = max(int(round(sec.L)), 3)
                sec_interp_loc_dict[idx] = interplocs(sec, np.linspace(0, 1, npts_interp), return_interpolant=True)
            sec_dict = {int(idx): sec for sec, idx in zip(seclst, secidxlst)}
            if len(sec_dict) > 1:
                sec_subgraph = sec_graph.subgraph(list(sec_dict.keys()))
                if len(sec_subgraph.edges()) > 0:
                    sec_roots = [n for n, d in sec_subgraph.in_degree() if d == 0]
                    sec_bfs_layers = list(nx.bfs_layers(sec_subgraph, sec_roots))
                    sec_order = [val for sublist in sec_bfs_layers for val in sublist]
                else:
                    sec_order = [idx for idx in list(sec_dict.keys())]
            else:
                sec_order = [idx for idx in list(sec_dict.keys())]
            sec_order = sorted(sec_order, key=lambda x: sec_syn_count[x])
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
                seg_syn_count_dict[sec_index] = np.zeros((len(seg_list),))
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
                for sec_index in sec_order:
                    interp_loc = sec_interp_loc_dict[sec_index]
                    seg_list = seg_dict[sec_index]
                    seg_syn_count = seg_syn_count_dict[sec_index]
                    sec_seg_layers = layers[sec_index]
                    sec_seg_density = seg_density[sec_index]
                    sec_seg_layer_set = set(sec_seg_layers)

                    syn_cluster_match_found = False
                    for layer in sec_seg_layer_set:
                        if (syn_type, swc_type, layer) in syn_cluster_dict:
                            syn_cluster_match_found = True
                    if not syn_cluster_match_found:
                        continue

                    current_syn_cluster_type = None
                    current_syn_cluster_id = None
                    current_cluster_syn_ids = []
                    current_cluster_syn_count = 0

                    interval = 0.
                    syn_loc = 0.
                    seg_order = np.argsort(seg_syn_count, kind='stable')
                    for seg_index in seg_order:

                        seg = seg_list[seg_index]
                        layer = sec_seg_layers[seg_index]
                        density = sec_seg_density[seg_index]
                        
                        if not density > 0.:
                            continue
                        
                        if current_syn_cluster_type != (syn_type, swc_type, layer):
                            current_syn_cluster_type = (syn_type, swc_type, layer)
                            if current_syn_cluster_type in syn_cluster_dict:
                                syn_clusters = syn_cluster_dict[current_syn_cluster_type]
                            else:
                                continue
                            if len(syn_clusters) == 0:
                                continue
                            current_syn_cluster_id = r.choice(list(syn_clusters.keys()), size=1)[0]
                            current_cluster_syn_ids = syn_clusters[current_syn_cluster_id]
                            current_cluster_syn_count = 0

                        seg_start = seg.x - (0.5 / seg.sec.nseg)
                        seg_end = seg.x + (0.5 / seg.sec.nseg)
                        L = seg.sec.L
                        L_seg_start = seg_start * L
                        L_seg_end = seg_end * L

                        beta = 1. / density
                        while True:
                            sample = r.exponential(beta)
                            if sample < (L_seg_end - L_seg_start):
                                break
                        interval = L_seg_end - sample
                        while interval > L_seg_start:
                            syn_loc = (interval / L)
                            assert ((syn_loc <= 1) and (syn_loc >= seg_start))
                            if syn_loc < 1.0:
                                if len(current_cluster_syn_ids) == 0 or (current_cluster_syn_count > cluster_syn_count_max):
                                    while len(current_cluster_syn_ids) == 0:
                                        if current_syn_cluster_type not in syn_cluster_dict:
                                            break
                                        syn_clusters = syn_cluster_dict[current_syn_cluster_type]
                                        if current_syn_cluster_id is not None:
                                            if (current_syn_cluster_id in syn_clusters) and (len(syn_clusters[current_syn_cluster_id]) == 0):
                                                del(syn_clusters[current_syn_cluster_id])
                                        if len(syn_clusters) == 0:
                                            break
                                        current_syn_cluster_id = r.choice(list(syn_clusters.keys()), size=1)[0]
                                        current_cluster_syn_ids = syn_clusters[current_syn_cluster_id]
                                        current_cluster_syn_count = 0
                                if len(current_cluster_syn_ids) == 0:
                                    break
                                syn_index = current_cluster_syn_ids.pop(0)
                                cluster_syn_ids_count -= 1
                                current_cluster_syn_count += 1
                                syn_cdist = math.sqrt(reduce(lambda a, b: a+b, ( interp_loc[i](syn_loc)**2 for i in range(3) )))
                                syn_cdists.append(syn_cdist)
                                syn_locs.append(syn_loc)
                                syn_ids.append(syn_index)
                                syn_secs.append(sec_index)
                                syn_layers.append(layer)
                                syn_types.append(syn_type)
                                swc_types.append(swc_type)
                                sec_syn_count[sec_index] += 1
                                seg_syn_count[seg_index] += 1
                            interval -= r.exponential(beta)

                    end_distance[sec_index] = (1.0 - syn_loc) * L

    assert (len(syn_ids) > 0)
    syn_dict = {'syn_ids': np.asarray(syn_ids, dtype='uint32'),
                'syn_locs': np.asarray(syn_locs, dtype='float32'),
                'syn_cdists': np.asarray(syn_cdists, dtype='float32'),
                'syn_secs': np.asarray(syn_secs, dtype='uint32'),
                'syn_layers': np.asarray(syn_layers, dtype='int8'),
                'syn_types': np.asarray(syn_types, dtype='uint8'),
                'swc_types': np.asarray(swc_types, dtype='uint8')}

    return (syn_dict, seg_density_per_sec)



def generate_log_normal_weights(weights_name, mu, sigma, seed, source_syn_dict, clip=None):
    """
    Generates log-normal synaptic weights by random sampling from a
    log-normal distribution with the given mu and sigma.

    :param weights_name: label to use for the weights namespace (must correspond to a synapse name)
    :param mu: mean of log-normal distribution
    :param sigma: standard deviation of log-normal distribution
    :param seed: seed for random number generator
    :param source_syn_dict: dictionary of the form { source_gid: <numpy uint32 array of synapse ids> }
    :param clip: if provided, specify min and max range for weight values
    :return: dictionary of the form:
    { 'syn_id': <numpy uint32 array of synapse ids>,
      weight_name: <numpy float array of weights>
    }

    """

    local_random = np.random.RandomState()
    local_random.seed(int(seed))
    source_weights = rejection_sampling(lambda n: local_random.lognormal(mu, sigma, n),
                                        len(source_syn_dict), clip)
    assert(len(source_weights) == len(source_syn_dict))
    syn_weight_dict = {}
    # weights are synchronized across all inputs from the same source_gid
    for this_source_gid, this_weight in zip(source_syn_dict, source_weights):
        for this_syn_id in source_syn_dict[this_source_gid]:
            syn_weight_dict[this_syn_id] = this_weight
    weights = np.array(list(syn_weight_dict.values())).astype('float32', copy=False)
    weights_dict = \
        {'syn_id': np.array(list(syn_weight_dict.keys())).astype('uint32', copy=False),
         weights_name: weights}
    return weights_dict


def generate_normal_weights(weights_name, mu, sigma, seed, source_syn_dict, clip=None):
    """
    Generates normal synaptic weights by random sampling from a
    normal distribution with the given mu and sigma.

    :param weights_name: label to use for the weights namespace (must correspond to a synapse name)
    :param mu: mean of log-normal distribution
    :param sigma: standard deviation of log-normal distribution
    :param clip: if provided, specify min and max range for weight values
    :param seed: seed for random number generator
    :param source_syn_dict: dictionary of the form { source_gid: <numpy uint32 array of synapse ids> }
    :return: dictionary of the form:
    { 'syn_id': <numpy uint32 array of synapse ids>,
      weight_name: <numpy float array of weights>
    }

    """

    local_random = np.random.RandomState()
    local_random.seed(int(seed))
    source_weights = rejection_sampling(lambda n: local_random.normal(mu, sigma, n),
                                        len(source_syn_dict), clip)
    assert(len(source_weights) == len(source_syn_dict))
    syn_weight_dict = {}
    # weights are synchronized across all inputs from the same source_gid
    for this_source_gid, this_weight in zip(source_syn_dict, source_weights):
        for this_syn_id in source_syn_dict[this_source_gid]:
            syn_weight_dict[this_syn_id] = this_weight
    weights = np.array(list(syn_weight_dict.values())).astype('float32', copy=False)
    if clip is not None:
        clip_min, clip_max = clip
        np.clip(weights, clip_min, clip_max, out=weights)
    normed_weights = weights
    weights_dict = \
        {'syn_id': np.array(list(syn_weight_dict.keys())).astype('uint32', copy=False),
         weights_name: normed_weights}
    return weights_dict


def generate_sparse_weights(weights_name, fraction, seed, source_syn_dict):
    """
    Generates sparse synaptic weights by random sampling where the given fraction of weights
    is 1 (and uniformly distributed) and the rest of the weights are 0.

    :param weights_name: label to use for the weights namespace (must correspond to a synapse name)
    :param fraction: fraction of weights to be 1.
    :param seed: seed for random number generator
    :param source_syn_dict: dictionary of the form { source_gid: <numpy uint32 array of synapse ids> }
    :return: dictionary of the form:
    { 'syn_id': <numpy uint32 array of synapse ids>,
      weight_name: <numpy float array of weights>
    }

    """
    local_random = np.random.RandomState()
    local_random.seed(int(seed))
    source_weights = [1.0 if x <= fraction else 0.0 for x in local_random.uniform(size=len(source_syn_dict))]
    syn_weight_dict = {}
    # weights are synchronized across all inputs from the same source_gid
    for this_source_gid, this_weight in zip(source_syn_dict, source_weights):
        for this_syn_id in source_syn_dict[this_source_gid]:
            syn_weight_dict[this_syn_id] = this_weight
    weights = np.array(list(syn_weight_dict.values())).astype('float32', copy=False)
    normed_weights = weights
    weights_dict = \
        {'syn_id': np.array(list(syn_weight_dict.keys())).astype('uint32', copy=False),
         weights_name: normed_weights}
    return weights_dict


def get_structured_input_arrays(structured_weights_dict, gid):

    target_map = structured_weights_dict['target_map']
    target_map_norm = target_map/target_map.max()
    target_act = np.argwhere(target_map_norm > 0.)

    initial_weight_dict = structured_weights_dict['initial_weight_dict']
    input_rate_map_dict = structured_weights_dict['input_rate_map_dict']
    non_structured_input_rate_map_dict = structured_weights_dict['non_structured_input_rate_map_dict']
    non_structured_weights_dict = structured_weights_dict['non_structured_weights_dict']
    syn_count_dict = structured_weights_dict['syn_count_dict']
    
    input_matrix = np.full((target_map.size, len(input_rate_map_dict)), np.nan,
                            dtype=np.float64)
    source_gid_array = np.full(len(input_rate_map_dict), -1, dtype=np.uint32)
    syn_count_array = np.full(len(input_rate_map_dict), 0, dtype=np.uint32)
    initial_weight_array = np.full(len(input_rate_map_dict), np.nan, dtype=np.float64)
    input_rank = np.full(len(input_rate_map_dict), np.nan, dtype=np.float32)
    for i, source_gid in enumerate(input_rate_map_dict):
        source_gid_array[i] = source_gid
        this_syn_count = syn_count_dict[source_gid]
        this_input = input_rate_map_dict[source_gid].ravel()
        input_matrix[:, i] = this_input * this_syn_count
        syn_count_array[i] = this_syn_count
        initial_weight_array[i] = initial_weight_dict[source_gid]
        this_input_max = np.max(this_input)
        this_input_norm = this_input_max if this_input_max > 0. else 1.
        this_input_normed = this_input/this_input_norm
        if np.sum(this_input_normed[target_act]) > 1e-6:
            input_rank[i] = np.clip(spatial.distance.correlation(this_input_normed[target_act].reshape((-1,)),
                                                                 target_map_norm[target_act].reshape((-1,))),
                                    0., None)
        else:
            input_rank[i] = 0.
    input_rank[np.isnan(input_rank)] = 0.

    input_rank_order = np.lexsort((syn_count_array, input_rank))
    
    non_structured_input_matrix = None
    if non_structured_input_rate_map_dict is not None:
        non_structured_input_matrix = np.full((target_map.size, len(non_structured_input_rate_map_dict)),
                                              np.nan, dtype=np.float32)
        non_structured_weight_array = np.full(len(non_structured_input_rate_map_dict), np.nan, dtype=np.float32)
        non_structured_source_gid_array = np.full(len(non_structured_input_rate_map_dict), -1, dtype=np.uint32)
        for i, source_gid in enumerate(non_structured_input_rate_map_dict):
            non_structured_source_gid_array[i] = source_gid
            this_syn_count = syn_count_dict[source_gid]
            this_input = non_structured_input_rate_map_dict[source_gid].ravel() * this_syn_count
            non_structured_input_matrix[:, i] = this_input
            non_structured_weight_array[i] = non_structured_weights_dict.get(source_gid, 1.0)
            
            
    return {'target_map': target_map,
            'input_matrix': input_matrix, 
            'initial_weight_array': initial_weight_array, 
            'non_structured_input_matrix': non_structured_input_matrix, 
            'non_structured_weight_array': non_structured_weight_array, 
            'non_structured_source_gid_array': non_structured_source_gid_array,
            'syn_count_array': syn_count_array, 
            'source_gid_array': source_gid_array,
            'input_rank': input_rank,
            'input_rank_order': input_rank_order}


def get_scaled_input_maps(target_amplitude, input_arrays_dict, gid):
    
    
    target_map = input_arrays_dict['target_map']
    initial_weight_array = input_arrays_dict['initial_weight_array']
    input_matrix = input_arrays_dict['input_matrix']
    non_structured_weight_array = input_arrays_dict['non_structured_weight_array']
    non_structured_input_matrix = np.asarray(input_arrays_dict['non_structured_input_matrix'], dtype=np.float64)
    input_rank_order = input_arrays_dict['input_rank_order']
    input_rank = input_arrays_dict['input_rank']
    
    initial_map = np.dot(input_matrix, initial_weight_array)
    if non_structured_input_matrix is not None:
        initial_map += np.dot(non_structured_input_matrix, non_structured_weight_array)
    if np.mean(initial_map)<= 0.:
       raise RuntimeError('generate_structured_delta_weights: initial weights must produce positive activation')
    

    input_matrix_norm = np.max(input_matrix)
    scaled_input_matrix = input_matrix / input_matrix_norm
    
    non_structured_input_norm = np.max(non_structured_input_matrix)
    scaled_non_structured_input_matrix = non_structured_input_matrix / non_structured_input_norm

    initial_weights_norm = np.mean(initial_map)
    normed_initial_weights = initial_weight_array / initial_weights_norm

    non_structured_weights_norm = np.mean(initial_map)
    normed_non_structured_weights = non_structured_weight_array / non_structured_weights_norm

    scaled_initial_map = np.dot(input_matrix, normed_initial_weights)
    if non_structured_input_matrix is not None:
        scaled_initial_map += np.dot(non_structured_input_matrix, normed_non_structured_weights)
    scaled_initial_map -= 1.
    
    target_map_scaling_factor = target_amplitude / (np.max(target_map) if np.max(target_map) > 0. else 1.)
    scaled_target_map = (target_map.flatten() * target_map_scaling_factor)

    return {'input_matrix' : input_matrix,
            'scaled_input_matrix' : scaled_input_matrix,
            'non_structured_input_matrix': non_structured_input_matrix,
            'scaled_non_structured_input_matrix': scaled_non_structured_input_matrix,
            'scaled_target_map': scaled_target_map,
            'scaled_initial_map': scaled_initial_map,
            'normed_initial_weights': normed_initial_weights,
            'normed_non_structured_weights': normed_non_structured_weights,
            'initial_weights_norm': initial_weights_norm,
            'non_structured_weights_norm': non_structured_weights_norm,
            'input_matrix_norm': input_matrix_norm,
            'input_rank_order': input_rank_order,
            'input_rank': input_rank
           }
    
def get_structured_delta_weights(initial_weight_array, normed_initial_weights, 
                                 non_structured_weight_array, normed_non_structured_weights,
                                 lsqr_weights, max_weight_decay_fraction):
    
    bounded_delta_weights = lsqr_weights - normed_initial_weights
                
    structured_delta_weights_lb = np.asarray([ -(max_weight_decay_fraction * x)
                                               for x in normed_initial_weights ])

    structured_delta_weights = np.clip(bounded_delta_weights, structured_delta_weights_lb, None)

    return structured_delta_weights


def generate_structured_weights(destination_gid, target_map, initial_weight_dict, input_rate_map_dict, syn_count_dict,
                                seed_offset=0,
                                target_amplitude=3.,
                                max_weight_decay_fraction = 1.,
                                arena_x=None, arena_y=None,
                                non_structured_input_rate_map_dict=None, 
                                non_structured_weights_dict=None, 
                                reference_weight_dict=None, reference_weights_are_delta=False,
                                reference_weights_namespace=None, 
                                optimize_tol=1e-6, max_opt_iter=1000,
                                verbose=False, plot=False, show_fig=False, save_fig=None,
                                fig_kwargs={}):
    """

    :param target_map: array
    :param initial_weight_dict: dict: {int: float}
    :param input_rate_map_dict: dict: {int: array}
    :param syn_count_dict: dict: {int: int}
    :param max_opt_iter: int
    :param target_amplitude: float
    :param arena_x: 2D array
    :param arena_y: 2D array
    :param reference_weight_dict: dict: {int: float}
    :param reference_weights_are_delta: bool
    :param reference_weights_namespace: str
    :param verbose: bool
    :param plot: bool
    :return: dict: {int: float}
    """

    if len(initial_weight_dict) != len(input_rate_map_dict):
        logger.warning(f"len(initial_weight_dict) = {len(initial_weight_dict)} len(input_rate_map_dict) = {len(input_rate_map_dict)}")
    assert(len(initial_weight_dict) == len(input_rate_map_dict))
    if non_structured_input_rate_map_dict is not None:
        assert(len(non_structured_weights_dict) == len(non_structured_input_rate_map_dict))

    assert((max_weight_decay_fraction >= 0.) and (max_weight_decay_fraction <= 1.))

    local_random = np.random.RandomState()
    local_random.seed(int(seed_offset + destination_gid))

    #np.seterr(all='raise')
    
    structured_weights_dict = { 'target_map': target_map,
                                'initial_weight_dict': initial_weight_dict,
                                'input_rate_map_dict': input_rate_map_dict,
                                'non_structured_input_rate_map_dict': non_structured_input_rate_map_dict,
                                'non_structured_weights_dict': non_structured_weights_dict,
                                'syn_count_dict': syn_count_dict,
                                }
    structured_input_arrays_dict = get_structured_input_arrays(structured_weights_dict, destination_gid)
    source_gid_array = structured_input_arrays_dict['source_gid_array']
    non_structured_source_gid_array = structured_input_arrays_dict['non_structured_source_gid_array']
    initial_weight_array = structured_input_arrays_dict['initial_weight_array']
    non_structured_weight_array = structured_input_arrays_dict.get('non_structured_weight_array', None)
    scaled_maps_dict = get_scaled_input_maps(target_amplitude, structured_input_arrays_dict, destination_gid)

    initial_weights_norm = scaled_maps_dict['initial_weights_norm']
    scaled_target_map = scaled_maps_dict['scaled_target_map']
    scaled_initial_map = scaled_maps_dict['scaled_initial_map']
    scaled_input_matrix = scaled_maps_dict['scaled_input_matrix']
    input_matrix = scaled_maps_dict['input_matrix']
    normed_initial_weights = scaled_maps_dict['normed_initial_weights']
    scaled_non_structured_input_matrix = scaled_maps_dict.get('scaled_non_structured_input_matrix', None)
    non_structured_input_matrix = scaled_maps_dict.get('non_structured_input_matrix', None)
    normed_non_structured_weights = scaled_maps_dict.get('normed_non_structured_weights', None)
    input_rank_order = scaled_maps_dict['input_rank_order']
    input_rank = scaled_maps_dict['input_rank']
    inverse_input_rank_order = np.empty_like(input_rank_order)
    inverse_input_rank_order[input_rank_order] = np.arange(input_rank_order.size)

    lsqr_target_map = np.clip(scaled_target_map + scaled_initial_map, 0.0, None)
    lsqr_target_map[lsqr_target_map > np.percentile(lsqr_target_map, 95)] = np.max(lsqr_target_map)
    if scaled_non_structured_input_matrix is not None:
        lsqr_target_map -= np.dot(scaled_non_structured_input_matrix, normed_non_structured_weights)

    csum = np.sum(initial_weight_array)
    n_variables = scaled_input_matrix.shape[1]
    D1 = np.diagflat(-1*np.ones(n_variables-1), 1)
    np.fill_diagonal(D1, 1)
    k1 = 2.0
    D2 = (np.diagflat(2*np.ones(n_variables-1), 1) + np.diagflat(-1*np.ones(n_variables-2), 2))
    np.fill_diagonal(D2, -1)
    k2 = 0.5
    W = (np.sort(local_random.lognormal(size=(1, n_variables), mean=0.0, sigma=0.5))[::-1])
    A = np.vstack((scaled_input_matrix[:,input_rank_order], k1*D1, k2*D2, W)).astype(np.float32)
    lsqr_target_map = np.concatenate((lsqr_target_map, np.zeros(n_variables), np.zeros(n_variables), csum*np.ones((1,)))).astype(np.float32)
    res = nnls_gdal(A, lsqr_target_map.reshape((-1,1)),
                    max_n_iter=max_opt_iter, epsilon=optimize_tol, verbose=verbose)
    lsqr_weights = np.asarray(res[inverse_input_rank_order], dtype=np.float32).reshape((res.shape[0],))
    logger.info(f'gid {destination_gid}: min/max/mean/sum LSQR weights: '
                f'{np.min(lsqr_weights)}/{np.max(lsqr_weights)}/{np.mean(lsqr_weights)}/{np.sum(lsqr_weights)} ')
    
    structured_delta_weights = \
        get_structured_delta_weights(initial_weight_array, normed_initial_weights, 
                                     non_structured_weight_array, normed_non_structured_weights,
                                     lsqr_weights, max_weight_decay_fraction)
    
    LTP_delta_weights_array = np.maximum(structured_delta_weights, 0.)
    LTD_delta_weights_array = np.minimum(structured_delta_weights, 0.)

    logger.info(f'gid {destination_gid}: '
                f'min/max/mean LTP delta weights: '
                f'{np.min(LTP_delta_weights_array)}/{np.max(LTP_delta_weights_array)}/{np.mean(LTP_delta_weights_array)} '
                f'min/max/mean LTD delta weights: '
                f'{np.min(LTD_delta_weights_array)}/{np.max(LTD_delta_weights_array)}/{np.mean(LTD_delta_weights_array)} ')
    
    structured_weights = LTP_delta_weights_array + LTD_delta_weights_array + normed_initial_weights
    assert(np.min(structured_weights) >= 0.)

    lb_LTP = np.min(LTP_delta_weights_array)
    ub_LTP = np.max(LTP_delta_weights_array)
    range_LTP = ub_LTP - lb_LTP
    output_LTP_delta_weights_array = (LTP_delta_weights_array - lb_LTP) / range_LTP
    output_LTD_delta_weights_array = LTD_delta_weights_array * initial_weights_norm * 0.999
    if not (np.min(output_LTD_delta_weights_array + initial_weight_array) >= 0.):
        logger.error(f'gid {destination_gid}: '
                     f'min(output_LTD_delta_weights_array + initial_weight_array) = '
                     f'{np.min(output_LTD_delta_weights_array + initial_weight_array)}')
    assert(np.min(output_LTD_delta_weights_array + initial_weight_array) >= 0.)

    logger.info(f'gid {destination_gid}: '
                f'min/max/mean output LTP delta weights: '
                f'{np.min(output_LTP_delta_weights_array)}/{np.max(output_LTP_delta_weights_array)}/{np.mean(output_LTP_delta_weights_array)} '
                f'min/max/mean LTD delta weights: '
                f'{np.min(output_LTD_delta_weights_array)}/{np.max(output_LTD_delta_weights_array)}/{np.mean(output_LTD_delta_weights_array)} ')

    LTP_delta_weights_dict = dict(zip(source_gid_array, output_LTP_delta_weights_array))
    LTD_delta_weights_dict = dict(zip(source_gid_array, output_LTD_delta_weights_array))
    input_rank_dict = dict(zip(source_gid_array, input_rank))
    
    structured_activation_map = np.dot(scaled_input_matrix, structured_weights)
    if scaled_non_structured_input_matrix is not None:
        structured_activation_map += np.dot(scaled_non_structured_input_matrix, normed_non_structured_weights)

    if plot:
        lsqr_map = np.dot(scaled_input_matrix, lsqr_weights)
        if scaled_non_structured_input_matrix is not None:
            lsqr_map += np.dot(scaled_non_structured_input_matrix, normed_non_structured_weights)

        plot_callback_structured_weights(arena_x = arena_x,
                                         arena_y = arena_y,
                                         initial_weight_array = initial_weight_array,
                                         normed_initial_weights = normed_initial_weights,
                                         non_structured_weight_array = non_structured_weight_array,
                                         normed_non_structured_weights = normed_non_structured_weights,
                                         initial_weights_norm = initial_weights_norm,
                                         lsqr_weights = lsqr_weights,
                                         LTP_delta_weights = LTP_delta_weights_array,
                                         LTD_delta_weights = LTD_delta_weights_array,
                                         input_matrix = input_matrix,
                                         non_structured_input_matrix = non_structured_input_matrix,
                                         scaled_input_matrix = scaled_input_matrix,
                                         scaled_non_structured_input_matrix = scaled_non_structured_input_matrix,
                                         scaled_target_map = scaled_target_map,
                                         scaled_initial_map = scaled_initial_map,
                                         lsqr_map = lsqr_map,
                                         structured_weights = structured_weights,
                                         structured_activation_map = structured_activation_map,
                                         show_fig=show_fig, save_fig=save_fig,
                                         **fig_kwargs)

    return {'LTP_delta_weights': LTP_delta_weights_dict,
            'LTD_delta_weights': LTD_delta_weights_dict,
            'structured_activation_map': structured_activation_map,
            'source_input_rank': input_rank_dict}


def plot_callback_structured_weights(**kwargs):
    import matplotlib as mpl
    import matplotlib.cm as cm
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    gid = kwargs['gid']
    font_size = kwargs.get('font_size', mpl.rcParams['font.size'])
    field_width = kwargs['field_width']
    show_fig = kwargs.get('show_fig', False)
    save_fig = kwargs.get('save_fig', None)
    arena_x = kwargs['arena_x']
    arena_y = kwargs['arena_y']
    
    initial_weight_array = kwargs['initial_weight_array']
    non_structured_weight_array = kwargs.get('non_structured_weight_array', None)
    normed_initial_weights = kwargs['normed_initial_weights']
    normed_non_structured_weights = kwargs['normed_non_structured_weights']
    input_matrix = kwargs['input_matrix']
    scaled_input_matrix = kwargs['scaled_input_matrix']
    non_structured_input_matrix = kwargs.get('non_structured_input_matrix', None)
    scaled_non_structured_input_matrix = kwargs.get('scaled_non_structured_input_matrix', None)
    scaled_initial_map = kwargs['scaled_initial_map']
    scaled_target_map = kwargs['scaled_target_map']
    lsqr_weights = kwargs['lsqr_weights']
    LTP_delta_weights = kwargs['LTP_delta_weights']
    LTD_delta_weights = kwargs['LTD_delta_weights']
    lsqr_map = kwargs['lsqr_map']
    structured_activation_map = kwargs['structured_activation_map']
    structured_weights = kwargs['structured_weights']
    unweighted_map = np.dot(input_matrix, np.ones((input_matrix.shape[1],)))
    
    initial_map = np.dot(input_matrix, initial_weight_array)
    if non_structured_input_matrix is not None:
        initial_map += np.dot(non_structured_input_matrix, non_structured_weight_array)

    
    min_vals = [np.min(scaled_target_map), np.min(lsqr_map)]
    max_vals = [np.max(scaled_target_map), np.max(lsqr_map)]

    vmin = min(min_vals)
    vmax = max(max_vals)

    fig = plt.figure(figsize=(15,8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    row = 0
    inner_grid = gs[row, 0].subgridspec(1, 2)

    ax = fig.add_subplot(inner_grid[0, 0])
    ax.plot(range(len(scaled_target_map)), scaled_initial_map, label='Scaled Initial', alpha=0.5, color='C0')
    ax.plot(range(len(scaled_target_map)), scaled_target_map, label='Target',
            alpha=0.5, color='C1')
    ax.set_ylabel('Normalized activity')
    ax.set_xlabel('Arena spatial bin')
    ax.legend(loc='best', #bbox_to_anchor=(0.6, 0.5), 
              frameon=False, framealpha=0.5, fontsize=8)

    ax = fig.add_subplot(inner_grid[0, 1])
    ax.plot(range(len(scaled_target_map)), scaled_target_map, label='Target',
            alpha=0.5, color='C1')
    ax.plot(range(len(scaled_target_map)), lsqr_map, label='NNLS',
            alpha=0.5, color='C2')
    ax.plot(range(len(scaled_target_map)), structured_activation_map, label='Structured',
            alpha=0.75, color='C3')
    ax.set_xlabel('Arena spatial bin')
    ax.legend(loc='best', #bbox_to_anchor=(0.6, 0.5), 
              frameon=False, framealpha=0.5, fontsize=8)
    
    ax = fig.add_subplot(gs[row, 1])
    p = ax.pcolormesh(arena_x, arena_y, structured_activation_map.reshape(arena_x.shape), shading='auto')
    ax.set_xlabel('Arena location (x)')
    ax.set_ylabel('Arena location (y)')
    ax.set_title('Structured activation map', fontsize=font_size)
    fig.colorbar(p, ax=ax)
    row += 1
    
    inner_grid = gs[row, 0].subgridspec(2, 2)
    
    ax = fig.add_subplot(inner_grid[0])
    ax.fill_between(np.arange(0, len(initial_weight_array)),
                    np.sort(initial_weight_array)[::-1], label='Initial')
    ax.set_ylabel('Weight')
    ax.set_title('Initial weights')
    
    if non_structured_weight_array is not None:
        ax = fig.add_subplot(inner_grid[1])
        ax.fill_between(np.arange(0, len(non_structured_weight_array)),
                        np.sort(non_structured_weight_array)[::-1], label='Non-structured')
        ax.set_title('Non-structured weights')

    ax = fig.add_subplot(inner_grid[2])
    ax.fill_between(np.arange(0, len(lsqr_weights)),
                    np.sort(lsqr_weights)[::-1], label='NNLS')
    ax.set_title('NNLS weights')
    
    ax = fig.add_subplot(inner_grid[3])
    ax.fill_between(np.arange(0, len(structured_weights)),
                    np.sort(structured_weights)[::-1], label='Structured')
    ax.set_title('Structured weights')

    inner_grid = gs[row, 1].subgridspec(2, 2)

    ax = fig.add_subplot(inner_grid[0])
    p = ax.pcolormesh(arena_x, arena_y, unweighted_map.reshape(arena_x.shape), shading='auto')
    ax.set_title('Unweighted', fontsize=font_size)
    fig.colorbar(p, ax=ax)

    ax = fig.add_subplot(inner_grid[1])
    p = ax.pcolormesh(arena_x, arena_y, scaled_initial_map.reshape(arena_x.shape), shading='auto')
    ax.set_title('Scaled Initial', fontsize=font_size)
    fig.colorbar(p, ax=ax)

    ax = fig.add_subplot(inner_grid[2])
    p = ax.pcolormesh(arena_x, arena_y, scaled_target_map.reshape(arena_x.shape), vmin=vmin, vmax=vmax, shading='auto')
    ax.set_title('Target', fontsize=font_size)
    ax.set_ylabel('Y position [cm]')
    fig.colorbar(p, ax=ax)

    ax = fig.add_subplot(inner_grid[3])
    p = ax.pcolormesh(arena_x, arena_y, lsqr_map.reshape(arena_x.shape), vmin=vmin, vmax=vmax, shading='auto')
    ax.set_title('NNLS', fontsize=font_size)
    ax.set_xlabel('X position [cm]')
    fig.colorbar(p, ax=ax)

    fig.suptitle(f'gid {gid}; field width is {field_width[0]:.02f} cm')

    if save_fig is not None:
        plt.savefig(save_fig)
        
    if show_fig:
        plt.show()

    return fig



    
    
    


    
