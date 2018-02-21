import itertools
from neuron import h
import numpy as np
import copy
import datetime
from dentate.neuron_utils import *
from dentate.utils import *
import btmorph


soma_swc_type, axon_swc_type, basal_swc_type, apical_swc_type, trunk_swc_type, tuft_swc_type, ais_swc_type, \
hillock_swc_type = 1, 2, 3, 4, 5, 6, 7, 8

swc_type_dict = {'soma': soma_swc_type, 'axon': axon_swc_type, 'basal': basal_swc_type, 'apical': apical_swc_type,
                 'trunk': trunk_swc_type, 'tuft': tuft_swc_type, 'ais': ais_swc_type, 'hillock': hillock_swc_type}

ordered_sec_types = ['soma', 'hillock', 'ais', 'axon', 'basal', 'trunk', 'apical', 'tuft', 'spine_neck', 'spine_head']


class HocCell(object):
    """
    A Python wrapper for neuronal cell objects specified in the NEURON language hoc.
    Extends btmorph.STree to provide an tree interface to facilitate:
    1) Iteration through connected neuronal compartments, and
    2) Specification of complex compartment attributes like gradients of ion channel density or synaptic properties.
    """
    def __init__(self, gid=0, population=None, hoc_cell=None, mech_file_path=None):
        """

        :param gid: int
        :param population: str
        :param hoc_cell: :class:'h.hocObject': instance of a NEURON cell template
        :param mech_file_path: str (path)
        """
        self._gid = gid
        self._population = population
        self.tree = btmorph.STree2()  # Builds a simple tree to store nodes of type 'SHocNode'
        self.count = 0  # Keep track of number of nodes
        self.nodes = {key: [] for key in swc_type_dict.keys() + ['spine_head', 'spine_neck']}
        self.mech_file_path = mech_file_path
        self.random = np.random.RandomState()
        self.random.seed(self.gid)
        self.spike_detector = None
        self.hoc_cell = hoc_cell
        if hoc_cell is not None:
            import_morphology_from_hoc(self, hoc_cell)
            if self.axon:
                self.spike_detector = connect2target(self, self.axon[-1].sec)
            elif self.soma:
                self.spike_detector = connect2target(self, self.soma[0].sec)
            if self.mech_file_path is not None:
                init_mechanisms(self, init_cable=True, from_file=True, mech_file_path=self.mech_file_path)

    def init_synaptic_mechanisms(self):
        """
        Attributes of potential synapses are stored in the synapse_mechanism_attributes dictionary within each node. Any
        time that synapse attributes are modified, this method can be called to synchronize those attributes with any
        synaptic point processes contained either within a parent section, or child spines.
        """
        for sec_type in ['soma', 'ais', 'basal', 'trunk', 'apical', 'tuft']:
            for node in self.get_nodes_of_subtype(sec_type):
                for syn in self.get_synapses(node):
                    if syn.id is not None and syn.id in node.synapse_mechanism_attributes:
                        for mech_name in (mech_name for mech_name in node.synapse_mechanism_attributes[syn.id]
                                          if mech_name in syn.targets):
                            for param_name, param_val in \
                                    node.synapse_mechanism_attributes[syn.id][mech_name].iteritems():
                                if hasattr(syn.target(mech_name), param_name):
                                    setattr(syn.target(mech_name), param_name, param_val)
                                elif hasattr(syn.netcon(mech_name), param_name):
                                    if param_name == 'weight':
                                        syn.netcon(mech_name).weight[0] = param_val
                                    else:
                                        setattr(syn.netcon(mech_name), param_name, param_val)

    def get_synapses(self, node, syn_type=None):
        """
        Returns a list of all synapse objects contained either directly in the specified node, or in attached spines.
        Can also filter by type of synaptic point process mechanism.
        :param node: :class:'SHocNode'
        :param syn_type: str
        :return: list of :class:'Synapse'
        """
        synapses = [syn for syn in node.synapses if syn_type is None or syn_type in syn.targets]
        for spine in node.spines:
            synapses.extend([syn for syn in spine.synapses if syn_type is None or syn_type in syn.targets])
        return synapses

    def sec_type_has_synapses(self, sec_type, syn_type=None):
        """
        Checks if any nodes of a given sec_type contain synapses, or spines with synapses. Can also check for a synaptic
        point process of a specific type.
        :param sec_type: str
        :param syn_type: str
        :return: boolean
        """
        for node in self.get_nodes_of_subtype(sec_type):
            if self.node_has_synapses(node, syn_type):
                return True
        return False

    def reinitialize_subset_mechanisms(self, sec_type, mech_name):
        """
        During parameter optimization, it is often convenient to reinitialize all the parameters for a single mechanism
        in a subset of compartments. For example, g_pas in basal dendrites that inherit the value from the soma after
        modifying the value in the soma compartment.
        :param sec_type: str
        :param mech_name: str
        :return:
        """
        if sec_type in self.mech_dict and mech_name in self.mech_dict[sec_type]:
            for node in self.get_nodes_of_subtype(sec_type):
                self._modify_mechanism(node, mech_name, self.mech_dict[sec_type][mech_name])

    def _parse_mech_content(self, node, mech_name, param_name, rules, syn_type=None):
        """
        This method loops through all the segments in a node and sets the value(s) for a single mechanism parameter by
        interpreting the rules specified in the mechanism dictionary. Properly handles ion channel gradients and
        inheritance of values from the closest segment of a specified type of section along the path from root. Also
        handles rules with distance boundaries, and rules to set synapse attributes. Gradients can be specified as
        linear, exponential, or sigmoidal. Custom functions can also be provided to specify arbitrary distributions.
        :param node: :class:'SHocNode'
        :param mech_name: str
        :param param_name: str
        :param rules: dict
        :param syn_type: str
        """
        if 'synapse' in mech_name:
            if syn_type is None:
                raise Exception('Cannot set %s mechanism parameter: %s without a specified point process' %
                                (mech_name, param_name))
        # an 'origin' with no 'value' inherits a starting parameter from the origin sec_type
        # a 'value' with no 'origin' is independent of other sec_types
        # an 'origin' with a 'value' uses the origin sec_type only as a reference point for applying a
        # distance-dependent gradient
        if 'origin' in rules:
            if rules['origin'] == 'parent':
                if node.type == 'spine_head':
                    donor = node.parent.parent.parent
                elif node.type == 'spine_neck':
                    donor = node.parent.parent
                else:
                    donor = node.parent
            elif rules['origin'] == 'branch_origin':
                donor = self.get_dendrite_origin(node)
            elif rules['origin'] in self.nodes:
                donor = self._get_node_along_path_to_root(node, rules['origin'])
            else:
                if 'synapse' in mech_name:
                    raise Exception('%s mechanism: %s parameter: %s cannot inherit from unknown origin: %s' %
                                    (mech_name, syn_type, param_name, rules['origin']))
                else:
                    raise Exception('Mechanism: {} parameter: {} cannot inherit from unknown origin: {}'.format(
                        mech_name, param_name, rules['origin']))
        else:
            donor = None
        if 'value' in rules:
            baseline = rules['value']
        elif donor is None:
            if 'synapse' in mech_name:
                raise Exception('Cannot set %s mechanism: %s parameter: %s without a specified origin or value' %
                                (mech_name, syn_type, param_name))
            else:
                raise Exception('Cannot set mechanism: {} parameter: {} without a specified origin or value'.format(
                    mech_name, param_name))
        else:
            if (mech_name == 'cable') and (param_name == 'spatial_res'):
                baseline = self._get_spatial_res(donor)
            elif 'synapse' in mech_name:
                baseline = self._inherit_mech_param(donor, mech_name, param_name, syn_type)
                if baseline is None:
                    raise Exception('Cannot inherit %s mechanism: %s parameter: %s from sec_type: %s' %
                                    (mech_name, syn_type, param_name, donor.type))
            else:
                baseline = self._inherit_mech_param(donor, mech_name, param_name)
        if mech_name == 'cable':  # cable properties can be inherited, but cannot be specified as gradients
            if param_name == 'spatial_res':
                node.init_nseg(baseline)
            else:
                setattr(node.sec, param_name, baseline)
                node.init_nseg(self._get_spatial_res(node))
            node.reinit_diam()
        else:
            if 'custom' in rules:
                if hasattr(self, rules['custom']['method']):
                    method_to_call = getattr(self, rules['custom']['method'])
                    method_to_call(node, mech_name, param_name, baseline, rules, syn_type, donor)
                else:
                    raise Exception('The custom method %s is not defined for this cell type.' %
                                    rules['custom']['method'])
            elif 'min_loc' in rules or 'max_loc' in rules or 'slope' in rules:
                if 'synapse' in mech_name:
                    if donor is None:
                        raise Exception('Cannot specify %s mechanism: %s parameter: %s without a provided origin' %
                                        (mech_name, syn_type, param_name))
                    else:
                        self._specify_synaptic_parameter(node, mech_name, param_name, baseline, rules, syn_type, donor)
                else:
                    if donor is None:
                        raise Exception('Cannot specify mechanism: %s parameter: %s without a provided origin' %
                                        (mech_name, param_name))
                    self._specify_mech_parameter(node, mech_name, param_name, baseline, rules, donor)
            elif mech_name == 'ions':
                setattr(node.sec, param_name, baseline)
            elif 'synapse' in mech_name:
                self._specify_synaptic_parameter(node, mech_name, param_name, baseline, rules, syn_type)
            else:
                node.sec.insert(mech_name)
                setattr(node.sec, param_name + "_" + mech_name, baseline)

    def _specify_mech_parameter(self, node, mech_name, param_name, baseline, rules, donor=None):
        """

        :param node: :class:'SHocNode'
        :param mech_name: str
        :param param_name: str
        :param baseline: float
        :param rules: dict
        :param donor: :class:'SHocNode' or None
        """
        if donor is None:
            raise Exception('Cannot specify mechanism: {} parameter: {} without a provided origin'.format(
                mech_name, param_name))
        if 'min_loc' in rules:
            min_distance = rules['min_loc']
        else:
            min_distance = None
        if 'max_loc' in rules:
            max_distance = rules['max_loc']
        else:
            max_distance = None
        min_seg_distance = self.get_distance_to_node(donor, node, 0.5 / node.sec.nseg)
        max_seg_distance = self.get_distance_to_node(donor, node, (0.5 + node.sec.nseg - 1) / node.sec.nseg)
        # if any part of the section is within the location constraints, insert the mechanism, and specify
        # the parameter at the segment level
        if (min_distance is None or max_seg_distance >= min_distance) and \
                (max_distance is None or min_seg_distance <= max_distance):
            if not mech_name == 'ions':
                node.sec.insert(mech_name)
            if min_distance is None:
                min_distance = 0.
            for seg in node.sec:
                seg_loc = self.get_distance_to_node(donor, node, seg.x)
                if seg_loc >= min_distance and (max_distance is None or seg_loc <= max_distance):
                    if 'slope' in rules:
                        seg_loc -= min_distance
                        if 'tau' in rules:
                            if 'xhalf' in rules:  # sigmoidal gradient
                                offset = baseline - rules['slope'] / (1. + np.exp(rules['xhalf'] / rules['tau']))
                                value = offset + rules['slope'] /\
                                                 (1. + np.exp((rules['xhalf'] - seg_loc) / rules['tau']))
                            else:  # exponential gradient
                                offset = baseline - rules['slope']
                                value = offset + rules['slope'] * np.exp(seg_loc / rules['tau'])
                        else:  # linear gradient
                            value = baseline + rules['slope'] * seg_loc
                        if 'min' in rules and value < rules['min']:
                            value = rules['min']
                        elif 'max' in rules and value > rules['max']:
                            value = rules['max']
                    else:
                        value = baseline
                # by default, if only some segments in a section meet the location constraints, the parameter inherits
                # the mechanism's default value. if another value is desired, it can be specified via an 'outside' key
                # in the mechanism dictionary entry
                elif 'outside' in rules:
                    value = rules['outside']
                else:
                    value = None
                if value is not None:
                    if mech_name == 'ions':
                        setattr(seg, param_name, value)
                    else:
                        setattr(getattr(seg, mech_name), param_name, value)

    def _specify_synaptic_parameter(self, node, mech_name, param_name, baseline, rules, syn_type, donor=None):
        """
        This method interprets an entry from the mechanism dictionary to set parameters for synapse_mechanism_attributes
        contained in this node. Appropriately implements slopes and inheritances.
        :param node: :class:'SHocNode'
        :param mech_name: str
        :param param_name: str
        :param baseline: float
        :param rules: dict
        :param syn_type: str
        :param donor: :class:'SHocNode' or None
        """
        syn_category = mech_name.split(' ')[0]
        if 'min_loc' in rules:
            min_distance = rules['min_loc']
        else:
            min_distance = 0.
        if 'max_loc' in rules:
            max_distance = rules['max_loc']
        else:
            max_distance = None
        if 'variance' in rules and rules['variance'] == 'normal':
            normal = True
        else:
            normal = False
        this_synapse_attributes = node.get_filtered_synapse_attributes(syn_category=syn_category)
        for i in xrange(len(this_synapse_attributes['syn_locs'])):
            loc = this_synapse_attributes['syn_locs'][i]
            this_syn_id = this_synapse_attributes['syn_id'][i]
            if this_syn_id not in node.synapse_mechanism_attributes:
                node.synapse_mechanism_attributes[this_syn_id] = {}
            if syn_type not in node.synapse_mechanism_attributes[this_syn_id]:
                node.synapse_mechanism_attributes[this_syn_id][syn_type] = {}
            if donor is None:
                value = baseline
            else:
                distance = self.get_distance_to_node(donor, node, loc)
                # If only some synapses in a section meet the location constraints, the synaptic parameter will
                # maintain its default value in all other locations. values for other locations must be specified
                # with an additional entry in the mechanism dictionary.
                if distance >= min_distance and (max_distance is None or distance <= max_distance):
                    if 'slope' in rules:
                        distance -= min_distance
                        if 'tau' in rules:
                            if 'xhalf' in rules:  # sigmoidal gradient
                                offset = baseline - rules['slope'] / (1. + np.exp(rules['xhalf'] / rules['tau']))
                                value = offset + rules['slope'] / (1. + np.exp((rules['xhalf'] - distance) /
                                                                               rules['tau']))
                            else:  # exponential gradient
                                offset = baseline - rules['slope']
                                value = offset + rules['slope'] * np.exp(distance / rules['tau'])
                        else:  # linear gradient
                            value = baseline + rules['slope'] * distance
                        if 'min' in rules and value < rules['min']:
                            value = rules['min']
                        elif 'max' in rules and value > rules['max']:
                            value = rules['max']
                    else:
                        value = baseline
            if normal:
                value = self.random.normal(value, value / 6.)
            node.synapse_mechanism_attributes[this_syn_id][syn_type][param_name] = value


    def get_dendrite_origin(self, node, parent_type=None):
        """
        This method determines the section type of the given node, and returns the node representing the primary branch
        point for the given section type. Basal and trunk sections originate at the soma, and apical and tuft dendrites
        originate at the trunk. For spines, recursively calls with parent node to identify the parent branch first.
        :param node: :class:'SHocNode'
        :return: :class:'SHocNode'
        """
        sec_type = node.type
        if sec_type in ['spine_head', 'spine_neck']:
            return self.get_dendrite_origin(node.parent, parent_type)
        elif parent_type is not None:
            return self._get_node_along_path_to_root(node.parent, parent_type)
        elif sec_type in ['basal', 'trunk', 'axon_hill', 'ais', 'axon']:
            return self._get_node_along_path_to_root(node, 'soma')
        elif sec_type in ['apical', 'tuft']:
            if self.nodes['trunk']:
                return self._get_node_along_path_to_root(node, 'trunk')
            else:
                return self._get_node_along_path_to_root(node, 'soma')
        elif sec_type == 'soma':
            return node

    def _get_node_along_path_to_root(self, node, sec_type):
        """
        This method follows the path from the given node to the root node, and returns the first node with section type
        sec_type.
        :param node: :class:'SHocNode'
        :param sec_type: str
        :return: :class:'SHocNode'
        """
        parent = node
        while not parent is None:
            if parent in self.soma and not sec_type == 'soma':
                parent = None
            elif parent.type == sec_type:
                return parent
            else:
                parent = parent.parent
        raise Exception('The path from node: {} to root does not contain sections of type: {}'.format(node.name,
                                                                                                      sec_type))

    def _get_closest_synapse(self, node, loc, syn_type=None, downstream=True):
        """
        This method finds the closest synapse to the specified location within or downstream of the provided node. Used
        for inheritance of synaptic mechanism parameters. Can also look upstream instead. Can also find the closest
        synapse containing a synaptic point_process of a specific type.
        :param node: :class:'SHocNode'
        :param loc: float
        :param syn_type: str
        :return: :class:'Synapse'
        """

        syn_list = [syn for syn in node.synapses if syn_type is None or syn_type in syn._syn]
        for spine in node.spines:
            syn_list.extend([syn for syn in spine.synapses if syn_type is None or syn_type in syn._syn])
        if not syn_list:
            if downstream:
                for child in [child for child in node.children if child.type == node.type]:
                    target_syn = self._get_closest_synapse(child, 0., syn_type)
                    if target_syn is not None:
                        return target_syn
                return None
            elif node.parent.type == node.type:
                return self._get_closest_synapse(node.parent, 1., syn_type, downstream=False)
            else:
                return None
        else:
            min_distance = 1.
            target_syn = None
            for syn in syn_list:
                distance = abs(syn.loc - loc)
                if distance < min_distance:
                    min_distance = distance
                    target_syn = syn
            return target_syn

    def _get_closest_synapse_attribute(self, node, loc, syn_category, syn_type=None, downstream=True):
        """
        This method finds the closest synapse_attribute to the specified location within or downstream of the specified
        node. Used for inheritance of synaptic mechanism parameters. Can also look upstream instead. Can also find the
        closest synapse_attribute specifying parameters of a synaptic point_process of a specific type.
        :param node: :class:'SHocNode'
        :param loc: float
        :param syn_category: str
        :param syn_type: str
        :param downstream: bool
        :return: tuple: (:class:'SHocNode', int) : node containing synapse, syn_id
        """
        min_distance = 1.
        target_index = None
        this_synapse_attributes = node.get_filtered_synapse_attributes(syn_category=syn_category, syn_type=syn_type)
        if this_synapse_attributes['syn_locs']:
            for i in xrange(len(this_synapse_attributes['syn_locs'])):
                this_syn_loc = this_synapse_attributes['syn_locs'][i]
                distance = abs(loc - this_syn_loc)
                if distance < min_distance:
                    min_distance = distance
                    target_index = this_synapse_attributes['syn_id'][i]
            return node, target_index
        else:
            if downstream:
                for child in (child for child in node.children if child.type not in ['spine_head', 'spine_neck']):
                    target_node, target_index = self._get_closest_synapse_attribute(child, 0., syn_category, syn_type)
                    if target_index is not None:
                        return target_node, target_index
                return node, None
            elif node.parent is not None:  # stop at the root
                return self._get_closest_synapse_attribute(node.parent, 1., syn_category, syn_type, downstream)
            else:
                return node, None

    def _inherit_mech_param(self, donor, mech_name, param_name, syn_type=None):
        """
        When the mechanism dictionary specifies that a node inherit a parameter value from a donor node, this method
        returns the value of that parameter found in the section or final segment of the donor node. For synaptic
        mechanism parameters, searches for the closest synapse_attribute in the donor node. If the donor node does not
        contain synapse_mechanism_attributes due to location constraints, this method searches first child nodes, then
        nodes along the path to root.
        :param donor: :class:'SHocNode'
        :param mech_name: str
        :param param_name: str
        :param syn_type: str
        :return: float
        """
        # accesses the last segment of the section
        loc = donor.sec.nseg / (donor.sec.nseg + 1.)
        try:
            if mech_name in ['cable', 'ions']:
                if mech_name == 'cable' and param_name == 'Ra':
                    return getattr(donor.sec, param_name)
                else:
                    return getattr(donor.sec(loc), param_name)
            elif 'synapse' in mech_name:
                # first look downstream for a nearby synapse, then upstream.
                syn_category = mech_name.split(' ')[0]
                target_node, target_index = self._get_closest_synapse_attribute(donor, 1., syn_category, syn_type,
                                                                                downstream=True)
                if target_index is None and donor.parent is not None:
                    target_node, target_index = self._get_closest_synapse_attribute(donor.parent, 1., syn_category,
                                                                                    syn_type, downstream=False)
                if target_index is not None \
                        and param_name in target_node.synapse_mechanism_attributes[target_index][syn_type]:
                    return target_node.synapse_mechanism_attributes[target_index][syn_type][param_name]
                else:
                    return None
            else:
                return getattr(getattr(donor.sec(loc), mech_name), param_name)
        except (AttributeError, NameError, KeyError):
            if syn_type is None:
                print 'Exception: Problem inheriting mechanism: {} parameter: {} from sec_type: {}'.format(
                    mech_name, param_name, donor.type)
            else:
                print 'Exception: Problem inheriting %s mechanism: %s parameter: %s from sec_type: %s' % \
                      (mech_name, syn_type, param_name, donor.type)
            raise KeyError

    def _get_spatial_res(self, node):
        """
        Checks the mechanism dictionary if the section type of this node has a specified spatial resolution factor.
        Used to scale the number of segments per section in the hoc model by a factor of an exponent of 3.
        :param node: :class:'SHocNode
        :return: int
        """
        try:  # if spatial_res has not been specified for the origin type of section, it defaults to 0
            rules = self.mech_dict[node.type]['cable']['spatial_res']
        except KeyError:
            return 0
        if 'value' in rules:
            return rules['value']
        elif 'origin' in rules:
            if rules['origin'] in self.nodes:  # if this sec_type also inherits the value, continue following the path
                return self._get_spatial_res(self._get_node_along_path_to_root(node, rules['origin']))
            else:
                print 'Exception: Spatial resolution cannot be inherited from sec_type: {}'.format(rules['origin'])
                raise KeyError
        else:
            print 'Exception: Cannot set spatial resolution without a specified origin or value'
            raise KeyError

    def modify_mech_param(self, sec_type, mech_name, param_name=None, value=None, origin=None, slope=None, tau=None,
                          xhalf=None, min=None, max=None, min_loc=None, max_loc=None, outside=None, syn_type=None,
                          variance=None, replace=True, custom=None):
        """
        Modifies or inserts new membrane mechanisms into hoc sections of type sec_type. First updates the mechanism
        dictionary, then sets the corresponding hoc parameters. This method is meant to be called manually during
        initial model specification, or during parameter optimization. For modifications to persist across simulations,
        the mechanism dictionary must be saved to a file using self.export_mech_dict() and re-imported during HocCell
        initialization.
        :param sec_type: str
        :param mech_name: str
        :param param_name: str
        :param value: float
        :param origin: str
        :param slope: float
        :param tau: float
        :param xhalf: float
        :param min: float
        :param max: float
        :param min_loc: float
        :param max_loc: float
        :param outside: float
        :param syn_type: str
        :param variance: str
        :param replace: bool
        :param custom: dict
        """
        global verbose
        if 'synapse' in mech_name:
            self._modify_synaptic_mech_param(sec_type, mech_name, param_name, value, origin, slope, tau, xhalf, min,
                                             max, min_loc, max_loc, outside, syn_type, variance, replace, custom)
            return
        backup_content = None
        mech_content = None
        if not sec_type in self.nodes:
            raise Exception('Cannot specify mechanism: {} parameter: {} for unknown sec_type: {}'.format(mech_name,
                                                                                                         param_name,
                                                                                                         sec_type))
        if param_name is None:
            if mech_name in ['cable', 'ions']:
                raise Exception('No parameter specified for mechanism: {}'.format(mech_name))
        if not param_name is None:
            if value is None and origin is None:
                raise Exception('Cannot set mechanism: {} parameter: {} without a specified origin or value'.format(
                    mech_name, param_name))
            rules = {}
            if not origin is None:
                if not origin in self.nodes + ['parent', 'branch_origin']:
                    raise Exception('Cannot inherit mechanism: {} parameter: {} from unknown origin: {}'.format(
                        mech_name, param_name, origin))
                else:
                    rules['origin'] = origin
            if not custom is None:
                rules['custom'] = custom
            if not value is None:
                rules['value'] = value
            if not slope is None:
                rules['slope'] = slope
            if not tau is None:
                rules['tau'] = tau
            if not xhalf is None:
                rules['xhalf'] = xhalf
            if not min is None:
                rules['min'] = min
            if not max is None:
                rules['max'] = max
            if not min_loc is None:
                rules['min_loc'] = min_loc
            if not max_loc is None:
                rules['max_loc'] = max_loc
            if not outside is None:
                rules['outside'] = outside
            # currently only implemented for synaptic parameters
            if not variance is None:
                rules['variance'] = variance
            mech_content = {param_name: rules}
        # No mechanisms have been inserted into this type of section yet
        if not sec_type in self.mech_dict:
            self.mech_dict[sec_type] = {mech_name: mech_content}
        # This mechanism has not yet been inserted into this type of section
        elif not mech_name in self.mech_dict[sec_type]:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            self.mech_dict[sec_type][mech_name] = mech_content
        # This mechanism has been inserted, but no parameters have been specified
        elif self.mech_dict[sec_type][mech_name] is None:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            self.mech_dict[sec_type][mech_name] = mech_content
        # This parameter has already been specified
        elif param_name is not None and param_name in self.mech_dict[sec_type][mech_name]:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            # Determine whether to replace or extend the current dictionary entry.
            if replace:
                self.mech_dict[sec_type][mech_name][param_name] = rules
            elif type(self.mech_dict[sec_type][mech_name][param_name]) == dict:
                self.mech_dict[sec_type][mech_name][param_name] = [self.mech_dict[sec_type][mech_name][param_name],
                                                                   rules]
            elif type(self.mech_dict[sec_type][mech_name][param_name]) == list:
                self.mech_dict[sec_type][mech_name][param_name].append(rules)
        # This mechanism has been inserted, but this parameter has not yet been specified
        elif param_name is not None:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            self.mech_dict[sec_type][mech_name][param_name] = rules

        try:
            nodes = self.get_nodes_of_subtype(sec_type)
            # all membrane mechanisms in sections of type sec_type must be reinitialized after changing cable properties
            if mech_name == 'cable':
                if param_name in ['Ra', 'cm', 'spatial_res']:
                    self._reinit_mech(nodes, reset_cable=True)
                else:
                    print 'Exception: Unknown cable property: {}'.format(param_name)
                    raise KeyError
            else:
                for node in nodes:
                    try:
                        self._modify_mechanism(node, mech_name, mech_content)
                    except (AttributeError, NameError, ValueError, KeyError):
                        raise KeyError
        except KeyError:
            if backup_content is None:
                del self.mech_dict[sec_type]
            else:
                self.mech_dict[sec_type] = copy.deepcopy(backup_content)
            if not param_name is None:
                raise Exception('Problem specifying mechanism: %s parameter: %s in node: %s' %
                                (mech_name, param_name, node.name))
            else:
                raise Exception('Problem specifying mechanism: %s in node: %s' %
                                (mech_name, node.name))

    def _modify_synaptic_mech_param(self, sec_type, mech_name=None, param_name=None, value=None, origin=None,
                                    slope=None, tau=None, xhalf=None, min=None, max=None, min_loc=None, max_loc=None,
                                    outside=None, syn_type=None, variance=None, replace=True, custom=None):

        """
        Attributes of synaptic point processes are stored in the synapse_mechanism_attributes dictionary of each node.
        This method first updates the mechanism dictionary, then replaces or creates synapse_mechanism_attributes in
        nodes of type sec_type. Handles special nested dictionary specification for synaptic parameters.
        :param sec_type: str
        :param mech_name: str
        :param param_name: str
        :param value: float
        :param origin: str
        :param slope: float
        :param tau: float
        :param xhalf: float
        :param min: float
        :param max: float
        :param min_loc: float
        :param max_loc: float
        :param outside: float
        :param syn_type: str
        :param variance: str
        :param replace: bool
        :param custom: dict
        """
        global verbose
        backup_content = None
        mech_content = None
        if syn_type is None:
            raise Exception('Cannot specify %s mechanism parameters without a specified type of synaptic point process.'
                            % mech_name)
        if not sec_type in self.nodes:
            raise Exception('Cannot specify %s mechanism: %s parameter: %s for unknown sec_type: %s' %
                            (mech_name, syn_type, param_name, sec_type))
        if not param_name is None:
            if value is None and origin is None:
                raise Exception('Cannot set %s mechanism: %s parameter: %s without a specified origin or value' %
                                (mech_name, syn_type, param_name))
            rules = {}
            if not origin is None:
                if not origin in self.nodes + ['parent', 'branch_origin']:
                    raise Exception('Cannot inherit %s mechanism: %s parameter: %s from unknown origin: %s' %
                                    (mech_name, syn_type, param_name, origin))
                else:
                    rules['origin'] = origin
            if not custom is None:
                rules['custom'] = custom
            if not value is None:
                rules['value'] = value
            if not slope is None:
                rules['slope'] = slope
            if not tau is None:
                rules['tau'] = tau
            if not xhalf is None:
                rules['xhalf'] = xhalf
            if not min is None:
                rules['min'] = min
            if not max is None:
                rules['max'] = max
            if not min_loc is None:
                rules['min_loc'] = min_loc
            if not max_loc is None:
                rules['max_loc'] = max_loc
            if not outside is None:
                rules['outside'] = outside
            if not variance is None:
                rules['variance'] = variance
            mech_content = {param_name: rules}
        # No mechanisms have been inserted into this type of section yet
        if not sec_type in self.mech_dict:
            self.mech_dict[sec_type] = {mech_name: {syn_type: mech_content}}
        # No synapse attributes have been specified in this type of section yet
        elif not mech_name in self.mech_dict[sec_type]:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            self.mech_dict[sec_type][mech_name] = {syn_type: mech_content}
        # This synaptic mechanism has not yet been specified in this type of section
        elif not syn_type in self.mech_dict[sec_type][mech_name]:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            self.mech_dict[sec_type][mech_name][syn_type] = mech_content
        # This synaptic mechanism has been specified, but no parameters have been specified
        elif self.mech_dict[sec_type][mech_name][syn_type] is None:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            self.mech_dict[sec_type][mech_name][syn_type] = mech_content
        # This parameter has already been specified.
        elif param_name is not None and param_name in self.mech_dict[sec_type][mech_name][syn_type]:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            # Determine whether to replace or extend the current dictionary entry.
            if replace:
                self.mech_dict[sec_type][mech_name][syn_type][param_name] = rules
            elif type(self.mech_dict[sec_type][mech_name][syn_type][param_name]) == dict:
                self.mech_dict[sec_type][mech_name][syn_type][param_name] = \
                    [self.mech_dict[sec_type][mech_name][syn_type][param_name], rules]
            elif type(self.mech_dict[sec_type][mech_name][syn_type][param_name]) == list:
                self.mech_dict[sec_type][mech_name][syn_type][param_name].append(rules)
        # This synaptic mechanism has been specified, but this parameter has not yet been specified
        elif param_name is not None:
            backup_content = copy.deepcopy(self.mech_dict[sec_type])
            self.mech_dict[sec_type][mech_name][syn_type][param_name] = rules

        for node in self.get_nodes_of_subtype(sec_type):
            try:
                self._modify_mechanism(node, mech_name, {syn_type: mech_content})
            except (AttributeError, NameError, ValueError, KeyError):
                if backup_content is None:
                    del self.mech_dict[sec_type]
                else:
                    self.mech_dict[sec_type] = copy.deepcopy(backup_content)
                if param_name is not None:
                    raise Exception('Problem specifying %s mechanism: %s parameter: %s in node: %s' %
                                    (mech_name, syn_type, param_name, node.name))
                else:
                    raise Exception('Problem specifying %s mechanism: %s in node: %s' %
                                    (mech_name, syn_type, node.name))

    def export_mech_dict(self, mech_file_path=None, output_dir=None):
        """
        Following modifications to the mechanism dictionary either during model specification or parameter optimization,
        this method stores the current mech_dict to a pickle file stamped with the date and time. This allows the
        current set of mechanism parameters to be recalled later.
        :param mech_file_path: str (path)
        :param output_dir: str (path)
        """
        if mech_file_path is None:
            mech_file_name = 'mech_dict_' + datetime.datetime.today().strftime('%Y%m%d_%H%M') + '.yaml'
            if output_dir is None:
                mech_file_path = mech_file_name
            elif os.path.isdir(output_dir):
                mech_file_path = output_dir+'/'+mech_file_name
        write_to_yaml(mech_file_path, self.mech_dict)
        print "Exported mechanism dictionary to " + mech_file_path

    def get_node_by_distance_to_soma(self, distance, sec_type):
        """
        Gets the first node of the given section type at least the given distance from a soma node.
        Not particularly useful, since it will always return the same node.
        :param distance: int or float
        :param sec_type: str
        :return: :class:'SHocNode'
        """
        nodes = self.nodes[sec_type]
        for node in nodes:
            if self.get_distance_to_node(self.tree.root, node) >= distance:
                return node
        raise Exception('No node is {} um from a soma node.'.format(distance))

    def get_distance_to_node(self, root, node, loc=None):
        """
        Returns the distance from the given location on the given node to its connection with a root node.
        :param root: :class:'SHocNode'
        :param node: :class:'SHocNode'
        :param loc: float
        :return: int or float
        """
        length = 0.
        if node in self.soma:
            return length
        if not loc is None:
            length += loc * node.sec.L
        if root in self.soma:
            while not node.parent in self.soma:
                node.sec.push()
                loc = h.parent_connection()
                h.pop_section()
                node = node.parent
                length += loc * node.sec.L
        elif self.node_in_subtree(root, node):
            while not node.parent is root:
                node.sec.push()
                loc = h.parent_connection()
                h.pop_section()
                node = node.parent
                length += loc * node.sec.L
        else:
            return None  # node is not connected to root
        return length

    def node_in_subtree(self, root, node):
        """
        Checks if a node is contained within a subtree of root.
        :param root: 'class':SNode2 or SHocNode
        :param node: 'class':SNode2 or SHocNode
        :return: boolean
        """
        nodelist = []
        self.tree._gather_nodes(root, nodelist)
        if node in nodelist:
            return True
        else:
            return False

    def get_path_length_swc(self, path):
        """
        Calculates the distance between nodes given a list of SNode2 nodes connected in a path.
        :param path: list : :class:'SNode2'
        :return: int or float
        """
        distance = 0.
        for i in xrange(len(path) - 1):
            distance += np.sqrt(np.sum((path[i].content['p3d'].xyz - path[i + 1].content['p3d'].xyz) ** 2.))
        return distance

    def get_node_length_swc(self, raw_node):
        """
        Calculates the distance between the center points of an SNode2 node and its parent.
        :param raw_node: :class:'SNode2'
        :return: float
        """
        if not raw_node.parent is None:
            return np.sqrt(np.sum((raw_node.content['p3d'].xyz - raw_node.parent.content['p3d'].xyz) ** 2.))
        else:
            return 0.

    def get_branch_order(self, node):
        """
        Calculates the branch order of a SHocNode node. The order is defined as 0 for all soma, axon, and apical trunk
        dendrite nodes, but defined as 1 for basal dendrites that branch from the soma, and apical and tuft dendrites
        that branch from the trunk. Increases by 1 after each additional branch point. Makes sure not to count spines.
        :param node: :class:'SHocNode'
        :return: int
        """
        if node.type in ['soma', 'axon_hill', 'ais', 'axon']:
            return 0
        elif node.type == 'trunk':
            children = [child for child in node.parent.children if not child.type == 'spine_neck']
            if len(children) > 1 and children[0].type == 'trunk' and children[1].type == 'trunk':
                return 1
            else:
                return 0
        else:
            order = 0
            path = [branch for branch in self.tree.path_between_nodes(node, self.get_dendrite_origin(node)) if
                    not branch.type in ['soma', 'trunk']]
            for node in path:
                if self.is_terminal(node):
                    order += 1
                elif len([child for child in node.parent.children if not child.type == 'spine_neck']) > 1:
                    order += 1
                elif node.parent.type == 'trunk':
                    order += 1
            return order

    def is_terminal(self, node):
        """
        Calculates if a node is a terminal dendritic branch.
        :param node: :class:'SHocNode'
        :return: bool
        """
        if node.type in ['soma', 'axon_hill', 'ais', 'axon']:
            return False
        else:
            return not bool([child for child in node.children if not child.type == 'spine_neck'])

    def is_bifurcation(self, node, child_type):
        """
        Calculates if a node bifurcates into at least two children of specified type.
        :param node: :class:'SHocNode'
        :param child_type: string
        :return: bool
        """
        return len([child for child in node.children if child.type == child_type]) >= 2

    def set_stochastic_synapses(self, value):
        """
        This method turns stochastic filtering of presynaptic release on or off for all synapses contained in this cell.
        :param value: int in [0, 1]
        """
        for nodelist in self.nodes.itervalues():
            for node in nodelist:
                for syn in node.synapses:
                    syn.stochastic = value

    def insert_spines(self, sec_type_list=None):
        """
        This method inserts explicit 'spine_head' and 'spine_neck' compartments at every pre-specified excitatory
        synapse location.
        :param syn_category: str
        :param sec_type_list: list of str
        """
        syn_category = 'excitatory'
        if sec_type_list is None:
            sec_type_list = ['basal', 'trunk', 'apical', 'tuft']
        for sec_type in sec_type_list:
            for node in self.get_nodes_of_subtype(sec_type):
                for loc in node.get_filtered_synapse_attributes(syn_category=syn_category)['syn_locs']:
                    self.insert_spine(node, loc)
        self._reinit_mech(self.spine)

    def append_synapse_attributes_by_density(self, node, density, syn_category):
        """
        Given a mean synapse density in /um, return a list of synapse locations at the specified density.
        :param node: :class:'SHocNode'
        :param density: float: mean density in /um
        :param syn_category: str
        """
        L = node.sec.L
        beta = 1. / density
        interval = self.random.exponential(beta)
        while interval < L:
            loc = interval / L
            node.append_synapse_attribute(syn_category, loc)
            interval += self.random.exponential(beta)

    def append_synapse_attributes_by_layer(self, node, density_dict, syn_category):
        """
        This method populates a node with putative synapse locations of the specified type following layer-specific
        rules for synapse density.
        TODO: Create a consistent way to specify and interpret rules and gradients in the density_dict.
        :param node: :class:'SHocNode'
        :param density_dict: dict
        :param syn_category: str
        """
        if node.get_layer() is None:
            raise Exception('Cannot specify synapse density by layer without first specifying dendritic layers.')
        distance = 0.
        x = 0.
        L = node.sec.L
        point_index = 0
        while distance <= L:
            layer = node.get_layer(x)
            while layer not in density_dict:
                while point_index < node.sec.n3d() and node.sec.arc3d(point_index) <= distance:
                    point_index += 1
                if point_index >= node.sec.n3d():
                    break
                distance = node.sec.arc3d(point_index)
                x = distance / L
                layer = node.get_layer(x)
            if layer not in density_dict:
                break
            density = density_dict[layer]
            interval = self.random.exponential(1. / density)
            distance += interval
            if distance > L:
                break
            x = distance / L
            node.append_synapse_attribute(syn_category, x)

    def insert_spine(self, node, parent_loc, child_loc=0):
        """
        Spines consist of two hoc sections: a cylindrical spine head and a cylindrical spine neck.
        :param node: :class:'SHocNode'
        :param parent_loc: float
        :param child_loc: int
        """
        neck = self.make_section('spine_neck')
        neck.connect(node, parent_loc, child_loc)
        neck.sec.L = 1.58
        neck.sec.diam = 0.077
        self._init_cable(neck)
        head = self.make_section('spine_head')
        head.connect(neck)
        node.spines.append(head)
        head.sec.L = 0.5  # open cylinder, matches surface area of sphere with diam = 0.5
        head.sec.diam = 0.5
        self._init_cable(head)

    def insert_synapses_in_spines(self, sec_type_list=None, syn_types=None, stochastic=False):
        """
        Inserts synapse of the specified type(s) in spines attached to nodes of the specified sec_types.
        :param sec_type_list: str
        :param syn_types: list of str
        :param stochastic: int
        """
        syn_category = 'excitatory'
        if sec_type_list is None:
            sec_type_list = ['basal', 'trunk', 'apical', 'tuft']
        if syn_types is None:
            syn_types = ['AMPA_KIN', 'NMDA_KIN5']
        for sec_type in sec_type_list:
            for node in self.get_nodes_of_subtype(sec_type):
                for i, syn_id in enumerate(node.get_filtered_synapse_attributes(syn_category=syn_category)['syn_id']):
                    spine = node.spines[i]
                    syn = Synapse(self, spine, type_list=syn_types, stochastic=stochastic, loc=0.5, id=syn_id)
        self.init_synaptic_mechanisms()

    def insert_synapses(self, syn_category=None, syn_types=None, sec_type_list=None, stochastic=False):
        """
        Inserts synapses of specified type(s) in nodes of the specified sec_types at the pre-determined putative
        synapse locations.
        :param syn_category: str
        :param syn_types: list of str
        :param sec_type_list: list of str
        :param stochastic: int
        """
        if syn_category is None:
            syn_category = 'excitatory'
        if sec_type_list is None:
            if syn_category == 'excitatory':
                sec_type_list = ['basal', 'trunk', 'apical', 'tuft']
            elif syn_category == 'inhibitory':
                sec_type_list = ['soma', 'ais', 'basal', 'trunk', 'apical', 'tuft']
        if syn_types is None:
            if syn_category == 'excitatory':
                syn_types = ['AMPA_KIN', 'NMDA_KIN5']
            elif syn_category == 'inhibitory':
                syn_types = ['GABA_A_KIN']
        for sec_type in sec_type_list:
            for node in self.get_nodes_of_subtype(sec_type):
                this_synapse_attribute = node.get_filtered_synapse_attributes(syn_category=syn_category)
                for syn_id  in this_synapse_attribute['syn_id']:
                    syn = Synapse(self, node, type_list=syn_types, stochastic=stochastic, id=syn_id)
        self.init_synaptic_mechanisms()

    def correct_g_pas_for_spines(self):
        """
        If not explicitly modeling spine compartments for excitatory synapses, this method scales g_pas in all
        dendritic sections proportional to the number of excitatory synapses contained in each section.
        """
        for sec_type in ['basal', 'trunk', 'apical', 'tuft']:
            for node in self.get_nodes_of_subtype(sec_type):
                node.correct_g_pas_for_spines()

    def correct_cm_for_spines(self):
        """
        If not explicitly modeling spine compartments for excitatory synapses, this method scales cm in all
        dendritic sections proportional to the number of excitatory synapses contained in each section.
        """
        for loop in xrange(2):
            for sec_type in ['basal', 'trunk', 'apical', 'tuft']:
                for node in self.get_nodes_of_subtype(sec_type):
                    node.correct_cm_for_spines()
                    if loop == 0:
                        node.init_nseg()
                        node.reinit_diam()
            if loop == 0:
                self.reinit_mechanisms()

    @property
    def gid(self):
        return self._gid

    @property
    def population(self):
        return self._population

    @property
    def soma(self):
        return self.nodes['soma']

    @property
    def axon(self):
        return self.nodes['axon']

    @property
    def basal(self):
        return self.nodes['basal']

    @property
    def apical(self):
        return self.nodes['apical']

    @property
    def trunk(self):
        return self.nodes['trunk']

    @property
    def tuft(self):
        return self.nodes['tuft']

    @property
    def spine(self):
        return self.nodes['spine_head']

    @property
    def spine_head(self):
        return self.nodes['spine_head']

    @property
    def spine_neck(self):
        return self.nodes['spine_neck']

    @property
    def ais(self):
        return self.nodes['ais']

    @property
    def hillock(self):
        return self.nodes['hillock']


# ------------------------------Extend SNode2 to interact with NEURON hoc sections------------------------


class SHocNode(btmorph.btstructs2.SNode2):
    """
    Extends SNode2 with some methods for storing and retrieving additional information in the node's content
    dictionary related to running NEURON models specified in the hoc language.
    """

    def __init__(self, index=0):
        """
        :param index: int : unique node identifier
        """
        btmorph.btstructs2.SNode2.__init__(self, index)
        self.content['spines'] = []
        self.content['synapses'] = []
        self.content['synapse_attributes'] = {'syn_locs': [],
                                              'syn_category': [],
                                              'syn_id': []}
        self.content['synapse_mechanism_attributes'] = {}

    def get_sec(self):
        """
        Returns the hoc section associated with this node, stored in the node's content dictionary.
        :return: :class:'neuron.h.Section'
        """
        if 'sec' in self.content:
            return self.content['sec']
        else:
            raise Exception('This node does not yet have an associated hoc section.')

    def set_sec(self, sec):
        """
        Stores the hoc section associated with this node in the node's content dictionary.
        :param sec: :class:'neuron.h.Section'
        """
        self.content['sec'] = sec

    sec = property(get_sec, set_sec)

    def reinit_diam(self):
        """
        For a node associated with a hoc section that is a tapered cylinder, every time the spatial resolution
        of the section (nseg) is changed, the section diameters must be reinitialized. This method checks the
        node's content dictionary for diameter boundaries and recalibrates the hoc section associated with this node.
        """
        if not self.get_diam_bounds() is None:
            [diam1, diam2] = self.get_diam_bounds()
            h('diam(0:1)={}:{}'.format(diam1, diam2), sec=self.sec)

    def append_synapse_attribute(self, syn_category, loc):
        """

        :param syn_category: str
        :param loc: float
        """
        self.synapse_attributes['syn_locs'].append(loc)
        self.synapse_attributes['syn_category'].append(syn_category_enumerator[syn_category])
        self.synapse_attributes['syn_id'].append(len(self.synapse_attributes['syn_id']))

    def get_filtered_synapse_attributes(self, syn_category=None, syn_type=None, layer=None):
        """
        Return dictionary containing attributes for all potential synapses that meet the query criterion. syn_category
        and layer can be specified as lists for a broad search.
        :param syn_category: str or list of str
        :param syn_type: str
        :param layer: int or list of int
        :return: dict
        """
        if type(syn_category) is list:
            syn_category_set = {syn_category_enumerator[item] for item in syn_category}
        elif syn_category is not None:
            syn_category_set = {syn_category_enumerator[syn_category]}
        if type(layer) is list:
            layer_set = set(layer)
        elif layer is not None:
            layer_set = {layer}
        filtered_attributes = {'syn_locs': [], 'syn_category': [], 'layer': [], 'syn_id': []}
        for i in xrange(len(self.synapse_attributes['syn_locs'])):
            this_syn_id = self.synapse_attributes['syn_id'][i]
            this_syn_loc = self.synapse_attributes['syn_locs'][i]
            this_syn_category = self.synapse_attributes['syn_category'][i]
            if not (syn_category is None or this_syn_category in syn_category_set):
                continue
            this_layer = self.get_layer(this_syn_loc)
            if not (layer is None or this_layer in layer_set):
                continue
            if not (syn_type is None or (this_syn_id in self.synapse_mechanism_attributes
                                         and syn_type in self.synapse_mechanism_attributes[this_syn_id])):
                continue
            filtered_attributes['syn_locs'].append(this_syn_loc)
            filtered_attributes['syn_category'].append(this_syn_category)
            filtered_attributes['layer'].append(this_layer)
            filtered_attributes['syn_id'].append(this_syn_id)
        return filtered_attributes

    def correct_cm_for_spines(self):
        """
        If not explicitly modeling spine compartments for excitatory synapses, this method scales cm in this
        dendritic section proportional to the number of excitatory synapses contained in the section.
        """
        # arrived at via optimization. spine neck appears to shield dendrite from spine head contribution to membrane
        # capacitance and time constant
        cm_fraction = 0.40
        SA_spine = math.pi * (1.58 * 0.077 + 0.5 * 0.5)
        this_syn_locs = self.get_filtered_synapse_attributes(syn_category='excitatory')['syn_locs']
        if this_syn_locs:
            this_syn_locs = np.array(this_syn_locs)
            seg_width = 1. / self.sec.nseg
            for i, segment in enumerate(self.sec):
                SA_seg = segment.area()
                num_spines = len(np.where((this_syn_locs >= i * seg_width) & (this_syn_locs < (i + 1) * seg_width))[0])
                cm_correction_factor = (SA_seg + cm_fraction * num_spines * SA_spine) / SA_seg
                self.sec(segment.x).cm *= cm_correction_factor

    def correct_g_pas_for_spines(self):
        """
        If not explicitly modeling spine compartments for excitatory synapses, this method scales g_pas in this
        dendritic section proportional to the number of excitatory synapses contained in the section.
        """
        SA_spine = math.pi * (1.58 * 0.077 + 0.5 * 0.5)
        this_syn_locs = self.get_filtered_synapse_attributes(syn_category='excitatory')['syn_locs']
        if this_syn_locs:
            this_syn_locs = np.array(this_syn_locs)
            seg_width = 1. / self.sec.nseg
            for i, segment in enumerate(self.sec):
                SA_seg = segment.area()
                num_spines = len(np.where((this_syn_locs >= i * seg_width) & (this_syn_locs < (i + 1) * seg_width))[0])
                soma_g_pas = self.sec.cell().mech_dict['soma']['pas']['g']['value']
                gpas_correction_factor = (SA_seg * self.sec(segment.x).g_pas + num_spines * SA_spine * soma_g_pas) / \
                                         (SA_seg * self.sec(segment.x).g_pas)
                self.sec(segment.x).g_pas *= gpas_correction_factor

    def get_diam_bounds(self):
        """
        If the hoc section associated with this node is a tapered cylinder, this method returns a list containing
        the values of the diameters at the 0 and 1 ends of the section, stored in the node's content dictionary.
        Otherwise, it returns None (for non-conical cylinders).
        :return: (list: int) or None
        """
        if 'diam' in self.content:
            return self.content['diam']
        else:
            return None

    def set_diam_bounds(self, diam1, diam2):
        """
        For a node associated with a hoc section that is a tapered cylinder, this stores a list containing the values
        of the diameters at the 0 and 1 ends of the section in the node's content dictionary.
        :param diam1: int
        :param diam2: int
        """
        self.content['diam'] = [diam1, diam2]
        self.reinit_diam()

    def get_type(self):
        """
        NEURON sections are assigned a node type for convenience in order to later specify membrane mechanisms and
        properties for each type of compartment.
        :return: str
        """
        if 'type' in self.content:
            return self.content['type']
        else:
            raise Exception('This node does not yet have a defined type.')

    def set_type(self, type):
        """
        Checks that type is a string in the list of defined section types, and stores the value in the node's content
        dictionary.
        :param type: str
        """
        if type in swc_type_dict:
            self.content['type'] = type
        else:
            raise Exception('That is not a defined type of section.')

    type = property(get_type, set_type)

    def get_layer(self, x=None):
        """
        NEURON sections can be assigned a layer type for convenience in order to later specify synaptic mechanisms and
        properties for each layer. If 3D points are used to specify cell morphology, each element in the list
        corresponds to the layer of the 3D point with the same index.
        :param x: float in [0, 1] : optional relative location in section
        :return: list or float or None
        """
        if 'layer' in self.content:
            if x is None:
                return self.content['layer']
            elif self.sec.n3d() == 0:
                return self.content['layer'][0]
            else:
                for i in xrange(self.sec.n3d()):
                    if self.sec.arc3d(i) / self.sec.L >= x:
                        return self.content['layer'][i]
        else:
            return None

    def append_layer(self, layer):
        """
        NEURON sections can be assigned a layer type for convenience in order to later specify synaptic mechanisms and
        properties for each layer. If 3D points are used to specify cell morphology, each element in the list
        corresponds to the layer of the 3D point with the same index.
        :param layer: int
        """
        if 'layer' in self.content:
            self.content['layer'].append(layer)
        else:
            self.content['layer'] = [layer]

    @property
    def name(self):
        """
        Returns a str containing the name of the hoc section associated with this node. Consists of a type descriptor
        and an index identifier.
        :return: str
        """
        if 'type' in self.content:
            return '{0.type}{0.index}'.format(self)
        else:
            raise Exception('This node does not yet have a defined type.')

    @property
    def spines(self):
        """
        Returns a list of the spine head sections attached to the hoc section associated with this node.
        :return: list of :class:'SHocNode' of sec_type == 'spine_head'
        """
        return self.content['spines']

    @property
    def synapses(self):
        """
        Returns a list of the objects of :class:'Synapse' associated with this node.
        :return: list of hoc objects, type depends on .mod file(s) used to implement synapses
        """
        return self.content['synapses']

    @property
    def synapse_attributes(self):
        """
        synapse_attributes is a dict specifying attributes of potential synapses, including 'syn_category'
        (e.g. 'excitatory' or 'inhibitory'), 'syn_locs', and 'syn_id' (unique index within each node).
        :return: dict of list
        """
        return self.content['synapse_attributes']

    @property
    def synapse_mechanism_attributes(self):
        """
        synapse_mechanism_attributes is a nested dict specifying parameters of synaptic point processes and netcon
        objects, indexed by syn_id.
        e.g. {syn_id: {'AMPA_KIN': {'gmax': float},
                                   {'weight': float}}}
        :return: dict of dict
        """
        return self.content['synapse_mechanism_attributes']

    @property
    def connection_loc(self):
        """
        Returns the location along the parent section of the connection with this section, except if the sec_type
        is spine_head, in which case it reports the connection_loc of the spine neck.
        :return: int or float
        """
        if self.type == 'spine_head':
            self.parent.sec.push()
        else:
            self.sec.push()
        loc = h.parent_connection()
        h.pop_section()
        return loc


# --------------------------------------------------------------------------------------------------------- #


def append_section(cell, sec_type, sec=None):
    """
    Places the specified hoc section within the tree structure of the Python HocCell wrapper. If sec is None, creates
    a new hoc section.
    :param cell: :class:'HocCell'
    :param sec_type: str
    :param sec: :class:'h.Section'
    :return node: :class:'SHocNode'
    """
    node = SHocNode(cell.count)
    if cell.count == 0:
        cell.tree.root = node
    cell.count += 1
    node.type = sec_type
    cell.nodes[sec_type].append(node)
    if sec is None:
        node.sec = h.Section(name=node.name, cell=cell)
    else:
        node.sec = sec
    return node


def connect_nodes(parent, child, parent_loc=1., child_loc=0., connect_hoc_sections=True):
    """
    Connects this SHocNode node to a parent node, and if specified, establishes a connection between their associated
    hoc sections.
    :param parent: :class:'SHocNode'
    :param child: :class:'SHocNode'
    :param parent_loc: float in [0,1] : connect to this end of the parent hoc section
    :param child_loc: float in [0,1] : connect this end of the child hoc section
    :param connect_hoc_sections: bool
    """
    child.parent = parent
    parent.add_child(child)
    if connect_hoc_sections:
        child.sec.connect(parent.sec, parent_loc, child_loc)


def append_child_sections(cell, parent_node, child_sec_list, sec_type_map):
    """
    Traverses the subtree of a parent section, and places each child hoc section within the tree structure of the
    Python HocCell wrapper
    :param cell: :class:'HocCell'
    :param parent_node: :class:'SHocNode'
    :param child_sec_list: list of :class:'h.Section'
    :param sec_type_map: dict {str: str}
    """
    for child in child_sec_list:
        sec_type = sec_type_map[child.hname()]
        node = append_section(cell, sec_type, child)
        connect_nodes(parent_node, node, connect_hoc_sections=False)
        append_child_sections(cell, node, child.children(), sec_type_map)


def get_dendrite_origin(cell, node, parent_type=None):
    """
    This method determines the section type of the given node, and returns the node representing the primary branch
    point for the given section type. Basal and trunk sections originate at the soma, and apical and tuft dendrites
    originate at the trunk. For spines, recursively calls with parent node to identify the parent branch first.
    :param node: :class:'SHocNode'
    :return: :class:'SHocNode'
    """
    sec_type = node.type
    if sec_type in ['spine_head', 'spine_neck']:
        return get_dendrite_origin(node.parent, parent_type)
    elif parent_type is not None:
        return get_node_along_path_to_root(node.parent, parent_type)
    elif sec_type in ['basal', 'trunk', 'hillock', 'ais', 'axon']:
        return get_node_along_path_to_root(node, 'soma')
    elif sec_type in ['apical', 'tuft']:
        if 'trunk' in cell.nodes and 'trunk' in cell.mech_dict:
            return get_node_along_path_to_root(node, 'trunk')
        else:
            return get_node_along_path_to_root(node, 'soma')
    elif sec_type == 'soma':
        return node


def get_node_along_path_to_root(node, sec_type):
    """
    This method follows the path from the given node to the root node, and returns the first node with section type
    sec_type.
    :param node: :class:'SHocNode'
    :param sec_type: str
    :return: :class:'SHocNode'
    """
    parent = node
    while not parent is None:
        if parent.type == 'soma' and not sec_type == 'soma':
            parent = None
        elif parent.type == sec_type:
            return parent
        else:
            parent = parent.parent
    raise Exception('The path from node: {} to root does not contain sections of type: {}'.format(node.name,
                                                                                                  sec_type))


def inherit_mech_param(cell, donor, mech_name, param_name, syn_type=None):
    """
    When the mechanism dictionary specifies that a node inherit a parameter value from a donor node, this method
    returns the value of that parameter found in the section or final segment of the donor node. For synaptic
    mechanism parameters, searches for the closest synapse_attribute in the donor node. If the donor node does not
    contain synapse_mechanism_attributes due to location constraints, this method searches first child nodes, then
    nodes along the path to root.
    :param donor: :class:'SHocNode'
    :param mech_name: str
    :param param_name: str
    :param syn_type: str
    :return: float
    """
    # accesses the last segment of the section
    loc = donor.sec.nseg / (donor.sec.nseg + 1.)
    try:
        if mech_name in ['cable', 'ions']:
            if mech_name == 'cable' and param_name == 'Ra':
                return getattr(donor.sec, param_name)
            else:
                return getattr(donor.sec(loc), param_name)
        elif 'synapse' in mech_name:
            # first look downstream for a nearby synapse, then upstream.
            syn_category = mech_name.split(' ')[0]
            target_node, target_index = get_closest_synapse_attribute(donor, 1., syn_category, syn_type,
                                                                            downstream=True)
            if target_index is None and donor.parent is not None:
                target_node, target_index = get_closest_synapse_attribute(donor.parent, 1., syn_category,
                                                                                syn_type, downstream=False)
            if target_index is not None \
                    and param_name in target_node.synapse_mechanism_attributes[target_index][syn_type]:
                return target_node.synapse_mechanism_attributes[target_index][syn_type][param_name]
            else:
                return None
        else:
            return getattr(getattr(donor.sec(loc), mech_name), param_name)
    except (AttributeError, NameError, KeyError):
        if syn_type is None:
            print 'Exception: Problem inheriting mechanism: {} parameter: {} from sec_type: {}'.format(
                mech_name, param_name, donor.type)
        else:
            print 'Exception: Problem inheriting %s mechanism: %s parameter: %s from sec_type: %s' % \
                  (mech_name, syn_type, param_name, donor.type)
        raise KeyError


def get_spatial_res(cell, node):
    """
    Checks the mechanism dictionary if the section type of this node has a specified spatial resolution factor.
    Used to scale the number of segments per section in the hoc model by a factor of an exponent of 3.
    :param node: :class:'SHocNode
    :return: int
    """
    try:  # if spatial_res has not been specified for the origin type of section, it defaults to 0
        rules = cell.mech_dict[node.type]['cable']['spatial_res']
    except KeyError:
        return 0
    if 'value' in rules:
        return rules['value']
    elif 'origin' in rules:
        if rules['origin'] in cell.nodes:  # if this sec_type also inherits the value, continue following the path
            return get_spatial_res(cell, get_node_along_path_to_root(node, rules['origin']))
        else:
            print 'Exception: Spatial resolution cannot be inherited from sec_type: {}'.format(rules['origin'])
            raise KeyError
    else:
        print 'Exception: Cannot set spatial resolution without a specified origin or value'
        raise KeyError


def import_morphology_from_hoc(cell, hoc_cell):
    """
    Append sections from an existing instance of a NEURON cell template to a Python cell wrapper.
    :param cell: :class:'HocCell'
    :param hoc_cell: :class:'h.hocObject': instance of a NEURON cell template
    """
    sec_types = ['soma', 'axon', 'basal', 'apical', 'trunk', 'tuft', 'ais', 'hillock']
    sec_type_map = {}
    for sec_type in sec_types:
        if hasattr(hoc_cell, sec_type):
            this_sec_list = list(getattr(hoc_cell, sec_type))
            if sec_type == 'soma':
                root_sec = this_sec_list[0]
            for sec in this_sec_list:
                sec_type_map[sec.hname()] = sec_type
    try:
        root_node = append_section(cell, 'soma', root_sec)
    except Exception:
        raise KeyError('import_morphology_from_hoc: problem locating soma section to act as root')
    append_child_sections(cell, root_node, root_sec.children(), sec_type_map)


def connect2target(cell, sec, loc=1., param='_ref_v', delay=None, weight=None, threshold=None, target=None):
    """
    Converts analog voltage in the specified section to digital spike output. Initializes and returns an h.NetCon
    object with voltage as a reference parameter connected to the specified target.
    :param cell: :class:'HocCell'
    :param sec: :class:'h.Section'
    :param loc: float
    :param param: str
    :param delay: float
    :param weight: float
    :param threshold: float
    :param target: object that can receive spikes
    :return: :class:'h.NetCon'
    """
    if cell.spike_detector is not None:
        if delay is None:
            delay = cell.spike_detector.delay
        if weight is None:
            weight = cell.spike_detector.weight[0]
        if threshold is None:
            threshold = cell.spike_detector.threshold
    else:
        if delay is None:
            delay = 0.
        if weight is None:
            weight = 1.
        if threshold is None:
            threshold = -30.
    this_netcon = h.NetCon(getattr(sec(loc), param), target, sec=sec)
    this_netcon.delay = delay
    this_netcon.weight[0] = weight
    this_netcon.threshold = threshold
    return this_netcon


def import_mech_dict_from_yaml(cell, mech_file_path=None):
    """
    Imports from a .yaml file a dictionary specifying parameters of NEURON cable properties, density mechanisms, and
    point processes for each type of section in a HocCell.
    :param cell: :class:'HocCell'
    :param mech_file_path: str (path)
    """
    if mech_file_path is None:
        if cell.mech_file_path is None:
            raise ValueError('import_mechanisms_from_yaml: missing mech_file_path')
        elif not os.path.isfile(cell.mech_file_path):
            raise ValueError('import_mechanisms_from_yaml: invalid mech_file_path: %s' % cell.mech_file_path)
    elif not os.path.isfile(mech_file_path):
        raise ValueError('import_mechanisms_from_yaml: invalid mech_file_path: %s' % mech_file_path)
    else:
        cell.mech_file_path = mech_file_path
    cell.mech_dict = read_from_yaml(cell.mech_file_path)


def init_mechanisms(cell, reset_cable=True, from_file=False, mech_file_path=None):
    """
    Consults a dictionary specifying parameters of NEURON cable properties, density mechanisms, and
    point processes for each type of section in a HocCell.
    :param cell: :class:'HocCell'
    :param reset_cable: bool
    :param from_file: bool
    :param mech_file_path: str (path)
    """
    if from_file:
        import_mech_dict_from_yaml(cell, mech_file_path)
    for sec_type in ordered_sec_types:
        if sec_type in cell.mech_dict and sec_type in cell.nodes:
            update_mechanisms_by_sec_type(cell, sec_type, reset_cable=reset_cable)


def update_mechanisms_by_sec_type(cell, sec_type, reset_cable=False):
    """
    This method loops through all sections of the specified type, and consults the mechanism dictionary to update
    mechanism properties. If the reset_cable flag is True, cable parameters are re-initialize first, then the
    ion channel mechanisms are updated.
    :param cell: :class:'HocCell'
    :param sec_type: str
    :param reset_cable: bool
    """
    if sec_type in cell.nodes and sec_type in cell.mech_dict:
        for node in cell.nodes[sec_type]:
            # cable properties must be set first, as they can change nseg, which will affect insertion of membrane
            # mechanism gradients
            if reset_cable and 'cable' in cell.mech_dict[sec_type]:
                reset_cable_by_node(cell, node)
            for mech_name in (mech_name for mech_name in cell.mech_dict[sec_type]
                              if not mech_name in ['cable', 'ions']):
                update_mechanism_by_node(cell, node, mech_name, cell.mech_dict[sec_type][mech_name])
            # ion-related parameters do not exist until after membrane mechanisms have been inserted
            if 'ions' in cell.mech_dict[sec_type]:
                update_mechanism_by_node(cell, node, 'ions', cell.mech_dict[sec_type]['ions'])


def reset_cable_by_node(cell, node):
    """
    Consults a dictionary specifying parameters of NEURON cable properties such as axial resistance ('Ra'),
    membrane specific capacitance ('cm'), and a spatial resolution parameter to specify the number of separate
    segments per section in a HocCell
    :param cell: :class:'HocCell'
    :param node: :class:'SHocNode'
    """
    sec_type = node.type
    if sec_type in cell.mech_dict and 'cable' in cell.mech_dict[sec_type]:
        mech_content = cell.mech_dict[sec_type]['cable']
        if mech_content is not None:
            update_mechanism_by_node(cell, node, 'cable', mech_content)
    else:
        init_nseg(node.sec)


def update_mechanism_by_node(cell, node, mech_name, mech_content):
    """
    This method loops through all the parameters for a single mechanism specified in the mechanism dictionary and
    calls parse_mech_content to interpret the rules and set the values for the given node.
    :param node: :class:'SHocNode'
    :param mech_name: str
    :param mech_content: dict
    """
    if mech_content is not None:
        if 'synapse' in mech_name:
            update_synapse_attributes_by_node(cell, node, mech_name, mech_content)
        else:
            for param_name in mech_content:
                # accommodate either a dict, or a list of dicts specifying multiple location constraints for
                # a single parameter
                if isinstance(mech_content[param_name], dict):
                    parse_mech_content(cell, node, mech_name, param_name, mech_content[param_name])
                elif isinstance(mech_content[param_name], Iterable):
                    for mech_content_entry in mech_content[param_name]:
                        parse_mech_content(cell, node, mech_name, param_name, mech_content_entry)
    else:
        node.sec.insert(mech_name)


def parse_mech_content(cell, node, mech_name, param_name, rules, syn_type=None):
    """
        This method loops through all the segments in a node and sets the value(s) for a single mechanism parameter by
        interpreting the rules specified in the mechanism dictionary. Properly handles ion channel gradients and
        inheritance of values from the closest segment of a specified type of section along the path from root. Also
        handles rules with distance boundaries, and rules to set synapse attributes. Gradients can be specified as
        linear, exponential, or sigmoidal. Custom functions can also be provided to specify arbitrary distributions.
        :param node: :class:'SHocNode'
        :param mech_name: str
        :param param_name: str
        :param rules: dict
        :param syn_type: str
    """
    if 'synapse' in mech_name:
        if syn_type is None:
            raise Exception('Cannot set %s mechanism parameter: %s without a specified point process' %
                            (mech_name, param_name))
    # an 'origin' with no 'value' inherits a starting parameter from the origin sec_type
    # a 'value' with no 'origin' is independent of other sec_types
    # an 'origin' with a 'value' uses the origin sec_type only as a reference point for applying a
    # distance-dependent gradient
    if 'origin' in rules:
        if rules['origin'] == 'parent':
            if node.type == 'spine_head':
                donor = node.parent.parent.parent
            elif node.type == 'spine_neck':
                donor = node.parent.parent
            else:
                donor = node.parent
        elif rules['origin'] == 'branch_origin':
            donor = get_dendrite_origin(cell, node)
        elif rules['origin'] in cell.nodes:
            donor = get_node_along_path_to_root(node, rules['origin'])
        else:
            if 'synapse' in mech_name:
                raise Exception('%s mechanism: %s parameter: %s cannot inherit from unknown origin: %s' %
                                (mech_name, syn_type, param_name, rules['origin']))
            else:
                raise Exception('Mechanism: {} parameter: {} cannot inherit from unknown origin: {}'.format(
                    mech_name, param_name, rules['origin']))
    else:
        donor = None
    if 'value' in rules:
        baseline = rules['value']
    elif donor is None:
        if 'synapse' in mech_name:
            raise Exception('Cannot set %s mechanism: %s parameter: %s without a specified origin or value' %
                            (mech_name, syn_type, param_name))
        else:
            raise Exception('Cannot set mechanism: {} parameter: {} without a specified origin or value'.format(
                mech_name, param_name))
    else:
        if (mech_name == 'cable') and (param_name == 'spatial_res'):
            baseline = get_spatial_res(cell, donor)
        elif 'synapse' in mech_name:
            baseline = inherit_mech_param(cell, donor, mech_name, param_name, syn_type)
            if baseline is None:
                raise Exception('Cannot inherit %s mechanism: %s parameter: %s from sec_type: %s' %
                                (mech_name, syn_type, param_name, donor.type))
        else:
            baseline = inherit_mech_param(cell, donor, mech_name, param_name)
    if mech_name == 'cable':  # cable properties can be inherited, but cannot be specified as gradients
        if param_name == 'spatial_res':
            init_nseg(node.sec, baseline)
        else:
            setattr(node.sec, param_name, baseline)
            init_nseg(node.sec, get_spatial_res(cell, node))
        node.reinit_diam()
    else:
        if 'custom' in rules:
            if rules['custom']['method'] in globals() and callable(globals()[rules['custom']['method']]):
                method_to_call = globals()[rules['custom']['method']]
                method_to_call(cell, node, mech_name, param_name, baseline, rules, syn_type, donor)
            else:
                raise Exception('The custom method %s is not defined for this cell type.' %
                                rules['custom']['method'])
        elif 'min_loc' in rules or 'max_loc' in rules or 'slope' in rules:
            if 'synapse' in mech_name:
                if donor is None:
                    raise Exception('Cannot specify %s mechanism: %s parameter: %s without a provided origin' %
                                    (mech_name, syn_type, param_name))
                else:
                    _specify_synaptic_parameter(node, mech_name, param_name, baseline, rules, syn_type, donor)
            else:
                if donor is None:
                    raise Exception('Cannot specify mechanism: %s parameter: %s without a provided origin' %
                                    (mech_name, param_name))
                specify_mech_parameter(cell, node, mech_name, param_name, baseline, rules, donor)
        elif mech_name == 'ions':
            setattr(node.sec, param_name, baseline)
        elif 'synapse' in mech_name:
            _specify_synaptic_parameter(node, mech_name, param_name, baseline, rules, syn_type)
        else:
            node.sec.insert(mech_name)
            setattr(node.sec, param_name + "_" + mech_name, baseline)


def specify_mech_parameter(cell, node, mech_name, param_name, baseline, rules, donor=None):
    """

    :param node: :class:'SHocNode'
    :param mech_name: str
    :param param_name: str
    :param baseline: float
    :param rules: dict
    :param donor: :class:'SHocNode' or None
    """
    if donor is None:
        raise Exception('Cannot specify mechanism: {} parameter: {} without a provided origin'.format(
            mech_name, param_name))
    if 'min_loc' in rules:
        min_distance = rules['min_loc']
    else:
        min_distance = None
    if 'max_loc' in rules:
        max_distance = rules['max_loc']
    else:
        max_distance = None
    min_seg_distance = get_distance_to_node(cell, donor, node, 0.5 / node.sec.nseg)
    max_seg_distance = get_distance_to_node(cell, donor, node, (0.5 + node.sec.nseg - 1) / node.sec.nseg)
    # if any part of the section is within the location constraints, insert the mechanism, and specify
    # the parameter at the segment level
    if (min_distance is None or max_seg_distance >= min_distance) and \
            (max_distance is None or min_seg_distance <= max_distance):
        if not mech_name == 'ions':
            node.sec.insert(mech_name)
        if min_distance is None:
            min_distance = 0.
        for seg in node.sec:
            seg_loc = get_distance_to_node(cell, donor, node, seg.x)
            if seg_loc >= min_distance and (max_distance is None or seg_loc <= max_distance):
                if 'slope' in rules:
                    seg_loc -= min_distance
                    if 'tau' in rules:
                        if 'xhalf' in rules:  # sigmoidal gradient
                            offset = baseline - rules['slope'] / (1. + np.exp(rules['xhalf'] / rules['tau']))
                            value = offset + rules['slope'] /\
                                             (1. + np.exp((rules['xhalf'] - seg_loc) / rules['tau']))
                        else:  # exponential gradient
                            offset = baseline - rules['slope']
                            value = offset + rules['slope'] * np.exp(seg_loc / rules['tau'])
                    else:  # linear gradient
                        value = baseline + rules['slope'] * seg_loc
                    if 'min' in rules and value < rules['min']:
                        value = rules['min']
                    elif 'max' in rules and value > rules['max']:
                        value = rules['max']
                else:
                    value = baseline
            # by default, if only some segments in a section meet the location constraints, the parameter inherits
            # the mechanism's default value. if another value is desired, it can be specified via an 'outside' key
            # in the mechanism dictionary entry
            elif 'outside' in rules:
                value = rules['outside']
            else:
                value = None
            if value is not None:
                if mech_name == 'ions':
                    setattr(seg, param_name, value)
                else:
                    setattr(getattr(seg, mech_name), param_name, value)


def update_synapse_attributes_by_node(cell, node, mech_name, mech_content):
    """
    Consults a dictionary to specify properties of synapses of the specified category. Only sets values in a nodes
    dictionary of synapse attributes. Must then call 'update_synapses' to modify properties of underlying hoc
    point process and netcon objects.
    :param node: :class:'SHocNode'
    :param mech_name: str
    :param mech_content: dict
    """
    syn_category = mech_name.split(' ')[0]
    # Only specify synapse attributes if this category of synapses has been specified in this node
    if get_synapse_attributes(node, syn_category=syn_category)['syn_locs']:
        for syn_type in mech_content:
            if mech_content[syn_type] is not None:
                for param_name in mech_content[syn_type]:
                    # accommodate either a dict, or a list of dicts specifying multiple location constraints for
                    # a single parameter
                    if isinstance(mech_content[syn_type][param_name], dict):
                        parse_mech_content(cell, node, mech_name, param_name, mech_content[syn_type][param_name], syn_type)
                    elif isinstance(mech_content[syn_type][param_name], Iterable):
                        for mech_content_entry in mech_content[syn_type][param_name]:
                            parse_mech_content(cell, node, mech_name, param_name, mech_content_entry, syn_type)


def init_nseg(sec, spatial_res=0):
    """
    Initializes the number of segments in this section (nseg) based on the AC length constant. Must be re-initialized
    whenever basic cable properties Ra or cm are changed. The spatial resolution parameter increases the number of
    segments per section by a factor of an exponent of 3.
    :param sec: :class:'h.Section'
    :param spatial_res: int
    """
    sugg_nseg = d_lambda_nseg(sec)
    print sec.hname(), sec.nseg, sugg_nseg
    sugg_nseg *= 3 ** spatial_res
    sec.nseg = int(sugg_nseg)


def get_distance_to_node(cell, root, node, loc=None):
    """
    Returns the distance from the given location on the given node to its connection with a root node.
    :param root: :class:'SHocNode'
    :param node: :class:'SHocNode'
    :param loc: float
    :return: int or float
    """
    length = 0.
    if node in cell.soma:
        return length
    if not loc is None:
        length += loc * node.sec.L
    if root in cell.soma:
        while not node.parent in cell.soma:
            node.sec.push()
            loc = h.parent_connection()
            h.pop_section()
            node = node.parent
            length += loc * node.sec.L
    elif node_in_subtree(cell, root, node):
        while not node.parent is root:
            node.sec.push()
            loc = h.parent_connection()
            h.pop_section()
            node = node.parent
            length += loc * node.sec.L
    else:
        return None  # node is not connected to root
    return length


def node_in_subtree(cell, root, node):
    """
    Checks if a node is contained within a subtree of root.
    :param root: 'class':SNode2 or SHocNode
    :param node: 'class':SNode2 or SHocNode
    :return: boolean
    """
    nodelist = []
    cell.tree._gather_nodes(root, nodelist)
    if node in nodelist:
        return True
    else:
        return False


def get_branch_order(cell, node):
    """
    Calculates the branch order of a SHocNode node. The order is defined as 0 for all soma, axon, and apical trunk
    dendrite nodes, but defined as 1 for basal dendrites that branch from the soma, and apical and tuft dendrites
    that branch from the trunk. Increases by 1 after each additional branch point. Makes sure not to count spines.
    :param node: :class:'SHocNode'
    :return: int
    """
    if node.type in ['soma', 'hillock', 'ais', 'axon']:
        return 0
    elif node.type == 'trunk':
        children = [child for child in node.parent.children if not child.type == 'spine_neck']
        if len(children) > 1 and children[0].type == 'trunk' and children[1].type == 'trunk':
            return 1
        else:
            return 0
    else:
        order = 0
        path = [branch for branch in cell.tree.path_between_nodes(node, get_dendrite_origin(cell, node)) if
                not branch.type in ['soma', 'trunk']]
        for node in path:
            if is_terminal(node):
                order += 1
            elif len([child for child in node.parent.children if not child.type == 'spine_neck']) > 1:
                order += 1
            elif node.parent.type == 'trunk':
                order += 1
        return order


def is_terminal(node):
    """
    Calculates if a node is a terminal dendritic branch.
    :param node: :class:'SHocNode'
    :return: bool
    """
    if node.type in ['soma', 'hillock', 'ais', 'axon']:
        return False
    else:
        return not bool([child for child in node.children if not child.type == 'spine_neck'])


def zero_na(cell):
    """
    Set na channel conductances to zero in all compartments. Used during parameter optimization.
    """
    for sec_type in ['soma', 'hillock', 'ais', 'axon', 'apical']:
        for na_type in (na_type for na_type in ['nas_kin', 'nat_kin', 'nas', 'nax'] if na_type in
                cell.mech_dict[sec_type]):
            cell.modify_mech_param(sec_type, na_type, 'gbar', 0.)


def custom_gradient_by_branch_order(cell, node, mech_name, param_name, baseline, rules, syn_type, donor=None):
    """

    :param node: :class:'SHocNode'
    :param mech_name: str
    :param param_name: str
    :param baseline: float
    :param rules: dict
    :param syn_type: str
    :param donor: :class:'SHocNode' or None
    """
    branch_order = int(rules['custom']['branch_order'])
    if get_branch_order(cell, node) >= branch_order:
        if 'synapse' in mech_name:
            _specify_synaptic_parameter(node, mech_name, param_name, baseline, rules, syn_type, donor)
        else:
            specify_mech_parameter(cell, node, mech_name, param_name, baseline, rules, donor)


def custom_gradient_by_terminal(cell, node, mech_name, param_name, baseline, rules, syn_type, donor=None):
    """

    :param node: :class:'SHocNode'
    :param mech_name: str
    :param param_name: str
    :param baseline: float
    :param rules: dict
    :param syn_type: str
    :param donor: :class:'SHocNode' or None
    """
    if is_terminal(node):
        start_val = baseline
        if 'min' in rules:
            end_val = rules['min']
            direction = -1
        elif 'max' in rules:
            end_val = rules['max']
            direction = 1
        else:
            raise Exception('custom_gradient_by_terminal: no min or max target value specified for mechanism: %s '
                            'parameter: %s' % (mech_name, param_name))
        slope = (end_val - start_val)/node.sec.L
        if 'slope' in rules:
            if direction < 0.:
                slope = min(rules['slope'], slope)
            else:
                slope = max(rules['slope'], slope)
        for seg in node.sec:
            value = start_val + slope * seg.x * node.sec.L
            if direction < 0:
                if value < end_val:
                    value = end_val
            else:
                if value < end_val:
                    value = end_val
            setattr(getattr(seg, mech_name), param_name, value)


# -------------------------------------------------------------------------------------------------------------------- #


def get_node_attribute (name, content, sec, secnodes, x=None):
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


def make_neurotree_cell (template_class, local_id=0, gid=0, dataset_path="", neurotree_dict={}):
    """

    :param template_class:
    :param local_id:
    :param gid:
    :param dataset_path:
    :param neurotree_dict:
    :return:
    """
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


def make_cell(template_class, local_id=0, gid=0, dataset_path=""):
    """

    :param template_class:
    :param local_id:
    :param gid:
    :param dataset_path:
    :return:
    """
    cell = template_class (local_id, gid, dataset_path)
    return cell
