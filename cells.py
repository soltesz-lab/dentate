import itertools
from dentate.neuron_utils import *
from neuroh5.h5py_io_utils import *
import btmorph
import networkx as nx


# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)


class BiophysCell(object):
    """
    A Python wrapper for neuronal cell objects specified in the NEURON language hoc.
    Extends btmorph.STree to provide an tree interface to facilitate:
    1) Iteration through connected neuronal compartments, and
    2) Specification of complex distributions of compartment attributes like gradients of ion channel density or
    synaptic properties.
    """
    def __init__(self, gid, pop_name, hoc_cell=None, mech_file_path=None, env=None):
        """

        :param gid: int
        :param pop_name: str
        :param hoc_cell: :class:'h.hocObject': instance of a NEURON cell template
        :param mech_file_path: str (path)
        :param env: :class:'Env'
        """
        self._gid = gid
        self._pop_name = pop_name
        self.tree = btmorph.STree2()  # Builds a simple tree to store nodes of type 'SHocNode'
        self.count = 0  # Keep track of number of nodes
        if env is not None:
            for sec_type in env.SWC_Types:
                if sec_type not in default_ordered_sec_types:
                    raise AttributeError('Warning! unexpected SWC Type definitions found in Env')
        self.nodes = {key: [] for key in default_ordered_sec_types}
        self.mech_file_path = mech_file_path
        self.mech_dict = dict()
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
                import_mech_dict_from_file(self, self.mech_file_path)
    
    @property
    def gid(self):
        return self._gid

    @property
    def pop_name(self):
        return self._pop_name

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
        self.content['spine_count'] = []

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
        if type in default_ordered_sec_types:
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
            return '%s%s' % (self.type, self.index)
        else:
            raise Exception('This node does not yet have a defined type.')

    @property
    def spines(self):
        """
        Returns a list of the spine head sections attached to the hoc section associated with this node.
        :return: list of :class:'SHocNode' of sec_type == 'spine_head'
        """
        return [head for neck in self.children if neck.type == 'spine_neck' for head in neck.children
                if head.type == 'spine_head']

    @property
    def spine_count(self):
        """
        Returns a list of the number of excitatory synaptic connections to the hoc section associated with this node.
        :return: list of int
        """
        return self.content['spine_count']

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


# ----------------------------- Methods to specify cell morphology --------------------------------------------------- #


def lambda_f(sec, f=freq):
    """
    Calculates the AC length constant for the given section at the frequency f
    Used to determine the number of segments per hoc section to achieve the desired spatial and temporal resolution
    :param sec: :class:'h.Section'
    :param f: int
    :return: int
    """
    diam = np.mean([seg.diam for seg in sec])
    Ra = sec.Ra
    cm = np.mean([seg.cm for seg in sec])
    return 1e5*math.sqrt(diam/(4.*math.pi*f*Ra*cm))


def d_lambda_nseg(sec, lam=d_lambda, f=freq):
    """
    The AC length constant for this section and the user-defined fraction is used to determine the maximum size of each
    segment to achieve the d esired spatial and temporal resolution. This method returns the number of segments to set
    the nseg parameter for this section. For tapered cylindrical sections, the diam parameter will need to be
    reinitialized after nseg changes.
    :param sec : :class:'h.Section'
    :param lam : int
    :param f : int
    :return : int
    """
    L = sec.L
    return int((L/(lam*lambda_f(sec, f))+0.9)/2)*2+1


def append_section(cell, sec_type, sec_index=None, sec=None):
    """
    Places the specified hoc section within the tree structure of the python BiophysCell wrapper. If sec is None,
    creates a new hoc section.
    :param cell: :class:'BiophysCell'
    :param sec_type: str
    :param sec_index: int
    :param sec: :class:'h.Section'
    :return node: :class:'SHocNode'
    """
    if sec_index is None:
        sec_index = cell.count
    node = SHocNode(sec_index)
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
    Python BiophysCell wrapper
    :param cell: :class:'BiophysCell'
    :param parent_node: :class:'SHocNode'
    :param child_sec_list: list of :class:'h.Section'
    :param sec_type_map: dict; {:class:'h.Section': (str, int)}
    """
    for child in child_sec_list:
        sec_type, sec_index = sec_type_map[child]
        node = append_section(cell, sec_type, sec_index, child)
        connect_nodes(parent_node, node, connect_hoc_sections=False)
        append_child_sections(cell, node, child.children(), sec_type_map)


def get_dendrite_origin(cell, node, parent_type=None):
    """
    This method determines the section type of the given node, and returns the node representing the primary branch
    point for the given section type. Basal and trunk sections originate at the soma, and apical and tuft dendrites
    originate at the trunk. For spines, recursively calls with parent node to identify the parent branch first.
    :param cell: :class:'BiophysCell'
    :param node: :class:'SHocNode'
    :param parent_type: str
    :return: :class:'SHocNode'
    """
    sec_type = node.type
    if node == cell.tree.root:
        if parent_type is None or parent_type == node.type:
            return node
        else:
            return None
    if sec_type in ['spine_head', 'spine_neck']:
        return get_dendrite_origin(cell, node.parent, parent_type)
    elif parent_type is not None:
        return get_node_along_path_to_root(cell, node.parent, parent_type)
    elif sec_type in ['basal', 'trunk', 'hillock', 'ais', 'axon']:
        return get_node_along_path_to_root(cell, node, 'soma')
    elif sec_type in ['apical', 'tuft']:
        if 'trunk' in cell.nodes and 'trunk' in cell.mech_dict:
            return get_node_along_path_to_root(cell, node, 'trunk')
        else:
            return get_node_along_path_to_root(cell, node, 'soma')
    elif sec_type == 'soma':
        return node


def get_node_along_path_to_root(cell, node, sec_type):
    """
    This method follows the path from the given node to the root node, and returns the first node with section type
    sec_type.
    :param cell: :class:'BiophysCell'
    :param node: :class:'SHocNode'
    :param sec_type: str
    :return: :class:'SHocNode'
    """
    parent = node
    if parent.type == sec_type:
        return parent
    elif parent is cell.tree.root:
        return None
    else:
        return get_node_along_path_to_root(cell, parent.parent, sec_type)


def get_spatial_res(cell, node):
    """
    Checks the mechanism dictionary if the section type of this node has a specified spatial resolution factor.
    Used to scale the number of segments per section in the hoc model by a factor of an exponent of 3.
    :param cell: :class:'BiophysCell'
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
        donor = get_donor(cell, node, rules['origin'])
        if donor is not None:
            return get_spatial_res(cell, donor)
        else:
            raise RuntimeError('get_spatial_res: node: %s cannot inherit spatial resolution from origin_type: %s' %
                               (node.name, rules['origin']))
    else:
        raise RuntimeError('get_spatial_res: cannot set spatial resolution in node: %s without a specified origin '
                           'or value' % node.name)


def import_morphology_from_hoc(cell, hoc_cell):
    """
    Append sections from an existing instance of a NEURON cell template to a Python cell wrapper.
    :param cell: :class:'BiophysCell'
    :param hoc_cell: :class:'h.hocObject': instance of a NEURON cell template
    """
    sec_type_map = {}
    for sec_type, sec_index_list in default_hoc_sec_lists.iteritems():
        if hasattr(hoc_cell, sec_type):
            sec_list = list(getattr(hoc_cell, sec_type))
            if hasattr(hoc_cell, sec_index_list):
                sec_indexes = list(getattr(hoc_cell, sec_index_list))
            else:
                raise AttributeError('import_morphology_from_hoc: %s is not an attribute of the hoc cell' %
                                     sec_index_list)
            if sec_type == 'soma':
                root_sec = sec_list[0]
                root_index = int(sec_indexes[0])
            for sec, index in zip(sec_list, sec_indexes):
                sec_type_map[sec] = (sec_type, int(index))
    try:
        root_node = append_section(cell, 'soma', root_index, root_sec)
    except Exception as e:
        raise KeyError, 'import_morphology_from_hoc: problem locating soma section to act as root\n%s' % e, \
            sys.exc_info()[2]
    append_child_sections(cell, root_node, root_sec.children(), sec_type_map)


def connect2target(cell, sec, loc=1., param='_ref_v', delay=None, weight=None, threshold=None, target=None):
    """
    Converts analog voltage in the specified section to digital spike output. Initializes and returns an h.NetCon
    object with voltage as a reference parameter connected to the specified target.
    :param cell: :class:'BiophysCell'
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


def init_nseg(sec, spatial_res=0, verbose=False):
    """
    Initializes the number of segments in this section (nseg) based on the AC length constant. Must be re-initialized
    whenever basic cable properties Ra or cm are changed. The spatial resolution parameter increases the number of
    segments per section by a factor of an exponent of 3.
    :param sec: :class:'h.Section'
    :param spatial_res: int
    :param verbose: bool
    """
    sugg_nseg = d_lambda_nseg(sec)
    sugg_nseg *= 3 ** spatial_res
    if verbose:
        print 'init_nseg: changed %s.nseg %i --> %i' % (sec.hname(), sec.nseg, sugg_nseg)
    sec.nseg = int(sugg_nseg)


def insert_spine(cell, node, parent_loc, child_loc=0, neck_L=1.58, neck_diam=0.077, head_L=0.5, head_diam=0.5):
    """
    Spines consist of two hoc sections: a cylindrical spine head and a cylindrical spine neck.
    :param cell: :class:'BiophysCell'
    :param node: :class:'SHocNode'
    :param parent_loc: float
    :param child_loc: int
    :param neck_L: float
    :param neck_diam: float
    :param head_L: float
    :param head_diam: float
    """
    neck = append_section(cell, 'spine_neck')
    connect_nodes(node, neck, parent_loc, child_loc)
    neck.sec.L = neck_L
    neck.sec.diam = neck_diam
    init_nseg(neck.sec)
    head = append_section(cell, 'spine_head')
    connect_nodes(neck, head)
    head.sec.L = head_L  # open cylinder, matches surface area of sphere with diam = 0.5
    head.sec.diam = head_diam
    init_nseg(head.sec)


def get_distance_to_node(cell, root, node, loc=None):
    """
    Returns the distance from the given location on the given node to its connection with a root node.
    :param root: :class:'SHocNode'
    :param node: :class:'SHocNode'
    :param loc: float
    :return: int or float
    """
    length = 0.
    if node is root:
        return length
    if loc is not None:
        length += loc * node.sec.L
    if not node_in_subtree(cell, root, node):
        return None  # node is not connected to root
    while node.parent is not root:
        loc = node.connection_loc
        node = node.parent
        length += loc * node.sec.L
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
    :param cell: :class:'BiophysCell'
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


def is_bifurcation(node, child_type):
    """
    Calculates if a node bifurcates into at least two children of specified type.
    :param node: :class:'SHocNode'
    :param child_type: string
    :return: bool
    """
    return len([child for child in node.children if child.type == child_type]) >= 2


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


def get_path_length_swc(path):
    """
    Calculates the distance between nodes given a list of SNode2 nodes connected in a path.
    :param path: list of :class:'SNode2'
    :return: int or float
    """
    distance = 0.
    for i in xrange(len(path) - 1):
        distance += np.sqrt(np.sum((path[i].content['p3d'].xyz - path[i + 1].content['p3d'].xyz) ** 2.))
    return distance


def get_node_length_swc(node):
    """
    Calculates the distance between the center points of an SNode2 node and its parent.
    :param node: :class:'SNode2'
    :return: float
    """
    if not node.parent is None:
        return np.sqrt(np.sum((node.content['p3d'].xyz - node.parent.content['p3d'].xyz) ** 2.))
    else:
        return 0.


# ------------------------------------ Methods to specify cell biophysics -------------------------------------------- #


def import_mech_dict_from_file(cell, mech_file_path=None):
    """
    Imports from a .yaml file a dictionary specifying parameters of NEURON cable properties, density mechanisms, and
    point processes for each type of section in a BiophysCell.
    :param cell: :class:'BiophysCell'
    :param mech_file_path: str (path)
    """
    if mech_file_path is None:
        if cell.mech_file_path is None:
            raise ValueError('import_mech_dict_from_file: missing mech_file_path')
        elif not os.path.isfile(cell.mech_file_path):
            raise IOError('import_mech_dict_from_file: invalid mech_file_path: %s' % cell.mech_file_path)
    elif not os.path.isfile(mech_file_path):
        raise IOError('import_mech_dict_from_file: invalid mech_file_path: %s' % mech_file_path)
    else:
        cell.mech_file_path = mech_file_path
    cell.mech_dict = read_from_yaml(cell.mech_file_path)


def export_mech_dict(cell, mech_file_path=None, output_dir=None):
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
    write_to_yaml(mech_file_path, cell.mech_dict, convert_scalars=True)
    print 'Exported mechanism dictionary to %s' % mech_file_path


def init_biophysics(cell, env=None, mech_file_path=None, reset_cable=True, from_file=False, correct_cm=False,
                    correct_g_pas=False):
    """
    Consults a dictionary specifying parameters of NEURON cable properties, density mechanisms, and point processes for
    each type of section in a BiophysCell. Traverses through the tree of SHocNode nodes following order of inheritance.
    Sets membrane mechanism parameters, including gradients and inheritance of parameters from nodes along the path from
    root. Warning! Do not reset cable after inserting synapses!
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param mech_file_path: str (path)
    :param reset_cable: bool
    :param from_file: bool
    :param correct_cm: bool
    :param correct_g_pas: bool
    """
    if from_file:
        import_mech_dict_from_file(cell, mech_file_path)
    if (correct_cm or correct_g_pas) and env is None:
        raise ValueError('init_biophysics: missing Env object; required to parse network configuration and count '
                         'synapses.')
    if reset_cable:
        for sec_type in default_ordered_sec_types:
            if sec_type in cell.mech_dict and sec_type in cell.nodes:
                for node in cell.nodes[sec_type]:
                    reset_cable_by_node(cell, node)
    if correct_cm:
        correct_cell_for_spines_cm(cell, env)
    else:
        for sec_type in default_ordered_sec_types:
            if sec_type in cell.mech_dict and sec_type in cell.nodes:
                if cell.nodes[sec_type]:
                    update_biophysics_by_sec_type(cell, sec_type)
    if correct_g_pas:
        correct_cell_for_spines_g_pas(cell, env)


def reset_cable_by_node(cell, node):
    """
    Consults a dictionary specifying parameters of NEURON cable properties such as axial resistance ('Ra'),
    membrane specific capacitance ('cm'), and a spatial resolution parameter to specify the number of separate
    segments per section in a BiophysCell
    :param cell: :class:'BiophysCell'
    :param node: :class:'SHocNode'
    """
    sec_type = node.type
    if sec_type in cell.mech_dict and 'cable' in cell.mech_dict[sec_type]:
        mech_content = cell.mech_dict[sec_type]['cable']
        if mech_content is not None:
            update_mechanism_by_node(cell, node, 'cable', mech_content)
    else:
        init_nseg(node.sec)
        node.reinit_diam()


def count_spines_per_seg(node, env, gid):
    """

    :param node: :class:'SHocNode'
    :param env: :class:'Env'
    :param gid: int
    """
    syn_attrs = env.synapse_attributes
    syn_id_attr_dict = syn_attrs.syn_id_attr_dict[gid]
    sec_index_map = env.synapse_attributes.sec_index_map[gid]
    node.content['spine_count'] = []
    sec_syn_indexes = np.array(sec_index_map[node.index])
    if len(sec_syn_indexes > 0):
        filtered_syn_indexes = syn_attrs.get_filtered_syn_indexes(gid, sec_syn_indexes,
                                                                  syn_types=[env.Synapse_Types['excitatory']])
        this_syn_locs = syn_id_attr_dict['syn_locs'][filtered_syn_indexes]
        seg_width = 1. / node.sec.nseg
        for i, seg in enumerate(node.sec):
            num_spines = len(np.where((this_syn_locs >= i * seg_width) & (this_syn_locs < (i + 1) * seg_width))[0])
            node.content['spine_count'].append(num_spines)
    else:
        node.content['spine_count'] = [0] * node.sec.nseg


def correct_node_for_spines_g_pas(node, env, gid, soma_g_pas, verbose=False):
    """
    If not explicitly modeling spine compartments for excitatory synapses, this method scales g_pas in this
    dendritic section proportional to the number of excitatory synapses contained in the section.
    :param node: :class:'SHocNode'
    :param env: :class:'Env'
    :param gid: int
    :param soma_g_pas: float
    :param verbose: bool
    """
    SA_spine = math.pi * (1.58 * 0.077 + 0.5 * 0.5)
    if len(node.spine_count) != node.sec.nseg:
        count_spines_per_seg(node, env, gid)
    for i, segment in enumerate(node.sec):
        SA_seg = segment.area()
        num_spines = node.spine_count[i]

        g_pas_correction_factor = (SA_seg * node.sec(segment.x).g_pas + num_spines * SA_spine * soma_g_pas) / \
                                 (SA_seg * node.sec(segment.x).g_pas)
        node.sec(segment.x).g_pas *= g_pas_correction_factor
        if verbose:
            print 'g_pas_correction_factor for %s seg %i: %.3f' % (node.name, i, g_pas_correction_factor)


def correct_node_for_spines_cm(node, env, gid, verbose=False):
    """
    If not explicitly modeling spine compartments for excitatory synapses, this method scales cm in this
    dendritic section proportional to the number of excitatory synapses contained in the section.
    :param node: :class:'SHocNode'
    :param env:  :class:'Env'
    :param gid: int
    :param verbose: bool
    """
    # arrived at via optimization. spine neck appears to shield dendrite from spine head contribution to membrane
    # capacitance and time constant:
    cm_fraction = 0.40
    SA_spine = math.pi * (1.58 * 0.077 + 0.5 * 0.5)
    if len(node.spine_count) != node.sec.nseg:
        count_spines_per_seg(node, env, gid)
    for i, segment in enumerate(node.sec):
        SA_seg = segment.area()
        num_spines = node.spine_count[i]
        cm_correction_factor = (SA_seg + cm_fraction * num_spines * SA_spine) / SA_seg
        node.sec(segment.x).cm *= cm_correction_factor
        if verbose:
            print 'cm_correction_factor for %s seg %i: %.3f' % (node.name, i, cm_correction_factor)


def correct_cell_for_spines_g_pas(cell, env):
    """
    If not explicitly modeling spine compartments for excitatory synapses, this method scales g_pas in all
    dendritic sections proportional to the number of excitatory synapses contained in each section.
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    """
    soma_g_pas = cell.mech_dict['soma']['pas']['g']['value']
    for sec_type in ['basal', 'trunk', 'apical', 'tuft']:
        for node in cell.nodes[sec_type]:
            correct_node_for_spines_g_pas(node, env, cell.gid, soma_g_pas)


def correct_cell_for_spines_cm(cell, env):
    """

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    """
    loop = 0
    while loop < 2:
        for sec_type in ['basal', 'trunk', 'apical', 'tuft']:
            for node in cell.nodes[sec_type]:
                correct_node_for_spines_cm(node, env, cell.gid)
                if loop == 0:
                    init_nseg(node.sec)  # , verbose=True)
                    node.reinit_diam()
        loop += 1
    init_biophysics(cell, env, reset_cable=False)


def update_biophysics_by_sec_type(cell, sec_type, reset_cable=False):
    """
    This method loops through all sections of the specified type, and consults the mechanism dictionary to update
    mechanism properties. If the reset_cable flag is True, cable parameters are re-initialized first, then the
    ion channel mechanisms are updated.
    :param cell: :class:'BiophysCell'
    :param sec_type: str
    :param reset_cable: bool
    """
    if sec_type in cell.nodes:
        if reset_cable:
            # cable properties must be set first, as they can change nseg, which will affect insertion of membrane
            # mechanism gradients
            for node in cell.nodes[sec_type]:
                reset_cable_by_node(cell, node)
        if sec_type in cell.mech_dict:
            for node in cell.nodes[sec_type]:
                for mech_name in (mech_name for mech_name in cell.mech_dict[sec_type]
                                  if mech_name not in ['cable', 'ions', 'synapses']):
                    update_mechanism_by_node(cell, node, mech_name, cell.mech_dict[sec_type][mech_name])
                # ion-related parameters do not exist until after membrane mechanisms have been inserted
                if 'ions' in cell.mech_dict[sec_type]:
                    update_mechanism_by_node(cell, node, 'ions', cell.mech_dict[sec_type]['ions'])


def update_mechanism_by_sec_type(cell, sec_type, mech_name):
    """
    During parameter optimization, it is often convenient to reinitialize all the parameters for a single mechanism
    in a subset of compartments. For example, g_pas in basal dendrites that inherit the value from the soma after
    modifying the value in the soma compartment.
    :param cell: :class:'BiophysCell'
    :param sec_type: str
    :param mech_name: str
    :return:
    """
    if sec_type in cell.nodes and sec_type in cell.mech_dict and mech_name in cell.mech_dict[sec_type]:
        for node in cell.nodes[sec_type]:
            update_mechanism_by_node(cell, node, mech_name, cell.mech_dict[sec_type][mech_name])


def get_mech_rules_dict(cell, **rules):
    """
    Used by modify_mech_param. Takes in a series of arguments and constructs a validated rules dictionary that will be
    used to update a cell's mechanism dictionary.
    :param cell: :class:'BiophysCell'
    :param rules: dict
    :return: dict
    """
    rules_dict = {name: rules[name] for name in
                  (name for name in ['value', 'origin', 'slope', 'tau', 'xhalf', 'min', 'max', 'min_loc', 'max_loc',
                                     'outside', 'custom'] if name in rules and rules[name] is not None)}
    if 'origin' in rules_dict:
        origin_type = rules_dict['origin']
        valid_sec_types = [sec_type for sec_type in cell.nodes if len(cell.nodes[sec_type]) > 0]
        if origin_type not in valid_sec_types + ['parent', 'branch_origin']:
            raise ValueError('modify_mech_param: cannot inherit from invalid origin type: %s' % origin_type)
    return rules_dict


def modify_mech_param(cell, sec_type, mech_name, param_name=None, value=None, origin=None, slope=None, tau=None,
                      xhalf=None, min=None, max=None, min_loc=None, max_loc=None, outside=None, custom=None,
                      append=False, verbose=False):
    """
    Modifies or inserts new membrane mechanisms into hoc sections of type sec_type. First updates the mechanism
    dictionary, then sets the corresponding hoc parameters. This method is meant to be called manually during
    initial model specification, or during parameter optimization. For modifications to persist across simulations,
    the mechanism dictionary must be saved to a file using export_mech_dict() and re-imported during BiophysCell
    initialization.
    :param cell: :class:'BiophysCell'
    :param sec_type: str
    :param mech_name: str
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
    :param verbose: bool
    """
    if sec_type not in cell.nodes:
        raise ValueError('modify_mech_param: sec_type: %s not in cell' % sec_type)
    if param_name is None:
        if mech_name in ['cable', 'ions']:
            raise ValueError('modify_mech_param: missing required parameter to modify mechanism: %s in sec_type: %s' %
                             (mech_name, sec_type))
        mech_content = None
    else:
        if value is None and origin is None:
            raise ValueError('modify_mech_param: mechanism: %s; parameter: %s; missing origin or value for '
                             'sec_type: %s' % (mech_name, param_name, sec_type))
        rules = get_mech_rules_dict(cell, value=value, origin=origin, slope=slope, tau=tau, xhalf=xhalf, min=min, max=max,
                                    min_loc=min_loc, max_loc=max_loc, outside=outside, custom=custom)
        mech_content = {param_name: rules}

    backup_mech_dict = copy.deepcopy(cell.mech_dict)

    # No mechanisms have been inserted into this type of section yet
    if sec_type not in cell.mech_dict:
        cell.mech_dict[sec_type] = {mech_name: mech_content}
    # This mechanism has not yet been inserted into this type of section, or has been inserted, but no parameters
    # have been specified
    elif mech_name not in cell.mech_dict[sec_type] or cell.mech_dict[sec_type][mech_name] is None:
        cell.mech_dict[sec_type][mech_name] = mech_content
    elif param_name is not None:
        # This parameter has already been specified, and the user wants to append a new rule set
        if param_name in cell.mech_dict[sec_type][mech_name] and append:
            if isinstance(cell.mech_dict[sec_type][mech_name][param_name], dict):
                cell.mech_dict[sec_type][mech_name][param_name] = \
                    [cell.mech_dict[sec_type][mech_name][param_name], rules]
            elif isinstance(cell.mech_dict[sec_type][mech_name][param_name], list):
                cell.mech_dict[sec_type][mech_name][param_name].append(rules)
        # This mechanism has been inserted, but this parameter has not yet been specified,
        # or the user wants to replace an existing rule set
        else:
            cell.mech_dict[sec_type][mech_name][param_name] = rules

    try:
        # all membrane mechanisms in sections of type sec_type must be reinitialized after changing cable properties
        if mech_name == 'cable':
            if param_name in ['Ra', 'cm', 'spatial_res']:
                update_biophysics_by_sec_type(cell, sec_type, reset_cable=True)
            else:
                raise AttributeError('modify_mech_param: unknown cable property: %s' % param_name)
        else:
            for node in cell.nodes[sec_type]:
                try:
                    update_mechanism_by_node(cell, node, mech_name, mech_content)
                except (AttributeError, NameError, ValueError, KeyError, RuntimeError, IOError) as e:
                    raise RuntimeError, e, sys.exc_info()[2]
    except RuntimeError as e:
        cell.mech_dict = copy.deepcopy(backup_mech_dict)
        if param_name is not None:
            raise RuntimeError, 'modify_mech_param: problem modifying mechanism: %s parameter: %s in node: %s\n%s' % \
                               (mech_name, param_name, node.name, e), sys.exc_info()[2]
        else:
            raise RuntimeError, 'modify_mech_param: problem modifying mechanism: %s in node: %s\n%s' % \
                                (mech_name, node.name, e), sys.exc_info()[2]


def update_mechanism_by_node(cell, node, mech_name, mech_content):
    """
    This method loops through all the parameters for a single mechanism specified in the mechanism dictionary and
    calls parse_mech_rules to interpret the rules and set the values for the given node.
    :param cell: :class:'BiophysCell'
    :param node: :class:'SHocNode'
    :param mech_name: str
    :param mech_content: dict
    """
    if mech_content is not None:
        for param_name in mech_content:
            # accommodate either a dict, or a list of dicts specifying multiple location constraints for
            # a single parameter
            if isinstance(mech_content[param_name], dict):
                parse_mech_rules(cell, node, mech_name, param_name, mech_content[param_name])
            elif isinstance(mech_content[param_name], list):
                for mech_content_entry in mech_content[param_name]:
                    parse_mech_rules(cell, node, mech_name, param_name, mech_content_entry)
    else:
        node.sec.insert(mech_name)


def get_donor(cell, node, origin_type):
    """

    :param cell: :class:'BiophysCell'
    :param node: :class:'SHocNode'
    :param origin_type: str
    :return: :class:'SHocNode'
    """
    if origin_type == 'parent':
        if node.type == 'spine_head':
            donor = node.parent.parent.parent
        elif node.type == 'spine_neck':
            donor = node.parent.parent
        else:
            donor = node.parent
    elif origin_type == 'branch_origin':
        donor = get_dendrite_origin(cell, node)
    elif origin_type in cell.nodes:
        donor = get_node_along_path_to_root(cell, node, origin_type)
    else:
        raise ValueError('get_donor: unknown origin_type: %s' % origin_type)
    return donor


def parse_mech_rules(cell, node, mech_name, param_name, rules, donor=None):
    """
    Provided a membrane density mechanism, a parameter, a node, and a dict of rules. Interprets the provided rules,
    including complex gradient and inheritance rules. Gradients can be specified as linear, exponential, or sigmoidal.
    Custom functions can also be provided to specify more complex distributions. Calls inherit_mech_param to retrieve a
    value from a donor node, if necessary. Calls set_mech_param to set the values of the actual hoc membrane density
    mechanisms.
    1) A 'value' with no 'origin' requires no further processing
    2) An 'origin' with no 'value' requires a donor node to inherit a baseline value
    3) An 'origin' with a 'value' requires a donor node to use as a reference point for applying a distance-dependent
    gradient
    :param cell: :class:'BiophysCell'
    :param node: :class:'SHocNode'
    :param mech_name: str
    :param param_name: str
    :param rules: dict
    """
    if 'origin' in rules and donor is None:
        donor = get_donor(cell, node, rules['origin'])
        if donor is None:
            raise RuntimeError('parse_syn_mech_rules: problem identifying donor of origin_type: %s for mechanism: '
                               '%s parameter: %s from origin: %s in sec_type: %s' %
                               (rules['origin'], mech_name, param_name, node.type))
    if 'value' in rules:
        baseline = rules['value']
    elif donor is None:
        raise RuntimeError('parse_mech_rules: cannot set mechanism: %s parameter: %s in sec_type: %s without a '
                           'specified origin or value' % (mech_name, param_name, node.type))
    else:
        if (mech_name == 'cable') and (param_name == 'spatial_res'):
            baseline = get_spatial_res(cell, donor)
        else:
            baseline = inherit_mech_param(donor, mech_name, param_name)
    if mech_name == 'cable':  # cable properties can be inherited, but cannot be specified as gradients
        if param_name == 'spatial_res':
            init_nseg(node.sec, baseline)
        else:
            setattr(node.sec, param_name, baseline)
            init_nseg(node.sec, get_spatial_res(cell, node))
        node.reinit_diam()
    elif 'custom' in rules:
        parse_custom_mech_rules(cell, node, mech_name, param_name, baseline, rules, donor)
    else:
        set_mech_param(cell, node, mech_name, param_name, baseline, rules, donor)


def inherit_mech_param(donor, mech_name, param_name):
    """
    When the mechanism dictionary specifies that a node inherit a parameter value from a donor node, this method
    returns the value of the requested parameter from the segment closest to the end of the section.
    :param donor: :class:'SHocNode'
    :param mech_name: str
    :param param_name: str
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
        else:
            return getattr(getattr(donor.sec(loc), mech_name), param_name)
    except (AttributeError, NameError, KeyError, ValueError, RuntimeError, IOError) as e:
        raise RuntimeError, 'inherit_mech_param: problem inheriting mechanism: %s parameter: %s from ' \
                            'sec_type: %s\n%s' % (mech_name, param_name, donor.type, e), sys.exc_info()[2]


def set_mech_param(cell, node, mech_name, param_name, baseline, rules, donor=None):
    """

    :param node: :class:'SHocNode'
    :param mech_name: str
    :param param_name: str
    :param baseline: float
    :param rules: dict
    :param donor: :class:'SHocNode' or None
    """
    if not ('min_loc' in rules or 'max_loc' in rules or 'slope' in rules):
        if mech_name == 'ions':
            setattr(node.sec, param_name, baseline)
        else:
            node.sec.insert(mech_name)
            setattr(node.sec, param_name + "_" + mech_name, baseline)
    elif donor is None:
        raise RuntimeError('set_mech_param: cannot set value of mechanism: %s parameter: %s in sec_type: %s '
                           'without a provided origin' % (mech_name, param_name, node.type))
    else:
        min_distance = rules['min_loc'] if 'min_loc' in rules else 0.
        max_distance = rules['max_loc'] if 'max_loc' in rules else None
        min_val = rules['min'] if 'min' in rules else None
        max_val = rules['max'] if 'max' in rules else None
        slope = rules['slope'] if 'slope' in rules else None
        tau = rules['tau'] if 'tau' in rules else None
        xhalf = rules['xhalf'] if 'xhalf' in rules else None
        outside = rules['outside'] if 'outside' in rules else None

        # No need to insert the mechanism into the section if no segment matches location constraints
        min_seg_distance = get_distance_to_node(cell, donor, node, 0.5 / node.sec.nseg)
        max_seg_distance = get_distance_to_node(cell, donor, node, (0.5 + node.sec.nseg - 1) / node.sec.nseg)
        if (min_distance is None or max_seg_distance > min_distance) and \
                (max_distance is None or min_seg_distance <= max_distance):
            # insert the mechanism first
            if not mech_name == 'ions':
                node.sec.insert(mech_name)
            if min_distance is None:
                min_distance = 0.
            for seg in node.sec:
                distance = get_distance_to_node(cell, donor, node, seg.x)
                value = get_param_val_by_distance(distance, baseline, slope, min_distance, max_distance, min_val, max_val,
                                  tau, xhalf, outside)
                if value is not None:
                    if mech_name == 'ions':
                        setattr(seg, param_name, value)
                    else:
                        setattr(getattr(seg, mech_name), param_name, value)


def get_param_val_by_distance(distance, baseline, slope, min_distance, max_distance=None, min_val=None, max_val=None,
                              tau=None, xhalf=None, outside=None):
    """
    By default, if only some segments or synapses in a section meet the location constraints, the parameter inherits the
    mechanism's default value. if another value is desired, it can be specified via an 'outside' key in the mechanism
    dictionary entry.
    :param distance: float
    :param baseline: float
    :param slope: float
    :param min_distance: float
    :param max_distance: float
    :param min_val: float
    :param max_val: float
    :param tau: float
    :param xhalf: float
    :param outside: float
    :return: float
    """
    if distance > min_distance and (max_distance is None or distance <= max_distance):
        if slope is not None:
            distance -= min_distance
            if tau is not None:
                if xhalf is not None:  # sigmoidal gradient
                    offset = baseline - slope / (1. + np.exp(xhalf / tau))
                    value = offset + slope / (1. + np.exp((xhalf - distance) / tau))
                else:  # exponential gradient
                    offset = baseline - slope
                    value = offset + slope * np.exp(distance / tau)
            else:  # linear gradient
                value = baseline + slope * distance
            if min_val is not None and value < min_val:
                value = min_val
            elif max_val is not None and value > max_val:
                value = max_val
        else:
            value = baseline
    elif outside is not None:
        value = outside
    else:
        value = None
    return value


def zero_na(cell):
    """
    Set na channel conductances to zero in all compartments. Used during parameter optimization.
    """
    for sec_type in (sec_type for sec_type in default_ordered_sec_types if sec_type in cell.nodes and
                     sec_type in cell.mech_dict):
        for na_type in (na_type for na_type in ['nas', 'nax'] if na_type in cell.mech_dict[sec_type]):
            modify_mech_param(cell, sec_type, na_type, 'gbar', 0.)


# --------------------------- Custom methods to specify subcellular mechanism gradients ------------------------------ #


def parse_custom_mech_rules(cell, node, mech_name, param_name, baseline, rules, donor):
    """
    If the provided node meets custom criteria, rules are modified and passed back to parse_mech_rules with the
    'custom' item removed. Avoids having to determine baseline and donor over again.
    :param cell: :class:'BiophysCell'
    :param node: :class:'SHocNode'
    :param mech_name: str
    :param param_name: str
    :param baseline: float
    :param rules: dict
    :param donor: :class:'SHocNode' or None
    """
    if 'func' not in rules['custom'] or rules['custom']['func'] is None:
        raise RuntimeError('parse_custom_mech_rules: no custom function provided for mechanism: %s parameter: %s in '
                           'sec_type: %s' % (mech_name, param_name, node.type))
    if rules['custom']['func'] in globals() and callable(globals()[rules['custom']['func']]):
        func = globals()[rules['custom']['func']]
    else:
        raise RuntimeError('parse_custom_mech_rules: problem locating custom function: %s for mechanism: %s '
                           'parameter: %s in sec_type: %s' %
                           (rules['custom']['func'], mech_name, param_name, node.type))
    custom = copy.deepcopy(rules['custom'])
    del custom['func']
    new_rules = copy.deepcopy(rules)
    del new_rules['custom']
    new_rules['value'] = baseline
    new_rules = func(cell, node, baseline, new_rules, donor, **custom)
    if new_rules:
        parse_mech_rules(cell, node, mech_name, param_name, new_rules, donor)


def custom_filter_by_branch_order(cell, node, baseline, rules, donor, branch_order, **kwargs):
    """
    Allows the provided rule to be applied if the provided node meets specified branch order criterion.
    :param cell: :class:'BiophysCell'
    :param node: :class:'SHocNode'
    :param baseline: float
    :param rules: dict
    :param donor: :class:'SHocNode' or None
    :param branch_order: int or float
    :return: dict or False
    """
    if branch_order is None:
        raise RuntimeError('custom_filter_by_branch_order: custom argument: branch_order not provided for sec_type: '
                           '%s' % node.type)
    branch_order = int(branch_order)
    if get_branch_order(cell, node) < branch_order:
        return False
    return rules


def custom_filter_by_terminal(cell, node, baseline, rules, donor, **kwargs):
    """
    Allows the provided rule to be applied if the provided node is a terminal branch. Adjusts the specified slope based
    on the length of the associated section.
    :param cell: :class:'BiophysCell'
    :param node: :class:'SHocNode'
    :param baseline: float
    :param rules: dict
    :param donor: :class:'SHocNode' or None
    :return: dict or False
    """
    if not is_terminal(node):
        return False
    start_val = baseline
    if 'min' in rules:
        end_val = rules['min']
        direction = -1
    elif 'max' in rules:
        end_val = rules['max']
        direction = 1
    else:
        raise RuntimeError('custom_filter_by_terminal: no min or max target value specified for sec_type: %s' %
                           node.type)
    slope = (end_val - start_val)/node.sec.L
    if 'slope' in rules:
        if direction < 0.:
            slope = min(rules['slope'], slope)
        else:
            slope = max(rules['slope'], slope)
    rules['slope'] = slope
    return rules


# ------------------- Methods to specify cells from hoc templates and neuroh5 trees ---------------------------------- #

def make_neurotree_graph(neurotree_dict, return_root=True):
    """
    Creates a graph of sections that follows the topological organization of the given neuron.
    :param neurotree_dict:
    :return: NetworkX.DiGraph
    """

    sec_nodes = neurotree_dict['section_topology']['nodes']
    sec_src   = neurotree_dict['section_topology']['src']
    sec_dst   = neurotree_dict['section_topology']['dst']

    sec_graph = nx.DiGraph()
    for i, j in itertools.izip(sec_src, sec_dst):
        sec_graph.add_edge(i, j)

    root=None
    if return_root:
        order = nx.topological_sort(sec_graph)
        root = order[0]

    if root:
        return (sec_graph, root)
    else:
        return sec_graph
    

def make_neurotree_cell(template_class, local_id=0, gid=0, dataset_path="", neurotree_dict={}):
    """

    :param template_class:
    :param local_id:
    :param gid:
    :param dataset_path:
    :param neurotree_dict:
    :return: hoc cell object
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
    cell     = template_class(local_id, gid, dataset_path, vlayer, vsrc, vdst, secnodes, vx, vy, vz, vradius, swc_type)
    return cell


def make_cell(template_class, local_id=0, gid=0, dataset_path=""):
    """

    :param template_class:
    :param local_id:
    :param gid:
    :param dataset_path:
    :return: hoc cell object
    """
    cell = template_class(local_id, gid, dataset_path)
    return cell
