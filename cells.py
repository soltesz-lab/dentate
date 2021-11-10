import collections, os, sys, traceback, copy, datetime, math, pprint
import numpy as np
from dentate.neuron_utils import h, d_lambda, default_hoc_sec_lists, default_ordered_sec_types, freq, make_rec, \
    load_cell_template, HocCellInterface, IzhiCellAttrs, default_izhi_cell_attrs_dict, PRconfig
from dentate.utils import get_module_logger, map, range, zip, zip_longest, viewitems, read_from_yaml, write_to_yaml, Promise
from neuroh5.io import read_cell_attribute_selection, read_graph_selection, read_tree_selection

# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)

##
## SNode2/STree2 structures from btmorph by B.T.Nielsen.
##
class P3D2(object):
    """
    Basic container to represent and store 3D information
    """

    def __init__(self, xyz, radius, type=7):
        """ Constructor.

        Parameters
        -----------

        xyz : numpy.array
            3D location
        radius : float
        type : int
            Type associated with the segment according to SWC standards
        """
        self.xyz = xyz
        self.radius = radius
        self.type = type

    def __str__(self):
        return "P3D2 [%.2f %.2f %.2f], R=%.2f" % (self.xyz[0], self.xyz[1], self.xyz[2], self.radius)


class SNode2(object):
    """
    Simple Node for use with a simple Tree (STree)
    
    By design, the "content" should be a dictionary. (2013-03-08)
    """

    def __init__(self, index):
        """
        Constructor.

        Parameters
        -----------
        index : int
           Index, unique name of the :class:`SNode2`
        """
        self.parent = None
        self.index = index
        self.children = []
        self.content = {}

    def get_parent(self):
        """
        Return the parent node of this one.

        Returns
        -------
        parent : :class:`SNode2`
           In case of the root, None is returned.Otherwise a :class:`SNode2` is returned
        """
        return self.__parent

    def set_parent(self, parent):
        """
        Set the parent node of a given other node

        Parameters
        ----------
        node : :class:`SNode2`
        """
        self.__parent = parent

    parent = property(get_parent, set_parent)

    def get_index(self):
        """
        Return the index of this node

        Returns
        -------
        index : int
        """
        return self.__index

    def set_index(self, index):
        """
        Set the unqiue name of a node

        Parameters
        ----------

        index : int
        """
        self.__index = index

    index = property(get_index, set_index)

    def get_children(self):
        """
        Return the children nodes of this one (if any)

        Returns
        -------
        children : list :class:`SNode2`
           In case of a leaf an empty list is returned
        """
        return self.__children

    def set_children(self, children):
        """
        Set the children nodes of this one

        Parameters
        ----------

        children: list :class:`SNode2`
        """
        self.__children = children

    children = property(get_children, set_children)

    def get_content(self):
        """
        Return the content dict of a :class:`SNode2`

        Returns
        -------
        parent : :class:`SNode2`
           In case of the root, None is returned.Otherwise a :class:`SNode2` is returned
        """
        return self.__content

    def set_content(self, content):
        """
        Set the content of a node. The content must be a dict

        Parameters
        ----------
        content : dict
            dict with content. For use in btmorph at least a 'p3d' entry should be present
        """
        if isinstance(content, dict):
            self.__content = content
        else:
            raise Exception("SNode2.set_content must receive a dict")

    content = property(get_content, set_content)

    def add_child(self, child_node):
        """
        add a child to the children list of a given node

        Parameters
        -----------
        node :  :class:`SNode2`
        """
        self.children.append(child_node)

    def make_empty(self):
        """
        Clear the node. Unclear why I ever implemented this. Probably to cover up some failed garbage collection
        """
        self.parent = None
        self.content = {}
        self.children = []

    def remove_child(self, child):
        """
        Remove a child node from the list of children of a specific node

        Parameters
        -----------
        node :  :class:`SNode2`
            If the child doesn't exist, you get into problems.
        """
        self.children.remove(child)

    def __str__(self):
        return 'SNode2 (ID: ' + str(self.index) + ')'

    def __lt__(self, other):
        if self.index < other.index:
            return True

    def __le__(self, other):
        if self.index <= other.index:
            return True

    def __gt__(self, other):
        if self.index > other.index:
            return True

    def __ge__(self, other):
        if self.index >= other.index:
            return True

    def __copy__(self):  # customization of copy.copy
        ret = SNode2(self.index)
        for child in self.children:
            ret.add_child(child)
        ret.content = self.content
        ret.parent = self.parent
        return ret


class STree2(object):
    '''
    Simple tree for use with a simple Node (:class:`SNode2`).

    While the class is designed to contain binary trees (for neuronal morphologies) the number of children is not limited.
    As such, this is a generic implementation of a tree structure as a linked list.
    '''

    def __init__(self):
        """
        Default constructor. No arguments are passed.
        """
        self.root = None

    def __iter__(self):
        nodes = []
        self._gather_nodes(self.root, nodes)
        for n in nodes:
            yield n

    def __getitem__(self, index):
        return self._find_node(self.root, index)

    def set_root(self, node):
        """
        Set the root node of the tree

        Parameters
        -----------
        node : :class:`SNode2`
            to-be-root node
        """
        if not node is None:
            node.parent = None
        self.__root = node

    def get_root(self):
        """
        Obtain the root node

        Returns
        -------
        root : :class:`SNode2`
        """
        return self.__root

    root = property(get_root, set_root)

    def is_root(self, node):
        """
        Check whether a node is the root node

        Returns
        --------
        is_root : boolean
            True is the queried node is the root, False otherwise
        """
        if node.parent is None:
            return True
        else:
            return False

    def is_leaf(self, node):
        """
        Check whether a node is a leaf node, i.e., a node without children

        Returns
        --------
        is_leaf : boolean
            True is the queried node is a leaf, False otherwise
        """
        if len(node.children) == 0:
            return True
        else:
            return False

    def add_node_with_parent(self, node, parent):
        """
        Add a node to the tree under a specific parent node

        Parameters
        -----------
        node : :class:`SNode2`
            node to be added
        parent : :class:`SNode2`
            parent node of the newly added node
        """
        node.parent = parent
        if not parent is None:
            parent.add_child(node)

    def remove_node(self, node):
        """
        Remove a node from the tree

        Parameters
        -----------
        node : :class:`SNode2`
            node to be removed
        """
        node.parent.remove_child(node)
        self._deep_remove(node)

    def _deep_remove(self, node):
        children = node.children
        node.make_empty()
        for child in children:
            self._deep_remove(child)

    def get_nodes(self):
        """
        Obtain a list of all nodes int the tree

        Returns
        -------
        all_nodes : list of :class:`SNode2`
        """
        n = []
        self._gather_nodes(self.root, n)
        return n

    def get_sub_tree(self, fake_root):
        """
        Obtain the subtree starting from the given node

        Parameters
        -----------
        fake_root : :class:`SNode2`
            Node which becomes the new root of the subtree

        Returns
        -------
        sub_tree :  STree2
            New tree with the node from the first argument as root node
        """
        ret = STree2()
        cp = fake_root.__copy__()
        cp.parent = None
        ret.root = cp
        return ret

    def _gather_nodes(self, node, node_list):
        if not node is None:
            node_list.append(node)
            for child in node.children:
                self._gather_nodes(child, node_list)

    def get_node_with_index(self, index):
        """
        Get a node with a specific name. The name is always an integer

        Parameters
        ----------
        index : int
            Name of the node to be found

        Returns
        -------
        node : :class:`SNode2`
            Node with the specific index
        """
        return self._find_node(self.root, index)

    def get_node_in_subtree(self, index, fake_root):
        """
        Get a node with a specific name in a the subtree rooted at fake_root. The name is always an integer

        Parameters
        ----------
        index : int
            Name of the node to be found
        fake_root: :class:`SNode2`
            Root node of the subtree in which the node with a given index is searched for

        Returns
        -------
        node : :class:`SNode2`
            Node with the specific index
        """
        return self._find_node(fake_root, index)

    def _find_node(self, node, index):
        """
        Sweet breadth-first/stack iteration to replace the recursive call. 
        Traverses the tree until it finds the node you are looking for.

        Parameters
        -----------

        
        Returns
        -------
        node : :class:`SNode2`
            when found and None when not found
        """
        stack = [];
        stack.append(node)
        while (len(stack) != 0):
            for child in stack:
                if child.index == index:
                    return child
                else:
                    stack.remove(child)
                    for cchild in child.children:
                        stack.append(cchild)
        return None  # Not found!

    def degree_of_node(self, node):
        """
        Get the degree of a given node. The degree is defined as the number of leaf nodes in the subtree rooted at this node.

        Parameters
        ----------
        node : :class:`SNode2`
            Node of which the degree is to be computed.

        Returns
        -------
        degree : int
        """
        sub_tree = self.get_sub_tree(node)
        st_nodes = sub_tree.get_nodes()
        leafs = 0
        for n in st_nodes:
            if sub_tree.is_leaf(n):
                leafs = leafs + 1
        return leafs

    def order_of_node(self, node):
        """
        Get the order of a given node. The order or centrifugal order is defined as 0 for the root and increased with any bifurcation.
        Hence, a node with 2 branch points on the shortest path between that node and the root has order 2.

        Parameters
        ----------
        node : :class:`SNode2`
            Node of which the order is to be computed.

        Returns
        -------
        order : int
        """
        ptr = self.path_to_root(node)
        order = 0
        for n in ptr:
            if len(n.children) > 1:
                order = order + 1
        # order is on [0,max_order] thus subtract 1 from this calculation
        return order - 1

    def path_to_root(self, node):
        """
        Find and return the path between a node and the root.

        Parameters
        ----------
        node : :class:`SNode2`
            Node at which the path starts

        Returns
        -------
        path : list of :class:`SNode2`
            list of :class:`SNode2` with the provided node and the root as first and last entry, respectively.
        """
        n = []
        self._go_up_from(node, n)
        return n

    def _go_up_from(self, node, n):
        n.append(node)
        if not node.parent is None:
            self._go_up_from(node.parent, n)

    def path_between_nodes(self, from_node, to_node):
        """
        Find the path between two nodes. The from_node needs to be of higher \
        order than the to_node. In case there is no path between the nodes, \
        the path from the from_node to the soma is given.

        Parameters
        -----------
        from_node : :class:`SNode2`
        to_node : :class:`SNode2`
        """
        n = []
        self._go_up_from_until(from_node, to_node, n)
        return n

    def _go_up_from_until(self, from_node, to_node, n):
        n.append(from_node)
        if from_node == to_node:
            return
        if not from_node.parent is None:
            self._go_up_from_until(from_node.parent, to_node, n)


    def _make_soma_from_cylinders(self, soma_cylinders, all_nodes):
        """Now construct 3-point soma
        step 1: calculate surface of all cylinders
        step 2: make 3-point representation with the same surface"""

        total_surf = 0
        for (node, parent_index) in soma_cylinders:
            n = node.content["p3d"]
            p = all_nodes[parent_index][1].content["p3d"]
            H = np.sqrt(np.sum((n.xyz - p.xyz) ** 2))
            surf = 2 * np.pi * p.radius * H
            total_surf = total_surf + surf
        logger.error("found 'multiple cylinder soma' w/ total soma surface=*.3f" % total_surf)

        # define apropriate radius
        radius = np.sqrt(total_surf / (4 * np.pi))

        s_node_1 = SNode2(2)
        r = self.root.content["p3d"]
        rp = r.xyz
        s_p_1 = P3D2(np.array([rp[0], rp[1] - radius, rp[2]]), radius, 1)
        s_node_1.content = {'p3d': s_p_1}
        s_node_2 = SNode2(3)
        s_p_2 = P3D2(np.array([rp[0], rp[1] + radius, rp[2]]), radius, 1)
        s_node_2.content = {'p3d': s_p_2}

        return s_node_1, s_node_2

    def _determine_soma_type(self, file_n):
        """
        Costly method to determine the soma type used in the SWC file.
        This method searches the whole file for soma entries.  

        Parameters
        ----------
        file_n : string
            Name of the file containing the SWC description

        Returns
        -------
        soma_type : int
            Integer indicating one of the su[pported SWC soma formats.
            1: Default three-point soma, 2: multiple cylinder description,
            3: otherwise [not suported in btmorph]
        """
        file = open(file_n, "r")
        somas = 0
        for line in file:
            if not line.startswith('#'):
                split = line.split()
                index = int(split[0].rstrip())
                s_type = int(split[1].rstrip())
                if s_type == 1:
                    somas = somas + 1
        file.close()
        if somas == 3:
            return 1
        elif somas < 3:
            return 3
        else:
            return 2

    def __str__(self):
        return "STree2 (" + str(len(self.get_nodes())) + " nodes)"


class PRneuron(object):
    """
    An implementation of a Pinsky-Rinzel-type reduced biophysical neuron model for simulation in NEURON.
    Conforms to the same API as BiophysCell.
    """
    def __init__(self, gid, pop_name, env=None, cell_config=None, mech_dict=None):
        """

        :param gid: int
        :param pop_name: str
        :param env: :class:'Env'
        :param cell_config: :namedtuple:'PRconfig'
        """
        self._gid = gid
        self._pop_name = pop_name
        self.tree = STree2()  # Builds a simple tree to store nodes of type 'SHocNode'
        self.count = 0  # Keep track of number of nodes
        if env is not None:
            for sec_type in env.SWC_Types:
                if sec_type not in default_ordered_sec_types:
                    raise AttributeError('Warning! unexpected SWC Type definitions found in Env')
        self.nodes = {key: [] for key in default_ordered_sec_types}
        self.mech_file_path = None
        self.init_mech_dict = dict(mech_dict) if mech_dict is not None else None
        self.mech_dict = dict(mech_dict) if mech_dict is not None else None
        
        self.random = np.random.RandomState()
        self.random.seed(self.gid)
        self.spike_detector = None
        self.spike_onset_delay = 0.
        self.is_reduced = True
        if not isinstance(cell_config, PRconfig):
            raise RuntimeError('PRneuron: argument cell_attrs must be of type PRconfig')

        param_dict = { 'pp': cell_config.pp,
                       'Ltotal': cell_config.Ltotal,
                       'gc': cell_config.gc,
                       'soma_gmax_Na': cell_config.soma_gmax_Na,
                       'soma_gmax_K': cell_config.soma_gmax_K,
                       'soma_g_pas': cell_config.soma_g_pas,
                       'dend_gmax_Ca': cell_config.dend_gmax_Ca,
                       'dend_gmax_KCa': cell_config.dend_gmax_KCa,
                       'dend_gmax_KAHP': cell_config.dend_gmax_KAHP,
                       'dend_g_pas':  cell_config.dend_g_pas,
                       'dend_d_Caconc':  cell_config.dend_d_Caconc,
                       'cm_ratio':  cell_config.cm_ratio,
                       'global_cm':  cell_config.global_cm,
                       'global_diam':  cell_config.global_diam,
        }

        PR_nrn = h.PR_nrn(param_dict)
        PR_nrn.soma.ic_constant = cell_config.ic_constant

        self.hoc_cell = PR_nrn

        soma_node = append_section(self, 'soma', sec_index=0, sec=PR_nrn.soma)
        apical_node = append_section(self, 'apical', sec_index=1, sec=PR_nrn.dend)
        connect_nodes(self.soma[0], self.apical[0], connect_hoc_sections=False)
        
        init_spike_detector(self, self.tree.root, loc=0.5, threshold=cell_config.V_threshold)


    def update_cell_attrs(self, **kwargs):
        for attr_name, attr_val in kwargs.items():
            if attr_name in PRconfig._fields:
                setattr(self.hoc_cell, attr_name, attr_val)

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

class IzhiCell(object):
    """
    An implementation of an Izhikevich adaptive integrate-and-fire-type cell model for simulation in NEURON.
    Conforms to the same API as BiophysCell.
    """
    def __init__(self, gid, pop_name, env=None, cell_type='RS', cell_attrs=None, mech_dict=None):
        """

        :param gid: int
        :param pop_name: str
        :param env: :class:'Env'
        :param cell_type: str
        :param cell_attrs: :namedtuple:'IzhiCellAttrs'
        """
        self._gid = gid
        self._pop_name = pop_name
        self.tree = STree2()  # Builds a simple tree to store nodes of type 'SHocNode'
        self.count = 0  # Keep track of number of nodes
        if env is not None:
            for sec_type in env.SWC_Types:
                if sec_type not in default_ordered_sec_types:
                    raise AttributeError('Warning! unexpected SWC Type definitions found in Env')
        self.nodes = {key: [] for key in default_ordered_sec_types}
        self.mech_file_path = None
        self.init_mech_dict = dict(mech_dict) if mech_dict is not None else None
        self.mech_dict = dict(mech_dict) if mech_dict is not None else None
        
        self.random = np.random.RandomState()
        self.random.seed(self.gid)
        self.spike_detector = None
        self.spike_onset_delay = 0.
        self.is_reduced = True
        if cell_attrs is not None:
            if not isinstance(cell_attrs, IzhiCellAttrs):
                raise RuntimeError('IzhiCell: argument cell_attrs must be of type IzhiCellAttrs')
            cell_type = 'custom'
        elif cell_type not in default_izhi_cell_attrs_dict:
            raise RuntimeError('IzhiCell: unknown value for cell_type: %s' % str(cell_type))
        else:
            cell_attrs = default_izhi_cell_attrs_dict[cell_type]
        self.cell_type = cell_type

        append_section(self, 'soma')
        sec = self.tree.root.sec
        sec.L, sec.diam = 10., 10.
        self.izh = h.Izhi2019(.5, sec=sec)
        self.base_cm = 31.831  # Produces membrane time constant of 8 ms for a RS cell with izh.C = 1. and izi.k = 0.7
        for attr_name, attr_val in cell_attrs._asdict().items():
            setattr(self.izh, attr_name, attr_val)
        sec.cm = self.base_cm * self.izh.C

        self.hoc_cell = HocCellInterface(sections=[sec], is_art=lambda: 0, is_reduced=True,
                                         all=[sec], soma=[sec], apical=[], basal=[], 
                                         axon=[], ais=[], hillock=[], state=[self.izh])

        init_spike_detector(self, self.tree.root, loc=0.5, threshold=self.izh.vpeak - 1.)


    def update_cell_attrs(self, **kwargs):
        for attr_name, attr_val in kwargs.items():
            if attr_name in IzhiCellAttrs._fields:
                setattr(self.izh, attr_name, attr_val)
            if attr_name == 'C':
                self.tree.root.sec.cm = self.base_cm * attr_val
            elif attr_name == 'vpeak':
                self.spike_detector.threshold = attr_val - 1.

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


class BiophysCell(object):
    """
    A Python wrapper for neuronal cell objects specified in the NEURON language hoc.
    Extends STree to provide an tree interface to facilitate:
    1) Iteration through connected neuronal compartments, and
    2) Specification of complex distributions of compartment attributes like gradients of ion channel density or
    synaptic properties.
    """
    def __init__(self, gid, pop_name, hoc_cell=None, neurotree_dict=None, mech_file_path=None, mech_dict=None, env=None):
        """

        :param gid: int
        :param pop_name: str
        :param hoc_cell: :class:'h.hocObject': instance of a NEURON cell template
        :param mech_file_path: str (path)
        :param env: :class:'Env'
        """
        self._gid = gid
        self._pop_name = pop_name
        self.tree = STree2()  # Builds a simple tree to store nodes of type 'SHocNode'
        self.count = 0  # Keep track of number of nodes
        if env is not None:
            for sec_type in env.SWC_Types:
                if sec_type not in default_ordered_sec_types:
                    raise AttributeError('Unexpected SWC Type definitions found in Env')
        self.nodes = {key: [] for key in default_ordered_sec_types}
        self.mech_file_path = mech_file_path
        self.init_mech_dict = dict(mech_dict) if mech_dict is not None else None
        self.mech_dict = dict(mech_dict) if mech_dict is not None else None
        self.random = np.random.RandomState()
        self.random.seed(self.gid)
        self.spike_detector = None
        self.spike_onset_delay = 0.
        self.hoc_cell = hoc_cell
        if hoc_cell is not None:
            import_morphology_from_hoc(self, hoc_cell)
        elif neurotree_dict is not None:
            import_morphology_from_neurotree_dict(self, neurotree_dict)
            import_morphology_from_hoc(self, hoc_cell)
            
        if (mech_dict is None) and (mech_file_path is not None):
            import_mech_dict_from_file(self, self.mech_file_path)
        elif mech_dict is None:
            # Allows for a cell to be created and for a new mech_dict to be constructed programmatically from scratch
            self.init_mech_dict = dict()
            self.mech_dict = dict()
        init_cable(self)
        init_spike_detector(self)

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


class SHocNode(SNode2):
    """
    Extends SNode2 with some methods for storing and retrieving additional information in the node's content
    dictionary related to running NEURON models specified in the hoc language.
    """

    def __init__(self, index=0):
        """
        :param index: int : unique node identifier
        """
        SNode2.__init__(self, index)
        self.content['spine_count'] = []
        self._connection_loc = None

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
                for i in range(self.sec.n3d()):
                    if (self.sec.arc3d(i) / self.sec.L) >= x:
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
        if self._connection_loc is None:
            if self.type == 'spine_head':
                self.parent.sec.push()
            else:
                self.sec.push()
                loc = h.parent_connection()
                h.pop_section()
            self._connection_loc = loc
        return self._connection_loc


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
    return 1e5 * math.sqrt(diam / (4. * math.pi * f * Ra * cm))


def d_lambda_nseg(sec, lam=d_lambda, f=freq):
    """
    The AC length constant for this section and the user-defined fraction is used to determine the maximum size of each
    segment to achieve the desired spatial and temporal resolution. This method returns the number of segments to set
    the nseg parameter for this section. For tapered cylindrical sections, the diam parameter will need to be
    reinitialized after nseg changes.
    :param sec : :class:'h.Section'
    :param lam : int
    :param f : int
    :return : int
    """
    L = sec.L
    return int(((L / (lam * lambda_f(sec, f))) + 0.9) / 2) * 2 + 1


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
    for sec_type, sec_index_list in viewitems(default_hoc_sec_lists):
        if hasattr(hoc_cell, sec_type) and (getattr(hoc_cell, sec_type) is not None):
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
        logger.error('import_morphology_from_hoc: problem locating soma section to act as root')
        raise e
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
    ps = getattr(sec(loc), param)
    this_netcon = h.NetCon(ps, target, sec=sec)
    this_netcon.delay = delay
    this_netcon.weight[0] = weight
    this_netcon.threshold = threshold
    return this_netcon


def init_spike_detector(cell, node=None, distance=100., threshold=-30, delay=0.05, onset_delay=0., loc=0.5):
    """
    Initializes the spike detector in the given cell according to the
    given arguments or a spike detector configuration of the mechanism
    dictionary of the cell, if one exists.

    :param cell: :class:'BiophysCell'
    :param node: :class:'SHocNode
    :param distance: float
    :param threshold: float
    :param delay: float
    :param onset_delay: float
    :param loc: float
    """
    if cell.mech_dict is not None:
        if 'spike detector' in cell.mech_dict:
            config = cell.mech_dict['spike detector']
            node = getattr(cell, config['section'])[0]
            loc = config['loc']
            distance = config['distance']
            threshold = config['threshold']
            delay = config['delay']
            onset_delay = config['onset delay']

    if node is None:
        if cell.axon:
            for node in cell.axon:
                sec_seg_locs = [seg.x for seg in node.sec]
                for loc in sec_seg_locs:
                    if get_distance_to_node(cell, cell.tree.root, node, loc=loc) >= distance:
                        break
                else:
                    continue
                break
            else:
                node = cell.axon[-1]
                loc = 1.
        elif cell.ais:
            node = cell.ais[0]
        elif cell.soma:
            node = cell.soma[0]
        else:
            raise RuntimeError('init_spike_detector: cell has neither soma nor axon compartment')

    cell.spike_detector = connect2target(cell, node.sec, loc=loc, delay=delay, threshold=threshold)

    cell.onset_delay = onset_delay
            
    return cell.spike_detector
        

def init_nseg(sec, spatial_res=0, verbose=True):
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
        logger.info('init_nseg: changed %s.nseg %i --> %i' % (sec.hname(), sec.nseg, sugg_nseg))
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
    if (node is root) or (node is None):
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
    TODO: this has to gather nodes below the variable root
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
    for i in range(len(path) - 1):
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
    cell.init_mech_dict = read_from_yaml(cell.mech_file_path)
    cell.mech_dict = copy.deepcopy(cell.init_mech_dict)


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
            mech_file_path = output_dir + '/' + mech_file_name
    write_to_yaml(mech_file_path, cell.mech_dict, convert_scalars=True)
    logger.info('Exported mechanism dictionary to %s' % mech_file_path)


def init_cable(cell, verbose=False):
    for sec_type in cell.nodes:
        for node in cell.nodes[sec_type]:
            reset_cable_by_node(cell, node, verbose=verbose)


def init_biophysics(cell, env=None, reset_cable=True, correct_cm=False, correct_g_pas=False, reset_mech_dict=False,
                    verbose=True):
    """
    Consults a dictionary specifying parameters of NEURON cable properties, density mechanisms, and point processes for
    each type of section in a BiophysCell. Traverses through the tree of SHocNode nodes following order of inheritance.
    Sets membrane mechanism parameters, including gradients and inheritance of parameters from nodes along the path from
    root. Warning! Do not reset cable after inserting synapses!
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param reset_cable: bool
    :param correct_cm: bool
    :param correct_g_pas: bool
    :param reset_mech_dict: bool
    :param verbose: bool
    """
    if (correct_cm or correct_g_pas) and env is None:
        raise ValueError('init_biophysics: missing Env object; required to parse network configuration and count '
                         'synapses.')
    if reset_mech_dict:
        cell.mech_dict = copy.deepcopy(cell.init_mech_dict)
    if reset_cable:
        for sec_type in default_ordered_sec_types:
            if sec_type in cell.mech_dict and sec_type in cell.nodes:
                for node in cell.nodes[sec_type]:
                    reset_cable_by_node(cell, node, verbose=verbose)
    if correct_cm:
        correct_cell_for_spines_cm(cell, env, verbose=verbose)
    else:
        for sec_type in default_ordered_sec_types:
            if sec_type in cell.mech_dict and sec_type in cell.nodes:
                if cell.nodes[sec_type]:
                    update_biophysics_by_sec_type(cell, sec_type)
    if correct_g_pas:
        correct_cell_for_spines_g_pas(cell, env, verbose=verbose)

def reset_cable_by_node(cell, node, verbose=True):
    """
    Consults a dictionary specifying parameters of NEURON cable properties such as axial resistance ('Ra'),
    membrane specific capacitance ('cm'), and a spatial resolution parameter to specify the number of separate
    segments per section in a BiophysCell
    :param cell: :class:'BiophysCell'
    :param node: :class:'SHocNode'
    :param verbose: bool
    """
    sec_type = node.type
    if sec_type in cell.mech_dict and 'cable' in cell.mech_dict[sec_type]:
        mech_content = cell.mech_dict[sec_type]['cable']
        if mech_content is not None:
            update_mechanism_by_node(cell, node, 'cable', mech_content, verbose=verbose)
    else:
        init_nseg(node.sec, verbose=verbose)
        node.reinit_diam()


def count_spines_per_seg(node, env, gid):
    """

    :param node: :class:'SHocNode'
    :param env: :class:'Env'
    :param gid: int
    """
    syn_attrs = env.synapse_attributes
    node.content['spine_count'] = []

    filtered_synapses = syn_attrs.filter_synapses(gid, syn_sections=[node.index], \
                                                  syn_types=[env.Synapse_Types['excitatory']])
    if len(filtered_synapses) > 0:
        this_syn_locs = np.asarray([syn.syn_loc for _, syn in viewitems(filtered_synapses)])
        seg_width = 1. / node.sec.nseg
        for i, seg in enumerate(node.sec):
            num_spines = len(np.where((this_syn_locs >= i * seg_width) & (this_syn_locs < (i + 1) * seg_width))[0])
            node.content['spine_count'].append(num_spines)
    else:
        node.content['spine_count'] = [0] * node.sec.nseg


def correct_node_for_spines_g_pas(node, env, gid, soma_g_pas, verbose=True):
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

        g_pas_correction_factor = ((SA_seg * node.sec(segment.x).g_pas + num_spines * SA_spine * soma_g_pas) /
                                   (SA_seg * node.sec(segment.x).g_pas))
        node.sec(segment.x).g_pas *= g_pas_correction_factor
        if verbose:
            logger.info('g_pas_correction_factor for gid: %i; %s seg %i: %.3f' %
                        (gid, node.name, i, g_pas_correction_factor))


def correct_node_for_spines_cm(node, env, gid, verbose=True):
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
            logger.info('cm_correction_factor for gid: %i; %s seg %i: %.3f' % (gid, node.name, i, cm_correction_factor))


def correct_cell_for_spines_g_pas(cell, env, verbose=False):
    """
    If not explicitly modeling spine compartments for excitatory synapses, this method scales g_pas in all
    dendritic sections proportional to the number of excitatory synapses contained in each section.
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param verbose: bool
    """
    if 'soma' in cell.mech_dict:
        soma_g_pas = cell.mech_dict['soma']['pas']['g']['value']
    elif hasattr(cell, 'hoc_cell'): 
        soma_g_pas = getattr(list(cell.hoc_cell.soma)[0], 'g_pas')
    else:
        raise RuntimeError("unable to determine soma g_pas")
    for sec_type in ['basal', 'trunk', 'apical', 'tuft']:
        for node in cell.nodes[sec_type]:
            correct_node_for_spines_g_pas(node, env, cell.gid, soma_g_pas, verbose=verbose)


def correct_cell_for_spines_cm(cell, env, verbose=False):
    """

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param verbose: bool
    """
    loop = 0
    while loop < 2:
        for sec_type in ['basal', 'trunk', 'apical', 'tuft']:
            for node in cell.nodes[sec_type]:
                correct_node_for_spines_cm(node, env, cell.gid, verbose=verbose)
                if loop == 0:
                    init_nseg(node.sec, verbose=verbose)
                    node.reinit_diam()
        loop += 1
    init_biophysics(cell, env, reset_cable=False, verbose=verbose)


def update_biophysics_by_sec_type(cell, sec_type, reset_cable=False, verbose=False):
    """
    This method loops through all sections of the specified type, and consults the mechanism dictionary to update
    mechanism properties. If the reset_cable flag is True, cable parameters are re-initialized first, then the
    ion channel mechanisms are updated.
    :param cell: :class:'BiophysCell'
    :param sec_type: str
    :param reset_cable: bool
    :param verbose: bool
    """
    if sec_type in cell.nodes:
        if reset_cable:
            # cable properties must be set first, as they can change nseg, which will affect insertion of membrane
            # mechanism gradients
            for node in cell.nodes[sec_type]:
                reset_cable_by_node(cell, node, verbose=verbose)
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
                      append=False, verbose=False, **kwargs):
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
        rules = get_mech_rules_dict(cell, value=value, origin=origin, slope=slope, tau=tau, xhalf=xhalf, min=min,
                                    max=max, min_loc=min_loc, max_loc=max_loc, outside=outside, custom=custom, **kwargs)
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
                update_biophysics_by_sec_type(cell, sec_type, reset_cable=True, verbose=verbose)
            else:
                raise AttributeError('modify_mech_param: unknown cable property: %s' % param_name)
        else:
            for node in cell.nodes[sec_type]:
                update_mechanism_by_node(cell, node, mech_name, mech_content, verbose=verbose)

    except Exception as e:
        cell.mech_dict = copy.deepcopy(backup_mech_dict)
        traceback.print_exc(file=sys.stderr)
        if param_name is not None:
            logger.error('modify_mech_param: problem modifying mechanism: %s parameter: %s in node: %s' %
                    (mech_name, param_name, node.name))
        else:
            logger.error('modify_mech_param: problem modifying mechanism: %s in node: %s' % (mech_name, node.name))
        sys.stderr.flush()
        raise e


def update_mechanism_by_node(cell, node, mech_name, mech_content, verbose=True):
    """
    This method loops through all the parameters for a single mechanism specified in the mechanism dictionary and
    calls apply_mech_rules to interpret the rules and set the values for the given node.
    :param cell: :class:'BiophysCell'
    :param node: :class:'SHocNode'
    :param mech_name: str
    :param mech_content: dict
    :param verbose: bool
    """
    if mech_content is not None:
        for param_name in mech_content:
            # accommodate either a dict, or a list of dicts specifying multiple location constraints for
            # a single parameter
            if isinstance(mech_content[param_name], dict):
                apply_mech_rules(cell, node, mech_name, param_name, mech_content[param_name], verbose=verbose)
            elif isinstance(mech_content[param_name], list):
                for mech_content_entry in mech_content[param_name]:
                    apply_mech_rules(cell, node, mech_name, param_name, mech_content_entry, verbose=verbose)
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


def apply_mech_rules(cell, node, mech_name, param_name, rules, donor=None, verbose=True):
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
    :param verbose: bool
    """
    if 'origin' in rules and donor is None:
        if node is None:
            donor = None
        donor = get_donor(cell, node, rules['origin'])
        if donor is None:
            raise RuntimeError('apply_mech_rules: problem identifying donor of origin_type: %s for mechanism: '
                               '%s parameter: %s in sec_type: %s' %
                               (rules['origin'], mech_name, param_name, node.type))
    if 'value' in rules:
        baseline = rules['value']
    elif donor is None:
        raise RuntimeError('apply_mech_rules: cannot set mechanism: %s parameter: %s in sec_type: %s without a '
                           'specified origin or value' % (mech_name, param_name, node.type))
    else:
        if (mech_name == 'cable') and (param_name == 'spatial_res'):
            baseline = get_spatial_res(cell, donor)
        elif node is not None:
            for target_seg in node.sec:
                break
            target_distance = get_distance_to_node(cell, cell.tree.root, node, loc=target_seg.x)
            baseline = inherit_mech_param(cell, donor, mech_name, param_name, target_distance=target_distance)
            if baseline is None:
                inherit_mech_param(cell, node, mech_name, param_name, target_distance=target_distance)

    if mech_name == 'cable':  # cable properties can be inherited, but cannot be specified as gradients
        if param_name == 'spatial_res':
            init_nseg(node.sec, baseline, verbose=verbose)
        else:
            setattr(node.sec, param_name, baseline)
            init_nseg(node.sec, get_spatial_res(cell, node), verbose=verbose)
        node.reinit_diam()
    elif 'custom' in rules:
        apply_custom_mech_rules(cell, node, mech_name, param_name, baseline, rules, donor, verbose=verbose)
    else:
        set_mech_param(cell, node, mech_name, param_name, baseline, rules, donor)


def inherit_mech_param(cell, donor, mech_name, param_name, target_distance=None):
    """
    When the mechanism dictionary specifies that a node inherit a parameter value from a donor node, this method
    returns the value of the requested parameter from the segment closest to the end of the section.
    :param cell: :class:'BiophysCell'
    :param donor: :class:'SHocNode'
    :param mech_name: str
    :param param_name: str
    :param target_distance: float
    :return: float
    """
    if target_distance is None:
        # accesses the last segment of the section
        loc = donor.sec.nseg / (donor.sec.nseg + 1.)
    else:
        locs = [seg.x for seg in donor.sec]
        locs.sort(key=lambda x: abs(target_distance - get_distance_to_node(cell, cell.tree.root, donor, loc=x)))
        loc = locs[0]
    try:
        if mech_name in ['cable', 'ions']:
            if mech_name == 'cable' and param_name == 'Ra':
                return getattr(donor.sec, param_name)
            else:
                return getattr(donor.sec(loc), param_name)
        else:
            return getattr(getattr(donor.sec(loc), mech_name), param_name)
    except Exception as e:
        logger.error('inherit_mech_param: problem inheriting mechanism: %s parameter: %s from sec_type: %s' %
              (mech_name, param_name, donor.type))
        raise e


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
            try:
                node.sec.insert(mech_name)
            except Exception:
                raise RuntimeError('set_mech_param: unable to insert mechanism: %s cell: %s in sec_type: %s ' \
                                   % (mech_name, str(cell), node.type))
            setattr(node.sec, param_name + "_" + mech_name, baseline)
    elif donor is None:
        raise RuntimeError('set_mech_param: cannot set value of mechanism: %s parameter: %s in sec_type: %s '
                           'without a provided origin' % (mech_name, param_name, node.type))
    else:
        min_distance = rules.get('min_loc', 0.)
        max_distance = rules.get('max_loc', None)
        min_val = rules.get('min', None)
        max_val = rules.get('max', None)
        slope = rules.get('slope', None)
        tau = rules.get('tau', None)
        xhalf = rules.get('xhalf', None)
        outside = rules.get('outside', None)

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
                value = get_param_val_by_distance(distance, baseline, slope, min_distance, max_distance, min_val,
                                                  max_val, tau, xhalf, outside)
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
                    offset = baseline - (slope / (1. + np.exp(xhalf / tau)))
                    value = offset + (slope / (1. + np.exp((xhalf - distance) / tau)))
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


def zero_h(cell):
    """
    Set Ih channel conductances to zero in all compartments. Used during parameter optimization.
    """
    for sec_type in (sec_type for sec_type in default_ordered_sec_types if sec_type in cell.nodes and
                                                                           sec_type in cell.mech_dict):
        for h_type in (h_type for h_type in ['h'] if h_type in cell.mech_dict[sec_type]):
            modify_mech_param(cell, sec_type, h_type, 'ghbar', 0.)


# --------------------------- Custom methods to specify subcellular mechanism gradients ------------------------------ #


def apply_custom_mech_rules(cell, node, mech_name, param_name, baseline, rules, donor, verbose=True):
    """
    If the provided node meets custom criteria, rules are modified and passed back to apply_mech_rules with the
    'custom' item removed. Avoids having to determine baseline and donor over again.
    :param cell: :class:'BiophysCell'
    :param node: :class:'SHocNode'
    :param mech_name: str
    :param param_name: str
    :param baseline: float
    :param rules: dict
    :param donor: :class:'SHocNode' or None
    :param verbose: bool
    """
    if 'func' not in rules['custom'] or rules['custom']['func'] is None:
        raise RuntimeError('apply_custom_mech_rules: no custom function provided for mechanism: %s parameter: %s in '
                           'sec_type: %s' % (mech_name, param_name, node.type))
    if rules['custom']['func'] in globals() and isinstance(globals()[rules['custom']['func']], collections.Callable):
        func = globals()[rules['custom']['func']]
    else:
        raise RuntimeError('apply_custom_mech_rules: problem locating custom function: %s for mechanism: %s '
                           'parameter: %s in sec_type: %s' %
                           (rules['custom']['func'], mech_name, param_name, node.type))
    custom = copy.deepcopy(rules['custom'])
    del custom['func']
    new_rules = copy.deepcopy(rules)
    del new_rules['custom']
    new_rules['value'] = baseline
    new_rules = func(cell, node, baseline, new_rules, donor, **custom)
    if new_rules:
        apply_mech_rules(cell, node, mech_name, param_name, new_rules, donor, verbose=verbose)


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


def custom_filter_modify_slope_if_terminal(cell, node, baseline, rules, donor, **kwargs):
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
        raise RuntimeError('custom_filter_modify_slope_if_terminal: no min or max target value specified for sec_type: '
                           '%s' % node.type)
    slope = (end_val - start_val) / node.sec.L
    if 'slope' in rules:
        if direction < 0.:
            slope = min(rules['slope'], slope)
        else:
            slope = max(rules['slope'], slope)
    rules['slope'] = slope
    return rules


def custom_filter_modify_slope_if_terminal(cell, node, baseline, rules, donor, **kwargs):
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
        raise RuntimeError('custom_filter_modify_slope_if_terminal: no min or max target value specified for sec_type: '
                           '%s' % node.type)
    slope = (end_val - start_val) / node.sec.L
    if 'slope' in rules:
        if direction < 0.:
            slope = min(rules['slope'], slope)
        else:
            slope = max(rules['slope'], slope)
    rules['slope'] = slope
    return rules


def custom_filter_if_terminal(cell, node, baseline, rules, donor, **kwargs):
    """
    Allows the provided rule to be applied if the provided node is a terminal branch.
    :param cell: :class:'BiophysCell'
    :param node: :class:'SHocNode'
    :param baseline: float
    :param rules: dict
    :param donor: :class:'SHocNode' or None
    :return: dict or False
    """
    if not is_terminal(node):
        return False
    return rules


def filter_nodes(cell, sections=None, layers=None, swc_types=None):
    """
    Returns a subset of the nodes of the given cell according to the given criteria.

    :param cell: 
    :param sections: sequence of int
    :param layers: list of enumerated type: layer
    :param swc_types: list of enumerated type: swc_type
    :return: list of nodes
    """
    matches = lambda items: all(
        map(lambda query_item: (query_item[0] is None) or (query_item[1] in query_item[0]), items))

    nodes = []
    if swc_types is None:
        sections = sorted(cell.nodes.keys())
    for swc_type in swc_types:
        nodes.extend(cell.nodes[swc_type])
            

    result = [v for v in nodes
                  if matches([(layers, v.get_layer()),
                              (sections, v.get_sec())])]

    return result


# ------------------- Methods to specify cells from hoc templates and neuroh5 trees ---------------------------------- #

def report_topology(cell, env, node=None):
    """
    Traverse a cell and report topology and number of synapses.
    :param cell:
    :param env:
    :param node:
    """
    if node is None:
        node = cell.tree.root
    syn_attrs = env.synapse_attributes
    num_exc_syns = len(syn_attrs.filter_synapses(cell.gid, syn_sections=[node.index],
                                                 syn_types=[env.Synapse_Types['excitatory']]))
    num_inh_syns = len(syn_attrs.filter_synapses(cell.gid, syn_sections=[node.index],
                                                 syn_types=[env.Synapse_Types['inhibitory']]))

    diams_str = ', '.join('%.2f' % node.sec.diam3d(i) for i in range(node.sec.n3d()))
    report = 'node: %s, L: %.1f, diams: [%s], nseg: %i, children: %i, exc_syns: %i, inh_syns: %i' % \
             (node.name, node.sec.L, diams_str, node.sec.nseg, len(node.children), num_exc_syns, num_inh_syns)
    if node.parent is not None:
        report += ', parent: %s; connection_loc: %.1f' % (node.parent.name, node.connection_loc)
    logger.info(report)
    for child in node.children:
        report_topology(cell, env, child)

        
def make_section_node_dict(neurotree_dict):
    """
    Creates a dictionary of node to section assignments.
    :param neurotree_dict:
    :return: dict
    """
    pt_sections = neurotree_dict['sections']
    num_sections = pt_sections[0]
    sec_nodes = {}
    i = 1
    section_idx = 0
    while i < len(pt_sections):
          num_points = pt_sections[i]
          i += 1
          sec_nodes[section_idx] = []
          for ip in range(num_points):
            p = pt_sections[i]
            sec_nodes[section_idx].append(p)
            i += 1
          section_idx += 1
    assert(section_idx == num_sections)
    return sec_nodes
  
    
    
def make_section_graph(neurotree_dict):
    """
    Creates a graph of sections that follows the topological organization of the given neuron.
    :param neurotree_dict:
    :return: NetworkX.DiGraph
    """
    import networkx as nx

    if 'section_topology' in neurotree_dict:
        sec_src = neurotree_dict['section_topology']['src']
        sec_dst = neurotree_dict['section_topology']['dst']
        sec_loc = neurotree_dict['section_topology']['loc']
    else:
        sec_src = neurotree_dict['src']
        sec_dst = neurotree_dict['dst']
        sec_loc = []
        sec_nodes = {}
        pt_sections = neurotree_dict['sections']
        pt_parents = neurotree_dict['parent']
        sec_nodes = make_section_node_dict(neurotree_dict)
        for src, dst in zip_longest(sec_src, sec_dst):
            src_pts = sec_nodes[src]
            dst_pts = sec_nodes[dst]
            dst_parent = pt_parents[dst_pts[0]]
            loc = np.argwhere(src_pts == dst_parent)[0]
            sec_loc.append(loc)
                
    sec_graph = nx.DiGraph()
    for i, j, loc in zip(sec_src, sec_dst, sec_loc):
        sec_graph.add_edge(i, j, loc=loc)

    return sec_graph


def make_morph_graph(biophys_cell, node_filters={}):
    """
    Creates a graph of 3d points that follows the morphological organization of the given neuron.
    :param neurotree_dict:
    :return: NetworkX.DiGraph
    """
    import networkx as nx

    nodes = filter_nodes(biophys_cell, **node_filters)

    sec_layers = {}
    src_sec = []
    dst_sec = []
    connection_locs = []
    pt_xs = []
    pt_ys = []
    pt_zs = []
    pt_locs = []
    pt_idxs = []
    pt_layers = []
    pt_idx = 0
    sec_pts = collections.defaultdict(list)

    for node in nodes:
        sec = node.sec
        nn = sec.n3d()
        L = sec.L
        for i in range(nn):
            pt_xs.append(sec.x3d(i))
            pt_ys.append(sec.y3d(i))
            pt_zs.append(sec.z3d(i))
            loc = sec.arc3d(i) / L
            pt_locs.append(loc)
            pt_layers.append(node.get_layer(loc))
            pt_idxs.append(pt_idx)
            sec_pts[node.index].append(pt_idx)
            pt_idx += 1

        for child in node.children:
            src_sec.append(node.index)
            dst_sec.append(child.index)
            connection_locs.append(h.parent_connection(sec=child.sec))
            
    sec_pt_idxs = {}
    edges = []
    for sec, pts in viewitems(sec_pts):
        sec_pt_idxs[pts[0]] = sec
        for i in range(1, len(pts)):
            sec_pt_idxs[pts[i]] = sec
            src_pt = pts[i-1]
            dst_pt = pts[i]
            edges.append((src_pt, dst_pt))

    for (s,d,parent_loc) in zip(src_sec, dst_sec, connection_locs):
        for src_pt in sec_pts[s]:
            if pt_locs[src_pt] >= parent_loc:
                break
        dst_pt = sec_pts[d][0]
        edges.append((src_pt, dst_pt))
        
#        print('connection %d -> %d: points %d( %.02f %.02f %.02f )   -> %d( %.02f %.02f %.02f ): ' % (s, d, src_pt, pt_xs[src_pt], pt_ys[src_pt], pt_zs[src_pt],
#                                                                                                          dst_pt, pt_xs[dst_pt], pt_ys[dst_pt], pt_zs[dst_pt]))

    morph_graph = nx.Graph()
    morph_graph.add_nodes_from([(i, {'x': x, 'y': y, 'z': z, 'sec': sec_pt_idxs[i], 'loc': loc, 'layer': layer})
                                    for (i,x,y,z,loc,layer) in zip(range(len(pt_idxs)), pt_xs, pt_ys, pt_zs, pt_locs, pt_layers)])
    for i, j in edges:
        morph_graph.add_edge(i, j)

    return morph_graph


def resize_tree_sections(neurotree_dict, max_section_length):
    """
    Given a neurotree dictionary, transforms section and point data such that 
    no section exceeds the length specified by parameter max_section_length.

    :param neurotree_dict: neurotree dictionary
    :param max_section_length: maximum section length
    :return: neurotree dict
    """
    import networkx as nx

    assert(max_section_length > 0)
    
    vx = copy.deepcopy(neurotree_dict['x'])
    vy = copy.deepcopy(neurotree_dict['y'])
    vz = copy.deepcopy(neurotree_dict['z'])
    vradius = copy.deepcopy(neurotree_dict['radius'])
    vlayer = copy.deepcopy(neurotree_dict['layer'])
    swc_type = copy.deepcopy(neurotree_dict['swc_type'])
    vparent = copy.deepcopy(neurotree_dict['parent'])
    vsrc = copy.deepcopy(neurotree_dict['src'])
    vdst = copy.deepcopy(neurotree_dict['dst'])
    sec_nodes_dict = make_section_node_dict(neurotree_dict)
    new_ndindex = len(vx)
    new_secindex = len(sec_nodes_dict)
    node_arrays = (vx,vy,vz,vradius,vlayer,swc_type,vparent)
    secg = make_section_graph(neurotree_dict)
    secq = sorted(sec_nodes_dict.keys(), reverse=True)
    while secq:
        secindex = secq.pop()
        nodes = np.asarray(sec_nodes_dict[secindex])
        nodes_xyz = np.column_stack((vx[nodes], vy[nodes], vz[nodes]))
        a = nodes_xyz[1:,:]
        b = nodes_xyz[:-1,:]
        nodes_dd = np.sqrt(np.sum((a - b) ** 2, axis=1))
        nodes_cd = np.cumsum(nodes_dd)
        nodes_dist = np.concatenate(([0.], nodes_cd))
        new_secnodes = nodes[np.argwhere(nodes_dist > max_section_length).flat]
        if len(new_secnodes) > 0:
            old_secnodes = nodes[np.argwhere(nodes_dist <= max_section_length).flat]
            sec_len = len(old_secnodes)+1
            sec_nodes_dict[new_secindex] = np.concatenate(([new_ndindex], new_secnodes))
            sec_nodes_dict[secindex] = np.concatenate((old_secnodes, [new_ndindex]))
            for a in node_arrays:
                a.resize(new_ndindex+1, refcheck=False)
            new_x = np.interp(max_section_length, nodes_dist, vx[nodes])
            new_y = np.interp(max_section_length, nodes_dist, vy[nodes])
            new_z = np.interp(max_section_length, nodes_dist, vz[nodes])
            new_radius = np.interp(max_section_length, nodes_dist, vradius[nodes])
            new_layer = vlayer[new_secnodes[0]]
            new_swctype = swc_type[new_secnodes[0]]
            new_parent = sec_nodes_dict[secindex][-1]
            for a,v in zip_longest(node_arrays,(new_x,new_y,new_z,new_radius,new_layer,new_swctype,new_parent)):
                np.concatenate((a[:-1],[v]),out=a)
            vparent[new_secnodes[0]] = new_ndindex
            secg.add_node(new_secindex)
            sec_out = secg.out_edges(secindex, data=True)
            for i,j,attr in sec_out:
                loc = attr['loc']
                if loc >= sec_len-1:
                    secg.remove_edge(i,j)
                    secg.add_edge(new_secindex, j, loc=loc-sec_len+1)
            secg.add_edge(secindex, new_secindex, loc=sec_len-1)
            secq.append(new_secindex)
            new_secindex += 1
            new_ndindex += 1
    num_secedges = secg.number_of_edges()
    vsrc = np.zeros((num_secedges,), dtype=np.uint16)
    vdst = np.zeros((num_secedges,), dtype=np.uint16)
    for i, (s, d, l) in enumerate(secg.edges.data('loc')):
        vsrc[i] = s
        vdst[i] = d
    sections = [np.asarray([len(sec_nodes_dict)], dtype=np.uint16)]
    for secindex, secnodes in viewitems(sec_nodes_dict):
        sections.append(np.asarray([len(secnodes)], dtype=np.uint16))
        sections.append(secnodes)
    vsection = np.asarray(np.concatenate(sections), dtype=np.uint16)
    
    new_tree_dict = { 'x': vx,
                      'y': vy,
                      'z': vz,
                      'radius': vradius,
                      'layer': vlayer,
                      'parent': vparent,
                      'swc_type': swc_type,
                      'sections': vsection,
                      'src': vsrc,
                      'dst': vdst }

    return new_tree_dict


def normalize_tree_topology(neurotree_dict, swc_type_defs):
    """
    Given a neurotree dictionary, perform topology normalization,
    where 1) all dendritic sections have as a parent either another
    dendritic sections, or the soma section; and 2) dendritic sections
    connected to the first point of another dendritic section are
    instead connected to the last point of the grandparent section.

    Note: This procedure assumes that all points that belong to a section
    have the same swc type.

    :param neurotree_dict:
    :param swc_type_defs:
    :return: neurotree dict
    
    """
    import networkx as nx
    
    pt_xs = copy.deepcopy(neurotree_dict['x'])
    pt_ys = copy.deepcopy(neurotree_dict['y'])
    pt_zs = copy.deepcopy(neurotree_dict['z'])
    pt_radius = copy.deepcopy(neurotree_dict['radius'])
    pt_layers = copy.deepcopy(neurotree_dict['layer'])
    pt_parents = copy.deepcopy(neurotree_dict['parent'])
    pt_swc_types = copy.deepcopy(neurotree_dict['swc_type'])
    pt_sections = copy.deepcopy(neurotree_dict['sections'])
    sec_src = copy.deepcopy(neurotree_dict['src'])
    sec_dst = copy.deepcopy(neurotree_dict['dst'])
    soma_pts = np.where(pt_swc_types == swc_type_defs['soma'])[0]
    hillock_pts = np.where(pt_swc_types == swc_type_defs['hillock'])[0]
    ais_pts = np.where(pt_swc_types == swc_type_defs['ais'])[0]
    axon_pts = np.where(pt_swc_types == swc_type_defs['axon'])[0]
    section_swc_types = []
    section_pt_dict = {}
    pt_section_dict = {}
    
    i = 1
    section_idx = 0
    soma_section_idx = None
    while i < len(pt_sections):
        num_points = pt_sections[i]
        i += 1
        section_pt_dict[section_idx] = []
        if (pt_sections[i] == soma_pts[0]) and (pt_sections[i+1] == soma_pts[1]):
            soma_section_idx = section_idx
        for ip in range(num_points):
            p = pt_sections[i]
            section_pt_dict[section_idx].append(p)
            if p not in pt_section_dict:
                pt_section_dict[p] = section_idx
            i += 1
        this_section_swc_type = pt_swc_types[pt_sections[i-1]]
        section_swc_types.append(this_section_swc_type)
        section_idx += 1

    sec_parents_dict = {}
    for src, dst in zip(sec_src, sec_dst):
        sec_parents_dict[dst] = src

    extra_edges = []
    for section_idx in sorted(section_pt_dict.keys()):
        section_pts = section_pt_dict[section_idx]
        section_swc_type = section_swc_types[section_idx]
        section_parent = sec_parents_dict.get(section_idx, None)
        ## Detect sections without parent where first point is part of the parent section
        if section_parent is None:
            candidate_parent = pt_section_dict[section_pts[0]]
            if candidate_parent != section_idx:
                section_parent = candidate_parent
                sec_parents_dict[section_idx] = section_parent
                extra_edges.append((section_parent, section_idx))
        ## Detect sections without parent and connect them
        if section_parent is None:
            if (section_swc_type == swc_type_defs['apical']) or (section_swc_type == swc_type_defs['basal']):
                pt_parents[section_pts[0]] = soma_pts[-1]
                extra_edges.append((soma_section_idx, section_idx))
                sec_parents_dict[section_idx] = soma_section_idx
            elif section_swc_type == swc_type_defs['hillock']:
                pt_parents[section_pts[0]] = soma_pts[0]
                extra_edges.append((soma_section_idx, section_idx))
                sec_parents_dict[section_idx] = soma_section_idx
            elif section_swc_type == swc_type_defs['ais']:
                pt_parents[section_pts[0]] = hillock_pts[-1]
                hillock_section_idx = pt_section_dict[hillock_pts[-1]]
                extra_edges.append((hillock_section_idx, section_idx))
                sec_parents_dict[section_idx] = hillock_section_idx
            elif section_swc_type == swc_type_defs['axon']:
                pt_parents[section_pts[0]] = ais_pts[-1]
                ais_section_idx = pt_section_dict[ais_pts[-1]]
                extra_edges.append((ais_section_idx, section_idx))
                sec_parents_dict[section_idx] = ais_section_idx
            elif section_swc_type == swc_type_defs['soma']:
                pass
            else:
                raise RuntimeError("normalize_tree_topology: section %d: unsupported section type %d without parent" % (section_idx, section_swc_type))

    sec_graph = nx.DiGraph()
    for i, j in zip(sec_src, sec_dst):
        sec_graph.add_edge(i, j)
    for i, j in extra_edges:
        sec_graph.add_edge(i, j)

    sec_graph_roots = [n for n,d in sec_graph.in_degree() if d==0]
    if len(sec_graph_roots) != 1:
        raise RuntimeError("normalize_tree_topology: section graph must be a rooted tree")

    
    edges_in_order = nx.dfs_edges(sec_graph, source=sec_graph_roots[0])
    
    for src, dst in edges_in_order:
        
        dst_pts = section_pt_dict[dst]
        src_pts = section_pt_dict[src]
        dst_pts_parents = [pt_parents[i] for i in dst_pts]
        
        ## detect sections that are connected to first point of their parent
        if dst_pts_parents[0] == src_pts[0]:
            ## obtain parent of src section
            src_parent = sec_parents_dict.get(src, None)
            if src_parent is not None:
                src_parent_pts = section_pt_dict[src_parent]
                pt_parents[dst_pts[0]] = src_parent_pts[-1]
                sec_parents_dict[dst] = src_parent

    ## Rebuild section graph in order to eliminate remaining inconsistencies
    sec_graph = nx.DiGraph()
    
    for section_idx in section_pt_dict:
        pts = section_pt_dict[section_idx]
        parent_pt = pt_parents[pts[0]]
        if parent_pt > -1:
            parent_section_idx = pt_section_dict[parent_pt]
            sec_graph.add_edge(parent_section_idx, section_idx)
        else:
            candidate_parent = pt_section_dict[pts[0]]
            if candidate_parent != section_idx:
                parent_section_idx = candidate_parent
                sec_graph.add_edge(parent_section_idx, section_idx)

    sec_nodes = [n for n in sec_graph.nodes]
    sec_nodes.sort()
    if len(sec_nodes) != len(section_pt_dict.keys()):
        raise RuntimeError("normalize_tree_topology: normalized section graph has fewer nodes (%d) than initial section graph (%d)" % (len(sec_nodes), len(section_pt_dict.keys())))
        
    soma_descendants = list(nx.descendants(sec_graph, source=soma_section_idx))
    if len(soma_descendants) < (len(sec_nodes) - 1):
        raise RuntimeError("normalize_tree_topology: not all nodes are reachable from soma in section graph")
            
    edges_in_order = list(nx.dfs_edges(sec_graph, source=sec_graph_roots[0]))
    sec_src = np.asarray([i for (i,j) in edges_in_order], dtype=np.uint16)
    sec_dst = np.asarray([j for (i,j) in edges_in_order], dtype=np.uint16)
    
    new_tree_dict = { 'x': pt_xs,
                      'y': pt_ys,
                      'z': pt_zs,
                      'radius': pt_radius,
                      'layer': pt_layers,
                      'parent': pt_parents,
                      'swc_type': pt_swc_types,
                      'sections': pt_sections,
                      'src': sec_src,
                      'dst': sec_dst }

    return new_tree_dict



def make_neurotree_hoc_cell(template_class, gid=0, dataset_path="", neurotree_dict={}):
    """

    :param template_class:
    :param local_id:
    :param gid:
    :param dataset_path:
    :param neurotree_dict:
    :return: hoc cell object
    """

    vx = neurotree_dict['x']
    vy = neurotree_dict['y']
    vz = neurotree_dict['z']
    vradius = neurotree_dict['radius']
    vlayer = neurotree_dict['layer']
    vsection = neurotree_dict['section']
    secnodes = neurotree_dict['section_topology']['nodes']
    vsrc = neurotree_dict['section_topology']['src']
    vdst = neurotree_dict['section_topology']['dst']
    vloc = neurotree_dict['section_topology']['loc']
    swc_type = neurotree_dict['swc_type']

    cell = template_class(gid, dataset_path, secnodes, vlayer, vsrc, vdst, vloc, vx, vy, vz, vradius, swc_type)
    return cell


def make_hoc_cell(env, pop_name, gid, neurotree_dict=False):
    """

    :param env:
    :param gid:
    :param pop_name:
    :return:
    """
    dataset_path = env.dataset_path if env.dataset_path is not None else ""
    data_file_path = env.data_file_path
    template_name = env.celltypes[pop_name]['template']
    assert (hasattr(h, template_name))
    template_class = getattr(h, template_name)

    if neurotree_dict:
        hoc_cell = make_neurotree_hoc_cell(template_class, neurotree_dict=neurotree_dict, gid=gid,
                                           dataset_path=dataset_path)
    else:
        if pop_name in env.cell_attribute_info and 'Trees' in env.cell_attribute_info[pop_name]:
            raise Exception('make_hoc_cell: morphology for population %s gid: %i is not provided' %
                            data_file_path, pop_name, gid)
        else:
            hoc_cell = template_class(gid, dataset_path)

    return hoc_cell


def make_input_cell(env, gid, pop_id, input_source_dict, spike_train_attr_name='t'):
    """
    Instantiates an input generator according to the given cell template.
    """

    input_sources = input_source_dict[pop_id]
    if 'spiketrains' in input_sources:
        cell = h.VecStim()
        spk_attr_dict = input_sources['spiketrains'].get(gid, None)
        if spk_attr_dict is not None:
            spk_ts = spk_attr_dict[spike_train_attr_name]
            if len(spk_ts) > 0:
                cell.play(h.Vector(spk_ts))
    elif 'generator' in input_sources:
        input_gen = input_sources['generator']
        template_name = input_gen['template']
        param_values = input_gen['params']
        template = getattr(h, template_name)
        params = [param_values[p] for p in env.netclamp_config.template_params[template_name]]
        cell = template(gid, *params)
    else:
        raise RuntimeError('cells.make_input_cell: unrecognized input cell configuration')
        
    return cell


def load_biophys_cell_dicts(env, pop_name, gid_set, load_connections=True, validate_tree=True):
    """
    Loads the data necessary to instantiate BiophysCell into the given dictionary.

    :param env: an instance of env.Env
    :param pop_name: population name
    :param gid: gid
    :param load_connections: bool
    :param validate_tree: bool

    Environment can be instantiated as:
    env = Env(config_file, template_paths, dataset_prefix, config_prefix)
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param config_prefix: str; path to directory containing network and cell mechanism config files
    """

    synapse_config = env.celltypes[pop_name]['synapses']

    has_weights = False
    weights_config = None
    if 'weights' in synapse_config:
        has_weights = True
        weights_config = synapse_config['weights']

    ## Loads cell morphological data, synaptic attributes and connection data

    tree_dicts = {}
    synapses_dicts = {}
    weight_dicts = {}
    connection_graphs = { gid: { pop_name: {} } for gid in gid_set }
    graph_attr_info = None
    
    gid_list = list(gid_set)
    tree_attr_iter, _ = read_tree_selection(env.data_file_path, pop_name,
                                            gid_list, comm=env.comm, 
                                            topology=True, validate=validate_tree)
    for gid, tree_dict in tree_attr_iter:
        tree_dicts[gid] = tree_dict

    if load_connections:
        synapses_iter = read_cell_attribute_selection(env.data_file_path, pop_name,
                                                      gid_list, 'Synapse Attributes',
                                                      mask=set(['syn_ids', 'syn_locs', 'syn_secs', 'syn_layers',
                                                                'syn_types', 'swc_types']),
                                                      comm=env.comm)
        for gid, attr_dict in synapses_iter:
            synapses_dicts[gid] = attr_dict

        if has_weights:
            for config in weights_config:
                weights_namespaces = config['namespace']
                cell_weights_iters = [read_cell_attribute_selection(env.data_file_path, pop_name, gid_list,
                                                                  weights_namespace, comm=env.comm)
                                      for weights_namespace in weights_namespaces]
                for weights_namespace, cell_weights_iter in zip_longest(weights_namespaces, cell_weights_iters):
                    for gid, cell_weights_dict in cell_weights_iter:
                        this_weights_dict = weight_dicts.get(gid, {})
                        this_weights_dict[weights_namespace] = cell_weights_dict
                        weight_dicts[gid] = this_weights_dict

        graph, graph_attr_info = read_graph_selection(file_name=env.connectivity_file_path, selection=gid_list,
                                                      namespaces=['Synapses', 'Connections'], comm=env.comm)
        if pop_name in graph:
            for presyn_name in graph[pop_name].keys():
                edge_iter = graph[pop_name][presyn_name]
                for (postsyn_gid, edges) in edge_iter:
                    connection_graphs[postsyn_gid][pop_name][presyn_name] = [(postsyn_gid, edges)]
        
        
    cell_dicts = {}
    for gid in gid_set:
        this_cell_dict = {}
        
        tree_dict = tree_dicts[gid]
        this_cell_dict['morph'] = tree_dict
        
        if load_connections:
            synapses_dict = synapses_dicts[gid]
            weight_dict = weight_dicts.get(gid, None)
            connection_graph = connection_graphs[gid]
            this_cell_dict['synapse'] = synapses_dict
            this_cell_dict['connectivity'] = connection_graph, graph_attr_info
            this_cell_dict['weight'] = weight_dict
        cell_dicts[gid] = this_cell_dict
        
    
    return cell_dicts


def init_circuit_context(env, pop_name, gid,
                         load_edges=False, connection_graph=None,
                         load_weights=False, weight_dict=None,
                         load_synapses=False, synapses_dict=None,
                         set_edge_delays=True, **kwargs):
    
    syn_attrs = env.synapse_attributes
    synapse_config = env.celltypes[pop_name]['synapses']

    has_weights = False
    weight_config = []
    if 'weights' in synapse_config:
        has_weights = True
        weight_config = synapse_config['weights']

    init_synapses = False
    init_weights = False
    init_edges = False
    if load_edges or (connection_graph is not None):
        init_synapses=True
        init_edges=True
    if has_weights and (load_weights or (weight_dict is not None)):
        init_synapses=True
        init_weights=True
    if load_synapses or (synapses_dict is not None):
        init_synapses=True

    if init_synapses:
        if synapses_dict is not None:
            syn_attrs.init_syn_id_attrs(gid, **synapses_dict)
        elif load_synapses or load_edges:
            if (pop_name in env.cell_attribute_info) and ('Synapse Attributes' in env.cell_attribute_info[pop_name]):
                synapses_iter = read_cell_attribute_selection(env.data_file_path, pop_name, [gid], 'Synapse Attributes',
                                                              mask=set(['syn_ids', 'syn_locs', 'syn_secs', 'syn_layers',
                                                                        'syn_types', 'swc_types']), comm=env.comm)
                syn_attrs.init_syn_id_attrs_from_iter(synapses_iter)
            else:
                raise RuntimeError('init_circuit_context: synapse attributes not found for %s: gid: %i' % (pop_name, gid))
        else:
            raise RuntimeError("init_circuit_context: invalid synapses parameters")
            

    if init_weights and has_weights:

        for weight_config_dict in weight_config:

            expr_closure = weight_config_dict.get('closure', None)
            weights_namespaces = weight_config_dict['namespace']

            cell_weights_dicts = {}
            if weight_dict is not None:
                for weights_namespace in weights_namespaces:
                    if weights_namespace in weight_dict:
                        cell_weights_dicts[weights_namespace] = weight_dict[weights_namespace]

            elif load_weights:
                if (env.data_file_path is None):
                    raise RuntimeError('init_circuit_context: load_weights=True but data file path is not specified ')
                
                for weights_namespace in weights_namespaces:
                    cell_weights_iter = read_cell_attribute_selection(env.data_file_path, pop_name, 
                                                                      selection=[gid], 
                                                                      namespace=weights_namespace, 
                                                                      comm=env.comm)
                    for cell_weights_gid, cell_weights_dict in cell_weights_iter:
                        assert(cell_weights_gid == gid)
                        cell_weights_dicts[weights_namespace] = cell_weights_dict

            else:
                raise RuntimeError("init_circuit_context: invalid weights parameters")
            if len(weights_namespaces) != len(cell_weights_dicts):
                logger.warning("init_circuit_context: Unable to load all weights namespaces: %s" % str(weights_namespaces))

            multiple_weights = 'error'
            append_weights = False
            for weights_namespace in weights_namespaces:
                if weights_namespace in cell_weights_dicts:
                    cell_weights_dict = cell_weights_dicts[weights_namespace]
                    weights_syn_ids = cell_weights_dict['syn_id']
                    for syn_name in (syn_name for syn_name in cell_weights_dict if syn_name != 'syn_id'):
                        weights_values = cell_weights_dict[syn_name]
                        syn_attrs.add_mech_attrs_from_iter(gid, syn_name,
                                                           zip_longest(weights_syn_ids,
                                                                       [{'weight': Promise(expr_closure, [x])} for x in weights_values]
                                                                       if expr_closure else [{'weight': x} for x in weights_values]),
                                                           multiple=multiple_weights, append=append_weights)
                        logger.info('init_circuit_context: gid: %i; found %i %s synaptic weights in namespace %s' %
                                    (gid, len(cell_weights_dict[syn_name]), syn_name, weights_namespace))
                        logger.info('weight_values min/max/mean: %.02f / %.02f / %.02f' %
                                    (np.min(weights_values), np.max(weights_values), np.mean(weights_values)))
                expr_closure = None
                append_weights = True
                multiple_weights='overwrite'


    if init_edges:
        if connection_graph is not None:
            (graph, a) = connection_graph
        elif load_edges:
            if env.connectivity_file_path is None:
                raise RuntimeError('init_circuit_context: load_edges=True but connectivity file path is not specified ')
            elif os.path.isfile(env.connectivity_file_path):
                (graph, a) = read_graph_selection(file_name=env.connectivity_file_path, selection=[gid],
                                                  namespaces=['Synapses', 'Connections'], comm=env.comm)
        else:
            raise RuntimeError('init_circuit_context: connection file %s not found' % env.connectivity_file_path)
    else:
        (graph, a) = None, None

    if graph is not None:
        if pop_name in graph:
            for presyn_name in graph[pop_name].keys():
                edge_iter = graph[pop_name][presyn_name]
                syn_attrs.init_edge_attrs_from_iter(pop_name, presyn_name, a, edge_iter, set_edge_delays)
        else:
            logger.error('init_circuit_context: connection attributes not found for %s: gid: %i' % (pop_name, gid))
            raise Exception
    
    

def make_biophys_cell(env, pop_name, gid, 
                      mech_file_path=None, mech_dict=None,
                      tree_dict=None,
                      load_synapses=False, synapses_dict=None, 
                      load_edges=False, connection_graph=None,
                      load_weights=False, weight_dict=None, 
                      set_edge_delays=True, bcast_template=True,
                      validate_tree=True,
                      **kwargs):
    """
    :param env: :class:'Env'
    :param pop_name: str
    :param gid: int
    :param tree_dict: dict
    :param synapses_dict: dict
    :param weight_dict: list of dict
    :param load_synapses: bool
    :param load_edges: bool
    :param load_weights: bool
    :param set_edge_delays: bool
    :param mech_file_path: str (path)
    :return: :class:'BiophysCell'
    """
    load_cell_template(env, pop_name, bcast_template=bcast_template)

    if tree_dict is None:
        tree_attr_iter, _ = read_tree_selection(env.data_file_path, pop_name, [gid], comm=env.comm, 
                                                topology=True, validate=validate_tree)
        _, tree_dict = next(tree_attr_iter)
        
    hoc_cell = make_hoc_cell(env, pop_name, gid, neurotree_dict=tree_dict)
    cell = BiophysCell(gid=gid, pop_name=pop_name, hoc_cell=hoc_cell, env=env,
                       mech_file_path=mech_file_path, mech_dict=mech_dict)
    circuit_flag = load_edges or load_weights or load_synapses or synapses_dict or weight_dict or connection_graph
    if circuit_flag:
        init_circuit_context(env, pop_name, gid, 
                             load_synapses=load_synapses, synapses_dict=synapses_dict,
                             load_edges=load_edges, connection_graph=connection_graph,
                             load_weights=load_weights, weight_dict=weight_dict, 
                             set_edge_delays=set_edge_delays, **kwargs)
    
    env.biophys_cells[pop_name][gid] = cell
    return cell


def make_PR_cell(env, pop_name, gid, mech_file_path=None, mech_dict=None,
                 tree_dict=None,  load_synapses=False, synapses_dict=None, 
                 load_edges=False, connection_graph=None,
                 load_weights=False, weight_dict=None, 
                 set_edge_delays=True, bcast_template=True, **kwargs):
    """
    :param env: :class:'Env'
    :param pop_name: str
    :param gid: int
    :param mech_file_path: str (path)
    :param mech_dict: dict
    :param synapses_dict: dict
    :param weight_dicts: list of dict
    :param load_synapses: bool
    :param load_edges: bool
    :param load_weights: bool
    :param set_edge_delays: bool
    :return: :class:'IzhikevichCell'
    """
    load_cell_template(env, pop_name, bcast_template=bcast_template)

    if mech_dict is None and mech_file_path is None:
        raise RuntimeError('make_PR_cell: mech_dict or mech_file_path must be specified')

    if mech_dict is None and mech_file_path is not None:
        mech_dict = read_from_yaml(mech_file_path)

    cell = PRneuron(gid=gid, pop_name=pop_name, env=env,
                    cell_config=PRconfig(**mech_dict['PinskyRinzel']),
                    mech_dict={ k: mech_dict[k] for k in mech_dict if k != 'PinskyRinzel' })

    circuit_flag = load_edges or load_weights or load_synapses or synapses_dict or weight_dict or connection_graph
    if circuit_flag:
        init_circuit_context(env, pop_name, gid, 
                             load_synapses=load_synapses,
                             synapses_dict=synapses_dict,
                             load_edges=load_edges, connection_graph=connection_graph,
                             load_weights=load_weights, weight_dict=weight_dict, 
                             set_edge_delays=set_edge_delays, **kwargs)
        
    env.biophys_cells[pop_name][gid] = cell
    return cell


def make_izhikevich_cell(env, pop_name, gid, mech_file_path=None, mech_dict=None,
                         tree_dict=None,  load_synapses=False, synapses_dict=None, 
                         load_edges=False, connection_graph=None,
                         load_weights=False, weight_dict=None, 
                         set_edge_delays=True, **kwargs):
    """
    :param env: :class:'Env'
    :param pop_name: str
    :param gid: int
    :param mech_file_path: str (path)
    :param mech_dict: dict
    :param synapses_dict: dict
    :param weight_dicts: list of dict
    :param load_synapses: bool
    :param load_edges: bool
    :param load_weights: bool
    :param set_edge_delays: bool
    :return: :class:'IzhikevichCell'
    """

    if mech_dict is None and mech_file_path is None:
        raise RuntimeError('make_izhikevich_cell: mech_dict or mech_file_path must be specified')

    if mech_dict is None and mech_file_path is not None:
        mech_dict = read_from_yaml(mech_file_path)

    cell = IzhiCell(gid=gid, pop_name=pop_name, env=env,
                    cell_attrs=IzhiCellAttrs(**mech_dict['izhikevich']),
                    mech_dict={ k: mech_dict[k] for k in mech_dict if k != 'izhikevich' })

    circuit_flag = load_edges or load_weights or load_synapses or synapses_dict or weight_dict or connection_graph
    if circuit_flag:
        init_circuit_context(env, pop_name, gid, 
                             load_synapses=load_synapses,
                             synapses_dict=synapses_dict,
                             load_edges=load_edges, connection_graph=connection_graph,
                             load_weights=load_weights, weight_dict=weight_dict, 
                             set_edge_delays=set_edge_delays, **kwargs)
        
    env.biophys_cells[pop_name][gid] = cell
    return cell


get_biophys_cell = make_biophys_cell


def register_cell(env, pop_name, gid, cell):
    """
    Registers a cell in a network environment.

    :param env: an instance of the `dentate.Env` class
    :param pop_name: population name
    :param gid: gid
    :param cell: cell instance
    """
    rank = env.comm.rank
    env.gidset.add(gid)
    env.pc.set_gid2node(gid, rank)
    hoc_cell = getattr(cell, 'hoc_cell', cell)
    env.cells[pop_name][gid] = hoc_cell
    if hoc_cell.is_art() > 0:
        env.artificial_cells[pop_name][gid] = hoc_cell
    # Tell the ParallelContext that this cell is a spike source
    # for all other hosts. NetCon is temporary.
    nc = getattr(cell, 'spike_detector', None)
    if nc is None:
        if hasattr(cell, 'connect2target'):
            nc = hoc_cell.connect2target(h.nil)
        elif cell.is_art() > 0:
            nc = h.NetCon(cell, None)
        else:
            raise RuntimeError('register_cell: unknown cell type')
    nc.delay = max(2*env.dt, nc.delay)
    env.pc.cell(gid, nc, 1)
    env.pc.outputcell(gid)
    # Record spikes of this cell
    env.pc.spike_record(gid, env.t_vec, env.id_vec)
    # if the spike detector is located in a compartment other than soma,
    # record the spike time delay relative to soma
    if hasattr(cell, 'spike_onset_delay'):
        env.spike_onset_delay[gid] = cell.spike_onset_delay

def is_cell_registered(env, gid):
    """
    Returns True if cell gid has already been registered, False otherwise.
    """
    return env.pc.gid_exists(gid)    

def record_cell(env, pop_name, gid, recording_profile=None):
    """
    Creates a recording object for the given cell, according to configuration in env.recording_profile.
    """
    recs = []
    if recording_profile is None:
        recording_profile = env.recording_profile
    if recording_profile is not None:
        syn_attrs = env.synapse_attributes
        cell = env.biophys_cells[pop_name].get(gid, None)
        if cell is not None:
            label = recording_profile['label']
            dt = recording_profile.get('dt', None)
            for reclab, recdict  in viewitems(recording_profile.get('section quantity', {})):
                recvar = recdict.get('variable', reclab)
                loc = recdict.get('loc', None)
                swc_types = recdict.get('swc_types', None)
                locdict = collections.defaultdict(lambda: 0.5)
                if (loc is not None) and (swc_types is not None):
                    for s,l in zip(swc_types,loc):
                        locdict[s] = l
                    
                nodes = filter_nodes(cell, layers=recdict.get('layers', None),
                                     swc_types=recdict.get('swc types', None))
                node_type_count = collections.defaultdict(int)
                for node in nodes:
                    node_type_count[node.type] += 1
                visited = set([])
                for node in nodes:
                    sec = node.get_sec()
                    if str(sec) not in visited:
                        if node_type_count[node.type] == 1:
                            rec_id = '%s' % (node.type)
                        else:
                            rec_id = '%s.%i' % (node.type, node.index)
                        rec = make_rec(rec_id, pop_name, gid, cell.hoc_cell, sec=sec, dt=dt,
                                       loc=locdict[node.type], param=recvar, label=reclab,
                                       description=node.name)
                        recs.append(rec)
                        env.recs_dict[pop_name][rec_id].append(rec)
                        env.recs_count += 1
                        visited.add(str(sec))
            for recvar, recdict  in viewitems(recording_profile.get('synaptic quantity', {})):
                syn_filters = recdict.get('syn_filters', {})
                syn_sections = recdict.get('sections', None)
                synapses = syn_attrs.filter_synapses(gid, syn_sections=syn_sections, **syn_filters)
                syn_names = recdict.get('syn names', syn_attrs.syn_name_index_dict.keys())
                for syn_id, syn in viewitems(synapses):
                    syn_swc_type_name = env.SWC_Type_index[syn.swc_type]
                    syn_section = syn.syn_section
                    for syn_name in syn_names:
                        pps = syn_attrs.get_pps(gid, syn_id, syn_name, throw_error=False)
                        if pps is not None:
                            rec_id = '%d.%s.%s' % (syn_id, syn_name, str(recvar))
                            label = '%s' % (str(recvar))
                            rec = make_rec(rec_id, pop_name, gid, cell.hoc_cell, ps=pps, dt=dt, param=recvar,
                                            label=label, description='%s' % label)
                            ns = '%s%d.%s' % (syn_swc_type_name, syn_section, syn_name)
                            env.recs_dict[pop_name][ns].append(rec)
                            env.recs_count += 1
                            recs.append(rec)
                
    return recs
    
def find_spike_threshold_minimum(cell, loc=0.5, sec=None, duration=10.0, delay=100.0, initial_amp=0.001):
    """
    Determines minimum stimulus sufficient to induce a spike in a cell. 
    Defines an IClamp with the specified duration, and an APCount to detect spikes.
    Uses NEURON's thresh.hoc to find threshold by bisection.

    :param cell: hoc cell
    :param sec: cell section for stimulation
    :param loc: location of stimulus
    :param duration: stimulus duration
    :param delay: stimulus delay
    :param initial_amp: initial stimulus amplitude (nA)
    """

    if sec is None:
        sec = list(cell.soma)[0]

    iclamp = h.IClamp(sec(loc))
    setattr(iclamp, 'del', delay)
    iclamp.dur = duration
    iclamp.amp = initial_amp

    apcount = h.APCount(sec(loc))
    apcount.thresh = -20
    apcount.time = 0.

    h.tstop = duration + delay
    h.cvode_active(1)

    h.load_file("stdrun.hoc")
    h.load_file("thresh.hoc")  ## nrn/lib/hoc
    thr = h.threshold(iclamp._ref_amp)

    return thr


def get_spike_shape(vm, spike_times, equilibrate=0., dt=0.025, th_dvdt=10.):
    """
    Given a voltage recording from a cell section, and a list of spike times recorded from a spike detector NetCon,
    report features of the spike shape, including the delay between the recorded section and the spike detector.
    :param vm: array
    :param spike_times: array
    :param equilibrate: float
    :param dt: float
    :param th_dvdt: float; slope of voltage change at spike threshold
    :return: dict
    """
    start = int((equilibrate + 1.) / dt)  # start time after equilibrate, expressed in time step
    vm = vm[start:]
    dvdt = np.gradient(vm, dt)  # slope of voltage change
    th_x_indexes = np.where(dvdt >= th_dvdt)[0]
    if th_x_indexes.any():
        try_th_x = th_x_indexes[0] - int(1.6 / dt)  # the true spike onset is before the slope threshold is crossed
        if try_th_x < 0:
            return None
        else:
            th_x = try_th_x
    else:
        th_x_indexes = np.where(vm > -30.)[0]
        if th_x_indexes.any():
            try_th_x = th_x_indexes[0] - int(2. / dt)
            if try_th_x < 0:
                return None
            else:
                th_x = try_th_x
        else:
            return None
    th_v = vm[th_x]
    v_before = np.mean(vm[th_x - int(0.1 / dt):th_x])

    spike_detector_delay = spike_times[0] - (equilibrate + 1. + th_x * dt)
    window_dur = 100.  # ms
    fAHP_window_dur = 20.  # ms
    ADP_min_start = 5.  # ms
    ADP_window_dur = 75. # ms
    if len(spike_times) > 1:
        window_dur = min(window_dur, spike_times[1] - spike_times[0])
    window_end = min(len(vm), th_x + int(window_dur / dt))
    fAHP_window_end = min(window_end, th_x + int(fAHP_window_dur / dt))
    ADP_min_start_len = min(window_end - th_x, int(ADP_min_start / dt))
    ADP_window_end = min(window_end, th_x + int(ADP_window_dur / dt))

    if window_end - th_x <= 0:
        return None
    x_peak = np.argmax(vm[th_x:window_end]) + th_x
    v_peak = vm[x_peak]

    # find fAHP trough
    rising_x = np.where(dvdt[x_peak+1:fAHP_window_end] > 0.)[0]
    if rising_x.any():
        fAHP_window_end = x_peak + 1 + rising_x[0]

    if fAHP_window_end - x_peak <= 0:
        return None
    x_fAHP = np.argmin(vm[x_peak:fAHP_window_end]) + x_peak
    v_fAHP = vm[x_fAHP]
    fAHP = v_before - v_fAHP

    # find ADP and mAHP
    rising_x = np.where(dvdt[x_fAHP:ADP_window_end] > 0.)[0]
    if not rising_x.any():
        ADP = 0.
        mAHP = 0.
    else:
        falling_x = np.where(dvdt[x_fAHP + rising_x[0]:ADP_window_end] < 0.)[0]
        if not falling_x.any():
            ADP = 0.
            mAHP = 0.
        else:
            x_ADP = np.argmax(vm[x_fAHP + rising_x[0]:x_fAHP + rising_x[0] + falling_x[0]]) + x_fAHP + rising_x[0]
            if x_ADP - th_x < ADP_min_start_len:
                ADP = 0.
                mAHP = 0.
            else:
                v_ADP = vm[x_ADP]
                ADP = v_ADP - v_fAHP
                mAHP = v_before - np.min(vm[x_ADP:window_end])

    return {'v_peak': v_peak, 'th_v': th_v, 'fAHP': fAHP, 'ADP': ADP, 'mAHP': mAHP,
            'spike_detector_delay': spike_detector_delay}
