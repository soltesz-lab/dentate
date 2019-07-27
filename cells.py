import collections, os, sys, traceback, copy, datetime, math
import numpy as np
from dentate.neuron_utils import h, d_lambda, default_hoc_sec_lists, default_ordered_sec_types, freq
from dentate.utils import get_module_logger, map, range, zip, zip_longest, viewitems, read_from_yaml, write_to_yaml
from neuroh5.h5py_io_utils import select_tree_attributes
from neuroh5.io import read_cell_attribute_selection, read_graph_selection

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
            Type asscoiated with the segment according to SWC standards
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
        print("found 'multiple cylinder soma' w/ total soma surface=*.3f" % total_surf)

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


class BiophysCell(object):
    """
    A Python wrapper for neuronal cell objects specified in the NEURON language hoc.
    Extends STree to provide an tree interface to facilitate:
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
        self.tree = STree2()  # Builds a simple tree to store nodes of type 'SHocNode'
        self.count = 0  # Keep track of number of nodes
        if env is not None:
            for sec_type in env.SWC_Types:
                if sec_type not in default_ordered_sec_types:
                    raise AttributeError('Warning! unexpected SWC Type definitions found in Env')
        self.nodes = {key: [] for key in default_ordered_sec_types}
        self.mech_file_path = mech_file_path
        self.init_mech_dict = dict()
        self.mech_dict = dict()
        self.random = np.random.RandomState()
        self.random.seed(self.gid)
        self.spike_detector = None
        self.spike_onset_delay = 0.
        self.hoc_cell = hoc_cell
        if hoc_cell is not None:
            import_morphology_from_hoc(self, hoc_cell)
        if self.mech_file_path is not None:
            import_mech_dict_from_file(self, self.mech_file_path)
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
    segment to achieve the d esired spatial and temporal resolution. This method returns the number of segments to set
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
        print('import_morphology_from_hoc: problem locating soma section to act as root')
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
    this_netcon = h.NetCon(getattr(sec(loc), param), target, sec=sec)
    this_netcon.delay = delay
    this_netcon.weight[0] = weight
    this_netcon.threshold = threshold
    return this_netcon


def init_spike_detector(cell, node=None, distance=100., threshold=-30, delay=0., onset_delay=0.):
    """
    Initializes the spike detector in the given cell according to the
    given arguments or a spike detector configuration of the mechanism
    dictionary of the cell, if one exists.

    :param cell: :class:'BiophysCell'
    :param node [optional]:  :class:'SHocNode
    :param distance: float
    :param threshold: float
    :param delay: float
    :param onset_delay: float
    """
    if 'spike detector' in cell.mech_dict:
        config = cell.mech_dict['spike detector']
        node = getattr(cell, config['section'])[0]
        distance = config['distance']
        threshold = config['threshold']
        delay = config['delay']
        onset_delay = config['onset delay']

    if node is None:
        if cell.axon:
            node = cell.axon[0]
        elif cell.soma:
            node = cell.soma[0]
        else:
            raise RuntimeError('init_spike_detector: cell has neither soma nor axon compartment')

    if node in cell.axon:
        sec_seg_locs = [seg.x for seg in node.sec]
        if get_distance_to_node(cell, cell.tree.root, node, loc=sec_seg_locs[-1]) < distance:
            cell.spike_detector = connect2target(cell, node.sec, loc=1., delay=delay, threshold=threshold)
        else:
            for loc in sec_seg_locs:
                if get_distance_to_node(cell, cell.tree.root, node, loc=loc) >= distance:
                    cell.spike_detector = connect2target(cell, node.sec, loc=loc, delay=delay, threshold=threshold)
                    break
    else:
        cell.spike_detector = connect2target(cell, node.sec, loc=0.5, delay=delay, threshold=threshold)

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


def correct_cell_for_spines_g_pas(cell, env, verbose):
    """
    If not explicitly modeling spine compartments for excitatory synapses, this method scales g_pas in all
    dendritic sections proportional to the number of excitatory synapses contained in each section.
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param verbose: bool
    """
    soma_g_pas = cell.mech_dict['soma']['pas']['g']['value']
    for sec_type in ['basal', 'trunk', 'apical', 'tuft']:
        for node in cell.nodes[sec_type]:
            correct_node_for_spines_g_pas(node, env, cell.gid, soma_g_pas, verbose=verbose)


def correct_cell_for_spines_cm(cell, env, verbose=True):
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
                      append=False):
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
                                    max=max, min_loc=min_loc, max_loc=max_loc, outside=outside, custom=custom)
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
                update_mechanism_by_node(cell, node, mech_name, mech_content)

    except Exception as e:
        cell.mech_dict = copy.deepcopy(backup_mech_dict)
        traceback.print_tb(sys.exc_info()[2])
        if param_name is not None:
            print('modify_mech_param: problem modifying mechanism: %s parameter: %s in node: %s' %
                  (mech_name, param_name, node.name))
        else:
            print('modify_mech_param: problem modifying mechanism: %s in node: %s' % (mech_name, node.name))
        raise e


def update_mechanism_by_node(cell, node, mech_name, mech_content, verbose=True):
    """
    This method loops through all the parameters for a single mechanism specified in the mechanism dictionary and
    calls parse_mech_rules to interpret the rules and set the values for the given node.
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
                parse_mech_rules(cell, node, mech_name, param_name, mech_content[param_name], verbose=verbose)
            elif isinstance(mech_content[param_name], list):
                for mech_content_entry in mech_content[param_name]:
                    parse_mech_rules(cell, node, mech_name, param_name, mech_content_entry, verbose=verbose)
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


def parse_mech_rules(cell, node, mech_name, param_name, rules, donor=None, verbose=True):
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
        donor = get_donor(cell, node, rules['origin'])
        if donor is None:
            raise RuntimeError('parse_syn_mech_rules: problem identifying donor of origin_type: %s for mechanism: '
                               '%s parameter: %s in sec_type: %s' %
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
            init_nseg(node.sec, baseline, verbose=verbose)
        else:
            setattr(node.sec, param_name, baseline)
            init_nseg(node.sec, get_spatial_res(cell, node), verbose=verbose)
        node.reinit_diam()
    elif 'custom' in rules:
        parse_custom_mech_rules(cell, node, mech_name, param_name, baseline, rules, donor, verbose=verbose)
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
    except Exception as e:
        print('inherit_mech_param: problem inheriting mechanism: %s parameter: %s from sec_type: %s' %
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


def parse_custom_mech_rules(cell, node, mech_name, param_name, baseline, rules, donor, verbose=True):
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
    :param verbose: bool
    """
    if 'func' not in rules['custom'] or rules['custom']['func'] is None:
        raise RuntimeError('parse_custom_mech_rules: no custom function provided for mechanism: %s parameter: %s in '
                           'sec_type: %s' % (mech_name, param_name, node.type))
    if rules['custom']['func'] in globals() and isinstance(globals()[rules['custom']['func']], collections.Callable):
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
        parse_mech_rules(cell, node, mech_name, param_name, new_rules, donor, verbose=verbose)


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
    num_exc_syns = len(syn_attrs.filter_synapses(cell.gid, \
                                                 syn_sections=[node.index], \
                                                 syn_types=[env.Synapse_Types['excitatory']]))
    num_inh_syns = len(syn_attrs.filter_synapses(cell.gid, \
                                                 syn_sections=[node.index], \
                                                 syn_types=[env.Synapse_Types['inhibitory']]))

    diams_str = ', '.join('%.2f' % node.sec.diam3d(i) for i in range(node.sec.n3d()))
    report = 'node: %s, L: %.1f, diams: [%s], children: %i, exc_syns: %i, inh_syns: %i' % \
             (node.name, node.sec.L, diams_str, len(node.children), num_exc_syns, num_inh_syns)
    if node.parent is not None:
        report += ', parent: %s' % node.parent.name
    logger.info(report)
    for child in node.children:
        report_topology(cell, env, child)


def make_neurotree_graph(neurotree_dict):
    """
    Creates a graph of sections that follows the topological organization of the given neuron.
    :param neurotree_dict:
    :return: NetworkX.DiGraph
    """
    import networkx as nx

    sec_nodes = neurotree_dict['section_topology']['nodes']
    sec_src = neurotree_dict['section_topology']['src']
    sec_dst = neurotree_dict['section_topology']['dst']

    sec_graph = nx.DiGraph()
    for i, j in zip(sec_src, sec_dst):
        sec_graph.add_edge(i, j)

    return sec_graph


def make_neurotree_cell(template_class, gid=0, dataset_path="", neurotree_dict={}):
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
    swc_type = neurotree_dict['swc_type']
    cell = template_class(gid, dataset_path, secnodes, vlayer, vsrc, vdst, vx, vy, vz, vradius, swc_type)
    return cell


def make_hoc_cell(env, pop_name, gid, neurotree_dict=False):
    """

    :param env:
    :param gid:
    :param population:
    :return:
    """
    dataset_path = env.dataset_path if env.dataset_path is not None else ""
    data_file_path = env.data_file_path
    template_name = env.celltypes[pop_name]['template']
    assert (hasattr(h, template_name))
    template_class = getattr(h, template_name)

    if neurotree_dict:
        hoc_cell = make_neurotree_cell(template_class, neurotree_dict=neurotree_dict, gid=gid,
                                       dataset_path=dataset_path)
    else:
        if pop_name in env.cellAttributeInfo and 'Trees' in env.cellAttributeInfo[pop_name]:
            raise Exception('make_hoc_cell: morphology for population %s gid: %i is not provided' %
                            data_file_path, pop_name, gid)
        else:
            hoc_cell = template_class(gid, dataset_path)

    return hoc_cell


def make_input_source(env, gid, pop_id, input_source_dict):
    """
    Instantiates an input generator according to the given cell template.
    """

    input_sources = input_source_dict[pop_id]
    input_gen = input_sources['gen']
    if input_gen is None:
        cell = h.VecStimCell(gid)
        if 'spiketrains' in input_sources:
            spk_inds = input_sources['spiketrains']['gid']
            spk_ts = input_sources['spiketrains']['t']
            data = spk_ts[np.where(spk_inds == gid)]
            cell.pp.play(h.Vector(data))
    else:
        template_name = input_gen['template']
        param_values = input_gen['params']
        template = getattr(h, template_name)
        params = [param_values[p] for p in env.netclamp_config.template_params[template_name]]
        cell = template(gid, *params)

    return cell


def get_biophys_cell(env, pop_name, gid, tree_dict=None, synapses_dict=None, load_synapses=True, load_edges=True,
                     load_weights=False, set_edge_delays=True, mech_file_path=None):
    """
    :param env: :class:'Env'
    :param pop_name: str
    :param gid: int
    :param tree_dict: dict
    :param synapses_dict: dict
    :param load_synapses: bool
    :param load_edges: bool
    :param load_weights: bool
    :param set_edge_delays: bool
    :param mech_file_path: str (path)
    :return: :class:'BiophysCell'
    """
    env.load_cell_template(pop_name)
    if tree_dict is None:
        tree_dict = select_tree_attributes(gid, env.comm, env.data_file_path, pop_name)
    hoc_cell = make_hoc_cell(env, pop_name, gid, neurotree_dict=tree_dict)
    cell = BiophysCell(gid=gid, pop_name=pop_name, hoc_cell=hoc_cell, env=env, mech_file_path=mech_file_path)
    syn_attrs = env.synapse_attributes
    synapse_config = env.celltypes[pop_name]['synapses']

    if load_weights and 'weights namespace' in synapse_config:
        weights_namespace = synapse_config['weights namespace']
    else:
        weights_namespace = None

    if load_synapses:
        if synapses_dict is not None:
            syn_attrs.init_syn_id_attrs(gid, synapses_dict)
        elif (pop_name in env.cellAttributeInfo) and ('Synapse Attributes' in env.cellAttributeInfo[pop_name]):
            synapses_iter = read_cell_attribute_selection(env.data_file_path, pop_name, [gid], 'Synapse Attributes',
                                                          comm=env.comm)
            syn_attrs.init_syn_id_attrs_from_iter(synapses_iter)

            if weights_namespace is not None:
                cell_weights_iter = read_cell_attribute_selection(env.data_file_path, pop_name, [gid],
                                                                  weights_namespace, comm=env.comm)
            else:
                cell_weights_iter = None

            if cell_weights_iter is not None:
                for gid, cell_weights_dict in cell_weights_iter:
                    weights_syn_ids = cell_weights_dict['syn_id']
                    for syn_name in (syn_name for syn_name in cell_weights_dict if syn_name != 'syn_id'):
                        weights_values = cell_weights_dict[syn_name]
                        syn_attrs.add_mech_attrs_from_iter(
                            gid, syn_name,
                            zip_longest(weights_syn_ids, map(lambda x: {'weight': x}, weights_values)))
                        logger.info('get_biophys_cell: gid: %i; found %i %s synaptic weights' %
                                    (gid, len(cell_weights_dict[syn_name]), syn_name))
        else:
            logger.error('get_biophys_cell: synapse attributes not found for %s: gid: %i' % (pop_name, gid))
            raise Exception

    if load_edges:
        if os.path.isfile(env.connectivity_file_path):
            (graph, a) = read_graph_selection(file_name=env.connectivity_file_path, selection=[gid],
                                              namespaces=['Synapses', 'Connections'], comm=env.comm)
            if pop_name in env.projection_dict:
                for presyn_name in env.projection_dict[pop_name]:
                    edge_iter = graph[pop_name][presyn_name]
                    syn_attrs.init_edge_attrs_from_iter(pop_name, presyn_name, a, edge_iter, set_edge_delays)
            else:
                logger.error('get_biophys_cell: connection attributes not found for %s: gid: %i' % (pop_name, gid))
                raise Exception
        else:
            logger.error('get_biophys_cell: connection file %s not found' % env.connectivity_file_path)
            raise Exception
    env.biophys_cells[pop_name][gid] = cell
    return cell


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
    env.cells.append(hoc_cell)
    # Tell the ParallelContext that this cell is a spike source
    # for all other hosts. NetCon is temporary.
    if hasattr(cell, 'spike_detector'):
        nc = cell.spike_detector
    else:
        nc = hoc_cell.connect2target(h.nil)
    nc.delay = max(env.dt, nc.delay)
    env.pc.cell(gid, nc, 1)
    # Record spikes of this cell
    env.pc.spike_record(gid, env.t_vec, env.id_vec)
    # if the spike detector is located in a compartment other than soma,
    # record the spike time delay relative to soma
    if hasattr(cell, 'spike_onset_delay'):
        env.spike_onset_delay[gid] = cell.spike_onset_delay

    
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
