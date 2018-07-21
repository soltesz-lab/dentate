
"""Procedures related to gap junction connectivity generation. """

import sys, time, string, math, itertools
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean
from mpi4py import MPI
from neuron import h
from neuroh5.io import read_population_ranges, read_tree_selection, append_graph
from dentate import cells

## This logger will inherit its setting from its root logger, dentate,
## which is created in module env
logger = utils.get_module_logger(__name__)


## Compartment weights
## comp_coeff        = [-0.0315,9.4210];
## HIPP_long_weights   = [37.5;112.5;187.5]*linear_coeff(1) + linear_coeff(2);
## HIPP_short_weights  = [25;75;125]*linear_coeff(1) + linear_coeff(2);
## Apical_weights      = [37.5;112.5;187.5;262.5]*linear_coeff(1) + linear_coeff(2);
## Basal_weights       = [25;75;125;175]*linear_coeff(1) + linear_coeff(2);


## Connections based on weighted distance
## selected = datasample(transpose(1:length(distance)),round(gj_prob(pre_type,post_type)*length(distance)),'Weights',prob,'Replace',false);

def filter_by_distance(gids_a, coords_a, gids_b, coords_b, bounds, params):
    coords_tree_a = cKDTree(coords_a)
    coords_tree_a = cKDTree(coords_b)

    res = coords_tree_a.query_ball_tree(coords_b, bounds[1])

    res_dict = {}
    for i, nn in enumerate(res):
        gid_a = gids_a[i]
        nngids  = np.asarray([ gids_b[j] for j in nn ], dtype=np.int32)
        nndists = np.asarray([ euclidean(coords_a[i], coords_b[j]) for j in nn ], dtype=np.float32)
        nnprobs = np.asarray([ np.polyval(d,params) for d in nndists ], dtype=np.float32)
        res_dict[gid_a] = (nngids, nndists, nnprobs)

    return res_dict

def distance_to_root(root, sec, loc):
    """
    Returns the distance from the given section location to the middle of the given root section.
    """
    distance = 0.0
    if sec is root:
        return distance
    
    distance += loc * sec.L
    while sec.parent is not root:
        sec = sec.parent
        distance += sec.L
    distance -= 0.5 * sec.L
        
    return distance

def choose_gj_locations(cell_a, cell_b):
    apical_sections_a = cell_a.apicalidx.to_python()
    basal_sections_a  = cell_a.basalidx.to_python()
    apical_sections_b = cell_b.apicalidx.to_python()
    basal_sections_b  = cell_b.basalidx.to_python()

    if ((len(apical_sections_a) > 0) and
        (len(basal_sections_a) > 0) and
        (len(apical_sections_b) > 0) and
        (len(basal_sections_a) > 0)):
        
        sec_type = ranstream_gj.random_sample()
        if sec_type > 0.5:
            sectionidx_a = np.randint(np.min(apical_sections_a), np.max(apical_sections_a)+1)
            sectionidx_b = np.randint(np.min(apical_sections_b), np.max(apical_sections_b)+1)
            section_a = cell_a.apical[sectionidx_a]
            section_b = cell_b.apical[sectionidx_b]
        else:
            sectionidx_a = np.randint(np.min(basal_sections_a), np.max(basal_sections_a)+1)
            sectionidx_b = np.randint(np.min(basal_sections_b), np.max(basal_sections_b)+1)
            section_a = cell_a.basal[sectionidx_a]
            section_b = cell_b.basal[sectionidx_b]
    elif ((len(apical_sections_a) > 0) and
          (len(apical_sections_b) > 0)):
        sectionidx_a = np.randint(np.min(apical_sections_a), np.max(apical_sections_a)+1)
        sectionidx_b = np.randint(np.min(apical_sections_b), np.max(apical_sections_b)+1)
        section_a = cell_a.apical[sectionidx_a]
        section_b = cell_b.apical[sectionidx_b]
    elif ((len(basal_sections_a) > 0) and
          (len(basal_sections_b) > 0)):
        sectionidx_a = np.randint(np.min(basal_sections_a), np.max(basal_sections_a)+1)
        sectionidx_b = np.randint(np.min(basal_sections_b), np.max(basal_sections_b)+1)
        section_a = cell_a.basal[sectionidx_a]
        section_b = cell_b.basal[sectionidx_b]
    else raise ValueError('Cells with incompatible section types')

    position_a = max(ranstream_gj.random_sample(), 0.01)
    position_b = max(ranstream_gj.random_sample(), 0.01)

    distance_a = distance_to_root(cell_a.soma, section_a, position_a)
    distance_b = distance_to_root(cell_b.soma, section_b, position_b)
    
    return sectionidx_a, position_a, distance_a, sectionidx_b, position_b, distance_b
        
    

def generate_gap_junctions(gj_config, ranstream_gj, gids_a, gids_b, gj_probs, gj_distances
                           cell_dict_a, cell_dict_b, gj_dict)
    k = round(gj_config.prob * len(gj_distances))
    selected = ranstream_gj.choice(np.arange(0, len(gj_distances)), size=k, replace=False, p=gj_probs)
    count = len(selected)
    
    gid_dict = defaultdict(list)
    for i in selected:
        gid_a = gids_a[i]
        gid_b = gids_b[i]
        gid_dict[gid_a].append(gid_b)

    sections_a   = []
    positions_a  = []
    sections_b   = []
    positions_b  = []
    couplings_a  = []
    couplings_b  = []
    
    for gid_a,gids_b in gid_dict.iteritems():
        cell_a = cell_dict_a[gid_a]
        
        for gid_b in gids_b:
            cell_b     = cell_dict_b[gid_b]

            section_a, position_a, distance_a, section_b, position_b, distance_b = choose_gj_locations(ranstream_gj, cell_a, cell_b)
            sections_a.append(section_a)
            positions_a.append(position_a)

            sections_b.append(section_b)
            positions_b.append(position_b)

            coupling_weight_a = np.polyval(distance_a, gj_config.coupling_parameters)
            coupling_weight_b = np.polyval(distance_b, gj_config.coupling_parameters)

            coupling_a = gj_config.coupling_coefficient * coupling_weight_a
            coupling_b = gj_config.coupling_coefficient * coupling_weight_b

            couplings_a.append(coupling_a)
            couplings_b.append(coupling_b)

    gj_dict[gid_a] = ( np.asarray(gids_b, dtype=np.uint32),
                       { 'Location' : { 'Source section':  np.asarray (sections_a, dtype=np.uint32),
                                        'Source position': np.asarray (positions_a, dtype=np.float32),
                                        'Target section':  np.asarray (sections_b, dtype=np.uint32),
                                        'Target position':  np.asarray (positions_b, dtype=np.float32) },
                         'Coupling strength' : { 'Source' : np.asarray (coupling_a, dtype=np.float32),
                                                 'Target' : np.asarray (coupling_b, dtype=np.float32) } } )
            
    return count

        

def generate_gj_connections(comm, gj_config_dict, gj_seed, connectivity_namespace, connectivity_path,
                            io_size, chunk_size, value_chunk_size, cache_size,
                            dry_run=False):
    
    """Generates gap junction connectivity based on Euclidean-distance-weighted probabilities.
    :param comm: mpi4py MPI communicator
    :param gj_config: connection configuration object (instance of env.GapjunctionConfig)
    :param gj_seed: random seed for determining gap junction connectivity
    :param connectivity_namespace: namespace of gap junction connectivity attributes
    :param connectivity_path: path to gap junction connectivity file
    :param io_size: number of I/O ranks to use for parallel connectivity append
    :param chunk_size: HDF5 chunk size for connectivity file (pointer and index datasets)
    :param value_chunk_size: HDF5 chunk size for connectivity file (value datasets)
    :param cache_size: how many cells to read ahead
    """
        
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)


    start_time = time.time()

    ranstream_gj = np.random.RandomState(gj_seed)

    population_pairs = gj_config['Connection Probabilities'].keys()
    cc_params = np.asarray(gj_config['Coupling Parameters'])
    connection_params = np.asarray(['Connection Parameters'])

    for pp in population_pairs:
        if rank == 0:
            logger.info('%s <-> %s:' % (pp[0], pp[1]))
                           
    total_count = 0
    gid_count   = 0

    for (i, (pp, gj_config)) in enumerate(gj_config_dict.iteritems()):

        ranstream_gj.seed(gj_seed + i)
        
        population_a = pp[0]
        population_b = pp[1]
        
        coords_a_iter = read_cell_attributes(coords_path, population_a, namespace=coords_namespace)
        coords_b_iter = read_cell_attributes(coords_path, population_b, namespace=coords_namespace)

        clst_a = []
        gid_a  = []
        for (gid, coords_dict) in coords_a_dict:
            cell_x = coords_dict['X Coordinate'][0]
            cell_y = coords_dict['Y Coordinate'][0]
            cell_z = coords_dict['Z Coordinate'][0]
            clst_a.append(np.asarray([gid, cell_x, cell_y, cell_z]))
            gid_a.append(gid)

        sortidx_a = np.argsort(np.asarray(gid_a))
        coords_a  = np.asarray([ clst_a[i] for i in sortidx_a ])
            
        clst_b = []
        gid_b  = []
        for (gid, coords_dict) in coords_b_dict:
            cell_x = coords_dict['X Coordinate'][0]
            cell_y = coords_dict['Y Coordinate'][0]
            cell_z = coords_dict['Z Coordinate'][0]
            clst_b.append(np.asarray([cell_x, cell_y, cell_z]))

        sortidx_b = np.argsort(np.asarray(gid_b))
        coords_b  = np.asarray([ clst_b[i] for i in sortidx_b ])
            
        gj_prob_dict = filter_by_distance(gid_a[sortidx_a], coords_a, gid_b[sortidx_b], coords_b, gj_config.connection_bounds, gj_config.connection_params)

        gj_probs = []
        gj_distances = []
        gids_a = []
        gids_b = []
        for k, v in gj_prob_dict.iteritems():
            if k % rank == 0:
                for (nngids,nndists,nnprobs) in v:
                    gids_a.append(np.full(nngids.shape,k,dtype=np.int32))
                    gids_b.append(nngids)
                    gj_probs.append(nnprobs)
                    gj_distances.append(nndists)

        gids_a = np.concatenate(gids_a)
        gids_b = np.concatenate(gids_b)
        gj_probs = np.concatenate(gj_probs)
        gj_probs = gj_probs / gj_probs.sum()
        gj_distances = np.concatenate(gj_distances)
        gids_a = np.asarray(gids_a, dtype=np.uint32)
        gids_b = np.asarray(gids_b, dtype=np.uint32)
                
        cell_dict_a = {}
        selection_a = set(gids_a)
        (tree_iter_a, _) = read_tree_selection(forest_path, population_a, list(gids_a))
        for (gid,tree_dict) in tree_iter_a:
            cell_dict_a[gid] = cells.make_neurotree_cell(template_class_a, neurotree_dict=tree_dict, gid=gid)
        
        cell_dict_b = {}
        selection_b = set(gids_b)
        (tree_iter_b, _) = read_tree_selection(forest_path, population_b, list(gids_b))
        for (gid,tree_dict) in tree_iter_b:
            cell_dict_b[gid] = cells.make_neurotree_cell(template_class_b, neurotree_dict=tree_dict, gid=gid)


        gj_dict = {}
        count = generate_gap_junctions(gj_config, ranstream_gj,
                                       gids_a, gids_b, gj_probs, gj_distances
                                       cell_dict_a, cell_dict_b,
                                       gj_dict)
        
        gj_graph_dict = { pp[0]: { pp[1]: gj_dict } }

        if not dry_run:
            append_graph(connectivity_path, gj_graph_dict, io_size=io_size, comm=comm)

        total_count += count
            
    global_count = comm.gather(total_count, root=0)
    if rank == 0:
        logger.info('%i ranks took %i s to generate %i edges' % (comm.size, time.time() - start_time, np.sum(global_count)))
