import sys, os, time, gc, random, click, logging
from pprint import pprint
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, read_population_ranges
import h5py
from nested.utils import Context
import dentate
from dentate.env import Env
import dentate.utils as utils
from dentate.utils import list_find, list_argsort, get_script_logger
from dentate.stimulus import generate_spatial_offsets

from optimize_DG_PP_features import calculate_field_distribution
from optimize_DG_PP_features import acquire_fields_per_cell
from optimize_DG_PP_features import _generate_mesh
from dentate.stimulus import generate_spatial_offsets
from dentate.stimulus import generate_spatial_ratemap

script_name = 'generate_DG_PP_features_v2.py'
utils.config_logging(True)
logger = utils.get_script_logger(script_name)

#  MEC is divided into discrete modules with distinct grid spacing and field width. Here we assume grid cells
#  sample uniformly from 10 modules with spacing that increases exponentially from 40 cm to 8 m. While organized
#  dorsal-ventrally, there is no organization in the transverse or septo-temporal extent of their projections to DG.
#  CA3 and LEC are assumed to exhibit place fields. Their field width varies septal-temporally. Here we assume a
#  continuous exponential gradient of field widths, with the same parameters as those controlling MEC grid width.

#  custom data type for type of feature feature
feature_grid  = 0
feature_place = 1

feature_MPP = 0
feature_LPP = 1

module_pi = [0.012, 0.313, 0.500, 0.654, 0.723, 0.783, 0.830, 0.852, 0.874, 0.890]
module_pr = [0.342, 0.274, 0.156, 0.125, 0.045, 0.038, 0.022, 0.018, 0.013, 0.004]

context = Context()


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--stimulus-id", type=int, default=0)
@click.option("--template-path",required=True, type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-path", required=True, type=click.Path(file_okay=True, dir_okay=False))
@click.option("--distances-namespace", type=str, default='Arc Distances')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--write-size", type=int, default=1)
@click.option("--verbose", '-v', is_flag=True)
@click.option("--dry-run", is_flag=True)
def main(config, stimulus_id, template_path, coords_path, output_path, distances_namespace, io_size, chunk_size, value_chunk_size, cache_size, write_size, verbose, dry_run):
    """

    :param config:
    :param coords_path:
    :param distances_namespace:
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    :param write_size:
    :param dry_run:
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank


    env = Env(comm=comm, configFile=config, templatePaths=template_path)
    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)

    if (not dry_run) and (rank==0):
        if not os.path.isfile(output_path):
            input_file  = h5py.File(coords_path,'r')
            output_file = h5py.File(output_path,'w')
            input_file.copy('/H5Types',output_file)
            input_file.close()
            output_file.close()
    comm.barrier()
    population_ranges = read_population_ranges(coords_path, comm)[0]
    context.update(locals())

    assign_cells_to_normalized_position() # Assign normalized u,v coordinates
    assign_cells_to_module(p_width=0.75, displace=0.0) # Determine which module a cell is in based on normalized u position
    total_num_fields = determine_cell_participation() # Determine if a cell is 1) active and; 2) how many fields? 
    build_cell_attributes(total_num_fields) # Determine additional cell properties (lambda, field_width, orientation, jitter, and rate map. This will also build the data structure ({<pop>: {<cell type>: <cells>}}) containing all cells.
    if not dry_run and rank == 0:
        if verbose:
            print('saving to %s' % output_path)
        save_to_h5()
        if verbose:
            print('h5 file saved')

    if verbose:
        plot_module_assignment_histogram()
    
def plot_module_assignment_histogram():

    module_bounds, module_counts, module_density = calculate_module_density()

    fig, (ax1, ax2, ax3) = plt.subplots(3,1)

    ax1.bar(np.arange(10)+1, module_counts)
    ax1.set_xlabel('Module')
    ax1.set_ylabel('Count')

    ax2.bar(np.arange(10)+1, module_density)
    ax2.set_xlabel('Module')
    ax2.set_ylabel('Density')

    for (i, bounds) in enumerate(module_bounds):
        ax3.plot([bounds[0],bounds[1]], [i+1,i+1], label='%i' % (i+1))
    ax3.set_xlabel('Normalized Bounds')
    ax3.set_ylabel('Module')
    ax3.legend(frameon=False, framealpha=0.5, loc='center left')

    fig, (ax1, ax2) = plt.subplots(2,1)
    normalized_u_positions = [norm_u for (norm_u,_,_,_) in context.gid_normed_distance.values()]
    absolute_u_positions   = [u for (_,_,u,_) in context.gid_normed_distance.values()]
    absolute_v_positions   = [v for (_,_,_,v) in context.gid_normed_distance.values()]
    hist_norm, edges_norm  = np.histogram(normalized_u_positions, bins=25)
    hist_abs, edges_abs    = np.histogram(absolute_u_positions, bins=100)
    hist_v_abs, edges_v_abs = np.histogram(absolute_v_positions, bins=100)

    ax1.plot(edges_norm[1:], hist_norm)
    ax1.set_xlabel('Normalized septo-temporal position')
    ax1.set_ylabel('Cell count')

    ax2.plot(edges_abs[1:], hist_abs)
    ax2.set_xlabel('Absolute septo-temporal position')
    ax2.set_ylabel('Cell Count')

    fig, ax = plt.subplots()
    module_pos_dictionary = dict()
    for gid in context.gid_normed_distance:
        norm_u,_,_,_ = context.gid_normed_distance[gid]
        module       = context.gid_module_assignments[gid]
        if module_pos_dictionary.has_key(module):
            module_pos_dictionary[module].append(norm_u)
        else:
            module_pos_dictionary[module] = [norm_u]

    for module in module_pos_dictionary:
        positions = module_pos_dictionary[module]
        hist_pos, _ = np.histogram(positions, bins=edges_norm)
        hist_pos = hist_pos.astype('float32')
        ax.plot(edges_norm[1:], hist_pos / hist_norm)
    ax.legend(['%i' % (i+1) for i in xrange(10)])

    plt.show()

def assign_cells_to_normalized_position():
    rank = context.comm.rank
    population_distances = []
    gid_arc_distance     = dict()
    gid_normed_distance  = dict()

    for population in ['MPP', 'LPP']:
        #(population_start, population_count) = context.population_ranges[population]
        attr_gen = NeuroH5CellAttrGen(context.coords_path, population, namespace=context.distances_namespace,
                                      comm=context.comm, io_size=context.io_size, cache_size=context.cache_size)

        for (gid, distances_dict) in attr_gen:
            if gid is None:
                break
            arc_distance_u = distances_dict['U Distance'][0]
            arc_distance_v = distances_dict['V Distance'][0]
            gid_arc_distance[gid] = (arc_distance_u, arc_distance_v)
            population_distances.append((arc_distance_u, arc_distance_v))

    population_distances = np.asarray(population_distances, dtype='float32')
        
    min_u, max_u = np.min(population_distances[:,0]), np.max(population_distances[:,0])
    min_v, max_v = np.min(population_distances[:,1]), np.max(population_distances[:,1])
    for (gid, (arc_distance_u, arc_distance_v)) in gid_arc_distance.iteritems():
            normalized_u   = (arc_distance_u - min_u) / (max_u - min_u)
            normalized_v   = (arc_distance_v - min_v) / (max_v - min_v)
            gid_normed_distance[gid] = (normalized_u, normalized_v, arc_distance_u, arc_distance_v)
    context.update({'gid_normed_distance': gid_normed_distance})

def assign_cells_to_module(p_width=2./3, displace=0.1):
    if not hasattr(context, 'gid_normed_distance'):
        raise Exception('Normed distances have not been calculated...')

    offsets   = np.linspace(-displace, 1. + displace, 10)
    positions = np.linspace(0., 1., 1000)
    p_module  = lambda width, offset: lambda x: np.exp(-((x - offset) / (width / 3. / np.sqrt(2.))) ** 2.)
    p_modules = np.array([p_module(p_width , offset)(positions) for (i,offset) in enumerate(offsets)], dtype='float32')
    p_sum     = np.sum(p_modules, axis=0)
    p_density = np.divide(p_modules, p_sum) # 10 x 1000
    #p_modules_max = np.max(p_density, axis=1)
    #mean_peak     = np.mean(p_modules_max[1:-1])

    left_offset  = 0
    right_offset = len(positions)
    valid_indices   = np.arange(left_offset, right_offset, 1)
    valid_positions = positions[valid_indices]
    renormalized_positions = (valid_positions - np.min(valid_positions)) / (np.max(valid_positions) - np.min(valid_positions))

    plt.figure()
    for i in xrange(len(p_density)):
        plt.plot(renormalized_positions, p_density[i][valid_indices])

    feature_seed_offset = int(context.env.modelConfig['Random Seeds']['Input Features'])
    local_random = np.random.RandomState()
    gid_module_assignments = dict()

    for gid, (u, _, _, _) in context.gid_normed_distance.iteritems():
        local_random.seed(gid + feature_seed_offset)
        interpolated_density_values = []
        for i in xrange(len(p_density)):
            module_density = p_density[i][valid_indices]
            interpolated_density_values.append(np.interp(u, renormalized_positions, module_density))
        remaining_density = 1. - np.sum(interpolated_density_values)
        max_density_index = np.argmax(interpolated_density_values)
        interpolated_density_values[max_density_index] += remaining_density
        module = local_random.choice(np.arange(len(p_density))+1, p=interpolated_density_values, size=(1,))
        gid_module_assignments[gid] = module[0]
    context.update({'gid_module_assignments': gid_module_assignments})

def calculate_module_density():
    if not hasattr(context, 'gid_module_assignments'):
        raise Exception('Cells need to be assigned to modules first...')
    if not hasattr(context, 'gid_normed_distance'):
        raise Exception('Arc distances have not been extracted...')

    module_bounds = [[1.0, 0.0] for _ in xrange(10)]
    module_counts = [0 for _ in xrange(10)]
    gid_module_assignments = context.gid_module_assignments
    gid_normed_distance    = context.gid_normed_distance
    
    for (gid,module) in gid_module_assignments.iteritems():
        normed_u, _, _, _ = gid_normed_distance[gid]
        if normed_u < module_bounds[module-1][0]:
            module_bounds[module - 1][0] = normed_u
        if normed_u > module_bounds[module - 1][1]:
            module_bounds[module - 1][1] = normed_u
        module_counts[module - 1] += 1

    module_widths  = [y-x for [x,y] in module_bounds]
    module_density = np.divide(module_counts, module_widths)
    return module_bounds, module_counts, module_density
        

def determine_cell_participation():
    if not hasattr(context, 'gid_module_assignments'):
        raise Exception('Cells need to be assigned to modules first...')

    input_config        = context.env.inputConfig[context.stimulus_id]
    feature_type_dict   = input_config['feature type']
    feature_seed_offset = int(context.env.modelConfig['Random Seeds']['Input Features'])
    feature_type_random = np.random.RandomState(feature_seed_offset - 1)
    num_field_random    = np.random.RandomState(feature_seed_offset - 1)

    gid_module_assignments = context.gid_module_assignments
    gid_attributes         = {gid: None for gid in gid_module_assignments}
    module_probabilities   = [calculate_field_distribution(pi, pr) for (pi, pr) in zip(module_pi, module_pr)]

    population_ranges = context.population_ranges
    total_num_fields  = 0
    for population in ['MPP', 'LPP']:
        (population_start, population_count) = population_ranges[population]
        if population == 'MPP':
            feature_population = feature_MPP
        elif population == 'LPP':
            feature_population = feature_LPP

        feature_type_values_lst = []
        feature_type_prob_lst   = []
        for t, p in feature_type_dict[population].iteritems():
            feature_type_values_lst.append(t)
            feature_type_prob_lst.append(p)
        feature_type_values = np.asarray(feature_type_values_lst)
        feature_type_probs  = np.asarray(feature_type_prob_lst)
        feature_types       = feature_type_random.choice(feature_type_values, p=feature_type_probs, size=(population_count,))    

        population_end = population_start + population_count
        gids = np.arange(population_start, population_end, 1)
        for (i,gid) in enumerate(gids):
            num_field_random.seed(feature_seed_offset + gid)
            cell   = {}
            module = gid_module_assignments[gid]
            cell['Module']     = np.array([module], dtype='uint8')
            cell['Population'] = np.array([feature_population], dtype='uint8')           
            cell['Cell Type']  = np.array([feature_types[i]], dtype='uint8')
            nfields = 1
            if feature_types[i] == feature_grid:
                cell['Num Fields'] = np.array([nfields], dtype='uint8')
            elif feature_types[i] == feature_place:
                field_probabilities = module_probabilities[module - 1]
                field_set = [i for i in xrange(field_probabilities.shape[0])]
                nfields   = num_field_random.choice(field_set, p=field_probabilities, size=(10,))[-1]
                cell['Num Fields'] = np.array([nfields], dtype='uint8')
            gid_attributes[gid] = cell
            total_num_fields += nfields
    context.update({'gid_attributes': gid_attributes})
    return total_num_fields

def _fields_per_module(gid_attributes, modules):
    fields_per_module_dict = {mod + 1: [0,0] for mod in modules}
    for gid in gid_attributes:
        module  = gid_attributes[gid]['Module'][0]
        nfields = gid_attributes[gid]['Num Fields'][0]
        #if gid_attributes[gid]['Cell Type'][0] == feature_place:
        fields_per_module_dict[module][0] += 1
        fields_per_module_dict[module][1] += nfields
    return fields_per_module_dict
    

def build_cell_attributes(total_num_fields):
    nmodules       = 10
    modules        = np.arange(nmodules)
    curr_module    = {mod + 1: int(0) for mod in modules}
    feature_seed_offset = int(context.env.modelConfig['Random Seeds']['Input Features'])
    local_random        = np.random.RandomState(feature_seed_offset - 1)
    grid_orientation    = [local_random.uniform(0., np.pi/3.) for i in xrange(nmodules)]
    field_width_params  = [35.0,   0.32]  # slope, tau
    field_width         = lambda x: 40. + field_width_params[0] * (np.exp(x / field_width_params[1]) - 1.)
    max_field_width     = field_width(1.)
    module_widths       = [field_width(float(module) / np.max(modules)) for module in modules]

    xp, yp = _generate_mesh()
    nx, ny = xp.shape
    ratemap_kwargs = {'a': 0.70 , 'b': -1.5, 'c': 0.90}

    gid_attributes = context.gid_attributes
    field_module_distribution = _fields_per_module(gid_attributes, modules)
    print(field_module_distribution)
    xy_offset_module_dict    = {mod + 1: None for mod in modules}
    for mod in field_module_distribution:
        module_width = module_widths[mod - 1]
        scale_factor  = 1. + (module_width * np.cos(np.pi/4.) / 100.)
        xy_offsets, _, _, _ = generate_spatial_offsets(field_module_distribution[mod][1], arena_dimension=100., scale_factor=scale_factor)
        local_random.shuffle(xy_offsets)
        xy_offset_module_dict[mod] = np.asarray(xy_offsets, dtype='float32')

    gid_attributes = context.gid_attributes
    for gid in gid_attributes:
        cell        = gid_attributes[gid]
        cell['gid'] = np.array([gid], dtype='int32')
        cell['Nx']  = np.array([nx], dtype='int32')
        cell['Ny']  = np.array([ny], dtype='int32')

        local_random.seed(feature_seed_offset + gid)
        ctype   = cell['Cell Type'][0]
        module  = cell['Module'][0]
        nfields = cell['Num Fields'][0]
        if ctype == feature_grid:
            cell_spacing     = []
            cell_orientation = []
            for n in xrange(nfields):
                this_spacing      = module_widths[module - 1]
                this_orientation  = grid_orientation[module - 1]
                delta_spacing     = local_random.uniform(-10., 10.)
                delta_orientation = local_random.uniform(-10., 10.)
                cell_spacing.append(this_spacing + delta_spacing)
                cell_orientation.append(this_orientation + delta_orientation)
            cell['Grid Spacing']     = np.asarray(cell_spacing, dtype='float32')
            cell['Grid Orientation'] = np.asarray(cell_orientation, dtype='float32')            

        elif ctype == feature_place:
            cell_width = []
            for n in xrange(nfields):
                this_width    = module_widths[module - 1]
                delta_spacing = local_random.uniform(-10., 10.)
                cell_width.append(this_width + delta_spacing)
            cell['Field Width'] = np.asarray(cell_width, dtype='float32')

        curr_n = curr_module[module]
        cell['X Offset'] = np.asarray(xy_offset_module_dict[module][curr_n:curr_n+nfields,0], dtype='float32')
        cell['Y Offset'] = np.asarray(xy_offset_module_dict[module][curr_n:curr_n+nfields,1], dtype='float32')
        curr_module[module] += nfields
        
        rate_map = generate_spatial_ratemap(cell['Cell Type'][0], cell, None, xp, yp, 20., 20., ramp_up_period=None, **ratemap_kwargs)
        cell['Rate Map'] = rate_map.reshape(-1,).astype('float32')

def save_to_h5():
    if not hasattr(context, 'gid_attributes'):
        raise Exception('Need gid attributes dictionary')
    gid_attributes = context.gid_attributes
    lpp_place_cells = {}
    mpp_place_cells, mpp_grid_cells = {}, {}

    for gid in gid_attributes:
        cell = gid_attributes[gid]

        if cell['Cell Type'][0] == feature_grid:
            mpp_grid_cells[gid] = cell
        elif cell['Cell Type'][0] == feature_place:
            if cell['Population'][0] == feature_LPP:
                lpp_place_cells[gid] = cell
            elif cell['Population'][0] == feature_MPP:
                mpp_place_cells[gid] = cell    

    lpp_place_gid = lpp_place_cells.keys()
    mpp_place_gid = mpp_place_cells.keys()
    mpp_grid_gid  = mpp_grid_cells.keys()

    assert(bool(set(lpp_place_gid) & set(mpp_place_gid)) == False)
    assert(bool(set(lpp_place_gid) & set(mpp_grid_gid)) == False)
    assert(bool(set(mpp_place_gid) & set(mpp_grid_gid)) == False)
    

    append_cell_attributes(context.output_path, 'MPP', mpp_grid_cells, namespace='Grid Input Features',\
                           comm=context.comm, io_size=context.io_size, chunk_size=context.chunk_size,\
                           value_chunk_size=context.value_chunk_size)


    append_cell_attributes(context.output_path, 'MPP', mpp_place_cells, namespace='Place Input Features',\
                           comm=context.comm, io_size=context.io_size, chunk_size=context.chunk_size,\
                           value_chunk_size=context.value_chunk_size)


    append_cell_attributes(context.output_path, 'LPP', lpp_place_cells, namespace='Place Input Features',\
                           comm=context.comm, io_size=context.io_size, chunk_size=context.chunk_size,\
                           value_chunk_size=context.value_chunk_size)
    

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
