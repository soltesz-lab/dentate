import sys, os, time, gc, random, click, logging
from pprint import pprint
import numpy as np
from mpi4py import MPI
#import matplotlib.pyplot as plt
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, read_population_ranges
import h5py
from nested.utils import Context
import dentate
from dentate.env import Env
from dentate.utils import *
from dentate.InputCell import *
from dentate.stimulus import generate_spatial_offsets, generate_spatial_ratemap
from optimize_DG_PP_features import _calculate_field_distribution
from dentate.stimulus import generate_spatial_offsets, generate_mesh

logger = get_script_logger(os.path.basename(__file__))

#  MEC is divided into discrete modules with distinct grid spacing and field width. Here we assume grid cells
#  sample uniformly from 10 modules with spacing that increases exponentially from 40 cm to 8 m. While organized
#  dorsal-ventrally, there is no organization in the transverse or septo-temporal extent of their projections to DG.
#  CA3 and LEC are assumed to exhibit place fields. Their field width varies septal-temporally. Here we assume a
#  continuous exponential gradient of field widths, with the same parameters as those controlling MEC grid width.

#  custom data type for type of feature feature
feature_grid  = 0
feature_place = 1
context = Context()


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--input-params-file-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
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
def main(config, input_params_file_path, stimulus_id, template_path, coords_path, output_path, distances_namespace, io_size, chunk_size, value_chunk_size, cache_size, write_size, verbose, dry_run):
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

    config_logging(verbose)

    env = Env(comm=comm, config_file=config, template_paths=template_path)
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

    input_params = read_from_yaml(input_params_file_path)
    nmodules = input_params['number modules']
    field_width_x1 = input_params['field width params']['x1']
    field_width_x2 = input_params['field width params']['x2']
    arena_dimension = input_params['arena dimension']
    resolution = input_params['resolution']
    module_pi = input_params['probability inactive']
    module_pr = input_params['probability remaining']
    context.update(locals()) 

    gid_normed_distances = assign_cells_to_normalized_position() # Assign normalized u,v coordinates
    gid_module_assignments = assign_cells_to_module(gid_normed_distances, p_width=0.75, displace=0.0) # Determine which module a cell is in based on normalized u position
    total_num_fields, gid_attributes = determine_cell_participation(gid_module_assignments) # Determine if a cell is 1) active and; 2) how many fields? 
    cell_attributes = build_cell_attributes(gid_attributes, gid_normed_distances, total_num_fields) # Determine additional cell properties (lambda, field_width, orientation, jitter, and rate map. This will also build the data structure ({<pop>: {<cell type>: <cells>}}) containing all cells.

    if not dry_run and rank == 0:
        save_to_h5(cell_attributes)

        
def assign_cells_to_normalized_position():
    
    rank = context.comm.rank
    population_distances  = []
    gid_arc_distance      = dict()
    gid_normed_distances  = dict()

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
    for (gid, (arc_distance_u, arc_distance_v)) in viewitems(gid_arc_distance):
            normalized_u   = (arc_distance_u - min_u) / (max_u - min_u)
            normalized_v   = (arc_distance_v - min_v) / (max_v - min_v)
            gid_normed_distances[gid] = (normalized_u, normalized_v, arc_distance_u, arc_distance_v)

    return gid_normed_distances

    
def assign_cells_to_module(gid_normed_distances, p_width=2./3, displace=0.1):

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

    #plt.figure()
    #for i in xrange(len(p_density)):
    #    plt.plot(renormalized_positions, p_density[i][valid_indices])

    feature_seed_offset = int(context.env.modelConfig['Random Seeds']['Input Features'])
    local_random = np.random.RandomState()
    gid_module_assignments = dict()

    for gid, (u, _, _, _) in viewitems(gid_normed_distances):
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
        
    return gid_module_assignments
        

def determine_cell_participation(gid_module_assignments):

    input_config        = context.env.inputConfig[context.stimulus_id]
    feature_type_dict   = input_config['feature type']
    feature_seed_offset = int(context.env.modelConfig['Random Seeds']['Input Features'])
    feature_type_random = np.random.RandomState(feature_seed_offset - 1)
    num_field_random    = np.random.RandomState(feature_seed_offset - 1)

    gid_attributes         = {}
    module_probabilities   = [ _calculate_field_distribution(pi, pr) for (pi, pr) \
                               in zip(context.module_pi, context.module_pr) ]

    population_ranges = context.population_ranges
    total_num_fields  = 0
    for population in ['MPP', 'LPP']:
        gid_attributes[population] = {}

        population_start = population_ranges[population][0]
        population_count = population_ranges[population][1]

        feature_type_values_lst = []
        feature_type_prob_lst   = []
        for t, p in viewitems(feature_type_dict[population]):
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
            cell['Feature Type']  = np.array([feature_types[i]], dtype='uint8')
            nfields = 1
            if feature_types[i] == feature_grid:
                cell['Num Fields'] = np.array([nfields], dtype='uint8')
            elif feature_types[i] == feature_place:
                field_probabilities = module_probabilities[module - 1]
                field_set = [i for i in xrange(field_probabilities.shape[0])]
                nfields = num_field_random.choice(field_set, p=field_probabilities, size=(10,))[-1]
                cell['Num Fields'] = np.array([nfields], dtype='uint8')
            gid_attributes[population][gid] = cell
            total_num_fields += nfields
            #logger.info('Rank %i: computed features for gid %i' % (context.env.comm.rank, gid))
            
    return total_num_fields, gid_attributes


def _fields_per_module(gid_attributes, modules):
    fields_per_module_dict = {mod + 1: [0,0] for mod in modules}
    for population in gid_attributes:
        this_gid_attributes = gid_attributes[population]
        for gid in this_gid_attributes:
            module  = this_gid_attributes[gid]['Module'][0]
            nfields = this_gid_attributes[gid]['Num Fields'][0]
            #if gid_attributes[gid]['Cell Type'][0] == feature_place:
            fields_per_module_dict[module][0] += 1
            fields_per_module_dict[module][1] += nfields
    return fields_per_module_dict

def build_cell_attributes(gid_attributes, gid_normed_distances, total_num_fields):
    
    modules        = np.arange(context.nmodules)
    curr_module    = {mod + 1: int(0) for mod in modules}
    feature_seed_offset = int(context.env.modelConfig['Random Seeds']['Input Features'])
    
    local_random        = np.random.RandomState(feature_seed_offset - 1)
    grid_orientation    = [ local_random.uniform(0., np.pi/3.) for i in range(context.nmodules) ]
    field_width_params  = [context.field_width_x1, context.field_width_x2]
    field_width         = lambda x: 40. + field_width_params[0] * (np.exp(x / field_width_params[1]) - 1.)
    max_field_width     = field_width(1.)
    module_widths       = [ field_width(float(module) / np.max(modules)) for module in modules ]

    xp, yp = generate_mesh(scale_factor=1., arena_dimension=context.arena_dimension, resolution=context.resolution)
    nx, ny = xp.shape
    ratemap_kwargs = {'a': 0.70 , 'b': -1.5, 'c': 0.90}

    field_module_distribution = _fields_per_module(gid_attributes, modules)
    xy_offset_module_dict    = { mod + 1: None for mod in modules }
    
    for mod in field_module_distribution:
        module_width = module_widths[mod - 1]
        scale_factor  = (module_width / 100. / 2.) + 1.

        xy_offsets, _, _, _ = generate_spatial_offsets(field_module_distribution[mod][1], arena_dimension=100., scale_factor=scale_factor)
        local_random.shuffle(xy_offsets)
        xy_offset_module_dict[mod] = np.asarray(xy_offsets, dtype='float32')

    for population in gid_attributes.keys():
        for gid, cell in viewitems(gid_attributes[population]):

            _, _, u, v = gid_normed_distances[gid]
            cell['U Distance'] = np.asarray([u], dtype='float32')
            cell['V Distance'] = np.asarray([v], dtype='float32')
            cell['gid'] = np.asarray([gid], dtype='int32')
            
            cell['Nx']  = np.array([nx], dtype='int32')
            cell['Ny']  = np.array([ny], dtype='int32')
            
            local_random.seed(feature_seed_offset + gid)
            ftype   = cell['Feature Type'][0]
            module  = cell['Module'][0]
            nfields = cell['Num Fields'][0]
            if ftype == feature_grid:
                cell_spacing     = []
                cell_orientation = []
                for n in xrange(nfields):
                    this_spacing      = module_widths[module - 1]
                    this_orientation  = grid_orientation[module - 1]
                    delta_spacing     = local_random.uniform(-10., 10.)
                    delta_orientation = local_random.uniform(np.deg2rad(-15.), np.deg2rad(15.))
                    cell_spacing.append(this_spacing + delta_spacing)
                    cell_orientation.append(this_orientation + delta_orientation)
                cell['Grid Spacing']     = np.asarray(cell_spacing, dtype='float32')
                cell['Grid Orientation'] = np.asarray(cell_orientation, dtype='float32')            

            elif ftype == feature_place:
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
            
            rate_map = generate_spatial_ratemap(cell['Feature Type'][0], cell, None, xp, yp, 20., 20., ramp_up_period=None, **ratemap_kwargs)
            cell['Rate Map'] = rate_map.reshape(-1,).astype('float32')

    return gid_attributes
        
def save_to_h5(cell_attributes):

    for population in cell_attributes.keys():
        place_cells, grid_cells = {}, {}
        for gid, cell in viewitems(cell_attributes[population]):

            if cell['Feature Type'][0] == feature_grid:
                grid_cells[gid] = cell
            elif cell['Feature Type'][0] == feature_place:
                place_cells[gid] = cell
                
        append_cell_attributes(context.output_path, population, grid_cells, namespace='Grid Input Features',\
                               comm=context.comm, io_size=context.io_size, chunk_size=context.chunk_size,\
                               value_chunk_size=context.value_chunk_size)

        append_cell_attributes(context.output_path, population, place_cells, namespace='Place Input Features',\
                               comm=context.comm, io_size=context.io_size, chunk_size=context.chunk_size,\
                               value_chunk_size=context.value_chunk_size)
    

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
