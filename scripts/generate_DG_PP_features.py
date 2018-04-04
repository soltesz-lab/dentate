import sys, os, time, gc
import numpy as np
from mpi4py import MPI
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, read_population_ranges
import h5py
import dentate
from dentate.env import Env
from dentate.DG_volume import DG_volume, make_volume, make_uvl_distance
from dentate.utils import list_find, list_argsort
from dentate.stimulus import generate_spatial_offsets
import random, click, logging
logging.basicConfig()

script_name = 'generate_DG_PP_features.py'
logger = logging.getLogger(script_name)


#  MEC is divided into discrete modules with distinct grid spacing and field width. Here we assume grid cells
#  sample uniformly from 10 modules with spacing that increases exponentially from 40 cm to 8 m. While organized
#  dorsal-ventrally, there is no organization in the transverse or septo-temporal extent of their projections to DG.
#  CA3 and LEC are assumed to exhibit place fields. Their field width varies septal-temporally. Here we assume a
#  continuous exponential gradient of field widths, with the same parameters as those controlling MEC grid width.

#  x varies from 0 to 1, corresponding either to module id or septo-temporal distance
field_width_params = [35.0,   0.32]  # slope, tau
field_width = lambda x: 40. + field_width_params[0] * (np.exp(x / field_width_params[1]) - 1.)
max_field_width = field_width(1.)

#  custom data type for type of feature feature
feature_grid = 0
feature_place_field = 1


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--stimulus-id", type=int, default=0)
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
def main(config, stimulus_id, coords_path, output_path, distances_namespace, io_size, chunk_size, value_chunk_size, cache_size, write_size, verbose, dry_run):
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
    if verbose:
        logger.setLevel(logging.INFO)

    comm = MPI.COMM_WORLD
    rank = comm.rank


    env = Env(comm=comm, configFile=config)
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

    feature_config = env.stimulusConfig[stimulus_id]
    feature_type_dict = feature_config['feature type']
    
    arena_dimension = int(feature_config['trajectory']['Distance to boundary'])  # minimum distance from origin to boundary (cm)

    # make sure random seeds are not being reused for various types of stochastic sampling
    feature_seed_offset = int(env.modelConfig['Random Seeds']['Input Features'])

    local_random = random.Random()
    local_random.seed(feature_seed_offset - 1)

    feature_type_random = np.random.RandomState(feature_seed_offset - 1)
    
    # every 60 degrees repeats in a hexagonal array
    modules = range(env.modelConfig['EC']['Grid Cells']['Number of modules'])
    grid_orientation = [local_random.uniform(0., np.pi / 3.) for i in range(len(modules))]

    population_ranges = read_population_ranges(coords_path, comm)[0]

    for population in ['MPP', 'LPP']:
        if rank == 0:
            logger.info('Generating features for population %s' % population)
        (population_start, population_count) = population_ranges[population]
        gid_count = 0
        start_time = time.time()
        feature_type_value_lst = []
        feature_type_prob_lst  = []
        for t, p in feature_type_dict[population].iteritems():
            feature_type_value_lst.append(t)
            feature_type_prob_lst.append(p)
            
        feature_type_values = np.asarray(feature_type_value_lst)
        feature_type_probs  = np.asarray(feature_type_prob_lst)

        feature_types = feature_type_random.choice(feature_type_values, p=feature_type_probs,
                                                   size=(population_count,) )

        ## Generate X-Y offsets that correspond approximately to 8x8 m space
        xy_offsets,_,_ = generate_spatial_offsets(population_count,arena_dimension=arena_dimension,\
                                                  scale_factor=6.0,maxit=40)
        
        grid_feature_dict = {}
        place_feature_dict = {}
        attr_gen = NeuroH5CellAttrGen(coords_path, population, namespace=distances_namespace,
                                      comm=comm, io_size=io_size, cache_size=cache_size)
        for gid, distances_dict in attr_gen:
            if gid is None:
                logger.info('Rank %i gid is None' % rank)
            else:
                logger.info('Rank %i received attributes for gid %i' % (rank, gid))
                local_time = time.time()

                arc_distance_u = distances_dict['U Distance'][0]
                arc_distance_v = distances_dict['V Distance'][0]

                local_random.seed(gid + feature_seed_offset)

                feature_type = feature_types[gid-population_start]
                
                if feature_type == feature_grid:
                    this_module = local_random.choice(modules)
                    this_grid_spacing = field_width(float(this_module)/float(max(modules)))
                    feature_dict = {}
                    feature_dict['Grid Spacing'] = np.array([this_grid_spacing], dtype='float32')
                    this_grid_orientation = grid_orientation[this_module]
                    feature_dict['Grid Orientation'] = np.array([this_grid_orientation], dtype='float32')
                    
                    x_offset = xy_offsets[gid_count,0]
                    y_offset = xy_offsets[gid_count,1]
                    feature_dict['X Offset'] = np.array([x_offset], dtype='float32')
                    feature_dict['Y Offset'] = np.array([y_offset], dtype='float32')
                    grid_feature_dict[gid] = feature_dict
                    
                elif feature_type == feature_place_field:
                    feature_dict = {}
                    # this_field_width = field_width(arc_distance_u / DG_surface.max_u)
                    this_field_width = field_width(local_random.random())
                    feature_dict['Field Width'] = np.array([this_field_width], dtype='float32')
                    
                    x_offset = xy_offsets[gid_count,0]
                    y_offset = xy_offsets[gid_count,1]
                    feature_dict['X Offset'] = np.array([x_offset], dtype='float32')
                    feature_dict['Y Offset'] = np.array([y_offset], dtype='float32')
                    place_feature_dict[gid] = feature_dict
                    
                logger.info('Rank %i: took %.2f s to compute feature parameters for %s gid %i (type %i)' % \
                            (rank, time.time() - local_time, population, gid, feature_type))
                gid_count += 1
                
            if gid_count % write_size == 0:
                if not dry_run:
                    append_cell_attributes(output_path, population, grid_feature_dict,
                                           namespace='Grid Input Features', comm=comm, io_size=io_size, 
                                           chunk_size=chunk_size, value_chunk_size=value_chunk_size)
                    append_cell_attributes(output_path, population, place_feature_dict,
                                           namespace='Place Input Features', comm=comm, io_size=io_size, 
                                           chunk_size=chunk_size, value_chunk_size=value_chunk_size)
                grid_feature_dict.clear()
                place_feature_dict.clear()
                gc.collect()

        if not dry_run:
            append_cell_attributes(output_path, population, grid_feature_dict,
                                   namespace='Grid Input Features', comm=comm, io_size=io_size, 
                                   chunk_size=chunk_size, value_chunk_size=value_chunk_size)
            append_cell_attributes(output_path, population, place_feature_dict,
                                   namespace='Place Input Features', comm=comm, io_size=io_size, 
                                   chunk_size=chunk_size, value_chunk_size=value_chunk_size)
        
        global_count = comm.gather(gid_count, root=0)
        if rank == 0:
            logger.info('%i ranks took %.2f s to compute feature parameters for %i cells in population %s' % \
                            (comm.size, time.time() - start_time, np.sum(global_count), population))


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
