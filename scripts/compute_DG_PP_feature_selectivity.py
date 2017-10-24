from function_lib import *
from mpi4py import MPI
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes
import click
import DG_surface

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


script_name = 'compute_DG_PP_feature_selectivity_parameters.py'

#  MEC is divided into discrete modules with distinct grid spacing and field width. Here we assume grid cells
#  sample uniformly from 10 modules with spacing that increases exponentially from 40 cm to 8 m. While organized
#  dorsal-ventrally, there is no organization in the transverse or septo-temporal extent of their projections to DG.
#  CA3 and LEC are assumed to exhibit place fields. Their field width varies septal-temporally. Here we assume a
#  continuous exponential gradient of field widths, with the same parameters as those controlling MEC grid width.

modules = range(10)
field_width_params = [35.00056621,   0.32020713]  # slope, tau

#  x varies from 0 to 1, corresponding either to module id or septo-temporal distance
field_width = lambda x: 40. + field_width_params[0] * (np.exp(x / field_width_params[1]) - 1.)

#  custom data type for type of feature selectivity
selectivity_grid = 0
selectivity_place_field = 1
selectivity_type_dict = {'MPP': selectivity_grid, 'LPP': selectivity_place_field}

@click.command()
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--distances-namespace", type=str, default='Arc Distance')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--seed", type=int, default=2)
@click.option("--debug", is_flag=True)
def main(coords_path, distances_namespace, origin_u, origin_v, io_size, chunk_size, value_chunk_size, cache_size, seed, debug):
    """

    :param coords_path:
    :param coords_namespace:
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    :param seed:
    :param debug:
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    srf = make_surface(l=3.)

    # make sure random seeds are not being reused for various types of stochastic sampling
    selectivity_seed_offset = int(seed * 2e6)

    local_random = random.Random()
    local_random.seed(selectivity_seed_offset - 1)
    # every 60 degrees repeats in a hexagonal array
    grid_orientation = [local_random.uniform(0., np.pi / 3.) for i in range(len(modules))]

    arena_dimension = 100.  # minimum distance from origin to boundary (cm)

    for population in ['MPP', 'LPP']:
        count = 0
        start_time = time.time()
        selectivity_type = selectivity_type_dict[population]
        attr_gen = NeuroH5CellAttrGen(comm, coords_path, population, io_size=io_size,
                                      cache_size=cache_size, namespace=coords_namespace):
        if debug:
            attr_gen_wrapper = (attr_gen.next() for i in xrange(2))
        else:
            attr_gen_wrapper = attr_gen
        for gid, distances_dict in attr_gen_wrapper:
            selectivity_dict = {}
            if gid is not None:
                local_time = time.time()
                selectivity_dict[gid] = {}

                arc_distance_u = distances_dict['U Distance']
                arc_distance_v = distances_dict['V Distance']
                    
                local_random.seed(gid + selectivity_seed_offset)
                selectivity_dict[gid]['Selectivity Type'] = np.array([selectivity_type], dtype='uint32')
                
                if selectivity_type == selectivity_grid:
                    this_module = local_random.choice(modules)
                    this_grid_spacing = field_width(float(this_module)/float(max(modules)))
                    selectivity_dict[gid]['Grid Spacing'] = np.array([this_grid_spacing],
                                                                    dtype='float32')
                    this_grid_orientation = grid_orientation[this_module]
                    selectivity_dict[gid]['Grid Orientation'] = np.array([this_grid_orientation], dtype='float32')
                    
                    # aiming for close to uniform input density inside the square arena
                    radius = (this_grid_spacing + np.sqrt(2.) * arena_dimension) * np.sqrt(local_random.random())
                    phi_offset = local_random.uniform(-np.pi, np.pi)
                    x_offset = radius * np.cos(phi_offset)
                    y_offset = radius * np.sin(phi_offset)
                    selectivity_dict[gid]['X Offset'] = np.array([x_offset], dtype='float32')
                    selectivity_dict[gid]['Y Offset'] = np.array([y_offset], dtype='float32')
                    
                elif selectivity_type == selectivity_place_field:
                    this_field_width = field_width(arc_distance_u / DG_surface.max_u)
                    selectivity_dict[gid]['Field Width'] = np.array([this_field_width], dtype='float32')
                    
                    # aiming for close to uniform input density inside the square arena
                    radius = (this_field_width + np.sqrt(2.) * arena_dimension) * np.sqrt(local_random.random())
                    phi_offset = local_random.uniform(-np.pi, np.pi)
                    x_offset = radius * np.cos(phi_offset)
                    y_offset = radius * np.sin(phi_offset)
                    selectivity_dict[gid]['X Offset'] = np.array([x_offset], dtype='float32')
                    selectivity_dict[gid]['Y Offset'] = np.array([y_offset], dtype='float32')
                print 'Rank %i: took %.2f s to compute selectivity parameters for %s gid %i' % \
                      (rank, time.time() - local_time, population, gid)
                count += 1
            if not debug:
                append_cell_attributes(comm, coords_path, population, selectivity_dict,
                                       namespace='Feature Selectivity', io_size=io_size, chunk_size=chunk_size,
                                       value_chunk_size=value_chunk_size)
            sys.stdout.flush()
            del selectivity_dict
            gc.collect()
        global_count = comm.gather(count, root=0)
        if rank == 0:
            print '%i ranks took %.2f s to compute selectivity parameters for %i %s cells' % \
                  (comm.size, time.time() - start_time, np.sum(global_count), population)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
