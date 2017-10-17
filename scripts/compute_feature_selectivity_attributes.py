from function_lib import *
from mpi4py import MPI
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes
import click


try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


script_name = 'compute_feature_selectivity_attributes.py'

#  MEC is divided into discrete modules with distinct grid spacing and field width. Here we assume grid cells
#  sample uniformly from 10 modules with spacing that increases exponentially from 40 cm to 8 m. While organized
#  dorsal-ventrally, organization in the transverse or septo-temporal extent of their projections to DG.
#  CA3 and LEC are assumed to exhibit place fields. Their field width varies septal-temporally. Here we assume a
#  continuous exponential gradient of field widths, with the same parameters as those controlling MEC grid width.

modules = range(10)
field_width_params = [35.00056621,   0.32020713]  # slope, tau

#  x varies from 0 to 1, corresponding either to module id or septo-temporal distance
field_width = lambda x: 40. + field_width_params[0] * (np.exp(x / field_width_params[1]) - 1.)

# make sure random seeds are not being reused for various types of stochastic sampling
selectivity_seed_offset = int(2 * 2e6)

local_random = random.Random()
local_random.seed(selectivity_seed_offset-1)
# every 60 degrees repeats in a hexagonal array
grid_orientation = [local_random.uniform(0., np.pi / 3.) for i in range(len(modules))]

#  custom data type for type of feature selectivity
selectivity_grid = 0
selectivity_place_field = 1
selectivity_type_dict = {'MPP': selectivity_grid, 'LPP': selectivity_place_field}

spatial_resolution = 1.  # um
max_u = 11690.
max_v = 2956.

du = (1.01*np.pi-(-0.016*np.pi))/max_u*spatial_resolution
dv = (1.425*np.pi-(-0.23*np.pi))/max_v*spatial_resolution
u = np.arange(-0.016*np.pi, 1.01*np.pi, du)
v = np.arange(-0.23*np.pi, 1.425*np.pi, dv)

U, V = np.meshgrid(u, v, indexing='ij')

# for the middle of the granule cell layer:
L = -1.
X = np.array(-500.* np.cos(U) * (5.3 - np.sin(U) + (1. + 0.138 * L) * np.cos(V)))
Y = np.array(750. * np.sin(U) * (5.5 - 2. * np.sin(U) + (0.9 + 0.114*L) * np.cos(V)))
Z = np.array(2500. * np.sin(U) + (663. + 114. * L) * np.sin(V - 0.13 * (np.pi-U)))

euc_coords = np.array([X.T, Y.T, Z.T]).T

del U
del V
del X
del Y
del Z
gc.collect()

delta_U = np.sqrt((np.diff(euc_coords, axis=0)**2.).sum(axis=2))
delta_V = np.sqrt((np.diff(euc_coords, axis=1)**2.).sum(axis=2))

distance_U = np.cumsum(np.insert(delta_U, 0, 0., axis=0), axis=0)
distance_V = np.cumsum(np.insert(delta_V, 0, 0., axis=1), axis=1)

del delta_U
del delta_V
gc.collect()


def get_array_index_func(val_array, this_val):
    """

    :param val_array: array
    :param this_val: float
    :return: int
    """
    indexes = np.where(val_array >= this_val)[0]
    if np.any(indexes):
        return indexes[0]
    else:
        return len(val_array) - 1


get_array_index = np.vectorize(get_array_index_func, excluded=[0])


@click.command()
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Sorted Coordinates')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
def main(coords_path, coords_namespace, io_size, chunk_size, value_chunk_size, cache_size):

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    for population in ['MPP', 'LPP']:
        count = 0
        start_time = time.time()
        selectivity_type = selectivity_type_dict[population]
        for gid, coords_dict in NeuroH5CellAttrGen(comm, coords_path, population, io_size=io_size,
                                                   cache_size=cache_size, namespace=coords_namespace):
            selectivity_dict = {}
            if gid is not None:
                # print 'Rank %i: %s gid %i' % (rank, population, gid)
                local_time = time.time()
                selectivity_dict[gid] = {}
                local_random.seed(gid + selectivity_seed_offset)
                selectivity_dict[gid]['Selectivity Type'] = np.array([selectivity_type], dtype='uint32')
                if selectivity_type == selectivity_grid:
                    this_module = local_random.choice(modules)
                    this_grid_spacing = field_width(float(this_module)/float(max(modules)))
                    selectivity_dict[gid]['Grid Spacing'] = np.array([this_grid_spacing],
                                                                    dtype='float32')
                    this_grid_orientation = grid_orientation[this_module]
                    selectivity_dict[gid]['Grid Orientation'] = np.array([this_grid_orientation], dtype='float32')
                    radius = this_grid_spacing * np.sqrt(local_random.random())
                    phi_offset = local_random.uniform(-np.pi, np.pi)
                    x_offset = radius * np.cos(phi_offset)
                    y_offset = radius * np.sin(phi_offset)
                    selectivity_dict[gid]['X Offset'] = np.array([x_offset], dtype='float32')
                    selectivity_dict[gid]['Y Offset'] = np.array([y_offset], dtype='float32')
                elif selectivity_type == selectivity_place_field:
                    this_u_index = get_array_index(u, coords_dict[coords_namespace]['U Coordinate'][0])
                    this_v_index = get_array_index(v, coords_dict[coords_namespace]['V Coordinate'][0])
                    this_u_distance = distance_U[this_u_index, this_v_index]
                    this_field_width = field_width(this_u_distance / max_u)
                    selectivity_dict[gid]['Field Width'] = np.array([this_field_width], dtype='float32')
                    # aiming for ~30% of cells to have a peak location inside a 200 cm x 200 cm arena, and the extreme
                    # ends of the arena to have the same input density, so distributing peak locations with radius
                    # 400 + 142 cm (widest field width plus longest distance from origin)
                    radius = 542. * np.sqrt(local_random.random())
                    phi_offset = local_random.uniform(-np.pi, np.pi)
                    x_offset = radius * np.cos(phi_offset)
                    y_offset = radius * np.sin(phi_offset)
                    selectivity_dict[gid]['X Offset'] = np.array([x_offset], dtype='float32')
                    selectivity_dict[gid]['Y Offset'] = np.array([y_offset], dtype='float32')
                print 'Rank %i: took %.2f s to compute selectivity parameters for %s gid %i' % \
                      (rank, time.time() - local_time, population, gid)
                count += 1
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
