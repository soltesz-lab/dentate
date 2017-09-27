
from mpi4py import MPI
from neuroh5.io import read_population_ranges, bcast_cell_attributes
from connection_generator import ConnectionProb, generate_uv_distance_connections
from bspline_surface import BSplineSurface
from env import Env
import utils
import click


script_name = 'generate_connections.py'

def DG_surface(u, v, l):
    x = np.array(-500.* np.cos(u) * (5.3 - np.sin(u) + (1. + 0.138 * l) * np.cos(v)))
    y = np.array(750. * np.sin(u) * (5.5 - 2. * np.sin(u) + (0.9 + 0.114*l) * np.cos(v)))
    z = np.array(2500. * np.sin(u) + (663. + 114. * l) * np.sin(v - 0.13 * (np.pi-u)))
    return np.array([x, y, z])


def make_surface(l=-1.): # default l is for the middle of the granule cell layer:
    spatial_resolution = 50.  # um
    max_u = 11690.
    max_v = 2956.
    
    du = (1.01*np.pi-(-0.016*np.pi))/max_u*spatial_resolution
    dv = (1.425*np.pi-(-0.23*np.pi))/max_v*spatial_resolution
    u = np.arange(-0.016*np.pi, 1.01*np.pi, du)
    v = np.arange(-0.23*np.pi, 1.425*np.pi, dv)

    u, v = np.meshgrid(u, v, indexing='ij')
    
    
    xyz = DG_surface (u, v, l)

    srf = BSplineSurface(np.linspace(0, 1, len(u)),
                         np.linspace(0, 1, xyz.shape[2]),
                         xyz)


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--connectivity-namespace", type=str, default='Connectivity')
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Sorted Coordinates')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
def main(config, forest_path, connectivity_namespace, coords_path, coords_namespace,
         io_size, chunk_size, value_chunk_size, cache_size):

    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, configFile=config)

    ip_surface  = make_surface()
    extent      = {}
    soma_coords = {}
    
    populations = read_population_ranges(comm, coords_path).keys()
    for population in populations:
        soma_coords[population] = bcast_cell_attributes(comm, 0, coords_path, population,
                                                        namespace=coords_namespace)
        extent[population] = { 'width': env.modelConfig['Connection Generator']['Axon Width'][population],
                               'offset': env.modelConfig['Connection Generator']['Axon Offset'][population] }

    for destination_population in populations:

        if env.modelConfig['Connection Generator']['Synapse Types']:

            connection_prob = ConnectionProb(destination_population, soma_coords, ip_surface, extent)

            synapse_layers      = 
            synapse_types       = 
            synapse_locations   = 
            synapse_proportions = 

            synapse_seed =
            synapse_namespace = 'Synapse Attributes'

            connectivity_seed =
            connectivity_namespace = 'Connections'

            generate_uv_distance_connections(comm, 
                                             connection_prob, forest_path,
                                             synapse_layers, synapse_types,
                                             synapse_locations, synapse_proportions,
                                             synapse_seed, synapse_namespace, 
                                             connectivity_seed, connectivity_namespace,
                                             io_size, chunk_size, value_chunk_size, cache_size)
        

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

