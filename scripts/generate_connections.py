import sys
from mpi4py import MPI
from neuroh5.io import read_population_ranges, bcast_cell_attributes
from connection_generator import ConnectionProb, generate_uv_distance_connections
from DG_surface import make_surface
from env import Env
import utils
import click

script_name = 'generate_connections.py'

@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--connectivity-path", required=True, type=click.Path())
@click.option("--connectivity-namespace", type=str, default='Connectivity')
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Sorted Coordinates')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
def main(config, forest_path, connectivity_path, connectivity_namespace, coords_path, coords_namespace,
         io_size, chunk_size, value_chunk_size, cache_size):

    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, configFile=config)

    ip_surface  = make_surface(l=3.0) ## Corresponds to OML boundary
    extent      = {}
    soma_coords = {}
    
    populations = read_population_ranges(comm, coords_path)[0].keys()
    for population in populations:
        soma_coords[population] = bcast_cell_attributes(comm, 0, coords_path, population,
                                                        namespace=coords_namespace)
        extent[population] = { 'width': env.modelConfig['Connection Generator']['Axon Width'][population],
                               'offset': env.modelConfig['Connection Generator']['Axon Offset'][population] }

    connectivity_synapse_types = env.modelConfig['Connection Generator']['Synapse Types']
    
    for destination_population in populations:

        connection_prob = ConnectionProb(destination_population, soma_coords, ip_surface, extent)

        synapse_seed        = int(env.modelConfig['Random Seeds']['Synapse Projection Partitions'])
        synapse_namespace   = 'Synapse Attributes'
        
        connectivity_seed = int(env.modelConfig['Random Seeds']['Distance-Dependent Connectivity'])
        connectivity_namespace = 'Connections'

        populations_dict = env.modelConfig['Definitions']['Populations']
        generate_uv_distance_connections(comm, populations_dict,
                                         env.connection_generator,
                                         connection_prob, forest_path,
                                         synapse_seed, synapse_namespace, 
                                         connectivity_seed, connectivity_namespace, connectivity_path,
                                         io_size, chunk_size, value_chunk_size, cache_size)
        

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

