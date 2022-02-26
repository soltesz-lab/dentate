import itertools, logging, os, sys, time
from collections import defaultdict
from mpi4py import MPI
import h5py
import numpy as np
import click
import dentate.utils as utils
import dentate.cells as cells
import dentate.synapses as synapses

from dentate.env import Env
from dentate.neuron_utils import configure_hoc_env, load_cell_template
from neuroh5.io import NeuroH5TreeGen, append_cell_attributes, read_population_ranges
from neuron import h

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
#sys.excepthook = mpi_excepthook

script_name=os.path.basename(__file__)

def get_distance_to_node(cell, source_sec, target_sec, loc=0.5):
    return h.distance(source_sec(0.5), target_sec(loc))


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-path", type=str)
@click.option("--output-path", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--populations", '-i', required=True, multiple=True, type=str)
@click.option("--distance-bin-size", type=float, default=10.)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=10000)
@click.option("--verbose", "-v", is_flag=True)
def main(config, template_path, output_path, forest_path, populations, distance_bin_size, io_size, chunk_size, value_chunk_size, cache_size, verbose):
    """

    :param config:
    :param template_path:
    :param forest_path:
    :param populations:
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    """

    utils.config_logging(verbose)
    logger = utils.get_script_logger(script_name)
        
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    env = Env(comm=MPI.COMM_WORLD, config_file=config, template_paths=template_path)
    configure_hoc_env(env)
    
    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)

    if output_path is None:
        output_path = forest_path

    if rank==0:
        if not os.path.isfile(output_path):
            input_file  = h5py.File(forest_path,'r')
            output_file = h5py.File(output_path,'w')
            input_file.copy('/H5Types',output_file)
            input_file.close()
            output_file.close()
    comm.barrier()
    
    layers = env.layers
    layer_idx_dict = { layers[layer_name]: layer_name 
                       for layer_name in ['GCL', 'IML', 'MML', 'OML', 'Hilus'] }
 
        
    (pop_ranges, _) = read_population_ranges(forest_path, comm=comm)
    start_time = time.time()
    for population in populations:
        logger.info('Rank %i population: %s' % (rank, population))
        count = 0
        (population_start, _) = pop_ranges[population]
        template_class = load_cell_template(env, population, bcast_template=True)
        measures_dict = {}
        for gid, morph_dict in NeuroH5TreeGen(forest_path, population, io_size=io_size, comm=comm, topology=True):
            if gid is not None:
                logger.info('Rank %i gid: %i' % (rank, gid))
                cell = cells.make_neurotree_hoc_cell(template_class, neurotree_dict=morph_dict, gid=gid)
                secnodes_dict = morph_dict['section_topology']['nodes']

                apicalidx = set(cell.apicalidx)
                basalidx  = set(cell.basalidx)
                
                dendrite_area_dict = { k: 0.0 for k in layer_idx_dict }
                dendrite_length_dict = { k: 0.0 for k in layer_idx_dict }
                dendrite_distances = []
                dendrite_diams = []
                for (i, sec) in enumerate(cell.sections):
                    if (i in apicalidx) or (i in basalidx):
                        secnodes = secnodes_dict[i]
                        for seg in sec.allseg():
                            L     = seg.sec.L
                            nseg  = seg.sec.nseg
                            seg_l = L / nseg
                            seg_area = h.area(seg.x)
                            seg_diam = seg.diam
                            seg_distance = get_distance_to_node(cell, list(cell.soma)[0], seg.sec, seg.x)
                            dendrite_diams.append(seg_diam)
                            dendrite_distances.append(seg_distance)
                            layer = synapses.get_node_attribute('layer', morph_dict, seg.sec, secnodes, seg.x)
                            dendrite_length_dict[layer] += seg_l
                            dendrite_area_dict[layer] += seg_area
                    
                dendrite_distance_array = np.asarray(dendrite_distances)
                dendrite_diam_array = np.asarray(dendrite_diams)
                dendrite_distance_bin_range = int(((np.max(dendrite_distance_array)) - np.min(dendrite_distance_array))/distance_bin_size)+1
                dendrite_distance_counts, dendrite_distance_edges = np.histogram(dendrite_distance_array, 
                                                                                 bins=dendrite_distance_bin_range, 
                                                                                 density=False)
                dendrite_diam_sums, _ = np.histogram(dendrite_distance_array, 
                                                     weights=dendrite_diam_array, 
                                                     bins=dendrite_distance_bin_range, 
                                                     density=False)
                dendrite_mean_diam_hist = np.zeros_like(dendrite_diam_sums)
                np.divide(dendrite_diam_sums, dendrite_distance_counts, 
                          where=dendrite_distance_counts>0,
                          out=dendrite_mean_diam_hist)

                dendrite_area_per_layer = np.asarray([ dendrite_area_dict[k] for k in sorted(dendrite_area_dict.keys()) ], dtype=np.float32)
                dendrite_length_per_layer = np.asarray([ dendrite_length_dict[k] for k in sorted(dendrite_length_dict.keys()) ], dtype=np.float32)

                measures_dict[gid] = { 'dendrite_distance_hist_edges': np.asarray(dendrite_distance_edges, dtype=np.float32),
                                       'dendrite_distance_counts': np.asarray(dendrite_distance_counts, dtype=np.int32),
                                       'dendrite_mean_diam_hist': np.asarray(dendrite_mean_diam_hist, dtype=np.float32),
                                       'dendrite_area_per_layer': dendrite_area_per_layer,
                                       'dendrite_length_per_layer': dendrite_length_per_layer }
                    
                del cell
                count += 1
            else:
                logger.info('Rank %i gid is None' % rank)
        append_cell_attributes(output_path, population, measures_dict,
                               namespace='Tree Measurements', comm=comm, io_size=io_size, chunk_size=chunk_size,
                               value_chunk_size=value_chunk_size, cache_size=cache_size)
    MPI.Finalize()



if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
