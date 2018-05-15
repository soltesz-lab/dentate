import sys, os, time
import itertools
from collections import defaultdict
import numpy as np
from mpi4py import MPI
from neuron import h
from dentate.utils import list_find
from neuroh5.io import NeuroH5TreeGen, read_population_ranges, append_cell_attributes
import h5py
import dentate
from dentate.env import Env
import dentate.cells as cells
import dentate.synapses as synapses
import click
import logging
logging.basicConfig()

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

script_name="measure_trees.py"
logger = logging.getLogger(script_name)


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-path", type=str)
@click.option("--output-path", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--populations", '-i', required=True, multiple=True, type=str)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=10000)
@click.option("--verbose", "-v", is_flag=True)
def main(config, template_path, output_path, forest_path, populations, io_size, chunk_size, value_chunk_size, cache_size, verbose):
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

    if verbose:
        logger.setLevel(logging.INFO)
        
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    env = Env(comm=MPI.COMM_WORLD, configFile=config, templatePaths=template_path)
    h('objref nil, pc, templatePaths')
    h.load_file("nrngui.hoc")
    h.load_file("./templates/Value.hoc")
    h.xopen("./lib.hoc")
    h.pc = h.ParallelContext()
    
    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)

    h.templatePaths = h.List()
    for path in env.templatePaths:
        h.templatePaths.append(h.Value(1,path))

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
        
    (pop_ranges, _) = read_population_ranges(forest_path, comm=comm)
    start_time = time.time()
    for population in populations:
        logger.info('Rank %i population: %s' % (rank, population))
        count = 0
        (population_start, _) = pop_ranges[population]
        template_name = env.celltypes[population]['template']
        h.find_template(h.pc, h.templatePaths, template_name)
        template_class = eval('h.%s' % template_name)
        measures_dict = {}
        for gid, morph_dict in NeuroH5TreeGen(forest_path, population, io_size=io_size, comm=comm, topology=True):
            if gid is not None:
                logger.info('Rank %i gid: %i' % (rank, gid))
                cell = cells.make_neurotree_cell(template_class, neurotree_dict=morph_dict, gid=gid)

                apicalidx = set(cell.apicalidx)
                basalidx  = set(cell.basalidx)

                GCLDidxs = set(cell.GCLDidxs)
                PDidxs = set(cell.PDidxs)
                MDidxs = set(cell.MDidxs)
                DDidxs = set(cell.DDidxs)
                
                dendrite_area_dict = { k+1: 0.0 for k in xrange(0, 4) }
                dendrite_length_dict = { k+1: 0.0 for k in xrange(0, 4) }
                for (i, sec) in enumerate(cell.sections):
                    if (i in apicalidx) or (i in basalidx):
                        l = None
                        if i in GCLDidxs:
                            l = 1
                        elif i in PDidxs:
                            l = 2
                        elif i in MDidxs:
                            l = 3
                        elif i in DDidxs:
                            l = 4
                        else:
                            raise Exception('Unknown layer index %i' % i)
                        dendrite_length_dict[l] = dendrite_length_dict[l] + sec.L
                        dendrite_sec_area = 0.
                        for seg in sec.allseg():
                            dendrite_sec_area = dendrite_sec_area + h.area(seg.x)
                        dendrite_area_dict[l] += dendrite_sec_area

                measures_dict[gid] = { 'dendrite_area': np.asarray([ dendrite_area_dict[k] for k in sorted(dendrite_area_dict.keys()) ], dtype=np.float32), \
                                       'dendrite_length': np.asarray([ dendrite_length_dict[k] for k in sorted(dendrite_length_dict.keys()) ], dtype=np.float32) }
                    
                del cell
                count += 1
            else:
                logger.info('Rank %i gid is None' % rank)
        append_cell_attributes(output_path, population, measures_dict,
                               namespace='Tree Measurements', comm=comm, io_size=io_size, chunk_size=chunk_size,
                               value_chunk_size=value_chunk_size, cache_size=cache_size)
    MPI.Finalize()



if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
