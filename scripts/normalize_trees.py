import os, sys, gc, logging, random
import click
import numpy as np
from mpi4py import MPI
import dentate
import dentate.utils as utils
import dentate.cells as cells
from dentate.env import Env
from neuroh5.io import NeuroH5TreeGen, append_cell_attributes, append_cell_trees, bcast_cell_attributes, read_population_ranges
import h5py

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook


@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), default='config')
@click.option("--population", required=True, type=str)
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-path", type=str)
@click.option("--output-path", required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--dry-run",  is_flag=True)
@click.option("--verbose", '-v', is_flag=True)
def main(config, config_prefix, population, forest_path, template_path, output_path, io_size, chunk_size, value_chunk_size, dry_run, verbose):
    """

    :param population: str
    :param forest_path: str (path)
    :param output_path: str (path)
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param verbose: bool
    """
    
    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info(f"{comm.size} ranks have been allocated")

    env = Env(comm=comm, config=config, config_prefix=config_prefix, template_paths=template_path)

    if rank==0:
        if not os.path.isfile(output_path):
            input_file  = h5py.File(forest_path,'r')
            output_file = h5py.File(output_path,'w')
            input_file.copy('/H5Types',output_file)
            input_file.close()
            output_file.close()
    comm.barrier()

    (forest_pop_ranges, _)  = read_population_ranges(forest_path)
    (forest_population_start, forest_population_count) = forest_pop_ranges[population]

    (pop_ranges, _)  = read_population_ranges(output_path)

    (population_start, population_count) = pop_ranges[population]

    new_trees_dict = {}
    for gid, tree_dict in NeuroH5TreeGen(forest_path, population, io_size=io_size, comm=comm, topology=False):
        if gid is not None:
            logger.info(f"Rank {rank} received gid {gid}")
            new_tree_dict = cells.normalize_tree_topology(tree_dict, env.SWC_Types)
            
            new_trees_dict[gid] = new_tree_dict

    if not dry_run:
        append_cell_trees(output_path, population, new_trees_dict, io_size=io_size, comm=comm)

    comm.barrier()
    if (not dry_run) and (rank == 0):
        logger.info(f"Appended normalized trees to {output_path}")


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
