import os, sys, gc, logging, random
from itertools import islice
import click
import numpy as np
from mpi4py import MPI
import dentate
import dentate.utils as utils
import dentate.cells as cells
from neuroh5.io import NeuroH5TreeGen, append_cell_attributes, append_cell_trees, append_cell_attributes, read_cell_attributes, read_tree_selection
import h5py
import scipy

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook


def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))

@click.command()
@click.option("--population", required=True, type=str)
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--phenotypes-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-path", required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--h5types-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--distances-namespace", type=str, default='Arc Distances')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=4000)
@click.option("--value-chunk-size", type=int, default=10000)
@click.option("--cache-size", type=int, default=10)
@click.option("--write-size", type=int, default=100)
@click.option("--dry-run",  is_flag=True)
@click.option("--verbose", '-v', is_flag=True)
@click.option("--debug", is_flag=True)
def main(population, forest_path, coords_path, phenotypes_path, output_path, h5types_path, distances_namespace, io_size, chunk_size, value_chunk_size, cache_size, write_size, dry_run, verbose, debug):
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

    color = 0
    if rank == 0:
         color = 1
    comm0 = comm.Split(color, 0)

    soma_distances = None
    phenotype_gids = None
    phenotypes_tree_dict = None
    phenotype_distances = None
    if rank==0:
        phenotypes_tree_dict = {}
        if not os.path.isfile(output_path):
            input_file  = h5py.File(h5types_path,'r')
            output_file = h5py.File(output_path,'w')
            input_file.copy('/H5Types',output_file)
            input_file.close()
            output_file.close()
        phenotypes = np.loadtxt(phenotypes_path)
        phenotype_gids = phenotypes[:,0].astype(int).tolist()
        phenotype_distances = np.column_stack((phenotypes[:,2], phenotypes[:,3]))
        phenotypes_tree_iter, _ = read_tree_selection(forest_path, population, phenotype_gids, 
                                                      comm=comm0, topology=False)
        for (gid,tree_dict) in phenotypes_tree_iter:
            phenotypes_tree_dict[gid] = tree_dict

        logger.info(f'Reading {population} coordinates...')
        distances_iter = read_cell_attributes(coords_path, population, comm=comm0,
                                              mask=set(['U Distance', 'V Distance']),
                                              namespace=distances_namespace)
        soma_distances = { k: (float(v['U Distance'][0]), 
                               float(v['V Distance'][0])) for (k,v) in distances_iter }

    comm.barrier()
    comm0.Free()

    phenotype_gids = comm.bcast(phenotype_gids, root=0)
    phenotype_distances = comm.bcast(phenotype_distances, root=0)
    phenotypes_tree_dict = comm.bcast(phenotypes_tree_dict, root=0)
    soma_distances = comm.bcast(soma_distances, root=0)

    phenotypes_kdt = scipy.spatial.cKDTree(phenotype_distances)

    phenotype_gid_dict = {}
    new_trees_dict = {}
    it = 0
    for gid, tree_dict in NeuroH5TreeGen(forest_path, population, io_size=io_size, cache_size=cache_size,
                                         comm=comm, topology=False):
        if gid is not None:
            logger.info(f"Rank {rank} received gid {gid}")
            this_soma_dist = soma_distances[gid]
            _, nn = phenotypes_kdt.query(this_soma_dist,  k=1)
            phenotype_gid = phenotype_gids[nn]
            logger.info(f"Rank {rank}: using phenotype {phenotype_gid} for gid {gid}")
            new_tree_dict = phenotypes_tree_dict[phenotype_gid]
            new_trees_dict[gid] = new_tree_dict
            phenotype_gid_dict[gid] = { 'phenotype_id': np.asarray([phenotype_gid],dtype=np.uint32) }
        if debug and it >= 10:
            break
        it += 1

    if not dry_run:
        comm.barrier()
        if write_size > 0:
            items = zip(split_every(write_size, new_trees_dict.items()), 
                        split_every(write_size, phenotype_gid_dict.items()))
            for chunk in items:
                new_trees_chunk = dict(chunk[0])
                phenotype_gid_chunk = dict(chunk[1])
                append_cell_trees(output_path, population, new_trees_chunk, io_size=io_size, 
                                  chunk_size=chunk_size, value_chunk_size=value_chunk_size, comm=comm)
                append_cell_attributes(output_path, population, phenotype_gid_chunk,
                                       namespace='Phenotype ID', io_size=io_size, 
                                       chunk_size=chunk_size, value_chunk_size=value_chunk_size, comm=comm)
                comm.barrier()
        else:
            append_cell_trees(output_path, population, new_trees_dict, io_size=io_size, 
                              chunk_size=chunk_size, value_chunk_size=value_chunk_size, comm=comm)
            append_cell_attributes(output_path, population, phenotype_gid_dict,
                                   namespace='Phenotype ID', io_size=io_size, 
                                   chunk_size=chunk_size, value_chunk_size=value_chunk_size, comm=comm)
            comm.barrier()
            
    if (not dry_run) and (rank == 0):
        logger.info(f"Appended phenotype trees to {output_path}")


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
