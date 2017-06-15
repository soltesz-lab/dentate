import sys
from mpi4py import MPI
import numpy as np
from neurotrees.io import append_cell_attributes, append_cell_trees, population_ranges
import click

def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None

@click.command()
@click.option("--population", required=True, type=str)
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--index-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
def main(population, trees_path, index_path, chunk_size, value_chunk_size):

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    f = open(index_path)
    lines = f.readlines()

    index = {}
    while lines:
        for l in lines:
            a = filter(None, l.split(" "))
            gid = int(a[0])
            newgid = int(a[1])
            index[gid] = newgid
        lines = f.readlines()

    f.close()

    
    selection_dict = {}
    for gid, morph_dict in NeurotreeGen(MPI._addressof(comm), forest_path, population, io_size=io_size):
        if index.has_key(gid):
            newgid = index[gid]
            selection_dict[newgid] = morph_dict
            
    append_cell_trees (MPI._addressof(comm), output_path, population, selection_dict, io_size=io_size)
        
        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("select_trees.py") != -1,sys.argv)+1):])
