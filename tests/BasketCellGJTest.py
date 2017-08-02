
import sys, os
import os.path
import click
import itertools
import numpy as np
from mpi4py import MPI # Must come before importing NEURON
from neuron import h
from neuroh5.io import read_tree_selection
from env import Env
import utils

def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None

def hoc_results_to_python(hoc_results):
    results_dict = {}
    for i in xrange(0, int(hoc_results.count())):
        vect   = hoc_results.o(i)
        gid    = int(vect.x[0])
        pyvect = vect.to_python()
        results_dict[gid] = pyvect[1:]
    hoc_results.remove_all()
    return results_dict

def write_results(results, filepath, header):
    f = open(filepath,'w')
    f.write(header+'\n')
    for item in results:
        for (gid, vect) in item.iteritems():
            f.write (str(gid)+"\t")
            f.write (("\t".join(['{:0.3f}'.format(i) for i in vect])) + "\n")
    f.close()

@click.command()
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(forest_path):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    #selection=[10422930, 10422670]
    popName = "BC"
    trees = read_tree_selection (comm, forest_path, popName, [1039000])

    for tree in trees:
        cell = new_cell ("BasketCell", neurotree_dict=tree)
        print "cell = ", cell
        
if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("BasketCellGJTest.py") != -1,sys.argv)+1):])
