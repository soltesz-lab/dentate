

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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

@click.command()
@click.option('--selection', callback=lambda _,__,x: map(int, x.split(',')) if x else [])
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(forest_path, selection):

    popName = "BC"
    trees = read_tree_selection (MPI._addressof(comm), forest_path, popName, selection, attributes=True)

    for tree in trees:
        cell = new_cell ("BasketCell", neurotree_dict=tree)

    
    

