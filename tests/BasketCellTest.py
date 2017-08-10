

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

    
def passive_test (tree, v_init):
    cell = utils.new_cell ("BasketCell", neurotree_dict=tree)

    prelength = 1000
    mainlength = 2000

    tstop = prelength+mainlength
    
    stimdur = 500.0
    
    stim1 = h.IClamp(cell.soma(0.5))
    stim1.delay = 250
    stim1.dur   = stimdur
    stim1.amp   = -0.1

    log_size = tstop/h.dt + 1
    
    h.tlog = h.Vector(log_size,0)
    h.tlog.record (h._ref_t)

    h.Vlog = h.Vector(log_size)
    h.Vlog.record (cell.soma(0.5)._ref_v)
    
    h.tstop = tstop

    utils.simulate(h, v_init, prelength,mainlength)

    ## compute membrane time constant
    vrest  = h.Vlog.x[int(h.tlog.indwhere(">=",prelength-1))]
    vmin   = h.Vlog.min()
    vmax   = vrest
    
    ## the time it takes the system's step response to reach 1-1/e (or
    ## 63.2%) of the peak value
    amp23  = 0.632 * abs (vmax - vmin)
    vtau0  = vrest - amp23
    tau0   = h.tlog.x[int(h.Vlog.indwhere ("<=", vtau0))] - prelength

    f=open("BasketCell_passive_results.dat",'w')
    
    f.write ("DC input resistance: %g MOhm\n" % h.rn(cell))
    f.write ("vmin: %g mV\n" % vmin)
    f.write ("vtau0: %g mV\n" % vtau0)
    f.write ("tau0: %g ms\n" % tau0)

    f.close()


@click.command()
@click.option("--template-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(template_path,forest_path):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    h('objref nil, pc, tlog, Vlog')
    h.load_file("nrngui.hoc")
    h.xopen("./lib.hoc")
    h.xopen ("./tests/rn.hoc")
    h.xopen(template_path+'/BasketCell.hoc')
    h.pc = h.ParallelContext()

    popName = "BC"
    (trees,_) = read_tree_selection (comm, forest_path, popName, [1039000])

    tree = trees.itervalues().next()

    print "tree = ", tree
    passive_test(tree,-60)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find("BasketCellTest.py") != -1,sys.argv)+1):])
