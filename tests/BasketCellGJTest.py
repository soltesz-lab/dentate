
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
@click.option("--template-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(template_path, forest_path):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    h('objref nil, pc, gjlist, cells, tlog, Vlog1, Vlog2')
    h.load_file("nrngui.hoc")
    h.xopen("./lib.hoc")
    h.xopen(template_path+'/BasketCell.hoc')
    h.pc = h.ParallelContext()
    h.cells  = h.List()
    h.gjlist = h.List()
    
    popName = "BC"
    (trees,_) = read_tree_selection (comm, forest_path, popName, [1039000])

    tree = trees.itervalues().next()
    cell1 = utils.new_cell ("BasketCell", neurotree_dict=tree)
    cell2 = utils.new_cell ("BasketCell", neurotree_dict=tree)

    h.cells.append(cell1)
    h.cells.append(cell2)
    
    ggid        = 20000000
    source      = 10422930
    destination = 10422670
    srcbranch   = 0
    srcsec      = 0
    dstbranch   = 1
    dstsec      = 0
    weight      = 5.4e-4

    stimdur     = 500
    tstop       = 2000
    
    h.pc.set_gid2node(source, int(h.pc.id()))
    nc = cell1.connect2target(h.nil)
    h.pc.cell(source, nc, 1)

    h.pc.set_gid2node(destination, int(h.pc.id()))
    nc = cell2.connect2target(h.nil)
    h.pc.cell(destination, nc, 1)

    stim1 = h.IClamp(cell1.soma(0.5))
    stim1.delay = 250
    stim1.dur = stimdur
    stim1.amp = -0.1

    stim2 = h.IClamp(cell2.soma(0.5))
    stim2.delay = 500+stimdur
    stim2.dur = stimdur
    stim2.amp = -0.1

    log_size = tstop/h.dt + 1
    
    h.tlog = h.Vector(log_size,0)
    h.tlog.record (h._ref_t)

    h.Vlog1 = h.Vector(log_size)
    h.Vlog1.record (cell1.soma(0.5)._ref_v)

    h.Vlog2 = h.Vector(log_size)
    h.Vlog2.record (cell2.soma(0.5)._ref_v)
    
    h.mkgap(h.pc, h.gjlist, source, srcbranch, srcsec, ggid, ggid+1, weight)
    h.mkgap(h.pc, h.gjlist, destination, dstbranch, dstsec, ggid+1, ggid, weight)

    h.pc.setup_transfer()
    h.pc.set_maxstep(10.0)

    h.stdinit()
    h.finitialize(-60)
    h.pc.barrier()

    h.tstop = tstop
    h.pc.psolve(h.tstop)

    f=open("BasketCellGJ.dat",'w')
    for (t,v1,v2) in itertools.izip(h.tlog,h.Vlog1,h.Vlog2):
        f.write("%f %f %f\n" % (t,v1,v2))
    f.close()
    
    
if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("BasketCellGJTest.py") != -1,sys.argv)+1):])
