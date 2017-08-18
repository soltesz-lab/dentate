

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
    h.dt = 0.025

    prelength = 1000
    mainlength = 2000

    tstop = prelength+mainlength
    
    stimdur = 500.0
    
    stim1 = h.IClamp(cell.sections[0](0.5))
    stim1.delay = prelength
    stim1.dur   = stimdur
    stim1.amp   = -0.1

    h.tlog = h.Vector()
    h.tlog.record (h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record (cell.sections[0](0.5)._ref_v)
    
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

def ap_rate_test (tree, v_init):

    cell = utils.new_cell ("BasketCell", neurotree_dict=tree)
    h.dt = 0.025

    prelength = 1000.0
    mainlength = 2000.0

    tstop = prelength+mainlength
    
    stimdur = 1000.0
    
    stim1 = h.IClamp(cell.sections[0](0.5))
    stim1.delay = prelength
    stim1.dur   = stimdur
    stim1.amp   = 0.2

    h.tlog = h.Vector()
    h.tlog.record (h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record (cell.sections[0](0.5)._ref_v)

    h.spikelog = h.Vector()
    nc = h.NetCon(cell.sections[0](0.5)._ref_v, h.nil)
    nc.threshold = -40.0
    nc.record(h.spikelog)
    
    h.tstop = tstop


    it = 1
    ## Increase the injected current until at least 60 spikes occur
    ## or up to 5 steps
    while (h.spikelog.size() < 50):

        utils.simulate(h, v_init, prelength,mainlength)
        
        if ((h.spikelog.size() < 50) & (it < 5)):
            print "ap_rate_test: stim1.amp = %g spikelog.size = %d\n" % (stim1.amp, h.spikelog.size())
            stim1.amp = stim1.amp + 0.1
            h.spikelog.clear()
            h.tlog.clear()
            h.Vlog.clear()
            it += 1
        else:
            break

    print "ap_rate_test: stim1.amp = %g spikelog.size = %d\n" % (stim1.amp, h.spikelog.size())

    isivect = h.Vector(h.spikelog.size()-1, 0.0)
    tspike = h.spikelog.x[0]
    for i in xrange(1,int(h.spikelog.size())):
        isivect.x[i-1] = h.spikelog.x[i]-tspike
        tspike = h.spikelog.x[i]
    
    print "ap_rate_test: isivect.size = %d\n" % isivect.size()
    isimean  = isivect.mean()
    isivar   = isivect.var()
    isistdev = isivect.stdev()
    
    isilast = int(isivect.size())-1
    if (isivect.size() > 10):
        isi10th = 10 
    else:
        isi10th = isilast
    
    ## Compute the last spike that is largest than the first one.
    ## This is necessary because some variants of the model generate spike doublets,
    ## (i.e. spike with very short distance between them, which confuse the ISI statistics.
    isilastgt = int(isivect.size())-1
    while (isivect.x[isilastgt] < isivect.x[1]):
        isilastgt = isilastgt-1
    
    if (not (isilastgt > 0)):
        isivect.printf()
        raise RuntimeError("Unable to find ISI greater than first ISI: forest_path = %s gid = %d" % (forest_path, gid))
    
    f=open("BasketCell_ap_rate_results.dat",'w')

    f.write ("## number of spikes: %g\n" % h.spikelog.size())
    f.write ("## FR mean: %g\n" % (1.0 / isimean))
    f.write ("## ISI mean: %g\n" % isimean) 
    f.write ("## ISI variance: %g\n" % isivar)
    f.write ("## ISI stdev: %g\n" % isistdev)
    f.write ("## ISI adaptation 1: %g\n" % (isivect.x[0] / isimean))
    f.write ("## ISI adaptation 2: %g\n" % (isivect.x[0] / isivect.x[isilast]))
    f.write ("## ISI adaptation 3: %g\n" % (isivect.x[0] / isivect.x[isi10th]))
    f.write ("## ISI adaptation 4: %g\n" % (isivect.x[0] / isivect.x[isilastgt]))

    f.close()

    f=open("BasketCell_voltage_trace.dat",'w')
    for i in xrange(0, int(h.tlog.size())):
        f.write('%g %g\n' % (h.tlog.x[i], h.Vlog.x[i]))
    f.close()

def fi_test (tree, v_init):

    cell = utils.new_cell ("BasketCell", neurotree_dict=tree)
    h.dt = 0.025

    prelength = 1000.0
    mainlength = 2000.0

    tstop = prelength+mainlength
    
    stimdur = 1000.0
    
    stim1 = h.IClamp(cell.sections[0](0.5))
    stim1.delay = prelength
    stim1.dur   = stimdur
    stim1.amp   = 0.2

    h.tlog = h.Vector()
    h.tlog.record (h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record (cell.sections[0](0.5)._ref_v)

    h.spikelog = h.Vector()
    nc = h.NetCon(cell.sections[0](0.5)._ref_v, h.nil)
    nc.threshold = -40.0
    nc.record(h.spikelog)
    
    h.tstop = tstop

    frs = []
    stim_amps = [stim1.amp]
    for it in range(1, 9):

        utils.simulate(h, v_init, prelength, mainlength)
        
        print "fi_test: stim1.amp = %g spikelog.size = %d\n" % (stim1.amp, h.spikelog.size())
        stim1.amp = stim1.amp + 0.1
        stim_amps.append(stim1.amp)
        frs.append(h.spikelog.size())
        h.spikelog.clear()
        h.tlog.clear()
        h.Vlog.clear()

    f=open("BasketCell_fi_results.dat",'w')

    for (fr,stim_amp) in itertools.izip(frs,stim_amps):
        f.write("%g %g\n" % (stim_amp,fr))

    f.close()


def gap_junction_test (tree, v_init):
    
    h('objref gjlist, cells, Vlog1, Vlog2')

    h.pc = h.ParallelContext()
    h.cells  = h.List()
    h.gjlist = h.List()
    
    cell1 = utils.new_cell ("BasketCell", neurotree_dict=tree)
    cell2 = utils.new_cell ("BasketCell", neurotree_dict=tree)

    h.cells.append(cell1)
    h.cells.append(cell2)

    ggid        = 20000000
    source      = 10422930
    destination = 10422670
    srcbranch   = 1
    dstbranch   = 2
    weight      = 5.4e-4

    stimdur     = 500
    tstop       = 2000
    
    h.pc.set_gid2node(source, int(h.pc.id()))
    nc = cell1.connect2target(h.nil)
    h.pc.cell(source, nc, 1)

    h.pc.set_gid2node(destination, int(h.pc.id()))
    nc = cell2.connect2target(h.nil)
    h.pc.cell(destination, nc, 1)

    stim1 = h.IClamp(cell1.sections[0](0.5))
    stim1.delay = 250
    stim1.dur = stimdur
    stim1.amp = -0.1

    stim2 = h.IClamp(cell2.sections[0](0.5))
    stim2.delay = 500+stimdur
    stim2.dur = stimdur
    stim2.amp = -0.1

    log_size = tstop/h.dt + 1
    
    h.tlog = h.Vector(log_size,0)
    h.tlog.record (h._ref_t)

    h.Vlog1 = h.Vector(log_size)
    h.Vlog1.record (cell1.sections[0](0.5)._ref_v)

    h.Vlog2 = h.Vector(log_size)
    h.Vlog2.record (cell2.sections[0](0.5)._ref_v)
    
    h.mkgap(h.pc, h.gjlist, source, srcbranch, ggid, ggid+1, weight)
    h.mkgap(h.pc, h.gjlist, destination, dstbranch, ggid+1, ggid, weight)

    h.pc.setup_transfer()
    h.pc.set_maxstep(10.0)

    h.stdinit()
    h.finitialize(v_init)
    h.pc.barrier()

    h.tstop = tstop
    h.pc.psolve(h.tstop)

    f=open("BasketCellGJ.dat",'w')
    for (t,v1,v2) in itertools.izip(h.tlog,h.Vlog1,h.Vlog2):
        f.write("%f %f %f\n" % (t,v1,v2))
    f.close()
    

def synapse_group_test (label, syntype, cell, w, v_holding, v_init):
    
    vv = h.Vector()
    vv.append(0,0,0,0,0,0)
    
    se = h.SEClamp(cell1.sections[0](0.5))
    
    v = cell.syntest(syntype,se,w,v_holding,v_init)
    vv = vv.add(v)
    
    amp     = vv.x[0]
    t_10_90 = vv.x[1]
    t_20_80 = vv.x[2]
    t_all   = vv.x[3]
    t_50    = vv.x[4]
    t_decay = vv.x[5]

    f=open("BasketCell_synapse_results.dat",'w')

    f.write("%s synapses: " % label)
    f.write("  Amplitude %f" % amp)
    f.write("  10-90 Rise Time %f" % t_10_90)
    f.write("  20-80 Rise Time %f" % t_20_80)
    f.write("  Decay Time Constant %f\n" % t_decay)

    f.close()


def synapse_test(tree, v_init):
    
    cell = utils.new_cell ("BasketCell", neurotree_dict=tree)
    synapse_group_test("Inhibitory", 1, cell, 0.0033, -70, v_init)
    synapse_group_test("Excitatory", 0, cell, 0.006, 0, v_init)

    

@click.command()
@click.option("--template-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(template_path,forest_path):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    h('objref nil, pc, tlog, Vlog, spikelog')
    h.load_file("nrngui.hoc")
    h.xopen("./lib.hoc")
    h.xopen ("./tests/rn.hoc")
    h.xopen(template_path+'/BasketCell.hoc')
    h.pc = h.ParallelContext()
    
    popName = "BC"
    (trees,_) = read_tree_selection (comm, forest_path, popName, [1039000])
    
    tree = trees.itervalues().next()
    
    passive_test(tree,-60)
    ap_rate_test(tree,-60)
    fi_test(tree,-60)
    gap_junction_test(tree,-60)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find("BasketCellTest.py") != -1,sys.argv)+1):])
