

import sys, os
import os.path
import click
import itertools
import numpy as np
from mpi4py import MPI # Must come before importing NEURON
from neuron import h
from neuroh5.io import read_tree_selection, read_cell_attribute_selection
import dentate
from dentate.env import Env
from dentate import neuron_utils, utils, cells

    
def passive_test (template_class, tree, v_init):

    cell = cells.make_neurotree_cell (template_class, neurotree_dict=tree)
    h.dt = 0.025

    prelength = 1000
    mainlength = 2000

    tstop = prelength+mainlength
    
    stimdur = 500.0

    soma = list(cell.soma)[0]

    stim1 = h.IClamp(soma(0.5))
    stim1.delay = prelength
    stim1.dur   = stimdur
    stim1.amp   = -0.1

    h.tlog = h.Vector()
    h.tlog.record (h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record (soma(0.5)._ref_v)
    
    h.tstop = tstop

    neuron_utils.simulate(h, v_init, prelength,mainlength)

    ## compute membrane time constant
    vrest  = h.Vlog.x[int(h.tlog.indwhere(">=",prelength-1))]
    vmin   = h.Vlog.min()
    vmax   = vrest
    
    ## the time it takes the system's step response to reach 1-1/e (or
    ## 63.2%) of the peak value
    amp23  = 0.632 * abs (vmax - vmin)
    vtau0  = vrest - amp23
    tau0   = h.tlog.x[int(h.Vlog.indwhere ("<=", vtau0))] - prelength

    f=open("MossyCell_passive_results.dat",'w')
    
    f.write ("DC input resistance: %g MOhm\n" % h.rn(cell))
    f.write ("vmin: %g mV\n" % vmin)
    f.write ("vtau0: %g mV\n" % vtau0)
    f.write ("tau0: %g ms\n" % tau0)

    f.close()

def ap_rate_test (template_class, tree, v_init):

    cell = cells.make_neurotree_cell (template_class, neurotree_dict=tree)
    h.dt = 0.025

    prelength = 2000.0
    mainlength = 2000.0

    tstop = prelength+mainlength
    
    stimdur = 1000.0

    soma = list(cell.soma)[0]
    
    stim1 = h.IClamp(soma(0.5))
    stim1.delay = prelength
    stim1.dur   = stimdur
    stim1.amp   = 0.1

    h.tlog = h.Vector()
    h.tlog.record (h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record (soma(0.5)._ref_v)

    h.spikelog = h.Vector()
    nc = h.NetCon(soma(0.5)._ref_v, h.nil)
    nc.threshold = -40.0
    nc.record(h.spikelog)
    
    h.tstop = tstop


    it = 1
    ## Increase the injected current until at least 40 spikes occur
    ## or up to 5 steps
    while (h.spikelog.size() < 40):

        neuron_utils.simulate(h, v_init, prelength,mainlength)
        
        if ((h.spikelog.size() < 40) & (it < 5)):
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
    
    f=open("MossyCell_ap_rate_results.dat",'w')

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

    f=open("MossyCell_voltage_trace.dat",'w')
    for i in xrange(0, int(h.tlog.size())):
        f.write('%g %g\n' % (h.tlog.x[i], h.Vlog.x[i]))
    f.close()
    

def synapse_group_test (label, syntype, cell, w, v_holding, v_init):
    
    vv = h.Vector()
    vv.append(0,0,0,0,0,0)

    soma = list(cell.soma)[0]
    se = h.SEClamp(soma(0.5))

    if syntype == 0:
        v = cell.syntest_exc(cell.syntypes.o(syntype),se,w,v_holding,v_init)
    else:
        v = cell.syntest_inh(cell.syntypes.o(syntype),se,w,v_holding,v_init)
        
    vv = vv.add(v)
    
    amp     = vv.x[0]
    t_10_90 = vv.x[1]
    t_20_80 = vv.x[2]
    t_all   = vv.x[3]
    t_50    = vv.x[4]
    t_decay = vv.x[5]

    f=open("MossyCell_%s_synapse_results.dat" % label,'w')

    f.write("%s synapses: \n" % label)
    f.write("  Amplitude %f\n" % amp)
    f.write("  10-90 Rise Time %f\n" % t_10_90)
    f.write("  20-80 Rise Time %f\n" % t_20_80)
    f.write("  Decay Time Constant %f\n" % t_decay)

    f.close()


def synapse_group_rate_test (label, syntype, cell, w, rate, v_init):

    res = cell.syntest_rate(cell.syntypes.o(syntype),w,rate,v_init)

    tlog = res.o(0)
    vlog = res.o(1)
    
    f=open("MossyCell_%s_synapse_rate.dat" % label,'w')

    for i in xrange(0, int(tlog.size())):
        f.write('%g %g\n' % (tlog.x[i], vlog.x[i]))

    f.close()
    

def synapse_test(template_class, tree, synapses, v_init, env, unique=True):
    
    cell = cells.make_neurotree_cell (template_class, neurotree_dict=tree)

    syn_ids      = synapses['syn_ids']
    syn_types    = synapses['syn_types']
    swc_types    = synapses['swc_types']
    syn_locs     = synapses['syn_locs']
    syn_sections = synapses['syn_secs']

    synapse_kinetics = {}
    synapse_kinetics[env.Synapse_Types['excitatory']] = env.celltypes['MC']['synapses']['kinetics']['GC']
    synapse_kinetics[env.Synapse_Types['inhibitory']] = env.celltypes['MC']['synapses']['kinetics']['BC']

    syn_params_dict = env.connection_config[postsyn_name]['MC'].mechanisms

    utils.mksyns(cell,syn_ids,syn_types,swc_types,syn_locs,syn_sections,synapse_kinetics,env)

    edge_syn_obj_dict = \
      synapses.mksyns(postsyn_gid, postsyn_cell, edge_syn_ids, syn_params_dict, env,
                      env.edge_count[postsyn_name][presyn_name],
                      add_synapse=synapses.add_unique_synapse if unique else synapses.add_shared_synapse)

    edge_syn_obj_dict = \
      synapses.mksyns(postsyn_gid, postsyn_cell, edge_syn_ids, syn_params_dict, env,
                      0, add_synapse=synapses.add_unique_synapse if unique else synapses.add_shared_synapse)

    print int(cell.syntypes.o(env.Synapse_Types['inhibitory']).count()), " inhibitory synapses"
    print int(cell.syntypes.o(env.Synapse_Types['excitatory']).count()), " excitatory synapses"
    
    synapse_group_test("Inhibitory", 1, cell, 0.0033, -70, 0)
    synapse_group_test("Excitatory", 0, cell, 0.006, 0, v_init)

    synapse_group_rate_test("Inhibitory", 1, cell, 0.0033, 150, v_init)
    synapse_group_rate_test("Excitatory", 0, cell, 0.006, 150, v_init)

    
@click.command()
@click.option("--config-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--synapses-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(config_path,template_path,forest_path,synapses_path):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    env = Env(comm=comm, configFile=config_path, templatePaths=template_path)

    h('objref nil, pc, tlog, Vlog, spikelog')
    h.load_file("nrngui.hoc")
    h.xopen("./lib.hoc")
    h.xopen ("./tests/rn.hoc")
    h.xopen(template_path+'/MossyCell.hoc')
    h.pc = h.ParallelContext()

    v_init = -75.0
    popName = "MC"
    gid = 1000000
    (trees,_) = read_tree_selection (forest_path, popName, [gid], comm=comm)
    if synapses_path is not None:
        synapses_dict = read_cell_attribute_selection (synapses_path, popName, [gid], "Synapse Attributes", comm=comm)
    else:
        synapses_dict = None

    gid, tree = trees.next()
    if synapses_dict is not None:
        (_, synapses) = synapses_dict.next()
    else:
        synapses = None

    template_class = getattr(h, "MossyCell")

    passive_test(template_class, tree, v_init)
    ap_rate_test(template_class, tree, v_init)

    if synapses is not None:
        synapse_test(template_class, tree, synapses, v_init, env)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find("MossyCellTest.py") != -1,sys.argv)+1):])
