

import sys, os, random, os.path, itertools
import click
import numpy as np
from collections import defaultdict
from mpi4py import MPI # Must come before importing NEURON
from neuron import h
from neuroh5.io import read_tree_selection, read_cell_attribute_selection
import dentate
from dentate.env import Env
from dentate import neuron_utils, utils, cells
from dentate.synapses import mksyns, mknetcon, config_syn, add_unique_synapse, add_shared_synapse
    
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

    neuron_utils.simulate(v_init, prelength,mainlength)

    ## compute membrane time constant
    vrest  = h.Vlog.x[int(h.tlog.indwhere(">=",prelength-1))]
    vmin   = h.Vlog.min()
    vmax   = vrest
    
    ## the time it takes the system's step response to reach 1-1/e (or
    ## 63.2%) of the peak value
    amp23  = 0.632 * abs (vmax - vmin)
    vtau0  = vrest - amp23
    tau0   = h.tlog.x[int(h.Vlog.indwhere ("<=", vtau0))] - prelength

    f=open("MossyCell_passive_trace.dat",'w')
    for i in xrange(0, int(h.tlog.size())):
        f.write('%g %g\n' % (h.tlog.x[i], h.Vlog.x[i]))
    f.close()

    f=open("MossyCell_passive_results.dat",'w')
    
    f.write ("DC input resistance: %g MOhm\n" % h.rn(cell))
    f.write ("vmin: %g mV\n" % vmin)
    f.write ("vmax: %g mV\n" % vmax)
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
    nc = cell.connect2target(h.nil)
    nc.threshold = -40.0
    nc.record(h.spikelog)
    
    h.tstop = tstop


    it = 1
    ## Increase the injected current until at least 50 spikes occur
    ## or up to 5 steps
    while (h.spikelog.size() < 50):

        neuron_utils.simulate(v_init, prelength,mainlength)
        
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
    

def synapse_group_test (env, presyn_name, gid, cell, syn_obj_dict, syn_params_dict, group_size, v_holding, v_init, tstart = 200.):

    syn_attrs = env.synapse_attributes
    
    vv = h.Vector()
    vv.append(0,0,0,0,0,0)

    ranstream = np.random.RandomState(0)

    syn_ids = syn_obj_dict.keys()

    if len(syn_ids) == 0:
        return
    
    selected = ranstream.choice(np.arange(0, len(syn_ids)), size=group_size, replace=False)
    selected_ids = [ syn_ids[i] for i in selected ]

    for syn_name in syn_params_dict.keys():
        synlst = []
        for syn_id in selected_ids:
            synlst.append(syn_obj_dict[syn_id][syn_name])
            
        print ('synapse_group_test: %s %s synapses: %i out of %i' % (presyn_name, syn_name, len(synlst), len(syn_ids)))

        ns = h.NetStim()
        ns.interval = 1000
        ns.number = 1
        ns.start  = 200
        ns.noise  = 0
        
        nclst = []
        for syn_id, syn in itertools.izip(selected_ids, synlst):
            this_nc = h.NetCon(ns,syn)
            syn_attrs.append_netcon(gid, syn_id, syn_name, this_nc)
            config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules,
                    mech_names=syn_attrs.syn_mech_names, nc=this_nc,
                    **syn_params_dict[syn_name])
            nclst.append(this_nc)

        if syn_name == 'SatAMPA':
            v = cell.syntest_exc(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'AMPA':
            v = cell.syntest_exc(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'SatGABA':
            v = cell.syntest_inh(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'GABA':
            v = cell.syntest_inh(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'SatGABA_A':
            v = cell.syntest_inh(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'GABA_A':
            v = cell.syntest_inh(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'SatGABA_B':
            v = cell.syntest_inh(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'GABA_B':
            v = cell.syntest_inh(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        else:
            raise RuntimeError('Unknown synapse mechanism type %s' % syn_name)
        vv = vv.add(v)
    
        amp     = vv.x[0]
        t_10_90 = vv.x[1]
        t_20_80 = vv.x[2]
        t_all   = vv.x[3]
        t_50    = vv.x[4]
        t_decay = vv.x[5]

        f=open("MossyCell_%s_%s_synapse_results_%i.dat" % (presyn_name, syn_name, group_size), 'w')

        f.write("%s synapses: \n" % syn_name)
        f.write("  Amplitude %f\n" % amp)
        f.write("  10-90 Rise Time %f\n" % t_10_90)
        f.write("  20-80 Rise Time %f\n" % t_20_80)
        f.write("  Decay Time Constant %f\n" % t_decay)
        
        f.close()


def synapse_group_rate_test (env, presyn_name, gid, cell, syn_obj_dict, syn_params_dict, group_size, rate, tstart = 200.):

    syn_attrs = env.synapse_attributes
    ranstream = np.random.RandomState(0)

    syn_ids = syn_obj_dict.keys()

    if len(syn_ids) == 0:
        return

    selected = ranstream.choice(np.arange(0, len(syn_ids)), size=group_size, replace=False)
    selected_ids = [ syn_ids[i] for i in selected ]

    for syn_name in syn_params_dict.keys():
        
        synlst = []
        for syn_id in selected_ids:
            synlst.append(syn_obj_dict[syn_id][syn_name])
    
        print ('synapse_group_rate_test: %s %s synapses: %i out of %i ' % (presyn_name, syn_name, len(synlst), len(syn_ids)))
    
        ns = h.NetStim()
        ns.interval = 1000./rate
        ns.number = rate
        ns.start  = 200
        ns.noise  = 0
        
        nclst = []
        for syn_id in selected_ids:
            for syn_name, syn in syn_obj_dict[syn_id].iteritems():
                this_nc = h.NetCon(ns,syn)
                syn_attrs.append_netcon(gid, syn_id, syn_name, this_nc)
                config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules,
                            mech_names=syn_attrs.syn_mech_names, nc=this_nc,
                            **syn_params_dict[syn_name])
                nclst.append(this_nc)

        if syn_name == 'SatAMPA':
            v_init = -75
        elif syn_name == 'AMPA':
            v_init = -75
        elif syn_name == 'SatGABA':
            v_init = 0
        elif syn_name == 'GABA':
            v_init = 0
        elif syn_name == 'SatGABA_A':
            v_init = 0
        elif syn_name == 'GABA_A':
            v_init = 0
        elif syn_name == 'SatGABA_B':
            v_init = 0
        elif syn_name == 'GABA_B':
            v_init = 0
        else:
            raise RuntimeError('Unknown synapse mechanism type %s' % syn_name)

        res = cell.syntest_rate(tstart,rate,v_init)

        tlog = res.o(0)
        vlog = res.o(1)
        
        f=open("MossyCell_%s_%s_synapse_rate_%.1f_%i.dat" % (presyn_name, syn_name, rate, group_size),'w')
        
        for i in xrange(0, int(tlog.size())):
            f.write('%g %g\n' % (tlog.x[i], vlog.x[i]))
            
        f.close()
    

def synapse_test(template_class, gid, tree, synapses, v_init, env, unique=True):
    
    postsyn_name = 'MC'
    presyn_names = ['GC', 'MC', 'HC', 'BC', 'AAC', 'HCC']
    
    cell = cells.make_neurotree_cell (template_class, neurotree_dict=tree)

    all_syn_ids = synapses['syn_ids']
    all_syn_layers = synapses['syn_layers']
    all_syn_sections = synapses['syn_secs']

    print ('Total %i %s synapses' % (len(all_syn_ids), postsyn_name))

    env.cells.append(cell)
    env.pc.set_gid2node(gid, env.comm.rank)

    syn_attrs = env.synapse_attributes
    syn_attrs.load_syn_id_attrs(gid, synapses)
    
    for presyn_name in presyn_names:

        syn_ids = []
        layers = env.connection_config[postsyn_name][presyn_name].layers
        proportions = env.connection_config[postsyn_name][presyn_name].proportions
        for syn_id, syn_layer in itertools.izip(all_syn_ids, all_syn_layers):
            i = utils.list_index(syn_layer, layers) 
            if i is not None:
                if random.random() <= proportions[i]:
                    syn_ids.append(syn_id)
        
        syn_params_dict = env.connection_config[postsyn_name][presyn_name].mechanisms
        syn_obj_dict = mksyns(gid, cell, syn_ids, syn_params_dict, env, 0,
                        add_synapse=add_unique_synapse if unique else add_shared_synapse)

        v_holding = -60
        synapse_group_test(env, presyn_name, gid, cell, syn_obj_dict, syn_params_dict, 1, v_holding, v_init)
        synapse_group_test(env, presyn_name, gid, cell, syn_obj_dict, syn_params_dict, 10, v_holding, v_init)
        synapse_group_test(env, presyn_name, gid, cell, syn_obj_dict, syn_params_dict, 100, v_holding, v_init)
        
        rate = 10
        synapse_group_rate_test(env, presyn_name, gid, cell, syn_obj_dict, syn_params_dict, 1, rate)
        synapse_group_rate_test(env, presyn_name, gid, cell, syn_obj_dict, syn_params_dict, 10, rate)

        rate = 30
        synapse_group_rate_test(env, presyn_name, gid, cell, syn_obj_dict, syn_params_dict, 1, rate)
        synapse_group_rate_test(env, presyn_name, gid, cell, syn_obj_dict, syn_params_dict, 10, rate)

        rate = 50
        synapse_group_rate_test(env, presyn_name, gid, cell, syn_obj_dict, syn_params_dict, 1, rate)
        synapse_group_rate_test(env, presyn_name, gid, cell, syn_obj_dict, syn_params_dict, 10, rate)

 
    
@click.command()
@click.option("--config-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--synapses-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(config_path,template_path,forest_path,synapses_path):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    env = Env(comm=comm, config_file=config_path, template_paths=template_path)

    h('objref nil, pc, tlog, Vlog, spikelog')
    h.load_file("nrngui.hoc")
    h.xopen ("./tests/rn.hoc")
    h.xopen(template_path+'/MossyCell.hoc')
    h.pc = h.ParallelContext()

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

    v_init = -75.0
    if (synapses is not None):
        synapse_test(template_class, gid, tree, synapses, v_init, env)

    v_init = -60
    passive_test(template_class, tree, v_init)
    ap_rate_test(template_class, tree, v_init)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find("MossyCellTest.py") != -1,sys.argv)+1):])
