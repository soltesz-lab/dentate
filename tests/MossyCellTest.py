

import sys, os, random, os.path, itertools
import click
import numpy as np
from collections import defaultdict
from mpi4py import MPI # Must come before importing NEURON
from neuron import h
from neuroh5.io import read_tree_selection, read_cell_attribute_selection
import dentate
from dentate.env import Env
from dentate import neuron_utils, utils, cells, synapses, network_clamp

    
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
    

def synapse_group_test (env, presyn_name, gid, cell, syn_params_dict, group_size, v_init, tstart = 200.):

    syn_attrs = env.synapse_attributes
    
    vv = h.Vector()
    vv.append(0,0,0,0,0,0)

    ranstream = np.random.RandomState(0)

    syn_ids = syn_attrs.filter_synapses(gid, sources=[presyn_name], cache=True).keys()

    if len(syn_ids) == 0:
        return
    
    selected = ranstream.choice(np.arange(0, len(syn_ids)), size=group_size, replace=False)
    selected_ids = [ syn_ids[i] for i in selected ]

    for syn_name in syn_params_dict.keys():

        print ('synapse_group_test: configuring %s %s synapses' % (presyn_name, syn_name))
        
        synlst = []
        for syn_id in selected_ids:
            synlst.append(syn_attrs.get_pps(gid, syn_id, syn_name))
            
        print ('synapse_group_test: %s %s synapses: %i out of %i' % (presyn_name, syn_name, len(synlst), len(syn_ids)))

        ns = h.NetStim()
        ns.interval = 1000
        ns.number = 1
        ns.start  = 200
        ns.noise  = 0
        
        nclst = []
        for syn_id, syn in itertools.izip(selected_ids, synlst):
            this_nc = h.NetCon(ns,syn)
            this_nc.weight[0] = 1.
            syn_attrs.append_netcon(gid, syn_id, syn_name, this_nc)
            config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules,
                       mech_names=syn_attrs.syn_mech_names, nc=this_nc,
                       **syn_params_dict[syn_name])
            nclst.append(this_nc)

        if syn_name == 'SatAMPA':
            v_holding = -75
            v = cell.syntest_exc(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'AMPA':
            v_holding = -75
            v = cell.syntest_exc(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'SatGABA':
            v_holding = 0
            v = cell.syntest_inh(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'GABA':
            v_holding = 0
            v = cell.syntest_inh(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'SatGABA_A':
            v_holding = 0
            v = cell.syntest_inh(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'AdGABA_A':
            v_holding = 0
            v = cell.syntest_inh(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'GABA_A':
            v_holding = 0
            v = cell.syntest_inh(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'AdGABA_B':
            v_holding = 0
            v = cell.syntest_inh(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'SatGABA_B':
            v_holding = 0
            v = cell.syntest_inh(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'GABA_B':
            v_holding = 0
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
    

def synapse_group_rate_test (env, presyn_name, gid, cell, syn_params_dict, group_size, rate, tstart = 200.):

    syn_attrs = env.synapse_attributes
    ranstream = np.random.RandomState(0)

    syn_ids = syn_attrs.filter_synapses(gid, sources=[env.Populations[presyn_name]], cache=True).keys()

    if len(syn_ids) == 0:
        return

    selected = ranstream.choice(np.arange(0, len(syn_ids)), size=group_size, replace=False)
    selected_ids = [ syn_ids[i] for i in selected ]

    for syn_name in syn_params_dict.keys():

        synlst = []
        for syn_id in selected_ids:
            synlst.append(syn_attrs.get_pps(gid, syn_id, syn_name))
    
        print ('synapse_group_rate_test: %s %s synapses: %i out of %i ' % (presyn_name, syn_name, len(synlst), len(syn_ids)))

        ns = h.NetStim()
        ns.interval = 1000./rate
        ns.number = rate
        ns.start  = 200
        ns.noise  = 0
        
        nclst = []
        for syn in synlst:
            this_nc = syn_attrs.get_netcon(gid, syn_id, syn_name, throw_error=False)
            if this_nc is None:
                this_nc = h.NetCon(ns,syn)
                this_nc.weight[0] = 1.
                syn_attrs.add_netcon(gid, syn_id, syn_name, this_nc)
            synapses.config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules,
                                mech_names=syn_attrs.syn_mech_names, nc=this_nc,
                                **syn_params_dict[syn_name])
            nclst.append(this_nc)

        print ('synapse_group_rate_test: %s %s synapses: %i netcons ' % (presyn_name, syn_name, len(nclst)))

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
        elif syn_name == 'AdGABA_A':
            v_init = 0
        elif syn_name == 'GABA_A':
            v_init = 0
        elif syn_name == 'SatGABA_B':
            v_init = 0
        elif syn_name == 'AdGABA_B':
            v_init = 0
        elif syn_name == 'GABA_B':
            v_init = 0
        else:
            raise RuntimeError('Unknown synapse mechanism type %s' % syn_name)

        res = cell.hoc_cell.syntest_rate(tstart,rate,v_init)

        tlog = res.o(0)
        vlog = res.o(1)
        
        f=open("MossyCell_%s_%s_synapse_rate_%i.dat" % (presyn_name, syn_name, group_size),'w')
        
        for i in xrange(0, int(tlog.size())):
            f.write('%g %g\n' % (tlog.x[i], vlog.x[i]))
            
        f.close()

def synapse_test(template_class, gid, tree, synapses_dict, v_init, env, unique=True):

    syn_attrs = env.synapse_attributes
    
    postsyn_name = 'MC'
    presyn_names = ['GC', 'MC', 'HC', 'BC', 'AAC', 'HCC']

    cell = network_clamp.load_cell(env, postsyn_name, gid, mech_file_path=None, \
                                   tree_dict=tree, load_synapses=False, \
                                   correct_for_spines=True, load_edges=False)
    
    network_clamp.register_cell(env, postsyn_name, gid, cell)


    all_syn_ids = synapses_dict['syn_ids']
    all_syn_layers = synapses_dict['syn_layers']
    all_syn_secs = synapses_dict['syn_secs']

    for presyn_name in presyn_names:

        syn_ids = []
        layers = env.connection_config[postsyn_name][presyn_name].layers
        proportions = env.connection_config[postsyn_name][presyn_name].proportions
        for syn_id, syn_layer, syn_sec in itertools.izip(all_syn_ids, all_syn_layers, all_syn_secs):
            i = utils.list_index(syn_layer, layers) 
            if i is not None:
                if (random.random() <= proportions[i]):
                        syn_ids.append(syn_id)

        syn_attrs.init_syn_id_attrs(gid, synapses_dict)
        synapses.init_syn_mech_attrs(cell, env)
        syn_attrs.init_edge_attrs(gid, presyn_name, np.zeros((len(syn_ids),)),
                                  syn_ids, delays=None)
        synapses.config_hoc_cell_syns(env, gid, postsyn_name, syn_ids=syn_ids,
                                      insert=True, verbose=True)

        syn_params_dict = env.connection_config[postsyn_name][presyn_name].mechanisms['default']

        rate = 20
        
        synapse_group_rate_test(env, presyn_name, gid, cell, syn_params_dict, 1, rate)
        #synapse_group_rate_test(env, presyn_name, gid, cell, syn_params_dict, 10, rate)
        #synapse_group_rate_test(env, presyn_name, gid, cell, syn_params_dict, 50, rate)

        del(syn_attrs.pps_dict[gid])
        syn_attrs.del_syn_id_attr_dict(gid)
        
        # synapse_group_test(env, presyn_name, gid, cell, syn_params_dict, 1, v_init)
        # synapse_group_test(env, presyn_name, gid, cell, syn_params_dict, 10, v_init)
        # synapse_group_test(env, presyn_name, gid, cell, syn_params_dict, 40, v_init)
        # synapse_group_test(env, presyn_name, gid, cell, syn_params_dict, 50, v_init)
        # synapse_group_test(env, presyn_name, gid, cell, syn_params_dict, 100, v_init)
        # synapse_group_test(env, presyn_name, gid, cell, syn_params_dict, 200, v_init)
        # synapse_group_test(env, presyn_name, gid, cell, syn_params_dict, 400, v_init)
    

 
    
@click.command()
@click.option("--config-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--synapses-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(config_path,template_path,forest_path,synapses_path):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    env = Env(comm=comm, config_file=config_path, template_paths=template_path, verbose=True)

    neuron_utils.configure_hoc_env(env)
    
    h.pc = h.ParallelContext()

    v_init = -75.0
    popName = "MC"
    gid = 1000000

    env.load_cell_template(popName)

    (trees,_) = read_tree_selection (forest_path, popName, [gid], comm=comm)
    if synapses_path is not None:
        synapses_iter = read_cell_attribute_selection (synapses_path, popName, [gid], "Synapse Attributes", comm=comm)
    else:
        synapses_iter = None

    gid, tree = trees.next()
    if synapses_iter is not None:
        (_, synapses_dict) = synapses_iter.next()
    else:
        synapses_dict = None

    template_class = getattr(h, "MossyCell")

    v_init = -75.0
    if (synapses is not None):
        synapse_test(template_class, gid, tree, synapses_dict, v_init, env)

#    v_init = -60
#    passive_test(template_class, tree, v_init)
#    ap_rate_test(template_class, tree, v_init)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find("MossyCellTest.py") != -1,sys.argv)+1):])
