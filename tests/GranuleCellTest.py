import itertools
import os
import os.path
import random
import sys
from collections import defaultdict

import numpy as np

import click
from dentate import cells
from dentate import network_clamp
from dentate import neuron_utils
from dentate import synapses
from dentate import utils
from dentate.env import Env
from dentate.utils import *
from mpi4py import MPI  # Must come before importing NEURON
from neuroh5.io import read_cell_attribute_selection
from neuroh5.io import read_tree_selection
from neuron import h


def synapse_group_test (env, presyn_name, gid, cell, syn_params_dict, group_size, v_init, tstart = 200.):

    syn_attrs = env.synapse_attributes
    
    vv = h.Vector()
    vv.append(0,0,0,0,0,0)

    ranstream = np.random.RandomState(0)

    syn_ids = list(syn_obj_dict.keys())

    if len(syn_ids) == 0:
        return
    
    selected = ranstream.choice(np.arange(0, len(syn_ids)), size=group_size, replace=False)
    selected_ids = [ syn_ids[i] for i in selected ]

    for syn_name in syn_params_dict:
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
        for syn_id, syn in zip(selected_ids, synlst):
            this_nc = h.NetCon(ns,syn)
            syn_attrs.append_netcon(gid, syn_id, syn_name, this_nc)
            config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules,
                    mech_names=syn_attrs.syn_mech_names, nc=this_nc,
                    **syn_params_dict[syn_name])
            nclst.append(this_nc)

        if syn_name == 'SatAMPA':
            v_holding = -75
            v = cell.syntest_exc(tstart,v_holding,v_init,"GranuleCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'AMPA':
            v_holding = -75
            v = cell.syntest_exc(tstart,v_holding,v_init,"GranuleCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'SatGABA':
            v_holding = 0
            v = cell.syntest_inh(tstart,v_holding,v_init,"GranuleCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'GABA':
            v_holding = 0
            v = cell.syntest_inh(tstart,v_holding,v_init,"GranuleCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'SatGABA_A':
            v_holding = 0
            v = cell.syntest_inh(tstart,v_holding,v_init,"GranuleCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'GABA_A':
            v_holding = 0
            v = cell.syntest_inh(tstart,v_holding,v_init,"GranuleCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'SatGABA_B':
            v_holding = 0
            v = cell.syntest_inh(tstart,v_holding,v_init,"GranuleCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name == 'GABA_B':
            v_holding = 0
            v = cell.syntest_inh(tstart,v_holding,v_init,"GranuleCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        else:
            raise RuntimeError('Unknown synapse mechanism type %s' % syn_name)
        vv = vv.add(v)
    
        amp     = vv.x[0]
        t_10_90 = vv.x[1]
        t_20_80 = vv.x[2]
        t_all   = vv.x[3]
        t_50    = vv.x[4]
        t_decay = vv.x[5]

        f=open("GranuleCell_%s_%s_synapse_results_%i.dat" % (presyn_name, syn_name, group_size), 'w')

        f.write("%s synapses: \n" % syn_name)
        f.write("  Amplitude %f\n" % amp)
        f.write("  10-90 Rise Time %f\n" % t_10_90)
        f.write("  20-80 Rise Time %f\n" % t_20_80)
        f.write("  Decay Time Constant %f\n" % t_decay)
        
        f.close()


def synapse_group_rate_test (env, presyn_name, gid, cell, syn_params_dict, group_size, rate, tstart = 200.):

    syn_attrs = env.synapse_attributes
    ranstream = np.random.RandomState(0)

    syn_ids = list(syn_obj_dict.keys())

    if len(syn_ids) == 0:
        return

    selected = ranstream.choice(np.arange(0, len(syn_ids)), size=group_size, replace=False)
    selected_ids = [ syn_ids[i] for i in selected ]

    for syn_name in syn_params_dict:
        
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
            for syn_name, syn in viewitems(syn_obj_dict[syn_id]):
                this_nc = h.NetCon(ns,syn)
                syn_attrs.append_netcon(gid, syn_id, syn_name, this_nc)
                config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules,
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
        
        f=open("GranuleCell_%s_%s_synapse_rate_%i.dat" % (presyn_name, syn_name, group_size),'w')
        
        for i in range(0, int(tlog.size())):
            f.write('%g %g\n' % (tlog.x[i], vlog.x[i]))
            
        f.close()
    

def synapse_test(template_class, mech_file_path, gid, tree, synapses, v_init, env, unique=True):
    
    postsyn_name = 'GC'
    presyn_names = ['MPP', 'LPP', 'MC', 'HC', 'BC', 'AAC', 'HCC']

    cell = network_clamp.load_cell(env, postsyn_name, gid, mech_file_path=mech_file_path, \
                                   tree_dict=tree, synapses_dict=synapses, \
                                   correct_for_spines=True, load_edges=False)

    all_syn_ids = synapses['syn_ids']
    all_syn_layers = synapses['syn_layers']
    all_syn_secs = synapses['syn_secs']
    print ('Total %i %s synapses' % (len(all_syn_ids), postsyn_name))
    env.cells.append(cell)
    env.pc.set_gid2node(gid, env.comm.rank)
    
    for presyn_name in presyn_names:

        syn_ids = []
        layers = env.connection_config[postsyn_name][presyn_name].layers
        proportions = env.connection_config[postsyn_name][presyn_name].proportions
        for syn_id, syn_layer, syn_sec in zip(all_syn_ids, all_syn_layers, all_syn_secs):
            i = utils.list_index(syn_layer, layers) 
            if i is not None:
                if (random.random() <= proportions[i]):
                    syn_ids.append(syn_id)
        
        syn_params_dict = env.connection_config[postsyn_name][presyn_name].mechanisms
        rate = 40
        
        synapse_group_rate_test(env, presyn_name, gid, cell, syn_params_dict, 1, rate)
        synapse_group_rate_test(env, presyn_name, gid, cell, syn_params_dict, 10, rate)
        synapse_group_rate_test(env, presyn_name, gid, cell, syn_params_dict, 50, rate)

        synapse_group_test(env, presyn_name, gid, cell, syn_params_dict, 1, v_init)
        synapse_group_test(env, presyn_name, gid, cell, syn_params_dict, 10, v_init)
        synapse_group_test(env, presyn_name, gid, cell, syn_params_dict, 40, v_init)
        synapse_group_test(env, presyn_name, gid, cell, syn_params_dict, 50, v_init)
        synapse_group_test(env, presyn_name, gid, cell, syn_params_dict, 100, v_init)
        synapse_group_test(env, presyn_name, gid, cell, syn_params_dict, 200, v_init)
        synapse_group_test(env, presyn_name, gid, cell, syn_params_dict, 400, v_init)
        
 
    
@click.command()
@click.option("--config-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-paths", required=True, type=str)
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--synapses-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(config_path,template_paths,forest_path,synapses_path):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    env = Env(comm=comm, config_file=config_path, template_paths=template_paths)

    neuron_utils.configure_hoc_env(env)
    
    h.pc = h.ParallelContext()

    v_init = -75.0
    popName = "GC"
    gid = 500500

    env.load_cell_template(popName)
    
    (trees,_) = read_tree_selection (forest_path, popName, [gid], comm=comm)
    if synapses_path is not None:
        synapses_dict = read_cell_attribute_selection (synapses_path, popName, [gid], \
                                                       "Synapse Attributes", comm=comm)
    else:
        synapses_dict = None

    gid, tree = next(trees)
    if synapses_dict is not None:
        (_, synapses) = next(synapses_dict)
    else:
        synapses = None

    if 'mech_file' in env.celltypes[popName]:
        mech_file_path = env.config_prefix + '/' + env.celltypes[popName]['mech_file']
    else:
        mech_file_path = None

    template_class = getattr(h, "DGC")

    if (synapses is not None):
        synapse_test(template_class, mech_file_path, gid, tree, synapses, v_init, env)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find("GranuleCellTest.py") != -1,sys.argv)+1):])
