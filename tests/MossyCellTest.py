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


def synapse_group_test (env, presyn_name, gid, cell, syn_ids, syn_mech_dict, group_size, v_init, tstart = 200.):

    syn_attrs = env.synapse_attributes
    
    vv = h.Vector()
    vv.append(0,0,0,0,0,0)

    ranstream = np.random.RandomState(0)

    if len(syn_ids) == 0:
        return
    
    selected = ranstream.choice(np.arange(0, len(syn_ids)), size=group_size, replace=False)
    selected_ids = [ syn_ids[i] for i in selected ]

    for syn_name in syn_mech_dict:
        nclst = []
            
        print('synapse_group_test: %s %s synapses: %i out of %i' % (presyn_name, syn_name, len(selected_ids), len(syn_ids)))

        ns = h.NetStim()
        ns.interval = 1000
        ns.number = 1
        ns.start  = 200
        ns.noise  = 0
        
        nclst = []
        first_syn_id = None
        for syn_id in selected_ids:
            if syn_attrs.has_netcon(gid, syn_id, syn_name):
                syn_index = syn_attrs.syn_name_index_dict[syn_name]
                del (syn_attrs.pps_dict[gid][syn_id].netcon[syn_index])
            syn = syn_attrs.get_pps(gid, syn_id, syn_name)
            this_nc = h.NetCon(ns,syn)
            syn_attrs.add_netcon(gid, syn_id, syn_name, this_nc)
            synapses.config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules,
                                mech_names=syn_attrs.syn_mech_names, nc=this_nc,
                                **syn_mech_dict[syn_name])
            nclst.append(this_nc)
            if first_syn_id is None:
                first_syn_id = syn_id
                print("%s netcon: %s" % (syn_name, str([this_nc.weight[i] for i in range(int(this_nc.wcnt()))])))

        for sec in list(cell.all):
            h.psection(sec=sec)

        v_holding_exc = -75
        v_holding_inh = -45
        
        if syn_name in ['NMDA', 'SatAMPA', 'AMPA']:
            v_holding = v_holding_exc
            v = cell.syntest_exc(tstart,v_holding,v_init,"MossyCell_%s_%s_synapse_trace_%i.dat" % (presyn_name, syn_name, group_size))
        elif syn_name in ['SatGABA', 'GABA', 'SatGABA_A', 'GABA_A', 'SatGABA_B', 'GABA_B']:
            v_holding = v_holding_inh
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


def synapse_group_rate_test (env, presyn_name, gid, cell, syn_ids, syn_mech_dict, group_size, rate, tstart = 200.):

    syn_attrs = env.synapse_attributes

    syn_attrs = env.synapse_attributes
    ranstream = np.random.RandomState(0)

    if len(syn_ids) == 0:
        return

    selected = ranstream.choice(np.arange(0, len(syn_ids)), size=group_size, replace=False)
    selected_ids = [ syn_ids[i] for i in selected ]

    for syn_name in syn_mech_dict:
        
        print('synapse_group_rate_test: %s %s synapses: %i out of %i ' % (presyn_name, syn_name, len(selected_ids), len(syn_ids)))

        ns = h.NetStim()
        ns.interval = 1000./rate
        ns.number = rate
        ns.start  = 200
        ns.noise  = 0
        
        nclst = []
        first_syn_id = None
        for syn_id in selected_ids:
            for syn_name in syn_mech_dict:
                if syn_attrs.has_netcon(gid, syn_id, syn_name):
                    syn_index = syn_attrs.syn_name_index_dict[syn_name]
                    del (syn_attrs.pps_dict[gid][syn_id].netcon[syn_index])
                syn = syn_attrs.get_pps(gid, syn_id, syn_name)
                this_nc = h.NetCon(ns,syn)
                syn_attrs.add_netcon(gid, syn_id, syn_name, this_nc)
                synapses.config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules,
                                    mech_names=syn_attrs.syn_mech_names, nc=this_nc,
                                    **syn_mech_dict[syn_name])
                nclst.append(this_nc)
                if first_syn_id is None:
                    print("%s netcon: %s" % (syn_name, str([this_nc.weight[i] for i in range(int(this_nc.wcnt()))])))
            if first_syn_id is None:
                first_syn_id = syn_id

        for sec in list(cell.all):
            h.psection(sec=sec)

        print('synapse_group_rate_test: %s %s synapses: %i netcons ' % (presyn_name, syn_name, len(nclst)))

        v_init_exc = -75
        v_init_inh = 0
        
        if syn_name in ['NMDA', 'SatAMPA', 'AMPA']:
            v_init = v_init_exc
        elif syn_name in ['SatGABA', 'GABA', 'SatGABA_A', 'GABA_A', 'SatGABA_B', 'GABA_B']:
            v_init = v_init_inh
        else:
            raise RuntimeError('Unknown synapse mechanism type %s' % syn_name)

        res = cell.syntest_rate(tstart,rate,v_init)

        tlog = res.o(0)
        vlog = res.o(1)
        
        f=open("MossyCell_%s_%s_synapse_rate_%i.dat" % (presyn_name, syn_name, group_size),'w')
        
        for i in range(0, int(tlog.size())):
            f.write('%g %g\n' % (tlog.x[i], vlog.x[i]))
            
        f.close()
    

def synapse_test(template_class, mech_file_path, gid, tree, synapses_dict, v_init, env, unique=True):
    
    
    postsyn_name = 'MC'
    presyn_names = ['GC', 'MC', 'HC', 'BC', 'AAC', 'HCC']

    syn_attrs = env.synapse_attributes

    cell = network_clamp.load_cell(env, postsyn_name, gid, mech_file_path=mech_file_path, \
                                   tree_dict=tree, synapses_dict=synapses_dict, \
                                   correct_for_spines=True, load_edges=False)

    network_clamp.register_cell(env, postsyn_name, gid, cell)

    

    all_syn_ids = synapses_dict['syn_ids']
    all_syn_layers = synapses_dict['syn_layers']
    all_syn_secs = synapses_dict['syn_secs']
    print('Total %i %s synapses' % (len(all_syn_ids), postsyn_name))

    
    for presyn_name in presyn_names:

        projection_config = env.connection_config[postsyn_name]
        syn_params_dict = projection_config[presyn_name].mechanisms

        syn_type = projection_config[presyn_name].type
        syn_layers = projection_config[presyn_name].layers
        syn_sections = projection_config[presyn_name].sections
                                                     


        if 'default' in syn_params_dict:
            syn_mech_dict = syn_params_dict['default']
            syn_ids = syn_attrs.get_filtered_syn_ids(gid, swc_types=syn_sections, layers=syn_layers,
                                                     syn_types=[syn_type])
            for syn_id in syn_ids:
                if syn_id in syn_attrs.pps_dict[gid]:
                    del(syn_attrs.pps_dict[gid][syn_id])
            synapses.insert_hoc_cell_syns(env, syn_params_dict, gid, cell.hoc_cell, syn_ids,
                                          insert_netcons=False)
            synapses.config_hoc_cell_syns(env, gid, postsyn_name, cell=cell.hoc_cell, syn_ids=syn_ids,
                                          insert=False, insert_netcons=False, throw_error=True, verbose=True)
        else:
            for syn_section, syn_layer in zip(syn_sections, syn_layers):
                syn_mech_dict = syn_params_dict[syn_section]
                syn_ids = syn_attrs.get_filtered_syn_ids(gid, swc_types=[syn_section], layers=[syn_layer],
                                                         syn_types=[syn_type])
                for syn_id in syn_ids:
                    if syn_id in syn_attrs.pps_dict[gid]:
                        del(syn_attrs.pps_dict[gid][syn_id])
                synapses.insert_hoc_cell_syns(env, syn_params_dict, gid, cell.hoc_cell, syn_ids,
                                            insert_netcons=False)
                synapses.config_hoc_cell_syns(env, gid, postsyn_name, cell=cell.hoc_cell, syn_ids=syn_ids,
                                            insert=False, insert_netcons=False, throw_error=True, verbose=True)
                
        
        rate = 1

        for sz in [1, 10, 50]:
            synapse_group_rate_test(env, presyn_name, gid, cell.hoc_cell, syn_ids, syn_mech_dict, sz, rate)

        for sz in [1, 10, 40, 50, 100, 200, 400]:
            synapse_group_test(env, presyn_name, gid, cell.hoc_cell, syn_ids, syn_mech_dict, sz, v_init)
        
 
    
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

    v_init = -65.0
    popName = "MC"
    gid = 1000000

    env.load_cell_template(popName)
    
    (trees,_) = read_tree_selection (forest_path, popName, [gid], comm=comm)
    if synapses_path is not None:
        synapses_iter = read_cell_attribute_selection (synapses_path, popName, [gid], \
                                                       "Synapse Attributes", comm=comm)
    else:
        synapses_iter = None

    gid, tree = next(trees)
    if synapses_iter is not None:
        (_, synapses) = next(synapses_iter)
    else:
        synapses = None

    if 'mech_file' in env.celltypes[popName]:
        mech_file_path = env.config_prefix + '/' + env.celltypes[popName]['mech_file']
    else:
        mech_file_path = None

    template_class = getattr(h, "MossyCell")

    if (synapses is not None):
        synapse_test(template_class, mech_file_path, gid, tree, synapses, v_init, env)

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find("MossyCellTest.py") != -1,sys.argv)+1):])
