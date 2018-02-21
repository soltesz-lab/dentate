import sys, os
import os.path
import click
import itertools
import numpy as np
from neuron import gui
from mpi4py import MPI # Must come before importing NEURON
from neuron import h
from neuroh5.io import read_tree_selection, read_cell_attribute_selection
from env import Env
import utils, cells, synapses



comm = MPI.COMM_WORLD
rank = comm.Get_rank()

template_path = "./templates"
h.xopen(template_path+'/AxoAxonicCell.hoc')

config_path = 'config/Full_Scale_Control.yaml'
forest_path = 'datasets/AAC_forest_syns_20170920.h5'
env = Env(comm=comm, configFile=config_path, templatePaths=template_path)
    
popName = "AAC"
(trees_dict,_) = read_tree_selection (comm, forest_path, popName, [1042800])
synapses_dict = read_cell_attribute_selection (comm, forest_path, popName, [1042800], "Synapse Attributes")
    
tree = trees_dict.itervalues().next()
synapses_vals = synapses_dict.itervalues().next()

synapse_kinetics=env.synapse_kinetics['AAC']

populations_dict = env.modelConfig['Definitions']['Populations']

syn_ids      = synapses_vals['syn_ids']
syn_types    = synapses_vals['syn_types']
swc_types    = synapses_vals['swc_types']
syn_locs     = synapses_vals['syn_locs']
syn_sections = synapses_vals['syn_secs']

cell = cells.make_neurotree_cell ("AxoAxonicCell", neurotree_dict=tree)

presyn_types = np.full((syn_ids.size,),0)
i = 0
for syn_type in syn_types:
    if syn_type == 0:
        presyn_types[i] = populations_dict['GC']
    elif syn_type == 1:
        presyn_types[i] = populations_dict['HC']
    i += 1
        

synapses.mksyns(cell,syn_ids,syn_types,swc_types,syn_locs,syn_sections,presyn_types,synapse_kinetics,env)

