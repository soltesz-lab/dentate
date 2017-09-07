import itertools
from collections import defaultdict
import sys, os.path, string
from neuron import h
import numpy as np
import utils

def synapse_relcounts(syn_type_dict, layer_density_dicts, seglist, seed, neurotree_dict=None):
    """Computes per-segment relative counts of synapse placement"""
    relcounts_dict  = {}
    relcount_total  = 0
    layers_dict     = {}
    relcount_total  = 0
    for (syn_type_label, layer_density_dict) in layer_density_dicts.iteritems():
        syn_type = syn_type_dict[syn_type_label] 
        rans = {}
        for (layer,density_dict) in layer_density_dict.iteritems():
            ran = h.Random(seed)
            ran.normal(density_dict['mean'], density_dict['variance'])
            rans[layer] = ran
        relcounts = []
        layers    = []
        for seg in seglist:
            L    = seg.sec.L
            nseg = seg.sec.nseg
            if neurotree_dict is not None:
                layer = get_node_attribute('layer', neurotree_dict, seg.sec, seg.x)
            else:
                layer = -1
            layers.append(layer)
            
            ran=None
            if layer > -1:
                if rans.has_key(layer):
                    ran = rans[layer]
            else:
                ran = rans['default']
                
            if ran is not None:
                l         = L/nseg
                rc        = ran.repick()*l
                relcount_total += rc
                relcounts.append(rc)
            else:
                relcounts.append(0)
                
        relcounts_dict[syn_type] = relcounts
        layers_dict[syn_type]    = layers
    return (relcounts_dict, relcount_total, layers_dict)
    
           
def distribute_uniform_synapses(seed, syn_type_dict, swc_type_dict, sec_layer_density_dict, neurotree_dict, sec_dict):
    """Computes uniformly-spaced synapse locations"""

    syn_ids    = []
    syn_locs   = []
    syn_secs   = []
    syn_layers = []
    syn_types  = []
    swc_types  = []
    syn_index  = 0

    for (sec_name, layer_density_dict) in sec_layer_density_dict.iteritems():

        swc_type = swc_type_dict[sec_name]
        seg_list = []
        sec_index_dict = {}
        sec_index = 0
        L_total   = 0
        for sec in sec_dict[sec_name]:
            L_total += sec.L
            sec_index_dict[sec] = sec_index
            sec_index    += 1
            for seg in sec:
                if seg.x < 1.0 and seg.x > 0.0:
                    seg_list.append(seg)
            
    
        relcounts_dict, total, layers_dict = synapse_relcounts(syn_type_dict, layer_density_dict, seg_list, seed, neurotree_dict=neurotree_dict)

        sample_size = total
        cumcount  = 0
        for (syn_type_label, _) in layer_density_dict.iteritems():
            syn_type  = syn_type_dict[syn_type_label]
            relcounts = relcounts_dict[syn_type]
            layers    = layers_dict[syn_type]
            for i in xrange(0,len(seg_list)):
                seg = seg_list[i]
                seg_start = seg.x - (0.5/seg.sec.nseg)
                seg_end   = seg.x + (0.5/seg.sec.nseg)
                seg_range = seg_end - seg_start
                rel_count = relcounts[i]
                int_rel_count = round(rel_count)
                layer = layers[i]
                syn_count = 0
                while syn_count < int_rel_count:
                    syn_loc = seg_start + seg_range * ((syn_count + 1) / rel_count)
                    syn_locs.append(syn_loc)
                    syn_ids.append(syn_index)
                    syn_secs.append(sec_index_dict[seg.sec])
                    syn_layers.append(layer)
                    syn_types.append(syn_type)
                    swc_types.append(swc_type)
                    syn_index += 1
                    syn_count += 1
                cumcount += syn_count

    syn_dict = {'syn_ids': np.asarray(syn_ids, dtype='uint32'),
                'syn_locs': np.asarray(syn_locs, dtype='float32'),
                'syn_secs': np.asarray(syn_secs, dtype='uint32'),
                'syn_layers': np.asarray(syn_layers, dtype='int8'),
                'syn_types': np.asarray(syn_types, dtype='uint8'),
                'swc_types': np.asarray(swc_types, dtype='uint8')}

    return syn_dict


def syn_in_seg(seg,syns_dict):
    if seg.sec not in syns_dict:
        return False
    if any(seg.sec(x) == seg for x in syns_dict[seg.sec]): return True
    return False


def add_shared_synapse(seg, syns_dict):
    """Returns the existing synapse in segment if any, else creates it."""
    if not syn_in_seg(seg,syns_dict):
        syn = h.Exp2Syn(seg)
        syns_dict[seg.sec][syn.get_segment().x] = syn
        return syn
    else:
        for x, syn in syns_dict[seg.sec].iteritems():
            if seg.sec(x) == seg:
               return syn

def add_unique_synapse(seg, syns_dict):
    """Creates a synapse in the segment"""
    syn = h.Exp2Syn(seg)
    return syn
    
def mksyns(cell,syn_ids,syn_types,swc_types,syn_locs,syn_sections,synapse_kinetics,env,add_synapse=add_shared_synapse,spines=False):
    sort_idx       = np.argsort(syn_ids,axis=0)
    syns_dict_dend = defaultdict(lambda: {})
    syns_dict_axon = defaultdict(lambda: {})
    syns_dict_soma = defaultdict(lambda: {})
    py_apical = [sec for sec in cell.apical]
    py_basal  = [sec for sec in cell.basal]
    py_ais = [sec for sec in cell.ais]
    for (syn_id,syn_type,swc_type,syn_loc,syn_section) in itertools.izip(syn_ids[sort_idx],syn_types[sort_idx],swc_types[sort_idx],syn_locs[sort_idx],syn_sections[sort_idx]):
      sref = None
      if swc_type == env.SWC_Types['apical']:
        syns_dict = syns_dict_dend
        sec = py_apical[syn_section]
        if spines and h.ismembrane('spines',sec=sec):
          sec(syn_loc).count_spines += 1
      elif swc_type == env.SWC_Types['basal']:
        syns_dict = syns_dict_dend
        sec = py_basal[syn_section]
        if spines and h.ismembrane('spines',sec=sec):
          sec(syn_loc).count_spines += 1
      elif swc_type == env.SWC_Types['axon']:
        syns_dict = syns_dict_axon
        sec = py_ais[syn_section]
      elif swc_type == env.SWC_Types['soma']:
        syns_dict = syns_dict_soma
        sec = cell.soma[syn_section]
      else: 
        raise RuntimeError ("Unsupported synapse SWC type %d" % swc_type)
      syn      = add_synapse(sec(syn_loc), syns_dict)
      syn.tau1 = synapse_kinetics[syn_type]['t_rise']
      syn.tau2 = synapse_kinetics[syn_type]['t_decay']
      syn.e    = synapse_kinetics[syn_type]['e_rev']
      cell.syns.append(syn)
      cell.syntypes.o(syn_type).append(syn)

           
def print_syn_summary (gid,syn_dict):
    print 'gid %d: ' % gid
    print '\t total %d synapses' % len(syn_dict['syn_ids'])
    print '\t %d excitatory synapses' % np.size(np.where(syn_dict['syn_types'] == syn_Excitatory))
    print '\t %d inhibitory synapses' % np.size(np.where(syn_dict['syn_types'] == syn_Inhibitory))
    print '\t %d apical excitatory synapses' % np.size(np.where((syn_dict['syn_types'] == syn_Excitatory) & (syn_dict['swc_types'] == swc_apical)))
    print '\t %d apical inhibitory synapses' % np.size(np.where((syn_dict['syn_types'] == syn_Inhibitory) & (syn_dict['swc_types'] == swc_apical)))
    print '\t %d soma excitatory synapses' % np.size(np.where((syn_dict['syn_types'] == syn_Excitatory) & (syn_dict['swc_types'] == swc_soma)))
    print '\t %d soma inhibitory synapses' % np.size(np.where((syn_dict['syn_types'] == syn_Inhibitory) & (syn_dict['swc_types'] == swc_soma)))
    print '\t %d ais excitatory synapses' % np.size(np.where((syn_dict['syn_types'] == syn_Excitatory) & (syn_dict['swc_types'] == swc_axon)))
    print '\t %d ais inhibitory synapses' % np.size(np.where((syn_dict['syn_types'] == syn_Inhibitory) & (syn_dict['swc_types'] == swc_axon)))
