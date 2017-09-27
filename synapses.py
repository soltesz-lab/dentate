import itertools
from collections import defaultdict
import sys, os.path, string, math
from neuron import h
import numpy as np
import utils, cells

def synapse_seg_counts(syn_type_dict, layer_density_dicts, sec_index_dict, seglist, seed, neurotree_dict=None):
    """Computes per-segment relative counts of synapse placement. """
    segcounts_dict  = {}
    segcount_total  = 0
    layers_dict     = {}
    segcount_total  = 0
    if neurotree_dict is not None:
        secnodes_dict = neurotree_dict['section_topology']['nodes']
    else:
        secnodes_dict = None
    for (syn_type_label, layer_density_dict) in layer_density_dicts.iteritems():
        syn_type = syn_type_dict[syn_type_label] 
        rans = {}
        for (layer,density_dict) in layer_density_dict.iteritems():
            ran = h.Random(seed)
            ran.normal(density_dict['mean'], density_dict['variance'])
            rans[layer] = ran
        segcounts = []
        layers    = []
        for seg in seglist:
            L    = seg.sec.L
            nseg = seg.sec.nseg
            if neurotree_dict is not None:
                secindex = sec_index_dict[seg.sec]
                secnodes = secnodes_dict[secindex]
                layer = cells.get_node_attribute('layer', neurotree_dict, seg.sec, secnodes, seg.x)
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
                segcount_total += rc
                segcounts.append(rc)
            else:
                segcounts.append(0)
                
        segcounts_dict[syn_type] = segcounts
        layers_dict[syn_type]    = layers
    return (segcounts_dict, segcount_total, layers_dict)
    
           
def distribute_uniform_synapses(seed, syn_type_dict, swc_type_dict, sec_layer_density_dict, neurotree_dict, sec_dict, secidx_dict):
    """Computes uniformly-spaced synapse locations. """

    syn_ids    = []
    syn_locs   = []
    syn_secs   = []
    syn_layers = []
    syn_types  = []
    swc_types  = []
    syn_index  = 0

    for (sec_name, layer_density_dict) in sec_layer_density_dict.iteritems():

        sec_index_dict = secidx_dict[sec_name]
        swc_type = swc_type_dict[sec_name]
        seg_list = []
        sec_obj_index_dict = {}
        L_total   = 0
        (seclst, maxdist) = sec_dict[sec_name]
        secidxlst         = secidx_dict[sec_name]
        for (sec, secindex) in itertools.izip(seclst, secidxlst):
            sec_obj_index_dict[sec] = secindex
            if maxdist is None:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0:
                        seg_list.append(seg)
            else:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0 and ((L_total + sec.L * seg.x) <= maxdist):
                        seg_list.append(seg)
            L_total += sec.L

        segcounts_dict, total, layers_dict = synapse_seg_counts(syn_type_dict, layer_density_dict, sec_obj_index_dict, seg_list, seed, neurotree_dict=neurotree_dict)

        sample_size = total
        cumcount  = 0
        for (syn_type_label, _) in layer_density_dict.iteritems():
            syn_type  = syn_type_dict[syn_type_label]
            segcounts = segcounts_dict[syn_type]
            layers    = layers_dict[syn_type]
            for i in xrange(0,len(seg_list)):
                seg = seg_list[i]
                seg_start = seg.x - (0.5/seg.sec.nseg)
                seg_end   = seg.x + (0.5/seg.sec.nseg)
                seg_range = seg_end - seg_start
                seg_count = segcounts[i]
                int_seg_count = round(seg_count)
                layer = layers[i]
                syn_count = 0
                while syn_count < int_seg_count:
                    syn_loc = seg_start + seg_range * ((syn_count + 1) / math.ceil(seg_count))
                    assert((syn_loc <= 1) & (syn_loc >= 0))
                    syn_locs.append(syn_loc)
                    syn_ids.append(syn_index)
                    syn_secs.append(sec_obj_index_dict[seg.sec])
                    syn_layers.append(layer)
                    syn_types.append(syn_type)
                    swc_types.append(swc_type)
                    syn_index += 1
                    syn_count += 1
                cumcount += syn_count

    assert(len(syn_ids) > 0)
    syn_dict = {'syn_ids': np.asarray(syn_ids, dtype='uint32'),
                'syn_locs': np.asarray(syn_locs, dtype='float32'),
                'syn_secs': np.asarray(syn_secs, dtype='uint32'),
                'syn_layers': np.asarray(syn_layers, dtype='int8'),
                'syn_types': np.asarray(syn_types, dtype='uint8'),
                'swc_types': np.asarray(swc_types, dtype='uint8')}

    return syn_dict


def syn_in_seg(seg, syns_dict):
    if seg.sec not in syns_dict:
        return False
    if any(seg.sec(x) == seg for x in syns_dict[seg.sec]): return True
    return False

def make_syn_mech(mech_name, seg):
    if mech_name == 'AMPA':
        syn = h.Exp2Syn(seg)
    elif mech_name == 'GABA_A':
        syn = h.Exp2Syn(seg)
    elif mech_name == 'GABA_B':
        syn = h.Exp2Syn(seg)
    else:
        syn = None
    return syn

def add_shared_synapse(mech_name, seg, syns_dict):
    """Returns the existing synapse in segment if any, otherwise creates it."""
    if not syn_in_seg(seg, syns_dict):
        syn = make_syn_mech(mech_name, seg)
        syns_dict[seg.sec][syn.get_segment().x] = syn
        return syn
    else:
        for x, syn in syns_dict[seg.sec].iteritems():
            if seg.sec(x) == seg:
               return syn

def add_unique_synapse(mech_name, seg, syns_dict):
    """Creates a synapse in the given segment."""
    syn = make_syn_mech(mech_name, seg)
    return syn
    
def mksyns(cell,syn_ids,syn_types,swc_types,syn_locs,syn_sections,presyn_ids,syn_kinetics,env,add_synapse=add_shared_synapse,spines=False):
    sort_idx       = np.argsort(syn_ids,axis=0)
    syns_dict_dend = defaultdict(lambda: defaultdict(lambda: {}))
    syns_dict_axon = defaultdict(lambda: defaultdict(lambda: {}))
    syns_dict_soma = defaultdict(lambda: defaultdict(lambda: {}))
    py_sections = [sec for sec in cell.sections]

    for (syn_id,syn_type,swc_type,syn_loc,syn_section,presyn_id) in itertools.izip(syn_ids[sort_idx],syn_types[sort_idx],swc_types[sort_idx],syn_locs[sort_idx],syn_sections[sort_idx],presyn_ids[sort_idx]):
      sref = None
      sec = py_sections[syn_section]
      if swc_type == env.SWC_Types['apical']:
        syns_dict = syns_dict_dend
        if spines and h.ismembrane('spines',sec=sec):
          sec(syn_loc).count_spines += 1
      elif swc_type == env.SWC_Types['basal']:
        syns_dict = syns_dict_dend
        if spines and h.ismembrane('spines',sec=sec):
          sec(syn_loc).count_spines += 1
      elif swc_type == env.SWC_Types['axon']:
        syns_dict = syns_dict_axon
      elif swc_type == env.SWC_Types['soma']:
        syns_dict = syns_dict_soma
      else: 
        raise RuntimeError ("Unsupported synapse SWC type %d" % swc_type)
      syn_kinetic_params = syn_kinetics[syn_type][presyn_id]
      for (syn_mech, params) in syn_kinetic_params.iteritems():
        syn = add_synapse(syn_mech, sec(syn_loc), syns_dict[presyn_id])
        syn.tau1 = params['t_rise']
        syn.tau2 = params['t_decay']
        syn.e    = params['e_rev']
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
