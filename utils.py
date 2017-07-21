import sys, os.path, string
from neuron import h
import numpy as np

def new_cell (template_name, local_id=0, gid=0, dataset_path="", neurotree_dict={}):
    h('objref cell, vx, vy, vz, vradius, vlayer, vsection, secnodes, vsrc, vdst')
    h.vx       = neurotree_dict['x']
    h.vy       = neurotree_dict['y']
    h.vz       = neurotree_dict['z']
    h.vradius  = neurotree_dict['radius']
    h.vlayer   = neurotree_dict['layer']
    h.vsection = neurotree_dict['section']
    h.secnodes = neurotree_dict['section_topology']['nodes']
    h.vsrc     = neurotree_dict['section_topology']['src']
    h.vdst     = neurotree_dict['section_topology']['dst']
    hstmt      = 'cell = new %s(%d, %d, "%s", vlayer, vsrc, vdst, secnodes, vx, vy, vz, vradius)' % (template_name, local_id, gid, dataset_path)
    h(hstmt)
    return h.cell


def hoc_results_to_python(hoc_results):
    results_dict = {}
    for i in xrange(0, int(hoc_results.count())):
        vect   = hoc_results.o(i)
        gid    = int(vect.x[0])
        pyvect = vect.to_python()
        results_dict[gid] = pyvect[1:]
    hoc_results.remove_all()
    return results_dict

    
def get_node_attribute (name, content, sec, x=None):
    if name in content:
        if x is None:
            return content[name]
        elif sec.n3d() == 0:
            return content[name][0]
        else:
            for i in xrange(sec.n3d()):
                if sec.arc3d(i)/sec.L >= x:
                    return content[name][i]
    else:
        return None



def synapse_relcounts(layer_density_dicts, seglist, seed, neurotree_dict=None):
    """Computes per-segment relative counts of synapse placement"""
    relcounts_dict  = {}
    relcount_total  = 0
    layers_dict     = {}
    relcount_total  = 0
    for (syn_type, layer_density_dict) in layer_density_dicts.iteritems():
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
                if layer in rans:
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
        layers_dict[syn_type]   = layers
    return (relcounts_dict, relcount_total, layers_dict)
    
           
def distribute_uniform_synapses(seed, sec_layer_density_dicts, sec_lists, sec_swc_types, neurotree_dicts):
    """Computes uniformly-spaced synapse locations"""

    syn_ids    = []
    syn_locs   = []
    syn_secs   = []
    syn_layers = []
    syn_types  = []
    swc_types  = []
    syn_index  = 0

    for (layer_density_dicts, sec_list, swc_type, neurotree_dict) in itertools.izip(sec_layer_density_dicts,
                                                                                    sec_lists,
                                                                                    sec_swc_types,
                                                                                    neurotree_dicts):
        
        seg_list = []
        sec_dict = {}
        sec_index = 0
        L_total   = 0
        for sec in sec_list:
            L_total += sec.L
            sec_dict[sec] = sec_index
            sec_index    += 1
            for seg in sec:
                if seg.x < 1.0 and seg.x > 0.0:
                    seg_list.append(seg)
            
    
        relcounts_dict, total, layers_dict = synapse_relcounts(layer_density_dicts, seg_list, seed, neurotree_dict=neurotree_dict)

        sample_size = total
        cumcount  = 0
        for (syn_type, _) in layer_density_dicts.iteritems():
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
                    syn_secs.append(sec_dict[seg.sec])
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
