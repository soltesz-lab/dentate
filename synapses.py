import itertools
from collections import defaultdict
import sys, os.path, string, math
from neuron import h
import numpy as np
import cells


def synapse_seg_density(syn_type_dict, layer_dict, layer_density_dicts, sec_index_dict, seglist, seed,
                        neurotree_dict=None):
    """
    Computes per-segment density of synapse placement.
    :param syn_type_dict:
    :param layer_dict:
    :param layer_density_dicts:
    :param sec_index_dict:
    :param seglist:
    :param seed:
    :param neurotree_dict:
    :return:
    """
    segdensity_dict  = {}
    layers_dict     = {}
    if neurotree_dict is not None:
        secnodes_dict = neurotree_dict['section_topology']['nodes']
    else:
        secnodes_dict = None
    for (syn_type_label, layer_density_dict) in layer_density_dicts.iteritems():
        syn_type = syn_type_dict[syn_type_label]
        rans = {}
        for (layer_label,density_dict) in layer_density_dict.iteritems():
            if layer_label == 'default':
                layer = layer_label
            else:
                layer = layer_dict[layer_label]
            ran = h.Random(seed)
            ran.normal(density_dict['mean'], density_dict['variance'])
            rans[layer] = ran
        segdensity = []
        layers     = []
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
                elif rans.has_key('default'):
                    ran = rans['default']
                else:
                    ran = None
            elif rans.has_key('default'):
                ran = rans['default']
            else:
                ran = None
            if ran is not None:
                dens      = ran.repick()
                segdensity.append(dens)
            else:
                segdensity.append(0)
                
        segdensity_dict[syn_type] = segdensity
        layers_dict[syn_type]     = layers
    return (segdensity_dict, layers_dict)


def synapse_seg_counts(syn_type_dict, layer_dict, layer_density_dicts, sec_index_dict, seglist, seed,
                       neurotree_dict=None):
    """
    Computes per-segment relative counts of synapse placement.
    :param syn_type_dict:
    :param layer_dict:
    :param layer_density_dicts:
    :param sec_index_dict:
    :param seglist:
    :param seed:
    :param neurotree_dict:
    :return:
    """
    segcounts_dict  = {}
    layers_dict     = {}
    segcount_total  = 0
    if neurotree_dict is not None:
        secnodes_dict = neurotree_dict['section_topology']['nodes']
    else:
        secnodes_dict = None
    for (syn_type_label, layer_density_dict) in layer_density_dicts.iteritems():
        syn_type = syn_type_dict[syn_type_label]
        rans = {}
        for (layer_label,density_dict) in layer_density_dict.iteritems():
            if layer_label == 'default':
                layer = layer_label
            else:
                layer = layer_dict[layer_label]
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
                elif rans.has_key('default'):
                    ran = rans['default']
                else:
                    ran = None
            elif rans.has_key('default'):
                ran = rans['default']
            else:
                ran = None
            if ran is not None:
                l         = L/nseg
                dens      = ran.repick()
                rc        = dens*l
                segcount_total += rc
                segcounts.append(rc)
            else:
                segcounts.append(0)
                
        segcounts_dict[syn_type] = segcounts
        layers_dict[syn_type]    = layers
    return (segcounts_dict, segcount_total, layers_dict)
    
           
def distribute_uniform_synapses(seed, syn_type_dict, swc_type_dict, layer_dict, sec_layer_density_dict, neurotree_dict,
                                sec_dict, secidx_dict):
    """
    Computes uniformly-spaced synapse locations.
    :param seed:
    :param syn_type_dict:
    :param swc_type_dict:
    :param layer_dict:
    :param sec_layer_density_dict:
    :param neurotree_dict:
    :param sec_dict:
    :param secidx_dict:
    :return:
    """
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
            sec_obj_index_dict[sec] = int(secindex)
            if maxdist is None:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0:
                        seg_list.append(seg)
            else:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0 and ((L_total + sec.L * seg.x) <= maxdist):
                        seg_list.append(seg)
            L_total += sec.L
        segcounts_dict, total, layers_dict = \
            synapse_seg_counts(syn_type_dict, layer_dict, layer_density_dict, sec_obj_index_dict, seg_list, seed,
                               neurotree_dict=neurotree_dict)

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
                int_seg_count = math.floor(seg_count)
                layer = layers[i]
                syn_count = 0
                while syn_count < int_seg_count:
                    syn_loc = seg_start + seg_range * ((syn_count + 1) / math.ceil(seg_count))
                    assert((syn_loc <= 1) & (syn_loc >= 0))
                    if syn_loc < 1.0:
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


def distribute_poisson_synapses(seed, syn_type_dict, swc_type_dict, layer_dict, sec_layer_density_dict, neurotree_dict,
                                sec_dict, secidx_dict, verbose):
    """
    Computes synapse locations according to a Poisson distribution.
    :param seed:
    :param syn_type_dict:
    :param swc_type_dict:
    :param layer_dict:
    :param sec_layer_density_dict:
    :param neurotree_dict:
    :param sec_dict:
    :param secidx_dict:
    :param verbose:
    :return:
    """
    syn_ids    = []
    syn_locs   = []
    syn_secs   = []
    syn_layers = []
    syn_types  = []
    swc_types  = []
    syn_index  = 0

    r = np.random.RandomState()
    
    for (sec_name, layer_density_dict) in sec_layer_density_dict.iteritems():

        sec_index_dict = secidx_dict[sec_name]
        swc_type = swc_type_dict[sec_name]
        seg_list = []
        sec_obj_index_dict = {}
        L_total   = 0
        (seclst, maxdist) = sec_dict[sec_name]
        secidxlst         = secidx_dict[sec_name]
        for (sec, secindex) in itertools.izip(seclst, secidxlst):
            sec_obj_index_dict[sec] = int(secindex)
            if maxdist is None:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0:
                        seg_list.append(seg)
            else:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0 and ((L_total + sec.L * seg.x) <= maxdist):
                        seg_list.append(seg)
            L_total += sec.L
        segdensity_dict, layers_dict = \
            synapse_seg_density(syn_type_dict, layer_dict, layer_density_dict, sec_obj_index_dict, seg_list, seed,
                                neurotree_dict=neurotree_dict)

        cumcount  = 0
        for (syn_type_label, _) in layer_density_dict.iteritems():
            syn_type   = syn_type_dict[syn_type_label]
            segdensity = segdensity_dict[syn_type]
            layers     = layers_dict[syn_type]
            for i in xrange(len(seg_list)):
                seg = seg_list[i]
                seg_start = seg.x - (0.5/seg.sec.nseg)
                seg_end   = seg.x + (0.5/seg.sec.nseg)
                seg_range = seg_end - seg_start
                L = seg.sec.L
                layer = layers[i]
                density = segdensity[i]
                syn_count = 0
                if density > 0.:
                    beta = 1./density
                    interval = r.exponential(beta)
                    while interval < seg_range*L:
                        syn_loc = seg_start + interval/L
                        assert((syn_loc <= 1) & (syn_loc >= 0))
                        if syn_loc < 1.0:
                            syn_locs.append(syn_loc)
                            syn_ids.append(syn_index)
                            syn_secs.append(sec_obj_index_dict[seg.sec])
                            syn_layers.append(layer)
                            syn_types.append(syn_type)
                            swc_types.append(swc_type)
                            syn_index += 1
                            syn_count += 1
                        interval += r.exponential(beta)
                cumcount += syn_count

    assert(len(syn_ids) > 0)
    syn_dict = {'syn_ids': np.asarray(syn_ids, dtype='uint32'),
                'syn_locs': np.asarray(syn_locs, dtype='float32'),
                'syn_secs': np.asarray(syn_secs, dtype='uint32'),
                'syn_layers': np.asarray(syn_layers, dtype='int8'),
                'syn_types': np.asarray(syn_types, dtype='uint8'),
                'swc_types': np.asarray(swc_types, dtype='uint8')}

    return syn_dict


def syn_in_seg(syn_name, seg, syns_dict):
    """
    If a synaptic mechanism of the specified type already exists in the specified segment, it is returned.
    Otherwise, it returns None.
    :param syn_name: str
    :param seg: hoc segment
    :param syns_dict: nested defaultdict
    :return: hoc point process or None
    """
    for sec in syns_dict:
        if seg.sec == sec:
            for x in syns_dict[sec]:
                if x == seg.x:
                    if syn_name in syns_dict[sec][x]:
                        syn = syns_dict[sec][x][syn_name]
                        return syn
    return None


def make_syn_mech(mech_name, seg):
    """
    :param mech_name: str (name of the point_process, specified by env.synapse_mech_names)
    :param seg: hoc segment
    :return: hoc point process
    """
    if hasattr(h, mech_name):
        syn = getattr(h, mech_name)(seg)
    else:
        raise ValueError('make_syn_mech: unrecognized synaptic mechanism name %s' % mech_name)
    return syn


def add_shared_synapse(syn_name, seg, mech_names=None, syns_dict=None):
    """
    If a synaptic mechanism of the specified type already exists in the specified segment, it is returned.
    Otherwise, it creates one and adds is to the provided syns_dict.
    :param syn_name: str
    :param seg: hoc segment
    :param mech_names: dict to convert syn_name to hoc mechanism name
    :param syns_dict: nested defaultdict
    :return: hoc point process
    """
    syn = syn_in_seg(syn_name, seg, syns_dict)
    if syn is None:
        if mech_names is not None:
            mech_name = mech_names[syn_name]
        else:
            mech_name = syn_name
        syn = make_syn_mech(mech_name, seg)
        syns_dict[seg.sec][seg.x][mech_name] = syn
    return syn


def add_unique_synapse(syn_name, seg, mech_names=None, syns_dict=None):
    """
    Creates a new synapse in the given segment.
    :param syn_name: str
    :param seg: hoc segment
    :param mech_names: dict to convert syn_name to hoc mechanism name
    :param syns_dict: nested defaultdict
    :return: hoc point process
    """
    if mech_names is not None:
        mech_name = mech_names[syn_name]
    else:
        mech_name = syn_name
    syn = make_syn_mech(mech_name, seg)
    return syn


def config_syn(syn_name, rules, mech_names=None, syn=None, nc=None, **params):
    """

    :param syn_name: str
    :param rules: dict to correctly parse params for specified hoc mechanism
    :param mech_names: dict to convert syn_name to hoc mechanism name
    :param syn: synaptic mechanism object
    :param nc: :class:'h.NetCon'
    :param params: dict
    """
    if mech_names is not None:
        mech_name = mech_names[syn_name]
    else:
        mech_name = syn_name
    for param, val in params.iteritems():
        failed = True
        if param in rules[mech_name]['mech_params']:
            if syn is None:
                failed = False
            elif hasattr(syn, param):
                setattr(syn, param, val)
                failed = False
        elif param in rules[mech_name]['netcon_params']:
            if nc is None:
                failed = False
            else:
                i = rules[mech_name]['netcon_params'][param]
                if nc.wcnt() >= i:
                    nc.weight[i] = val
                    failed = False
        if failed:
            raise AttributeError('config_syn: problem setting attribute: %s for synaptic mechanism: %s' %
                                 (param, mech_name))


def mksyns(gid, cell, syn_ids, syn_types, swc_types, syn_locs, syn_sections, syn_kinetic_params, env,
           add_synapse=add_shared_synapse, spines=False):
    """
    20180510: Aaron modified add_shared_synapse to allow at most synaptic mechanism OF EACH TYPE per segment.
    :param gid:
    :param cell:
    :param syn_ids:
    :param syn_types:
    :param swc_types:
    :param syn_locs:
    :param syn_sections:
    :param syn_kinetic_params:
    :param env:
    :param add_synapse:
    :param spines:
    :return: nested dict of hoc point processes
    """
    syns_dict_dend = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_axon = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_ais  = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_hill = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_soma = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    py_sections = [sec for sec in cell.sections]

    syn_type_excitatory = env.Synapse_Types['excitatory']
    syn_type_inhibitory = env.Synapse_Types['inhibitory']

    swc_type_apical = env.SWC_Types['apical']
    swc_type_basal  = env.SWC_Types['basal']
    swc_type_soma   = env.SWC_Types['soma']
    swc_type_axon   = env.SWC_Types['axon']
    swc_type_ais   = env.SWC_Types['ais']
    swc_type_hill  = env.SWC_Types['hillock']
    
    syn_obj_dict = {}

    for i in xrange(syn_ids.size):
        syn_id = syn_ids[i]
        if not (syn_id < syn_types.size):
            print 'mksyns: syn_ids for gid %i: ' % gid, syn_ids
            raise ValueError('mksyns: cell %i received invalid syn_id %d' % (gid, syn_id))

        syn_type = syn_types[syn_id]
        swc_type = swc_types[syn_id]
        syn_loc = syn_locs[syn_id]
        syn_section = syn_sections[syn_id]

        sref = None
        sec = py_sections[syn_section]
        if swc_type == swc_type_apical:
            syns_dict = syns_dict_dend
            if syn_type == syn_type_excitatory:
                if spines and h.ismembrane('spines', sec=sec):
                    sec(syn_loc).count_spines += 1
        elif swc_type == swc_type_basal:
            syns_dict = syns_dict_dend
            if syn_type == syn_type_excitatory:
                if spines and h.ismembrane('spines', sec=sec):
                    sec(syn_loc).count_spines += 1
        elif swc_type == swc_type_axon:
            syns_dict = syns_dict_axon
        elif swc_type == swc_type_ais:
            syns_dict = syns_dict_ais
        elif swc_type == swc_type_hill:
            syns_dict = syns_dict_hill
        elif swc_type == swc_type_soma:
            syns_dict = syns_dict_soma
        else:
            raise RuntimeError("Unsupported synapse SWC type %d" % swc_type)
        syn_mech_dict = {}
        for (syn_name, params) in syn_kinetic_params.iteritems():
            syn = add_synapse(syn_name=syn_name, seg=sec(syn_loc), mech_names=env.synapse_mech_names,
                              syns_dict=syns_dict)
            config_syn(syn_name=syn_name, rules=env.synapse_param_rules, mech_names=env.synapse_mech_names, syn=syn,
                       **params)
            cell.syns.append(syn)
            cell.syntypes.o(syn_type).append(syn)
            syn_mech_dict[syn_name] = syn
        syn_obj_dict[syn_id] = syn_mech_dict

    if spines:
        cell.correct_for_spines()

    return syn_obj_dict


def print_syn_summary(gid, syn_dict):
    """

    :param gid:
    :param syn_dict:
    """
    print 'gid %d: ' % gid
    print '\t total %d synapses' % len(syn_dict['syn_ids'])
    print '\t %d excitatory synapses' % np.size(np.where(syn_dict['syn_types'] == syn_Excitatory))
    print '\t %d inhibitory synapses' % np.size(np.where(syn_dict['syn_types'] == syn_Inhibitory))
    print '\t %d apical excitatory synapses' % \
          np.size(np.where((syn_dict['syn_types'] == syn_Excitatory) & (syn_dict['swc_types'] == swc_apical)))
    print '\t %d apical inhibitory synapses' % \
          np.size(np.where((syn_dict['syn_types'] == syn_Inhibitory) & (syn_dict['swc_types'] == swc_apical)))
    print '\t %d soma excitatory synapses' % \
          np.size(np.where((syn_dict['syn_types'] == syn_Excitatory) & (syn_dict['swc_types'] == swc_soma)))
    print '\t %d soma inhibitory synapses' % \
          np.size(np.where((syn_dict['syn_types'] == syn_Inhibitory) & (syn_dict['swc_types'] == swc_soma)))
    print '\t %d ais excitatory synapses' % \
          np.size(np.where((syn_dict['syn_types'] == syn_Excitatory) & (syn_dict['swc_types'] == swc_axon)))
    print '\t %d ais inhibitory synapses' % \
          np.size(np.where((syn_dict['syn_types'] == syn_Inhibitory) & (syn_dict['swc_types'] == swc_axon)))
