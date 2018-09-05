
from collections import defaultdict
from dentate.utils import *
from dentate.neuron_utils import *
from neuroh5.h5py_io_utils import *
from dentate.env import Env
from dentate.cells import *
from dentate.synapses import *
from dentate import spikedata


def load_cell(env, pop_name, gid, mech_file=None, correct_for_spines=False):
    """

    :param gid: int
    :param pop_name: str
    :param config_file: str; model configuration file name
    :param mech_file: str; cell mechanism config file name
    :param correct_for_spines: bool

    Environment can be instantiated as:
    env = Env(config_file, template_paths, dataset_prefix, config_prefix)
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param config_prefix: str; path to directory containing network and cell mechanism config files
    
    """
    configure_hoc_env(env)

    cell = get_biophys_cell(env, pop_name, gid)
    if env.configPrefix is not None:
        mech_file_path = env.configPrefix + '/' + mech_file
    else:
        mech_file_path = mech_file

    init_biophysics(cell, reset_cable=True, from_file=True, mech_file_path=mech_file_path,
                    correct_cm=correct_for_spines, correct_g_pas=correct_for_spines, env=env)
    init_syn_mech_attrs(cell, env)
    config_syns_from_mech_attrs(gid, env, pop_name, insert=True)
    return cell





def add_rec(self, recid, cell, sec=None, dt=h.dt, loc=None, param='v', description=''):
        """

        :param recid: integer
        :param cell: :class:'BiophysCell'
        :param sec: :class:'HocObject'
        :param dt: float
        :param loc: float
        :param param: str
        :param ylabel: str
        :param description: str
        """
        vec = h.Vector()
        name = 'rec%i' % recid
        if loc is None:
           loc = 0.5
        vec.record(getattr(sec(loc), '_ref_%s' % param), dt)

        rec_dict = { 'name': name,
                     'cell': cell,
                     'loc': loc,
                     'sec': sec,
                     'description': description,
                     'vec': vec }
                
        return rec_dict

    


def register_cell(env, pop_name, gid, cell):
    """
    Registers a cell in a network environment.
    :param env: an instance of env.Env
    :param pop_name: population name
    :param gid: gid
    :param cell: cell instance
    """
    rank = env.comm.rank
    env.gidlist.append(gid)
    env.cells.append(cell)
    env.pc.set_gid2node(gid, rank)
    # Tell the ParallelContext that this cell is a spike source
    # for all other hosts. NetCon is temporary.
    nc = cell.connect2target(h.nil)
    env.pc.cell(gid, nc, 1)
    # Record spikes of this cell
    env.pc.spike_record(gid, env.t_vec, env.id_vec)


    
def init(env, pop_name, gid, spike_events_path, spike_events_namespace='Spike Events', t_var='t', t_min=None, t_max=None, spike_generator_dict={}):
    """
    Instantiates a cell and all its synapses and connections and
    loads or generates spike times for all synaptic connections.

    :param env: an instance of env.Env
    :param pop_name: population name
    :param gid: gid
    :param spike_events_path:
    """

    ## If specified, presynaptic spikes that only fall within this time range
    ## will be loaded or generated
    if t_max is None:
        t_range = None
    else:
        if t_min is None:
            t_range = [0.0, t_max]
        else:
            r_range = [t_min, t_max]

    ## Attribute namespace that contains recorded spike events
    if not str(env.resultsId):
        namespace_id = spike_events_namespace
    else:
        namespace_id = "%s %s" % (spike_events_namespace, str(env.resultsId))

    ## Determine if a mechanism configuration file exists for this cell type
    if 'mech_file' in env.celltypes[pop_name]:
        mech_file = env.celltypes[pop_name]['mech_file']
    else:
        mech_file = None

    ## Determine if correct_for_spines flag has been specified for this cell type
    synapse_config = env.celltypes[pop_name]['synapses']
    if 'correct_for_spines' in synapse_config:
        correct_for_spines_flag = synapse_config['correct_for_spines']
    else:
        correct_for_spines_flag = False

    ## Determine presynaptic populations that connect to this cell type
    presyn_names = env.projection_dict[pop_name]

    ## Load cell gid and its synaptic attributes and connection data
    cell = load_cell(env, pop_name, gid, mech_file=mech_file, correct_for_spines=correct_for_spines_flag)
    register_cell(env, pop_name, gid, cell)

    ## Load spike times of presynaptic cells
    spkdata = spikedata.read_spike_events (spike_events_path, \
                                           presyn_names, \
                                           spike_events_namespace, \
                                           timeVariable=t_var, \
                                           timeRange=t_range)
    spkindlst = spkdata['spkindlst']
    spktlst   = spkdata['spktlst']
    spkpoplst = spkdata['spkpoplst']
    
    ## Organize spike times by index of presynaptic population and gid
    spk_source_dict = {}
    for presyn_name in presyn_names:
        presyn_index = int(env.pop_dict[presyn_name])
        spk_pop_index = list_index(presyn_name, spkpoplst)
        spk_inds   = spkindlst[spk_pop_index]
        spk_ts     = spktlst[spk_pop_index]
        if presyn_name in spike_generator_dict:
            spike_generator = spike_generator_dict[presyn_name]
        else:
            spike_generator = None
        spk_source_dict[presyn_index] = { 'gid': spk_inds, 't': spk_ts, 'gen': spike_generator }

    ## Load presynaptic spike times into the VecStim for each synapse;
    ## if spike_generator_dict contains an entry for the respective presynaptic population,
    ## then use the given generator function to generate spikes.
    syn_attrs = env.synapse_attributes
    this_syn_attrs = syn_attrs.syn_id_attr_dict[gid]
    this_syn_ids   = syn_attrs.syn_id_attr_dict[gid]['syn_ids']
    for syn_id, presyn_id, presyn_gid in zip(this_syn_ids, \
                                             this_syn_attrs['syn_sources'], \
                                             this_syn_attrs['syn_source_gids']):
    
        vss = [ value['vecstim'] for _,value in viewitems(syn_attrs.syn_mech_attr_dict[gid][syn_id]) ]
        spk_sources = spk_source_dict[presyn_id]
        gen = spk_sources['gen']
        if gen is None:
            spk_inds = spk_sources['gid']
            spk_ts = spk_sources['t']
            data = spk_ts[np.where(spk_inds == presyn_gid)]
        else:
            data = gen(presyn_gid, t_range)
        for vs in vss:
            vs.play(data)

        
