"""
Routines for Network Clamp simulation.
"""

import click
from collections import defaultdict
import dentate
from dentate.utils import *
from dentate.neuron_utils import *
from dentate.env import Env
from dentate.cells import *
from dentate import spikedata, io_utils, synapses


def generate_weights(env, weight_source_rules, this_syn_attrs):

    weights_dict = {}

    if len(weight_source_rules) > 0:

        for presyn_id, weight_rule in viewitems(weight_source_rules):
            if weight_rule['class'] == 'Log-Normal':
                
                source_syn_dict = defaultdict(list)
    
                for syn_id, syn in viewitems(this_syn_attrs):
                    this_presyn_id = syn.source.population
                    this_presyn_gid = syn.source.gid
                    if this_presyn_id == presyn_id:
                        source_syn_dict[this_presyn_gid].append(syn_id)

                weights_name = weight_rule['name']
                rule_params = weight_rule['params']
                mu = rule_params['mu']
                sigma = rule_params['sigma']
                seed_offset = int(env.modelConfig['Random Seeds']['GC Log-Normal Weights 1'])
                seed = int(seed_offset + 1)
                weights_dict[presyn_id] = \
                  synapses.generate_log_normal_weights(weights_name, mu, sigma, seed, source_syn_dict)
            else:
                raise RuntimeError('network_clamp.generate_weights: unknown weight generator rule class %s' % \
                                   weight_rule['class'])

    return weights_dict
        

def make_input_cell(env, gid, gen):
    template_name = gen['template']
    param_values  = gen['params']
    template = getattr(h, template_name)
    params = [ param_values[p] for p in env.netclampConfig.template_params[template_name] ]
    cell = template(gid, *params)
    return cell

def load_cell(env, pop_name, gid, mech_file=None, correct_for_spines=False, load_edges=True, tree_dict=None, synapses_dict=None):
    """
    Instantiates the mechanisms of a single cell.

    :param env: env.Env
    :param pop_name: str
    :param gid: int
    :param mech_file: str; cell mechanism config file name
    :param correct_for_spines: bool

    Environment can be instantiated as:
    env = Env(config_file, template_paths, dataset_prefix, config_prefix)
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param config_prefix: str; path to directory containing network and cell mechanism config files
    
    """
    configure_hoc_env(env)
    
    cell = get_biophys_cell(env, pop_name, gid, load_edges=load_edges, \
                            tree_dict=tree_dict, synapses_dict=synapses_dict)
    if mech_file is not None:
        if env.configPrefix is not None:
            mech_file_path = env.configPrefix + '/' + mech_file
        else:
            mech_file_path = mech_file
    else:
        mech_file_path = None

    if mech_file_path is not None:
        init_biophysics(cell, reset_cable=True, from_file=True, mech_file_path=mech_file_path,
                        correct_cm=correct_for_spines, correct_g_pas=correct_for_spines, env=env)
    synapses.init_syn_mech_attrs(cell, env)
    
    return cell

def register_cell(env, population, gid, cell):
    """
    Registers a cell in a ParallelContext network environment.

    :param env: an instance of env.Env
    :param population: population name
    :param gid: gid
    :param cell: cell instance
    """
    rank = env.comm.rank
    env.gidset.add(gid)
    env.cells.append(cell)
    env.pc.set_gid2node(gid, rank)
    # Tell the ParallelContext that this cell is a spike source
    # for all other hosts. NetCon is temporary.
    hoc_cell = getattr(cell, "hoc_cell", None)
    if hoc_cell is None:
        nc = cell.connect2target(h.nil)
    else:
        nc = cell.hoc_cell.connect2target(h.nil)
    env.pc.cell(gid, nc, 1)
    # Record spikes of this cell
    env.pc.spike_record(gid, env.t_vec, env.id_vec)

    
def init_cell(env, pop_name, gid, load_edges=True):
    """
    Instantiates a cell and all its synapses

    :param env: an instance of env.Env
    :param pop_name: population name
    :param gid: gid
    """
    
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
    cell = load_cell(env, pop_name, gid, mech_file=mech_file, \
                     correct_for_spines=correct_for_spines_flag, \
                     load_edges=load_edges)
    register_cell(env, pop_name, gid, cell)

    
    env.recs_dict[pop_name][0] = make_rec(0, pop_name, gid, cell, \
                                          sec=cell.soma[0].sec, loc=0.5, param='v', \
                                          dt=h.dt, description='Soma recording')
     
    if env.verbose:
        report_topology(cell, env)

    return cell

def init(env, pop_name, gid, spike_events_path, generate_inputs_pops=set([]), generate_weights_pops=set([]), spike_events_namespace='Spike Events', t_var='t', t_min=None, t_max=None):
    """Instantiates a cell and all its synapses and connections and loads
    or generates spike times for all synaptic connections.

    :param env: an instance of env.Env
    :param pop_name: population name
    :param gid: gid
    :param spike_events_path:

    """
    io_utils.mkout(env, env.results_file_path)

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
    if env.results_id is None:
        namespace_id = spike_events_namespace
    else:
        namespace_id = "%s %s" % (spike_events_namespace, str(env.results_id))

    ## Determine presynaptic populations that connect to this cell type
    presyn_names = env.projection_dict[pop_name]

    ## Load cell gid and its synaptic attributes and connection data
    cell = init_cell(env, pop_name, gid)

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
    input_source_dict = {}
    weight_source_dict = {}
    for presyn_name in presyn_names:
        presyn_index = int(env.pop_dict[presyn_name])
        spk_pop_index = list_index(presyn_name, spkpoplst)
        if spk_pop_index is None:
            logger.warning("No spikes found for population %s in file %s" % (presyn_name, spike_events_path))
            continue
        spk_inds   = spkindlst[spk_pop_index]
        spk_ts     = spktlst[spk_pop_index]
        
        if presyn_name in generate_inputs_pops:
            if (presyn_name in env.netclampConfig.input_generators):
                spike_generator = env.netclampConfig.input_generators[presyn_name]
            else:
                raise RuntimeError('network_clamp.init: no input generator specified for population %s' % presyn_name)
        else:
            spike_generator = None
            
        input_source_dict[presyn_index] = { 'gid': spk_inds, 't': spk_ts, 'gen': spike_generator }

        if presyn_name in generate_weights_pops:
            if (presyn_name in env.netclampConfig.weight_generators[pop_name]):
                weight_rule = env.netclampConfig.weight_generators[pop_name][presyn_name]
            else:
                raise RuntimeError('network_clamp.init: no weights generator rule specified for population %s' % presyn_name)
        else:
            weight_rule = None

        if weight_rule is not None:
            weight_source_dict[presyn_index] = weight_rule
        

    min_delay = float('inf')
    syn_attrs = env.synapse_attributes
    this_syn_attrs = syn_attrs[gid]
    source_syn_dict = defaultdict(lambda: defaultdict(list))
    weight_params = defaultdict(dict)
    for syn_id, syn in viewitems(this_syn_attrs):
        presyn_id = syn.source.population
        presyn_gid = syn.source.gid
        delay = syn.source.delay
        if presyn_id in input_source_dict:
            ## Load presynaptic spike times into the VecStim for each synapse;
            ## if spike_generator_dict contains an entry for the respective presynaptic population,
            ## then use the given generator to generate spikes.
            if not (presyn_gid in env.gidset):
                input_sources = input_source_dict[presyn_id]
                input_gen = input_sources['gen']
                if input_gen is None:
                    spk_inds = input_sources['gid']
                    spk_ts = input_sources['t']
                    data = spk_ts[np.where(spk_inds == presyn_gid)]
                    cell = h.VecStimCell(presyn_gid)
                    cell.pp.play(h.Vector(data))
                else:
                    cell = make_input_cell(env, presyn_gid, input_gen)
                register_cell(env, presyn_id, presyn_gid, cell)

    source_weight_params = generate_weights(env, weight_source_dict, this_syn_attrs)
    
    for presyn_id, weight_params in viewitems(source_weight_params):
        weights_syn_ids = weight_params['syn_id']
        for syn_name in (syn_name for syn_name in weight_params if syn_name != 'syn_id'):
            weights_values  = weight_params[syn_name]
            syn_attrs.add_netcon_weights_from_iter(gid, syn_name, \
                                                   zip_longest(weights_syn_ids, \
                                                               weights_values))
        
    synapses.config_biophys_cell_syns(env, gid, pop_name, insert=True, insert_netcons=True)

    env.pc.set_maxstep(10)
    h.stdinit()
    h.finitialize(env.v_init)


def run(env, output=True):
    """
    Runs network clamp simulation. Assumes that procedure `init` has been
    called with the network configuration provided by the `env`
    argument.

    :param env:
    :param output: bool

    """
    
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())
    
    h.tstop = env.tstop
    if rank == 0:
        logger.info("*** Running simulation with dt = %f and tstop = %f" % (h.dt, h.tstop))

    env.pc.barrier()
    env.pc.psolve(h.tstop)

    if rank == 0:
        logger.info("*** Simulation completed")
    env.pc.barrier()

    if output:
        if rank == 0:
            logger.info("*** Writing spike data")
        io_utils.spikeout(env, env.results_file_path)
        if rank == 0:
            logger.info("*** Writing intracellular data")
        io_utils.recsout(env, env.results_file_path)

    comptime = env.pc.step_time()
    cwtime   = comptime + env.pc.step_wait()
    maxcw    = env.pc.allreduce(cwtime, 2)
    avgcomp  = env.pc.allreduce(comptime, 1)/nhosts
    maxcomp  = env.pc.allreduce(comptime, 2)

    if rank == 0:
        logger.info("Host %i  ran simulation in %g seconds" % (rank, comptime))
        if maxcw > 0:
            logger.info("  load balance = %g" % (avgcomp/maxcw))

    env.pc.runworker()
    env.pc.done()


@click.command()
@click.option("--config-file", '-c', required=True, type=str)
@click.option("--population", '-p', required=True, type=str, default='GC')
@click.option("--gid", '-g', required=True, type=int, default=0)
@click.option("--template-paths", type=str, required=True)
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), default='config')
@click.option('--verbose', '-v', is_flag=True)
def show(config_file, population, gid, tstop, template_paths, dataset_prefix, config_prefix, spike_events_path, spike_events_namespace, verbose):
    """
    Show configuration for the specified cell gid.

    :param config_file: str; model configuration file name
    :param population: str
    :param gid: int
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param config_prefix: str; path to directory containing network and cell mechanism config files
    :param verbose: bool
    """

    comm = MPI.COMM_WORLD
    np.seterr(all='raise')

    env = Env(comm=comm, tstop=tstop, config_file=config_file, template_paths=template_paths, \
              dataset_prefix=dataset_prefix, config_prefix=config_prefix, \
              verbose=True)
    configure_hoc_env(env)
    
    init_cell(env, population, gid, load_edges=False)


@click.command()
@click.option("--config-file", '-c', required=True, type=str)
@click.option("--population", '-p', required=True, type=str, default='GC')
@click.option("--gid", '-g', required=True, type=int, default=0)
@click.option("--generate-inputs", '-e', required=False, type=str, multiple=True)
@click.option("--generate-weights", '-w', required=False, type=str, multiple=True)
@click.option("--tstop", '-t', type=float, default=150.0)
@click.option("--template-paths", type=str, required=True)
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), default='config')
@click.option("--spike-events-path", '-s', required=True, type=click.Path())
@click.option("--spike-events-namespace", type=str, default='Spike Events')
@click.option("--results-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), \
                  help='path to directory where output files will be written')
@click.option("--results-id", type=str, required=False, default=None, \
                  help='identifier that is used to name neuroh5 namespaces that contain output spike and intracellular trace data')
@click.option('--verbose', '-v', is_flag=True)
@click.option('--profile', is_flag=True)
def main(config_file, population, gid, generate_inputs, generate_weights, tstop, template_paths, dataset_prefix, config_prefix, spike_events_path, spike_events_namespace, results_path, results_id, verbose, profile):
    """
    Runs network clamp simulation for the specified cell gid.

    :param config_file: str; model configuration file name
    :param population: str
    :param gid: int
    :param tstop: float
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param config_prefix: str; path to directory containing network and cell mechanism config files
    :param spike_events_path: str; path to file containing spike times
    :param spike_events_namespace: str; namespace containing spike times
    :param verbose: bool
    """

    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    params = dict(locals())
    env = Env(**params)
    configure_hoc_env(env)

    init(env, population, gid, spike_events_path, \
         generate_inputs_pops=set(generate_inputs), \
         generate_weights_pops=set(generate_weights), \
         spike_events_namespace=spike_events_namespace, \
         t_var='t', t_min=None, t_max=None)

    run(env)

    if profile:
        from guppy import hpy
        h = hpy()
        
        logger.info(h.heap())
    
    

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
