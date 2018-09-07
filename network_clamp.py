"""
Routines for Network Clamp simulation.
"""

import click
from collections import defaultdict
from dentate.utils import *
from dentate.neuron_utils import *
from neuroh5.h5py_io_utils import *
from dentate.env import Env
from dentate.cells import *
from dentate.synapses import *
from dentate import spikedata, io_utils


def load_cell(env, pop_name, gid, mech_file=None, correct_for_spines=False):
    """
    Instantiates the mechanisms, synapses, and connections of a single cell.

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

    cell = get_biophys_cell(env, pop_name, gid)
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
    init_syn_mech_attrs(cell, env)
    config_syns_from_mech_attrs(gid, env, pop_name, insert=True)

    return cell



def add_rec(recid, population, cell, sec, dt=h.dt, loc=None, param='v', description=''):
        """
        Adds a recording vector for the specified quantity in the specified section and location.

        :param recid: integer
        :param population: str
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
                     'population': population,
                     'loc': loc,
                     'sec': sec,
                     'description': description,
                     'vec': vec }
                
        return rec_dict


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


    
def init(env, pop_name, gid, spike_events_path, spike_events_namespace='Spike Events', t_var='t', t_min=None, t_max=None, spike_generator_dict={}):
    """
    Instantiates a cell and all its synapses and connections and loads or generates spike times for all synaptic connections.

    :param env: an instance of env.Env
    :param pop_name: population name
    :param gid: gid
    :param spike_events_path:
    """
    io_utils.mkout(env, env.resultsFilePath)

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

    recs = []
    recs.append(add_rec(0, pop_name, cell, sec=cell.soma[0].sec, dt=h.dt, loc=0.5, param='v', description='Soma recording'))
     
    if env.verbose:
        report_topology(cell, env)
        
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
        if spk_pop_index is None:
            logger.warning("No spikes found for population %s in file %s" % (presyn_name, spike_events_path))
            continue
        spk_inds   = spkindlst[spk_pop_index]
        spk_ts     = spktlst[spk_pop_index]
        if presyn_name in spike_generator_dict:
            spike_generator = spike_generator_dict[presyn_name]
        else:
            spike_generator = None
        spk_source_dict[presyn_index] = { 'gid': spk_inds, 't': spk_ts, 'gen': spike_generator }


    min_delay = float('inf')
    syn_attrs = env.synapse_attributes
    this_syn_attrs = syn_attrs.syn_id_attr_dict[gid]
    this_syn_ids   = syn_attrs.syn_id_attr_dict[gid]['syn_ids']
    for syn_id, presyn_id, presyn_gid, delay in zip(this_syn_ids, \
                                                    this_syn_attrs['syn_sources'], \
                                                    this_syn_attrs['syn_source_gids'],
                                                    this_syn_attrs['delays']):
        if presyn_id in spk_source_dict:
            ## Load presynaptic spike times into the VecStim for each synapse;
            ## if spike_generator_dict contains an entry for the respective presynaptic population,
            ## then use the given generator function to generate spikes.
             if not (presyn_gid in env.gidset):
                spk_sources = spk_source_dict[presyn_id]
                gen = spk_sources['gen']
                if gen is None:
                    spk_inds = spk_sources['gid']
                    spk_ts = spk_sources['t']
                    data = spk_ts[np.where(spk_inds == presyn_gid)]
                else:
                    data = gen(presyn_gid, t_range)
                vs = h.VecStimCell(presyn_gid)
                vs.pp.play(h.Vector(data))
                register_cell(env, presyn_id, presyn_gid, vs)

        ncs = [ value['netcon'] for _,value in viewitems(syn_attrs.syn_mech_attr_dict[gid][syn_id]) ]
                
        for nc in ncs:
            nc.delay = delay
            env.pc.gid_connect(presyn_gid, nc.syn(), nc)
            min_delay = min(min_delay, delay)

    assert(min_delay > 0.0)
    env.pc.set_maxstep(10)
    h.stdinit()
    h.finitialize(env.v_init)

    return recs


def run(env, recs, output=True):
    """
    Runs network clamp simulation. Assumes that procedure `init` has been
    called with the network configuration provided by the `env`
    argument.

    :param env:
    :param recs:
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
    del env.cells
    env.pc.barrier()
    if rank == 0:
        logger.info("*** Writing spike data")
    if output:
        io_utils.spikeout(env, env.resultsFilePath, np.array(env.t_vec, dtype=np.float32), np.array(env.id_vec, dtype=np.uint32))
        io_utils.recsout(env, env.resultsFilePath, np.array(env.t_vec, dtype=np.float32), recs)

    comptime = env.pc.step_time()
    cwtime   = comptime + env.pc.step_wait()
    maxcw    = env.pc.allreduce(cwtime, 2)
    avgcomp  = env.pc.allreduce(comptime, 1)/nhosts
    maxcomp  = env.pc.allreduce(comptime, 2)

    if rank == 0:
        logger.info("Execution time summary for host %i:" % rank)
        logger.info("  created cells in %g seconds" % env.mkcellstime)
        logger.info("  connected cells in %g seconds" % env.connectcellstime)
        logger.info("  created gap junctions in %g seconds" % env.connectgjstime)
        logger.info("  ran simulation in %g seconds" % comptime)
        if maxcw > 0:
            logger.info("  load balance = %g" % (avgcomp/maxcw))

    env.pc.runworker()
    env.pc.done()



@click.command()
@click.option("--config-file", '-c', required=True, type=str)
@click.option("--population", '-p', required=True, type=str, default='GC')
@click.option("--gid", '-g', required=True, type=int, default=0)
@click.option("--tstop", '-t', type=float, default=150.0)
@click.option("--template-paths", type=str, required=True)
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), default='config')
@click.option("--spike-events-path", '-s', required=True, type=click.Path())
@click.option("--spike-events-namespace", type=str, default='Spike Events')
@click.option('--verbose', '-v', is_flag=True)
def main(config_file, population, gid, tstop, template_paths, dataset_prefix, config_prefix, spike_events_path, spike_events_namespace, verbose):
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

    env = Env(comm=comm, tstop=tstop, configFile=config_file, templatePaths=template_paths, \
                  datasetPrefix=dataset_prefix, configPrefix=config_prefix, \
                  verbose=verbose)
    configure_hoc_env(env)
    
    recs = init(env, population, gid, spike_events_path, spike_events_namespace=spike_events_namespace, \
                t_var='t', t_min=None, t_max=None, spike_generator_dict={})

    run(env, recs)
    
    

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
