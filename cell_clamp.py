import os, os.path, itertools, random, sys, uuid, pprint
import numpy as np
import click
from mpi4py import MPI  # Must come before importing NEURON
from neuroh5.io import append_cell_attributes
from neuron import h
from dentate import cells, synapses, utils, neuron_utils, io_utils
from dentate.env import Env
from dentate.synapses import config_syn
from dentate.utils import get_module_logger
from dentate.neuron_utils import h, configure_hoc_env

# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)

def load_biophys_cell(env, pop_name, gid, mech_file_path=None, mech_dict=None, correct_for_spines=False,
                      tree_dict=None, load_synapses=True, synapses_dict=None,
                      load_connections=True, connection_graph=None,
                      load_weights=True, weight_dict=None):
    """
    Instantiates the mechanisms of a single BiophysCell instance.

    :param env: env.Env
    :param pop_name: str
    :param gid: int
    :param mech_file_path: str; path to cell mechanism config file
    :param correct_for_spines: bool

    Environment can be instantiated as:
    env = Env(config_file, template_paths, dataset_prefix, config_prefix)
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param config_prefix: str; path to directory containing network and cell mechanism config files
    
    """

    cell = cells.get_biophys_cell(env, pop_name, gid, tree_dict=tree_dict,
                                  load_synapses=load_synapses,
                                  synapses_dict=synapses_dict,
                                  load_weights=load_weights,
                                  weight_dict=weight_dict,
                                  load_edges=load_connections,
                                  connection_graph=connection_graph,
                                  mech_file_path=mech_file_path,
                                  mech_dict=mech_dict)

    # init_spike_detector(cell)
    if mech_file_path is not None:
        cells.init_biophysics(cell, reset_cable=True, 
                              correct_cm=correct_for_spines,
                              correct_g_pas=correct_for_spines, env=env)
    synapses.init_syn_mech_attrs(cell, env)
    
    return cell



def init_biophys_cell(env, pop_name, gid, load_connections=True, register_cell=True, write_cell=False,
                      cell_dict={}):
    """
    Instantiates a BiophysCell instance and all its synapses.

    :param env: an instance of env.Env
    :param pop_name: population name
    :param gid: gid
    """

    rank = int(env.pc.id())

    ## Determine if a mechanism configuration file exists for this cell type
    if 'mech_file_path' in env.celltypes[pop_name]:
        mech_dict = env.celltypes[pop_name]['mech_dict']
    else:
        mech_dict = None

    ## Determine if correct_for_spines flag has been specified for this cell type
    synapse_config = env.celltypes[pop_name]['synapses']
    if 'correct_for_spines' in synapse_config:
        correct_for_spines_flag = synapse_config['correct_for_spines']
    else:
        correct_for_spines_flag = False

    ## Determine presynaptic populations that connect to this cell type
    presyn_names = env.projection_dict[pop_name]

    ## Load cell gid and its synaptic attributes and connection data
    cell = load_biophys_cell(env, pop_name, gid, mech_dict=mech_dict, \
                             correct_for_spines=correct_for_spines_flag, \
                             load_connections=load_connections,
                             tree_dict=cell_dict.get('morph', None),
                             synapses_dict=cell_dict.get('synapse', None),
                             connection_graph=cell_dict.get('connectivity', None),
                             weight_graph=cell_dict.get('weight', None))
                             
    if register_cell:
        cells.register_cell(env, pop_name, gid, cell)
    
    cells.report_topology(cell, env)

    env.cell_selection[pop_name] = [gid]
    if write_cell:
        write_selection_file_path =  "%s/%s_%d.h5" % (env.results_path, env.modelName, gid)
        if rank == 0:
            io_utils.mkout(env, write_selection_file_path)
        env.comm.barrier()
        io_utils.write_cell_selection(env, write_selection_file_path)
        if load_connections:
            io_utils.write_connection_selection(env, write_selection_file_path)
    
    return cell

def measure_passive (gid, pop_name, v_init, env, prelength=1000.0, mainlength=2000.0, stimdur=500.0):

    biophys_cell = init_biophys_cell(env, pop_name, gid, register_cell=False)
    hoc_cell = biophys_cell.hoc_cell

    h.dt = env.dt

    tstop = prelength+mainlength
    
    soma = list(hoc_cell.soma)[0]
    stim1 = h.IClamp(soma(0.5))
    stim1.delay = prelength
    stim1.dur   = stimdur
    stim1.amp   = -0.1

    h('objref tlog, Vlog')
    
    h.tlog = h.Vector()
    h.tlog.record (h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record (soma(0.5)._ref_v)
    
    h.tstop = tstop

    Rin = h.rn(hoc_cell)
    
    neuron_utils.simulate(v_init, prelength, mainlength)

    ## compute membrane time constant
    vrest  = h.Vlog.x[int(h.tlog.indwhere(">=",prelength-1))]
    vmin   = h.Vlog.min()
    vmax   = vrest
    
    ## the time it takes the system's step response to reach 1-1/e (or
    ## 63.2%) of the peak value
    amp23  = 0.632 * abs (vmax - vmin)
    vtau0  = vrest - amp23
    tau0   = h.tlog.x[int(h.Vlog.indwhere ("<=", vtau0))] - prelength

    results = {'Rin': np.asarray([Rin], dtype=np.float32),
               'vmin': np.asarray([vmin], dtype=np.float32),
               'vmax': np.asarray([vmax], dtype=np.float32),
               'vtau0': np.asarray([vtau0], dtype=np.float32),
               'tau0': np.asarray([tau0], dtype=np.float32)
               }

    env.synapse_attributes.del_syn_id_attr_dict(gid)
    if gid in env.biophys_cells[pop_name]:
        del env.biophys_cells[pop_name][gid]
        
    return results


def measure_ap (gid, pop_name, v_init, env, cell_dict={}):


    biophys_cell = init_biophys_cell(env, pop_name, gid, register_cell=False, cell_dict=cell_dict)
    hoc_cell = biophys_cell.hoc_cell

    h.dt = env.dt

    prelength = 100.0
    stimdur = 10.0
    
    soma = list(hoc_cell.soma)[0]
    initial_amp = 0.05

    h.tlog = h.Vector()
    h.tlog.record (h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record (soma(0.5)._ref_v)

    thr = cells.find_spike_threshold_minimum(hoc_cell,loc=0.5,sec=soma,duration=stimdur,initial_amp=initial_amp)

    results = { 'spike threshold current': np.asarray([thr], dtype=np.float32),
                'spike threshold trace t': np.asarray(h.tlog.to_python(), dtype=np.float32),
                'spike threshold trace v': np.asarray(h.Vlog.to_python(), dtype=np.float32) }

    env.synapse_attributes.del_syn_id_attr_dict(gid)
    if gid in env.biophys_cells[pop_name]:
        del env.biophys_cells[pop_name][gid]

    return results
    
def measure_ap_rate (gid, pop_name, v_init, env, prelength=1000.0, mainlength=2000.0, stimdur=1000.0, minspikes=50, maxit=5, cell_dict={}):

    biophys_cell = init_biophys_cell(env, pop_name, gid, register_cell=False, cell_dict=cell_dict)
    hoc_cell = biophys_cell.hoc_cell

    h.dt = env.dt

    tstop = prelength+mainlength
    
    soma = list(hoc_cell.soma)[0]
    stim1 = h.IClamp(soma(0.5))
    stim1.delay = prelength
    stim1.dur   = stimdur
    stim1.amp   = 0.2

    h('objref nil, tlog, Vlog, spikelog')

    h.tlog = h.Vector()
    h.tlog.record (h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record (soma(0.5)._ref_v)

    h.spikelog = h.Vector()
    nc = biophys_cell.spike_detector
    nc.record(h.spikelog)
    
    h.tstop = tstop

    it = 1
    ## Increase the injected current until at least maxspikes spikes occur
    ## or up to maxit steps
    while (h.spikelog.size() < minspikes):

        neuron_utils.simulate(v_init, prelength,mainlength)
        
        if ((h.spikelog.size() < minspikes) & (it < maxit)):
            logger.info("ap_rate_test: stim1.amp = %g spikelog.size = %d\n" % (stim1.amp, h.spikelog.size()))
            stim1.amp = stim1.amp + 0.1
            h.spikelog.clear()
            h.tlog.clear()
            h.Vlog.clear()
            it += 1
        else:
            break

    logger.info("ap_rate_test: stim1.amp = %g spikelog.size = %d\n" % (stim1.amp, h.spikelog.size()))

    isivect = h.Vector(h.spikelog.size()-1, 0.0)
    tspike = h.spikelog.x[0]
    for i in range(1,int(h.spikelog.size())):
        isivect.x[i-1] = h.spikelog.x[i]-tspike
        tspike = h.spikelog.x[i]
    
    isimean  = isivect.mean()
    isivar   = isivect.var()
    isistdev = isivect.stdev()
    
    isilast = int(isivect.size())-1
    if (isivect.size() > 10):
        isi10th = 10 
    else:
        isi10th = isilast
    
    ## Compute the last spike that is largest than the first one.
    ## This is necessary because some models generate spike doublets,
    ## (i.e. spike with very short distance between them, which confuse the ISI statistics.
    isilastgt = int(isivect.size())-1
    while (isivect.x[isilastgt] < isivect.x[1]):
        isilastgt = isilastgt-1
    
    if (not (isilastgt > 0)):
        isivect.printf()
        raise RuntimeError("Unable to find ISI greater than first ISI")


    results = {'spike_count': np.asarray([h.spikelog.size()], dtype=np.uint32),
               'FR_mean': np.asarray([1.0 / isimean], dtype=np.float32),
               'ISI_mean': np.asarray([isimean], dtype=np.float32),
               'ISI_var': np.asarray([isivar], dtype=np.float32),
               'ISI_stdev': np.asarray([isistdev], dtype=np.float32),
               'ISI_adaptation_1': np.asarray([isivect.x[0] / isimean], dtype=np.float32),
               'ISI_adaptation_2': np.asarray([isivect.x[0] / isivect.x[isilast]], dtype=np.float32),
               'ISI_adaptation_3': np.asarray([isivect.x[0] / isivect.x[isi10th]], dtype=np.float32),
               'ISI_adaptation_4': np.asarray([isivect.x[0] / isivect.x[isilastgt]], dtype=np.float32)
               }

    env.synapse_attributes.del_syn_id_attr_dict(gid)
    if gid in env.biophys_cells[pop_name]:
        del env.biophys_cells[pop_name][gid]

    return results

    
def measure_fi (gid, pop_name, v_init, env, cell_dict={}):

    biophys_cell = init_biophys_cell(env, pop_name, gid, register_cell=False, cell_dict=cell_dict)
    hoc_cell = biophys_cell.hoc_cell

    soma = list(hoc_cell.soma)[0]
    h.dt = 0.025

    prelength = 1000.0
    mainlength = 2000.0

    tstop = prelength+mainlength
    
    stimdur = 1000.0

    
    stim1 = h.IClamp(soma(0.5))
    stim1.delay = prelength
    stim1.dur   = stimdur
    stim1.amp   = 0.2

    h('objref tlog, Vlog, spikelog')

    h.tlog = h.Vector()
    h.tlog.record (h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record (soma(0.5)._ref_v)

    h.spikelog = h.Vector()
    nc = biophys_cell.spike_detector
    nc.record(h.spikelog)
    
    h.tstop = tstop

    frs = []
    stim_amps = [stim1.amp]
    for it in range(1, 9):

        neuron_utils.simulate(v_init, prelength, mainlength)
        
        logger.info("fi_test: stim1.amp = %g spikelog.size = %d\n" % (stim1.amp, h.spikelog.size()))
        stim1.amp = stim1.amp + 0.1
        stim_amps.append(stim1.amp)
        frs.append(h.spikelog.size())
        h.spikelog.clear()
        h.tlog.clear()
        h.Vlog.clear()


    results = {'FI_curve_amplitude': np.asarray(stim_amps, dtype=np.float32),
               'FI_curve_frequency': np.asarray(frs, dtype=np.float32) }

    env.synapse_attributes.del_syn_id_attr_dict(gid)
    if gid in env.biophys_cells[pop_name]:
        del env.biophys_cells[pop_name][gid]

    return results


def measure_gap_junction_coupling (env, template_class, tree, v_init, cell_dict={}):
    
    h('objref gjlist, cells, Vlog1, Vlog2')

    pc = env.pc
    h.cells  = h.List()
    h.gjlist = h.List()
    
    cell1 = cells.make_neurotree_cell (template_class, neurotree_dict=tree)
    cell2 = cells.make_neurotree_cell (template_class, neurotree_dict=tree)

    h.cells.append(cell1)
    h.cells.append(cell2)

    ggid        = 20000000
    source      = 10422930
    destination = 10422670
    weight      = 5.4e-4
    srcsec   = int(cell1.somaidx.x[0])
    dstsec   = int(cell2.somaidx.x[0])

    stimdur     = 500
    tstop       = 2000
    
    pc.set_gid2node(source, int(pc.id()))
    nc = cell1.connect2target(h.nil)
    pc.cell(source, nc, 1)
    soma1 = list(cell1.soma)[0]

    pc.set_gid2node(destination, int(pc.id()))
    nc = cell2.connect2target(h.nil)
    pc.cell(destination, nc, 1)
    soma2 = list(cell2.soma)[0]

    stim1 = h.IClamp(soma1(0.5))
    stim1.delay = 250
    stim1.dur = stimdur
    stim1.amp = -0.1

    stim2 = h.IClamp(soma2(0.5))
    stim2.delay = 500+stimdur
    stim2.dur = stimdur
    stim2.amp = -0.1

    log_size = old_div(tstop,h.dt) + 1
    
    h.tlog = h.Vector(log_size,0)
    h.tlog.record (h._ref_t)

    h.Vlog1 = h.Vector(log_size)
    h.Vlog1.record (soma1(0.5)._ref_v)

    h.Vlog2 = h.Vector(log_size)
    h.Vlog2.record (soma2(0.5)._ref_v)


    gjpos = 0.5
    neuron_utils.mkgap(env, cell1, source, gjpos, srcsec, ggid, ggid+1, weight)
    neuron_utils.mkgap(env, cell2, destination, gjpos, dstsec, ggid+1, ggid, weight)

    pc.setup_transfer()
    pc.set_maxstep(10.0)

    h.stdinit()
    h.finitialize(v_init)
    pc.barrier()

    h.tstop = tstop
    pc.psolve(h.tstop)
    

    

@click.command()
@click.option("--config-file", '-c', required=True, type=str, help='model configuration file name')
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config',
              help='path to directory containing network and cell mechanism config files')
@click.option("--population", '-p', required=True, type=str, default='GC', help='target population')
@click.option("--gid", '-g', required=True, type=int, default=0, help='target cell gid')
@click.option("--template-paths", type=str, required=True,
              help='colon-separated list of paths to directories containing hoc cell templates')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='path to directory containing required neuroh5 data files')
@click.option("--results-path", required=False, type=click.Path(exists=True, file_okay=False, dir_okay=True), \
              help='path to directory where output files will be written')
@click.option("--results-file-id", type=str, required=False, default=None, \
              help='identifier that is used to name neuroh5 files that contain output spike and intracellular trace data')
@click.option("--results-namespace-id", type=str, required=False, default=None, \
              help='identifier that is used to name neuroh5 namespaces that contain output spike and intracellular trace data')
@click.option("--v-init", type=float, default=-75.0, help='initialization membrane potential (mV)')
def main(config_file, config_prefix, population, gid, template_paths, dataset_prefix, results_path, results_file_id, results_namespace_id, v_init):

    if results_file_id is None:
        results_file_id = uuid.uuid4()
    if results_namespace_id is None:
        results_namespace_id = 'Cell Clamp Results'
    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    verbose = True
    params = dict(locals())
    env = Env(**params)
    configure_hoc_env(env)
    io_utils.mkout(env, env.results_file_path)
    env.cell_selection = {}

    attr_dict = {}
    attr_dict[gid] = {}
    attr_dict[gid].update(measure_passive(gid, population, v_init, env))
    attr_dict[gid].update(measure_ap(gid, population, v_init, env))
    attr_dict[gid].update(measure_ap_rate(gid, population, v_init, env))
    attr_dict[gid].update(measure_fi(gid, population, v_init, env))

    pprint.pprint(attr_dict)

    if results_path is not None:
        append_cell_attributes(env.results_file_path, population, attr_dict,
                               namespace=env.results_namespace_id,
                               comm=env.comm, io_size=env.io_size)
        
    #gap_junction_test(gid, population, v_init, env)
    #synapse_test(gid, population, v_init, env)
    

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
