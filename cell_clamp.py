import os, os.path, itertools, random, sys
import numpy as np
import click
from mpi4py import MPI  # Must come before importing NEURON
from neuroh5.io import read_cell_attribute_selection, read_tree_selection
from neuron import h
from dentate import cells, synapses, utils, neuron_utils
from dentate.env import Env
from dentate.synapses import config_syn


def measure_passive (template_class, tree, v_init, dt=0.025, prelength=1000.0, mainlength=2000.0, stimdur=500.0):

    cell = cells.make_neurotree_cell (template_class, neurotree_dict=tree)
    h.dt = dt

    tstop = prelength+mainlength
    
    soma = list(cell.soma)[0]
    stim1 = h.IClamp(soma(0.5))
    stim1.delay = prelength
    stim1.dur   = stimdur
    stim1.amp   = -0.1

    h.tlog = h.Vector()
    h.tlog.record (h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record (soma(0.5)._ref_v)
    
    h.tstop = tstop

    Rin = h.rn(cell)
    
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

    results = {'Rin': Rin,
               'vmin': vmin,
               'vmax': vmax,
               'vtau0': vtau0,
               'tau0': tau0
               }

    return results


def ap_test (template_class, tree, v_init):

    cell = cells.make_neurotree_cell (template_class, neurotree_dict=tree)
    h.dt = 0.025

    prelength = 100.0
    stimdur = 10.0
    
    soma = list(cell.soma)[0]
    initial_amp = 0.05

    h.tlog = h.Vector()
    h.tlog.record (h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record (soma(0.5)._ref_v)

    thr = cells.find_spike_threshold_minimum(cell,loc=0.5,sec=soma,duration=stimdur,initial_amp=initial_amp)
    
    f=open("BasketCell_ap_results.dat",'w')
    f.write ("## current amplitude: %g\n" % thr)
    f.close()

    f=open("BasketCell_voltage_trace.dat",'w')
    for i in range(0, int(h.tlog.size())):
        f.write('%g %g\n' % (h.tlog.x[i], h.Vlog.x[i]))
    f.close()

    
def measure_ap (template_class, tree, v_init, dt=0.025, prelength=1000.0, mainlength=2000.0, stimdur=1000.0, minspikes=50, maxit=5):

    cell = cells.make_neurotree_cell (template_class, neurotree_dict=tree)
    h.dt = dt

    tstop = prelength+mainlength
    
    soma = list(cell.soma)[0]
    stim1 = h.IClamp(soma(0.5))
    stim1.delay = prelength
    stim1.dur   = stimdur
    stim1.amp   = 0.2

    h.tlog = h.Vector()
    h.tlog.record (h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record (soma(0.5)._ref_v)

    h.spikelog = h.Vector()
    nc = h.NetCon(soma(0.5)._ref_v, h.nil)
    nc.threshold = -40.0
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


    results = {'spike_count': h.spikelog.size(),
               'FR_mean': (1.0 / isimean),
               'ISI_mean': isimean,
               'ISI_var': isivar,
               'ISI_stdev': isistdev,
               'ISI_adaptation_1': isivect.x[0] / isimean,
               'ISI_adaptation_2': isivect.x[0] / isivect.x[isilast]
               'ISI_adaptation_3': isivect.x[0] / isivect.x[isi10th]
               'ISI_adaptation_4': isivect.x[0] / isivect.x[isilastgt]
               }

    return results

    
def measure_fi (template_class, tree, v_init):

    cell = cells.make_neurotree_cell (template_class, neurotree_dict=tree)
    soma = list(cell.soma)[0]
    h.dt = 0.025

    prelength = 1000.0
    mainlength = 2000.0

    tstop = prelength+mainlength
    
    stimdur = 1000.0

    
    stim1 = h.IClamp(soma(0.5))
    stim1.delay = prelength
    stim1.dur   = stimdur
    stim1.amp   = 0.2

    h.tlog = h.Vector()
    h.tlog.record (h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record (soma(0.5)._ref_v)

    h.spikelog = h.Vector()
    nc = h.NetCon(soma(0.5)._ref_v, h.nil)
    nc.threshold = -40.0
    nc.record(h.spikelog)
    
    h.tstop = tstop

    frs = []
    stim_amps = [stim1.amp]
    for it in range(1, 9):

        neuron_utils.simulate(v_init, prelength, mainlength)
        
        print("fi_test: stim1.amp = %g spikelog.size = %d\n" % (stim1.amp, h.spikelog.size()))
        stim1.amp = stim1.amp + 0.1
        stim_amps.append(stim1.amp)
        frs.append(h.spikelog.size())
        h.spikelog.clear()
        h.tlog.clear()
        h.Vlog.clear()

    f=open("BasketCell_fi_results.dat",'w')

    for (fr,stim_amp) in zip(frs,stim_amps):
        f.write("%g %g\n" % (stim_amp,fr))

    f.close()


def measure_gap_junction_coupling (env, template_class, tree, v_init):
    
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

    f=open("BasketCellGJ.dat",'w')
    for (t,v1,v2) in zip(h.tlog,h.Vlog1,h.Vlog2):
        f.write("%f %f %f\n" % (t,v1,v2))
    f.close()
    

    

@click.command()
@click.option("--template-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--synapses-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--config-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(template_path,forest_path,synapses_path,config_path):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    env = Env(comm=comm, config_file=config_path, template_paths=template_path)

    h('objref nil, pc, tlog, Vlog, spikelog')
    h.load_file("nrngui.hoc")
    h.xopen ("./tests/rn.hoc")
    h.xopen(template_path+'/BasketCell.hoc')
    
    pop_name = "BC"
    gid = 1039000
    (trees_dict,_) = read_tree_selection (forest_path, pop_name, [gid], comm=env.comm)
    synapses_dict = read_cell_attribute_selection (synapses_path, pop_name, [gid], "Synapse Attributes", comm=env.comm)

    (_, tree) = next(trees_dict)
    (_, synapses) = next(synapses_dict)

    v_init = -60
    
    template_class = getattr(h, "BasketCell")

    ap_test(template_class, tree, v_init)
    passive_test(template_class, tree, v_init)
    ap_rate_test(template_class, tree, v_init)
    fi_test(template_class, tree, v_init)
    gap_junction_test(env, template_class, tree, v_init)
    synapse_test(template_class, gid, tree, synapses, v_init, env)
    
if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find("BasketCellTest.py") != -1,sys.argv)+1):])
