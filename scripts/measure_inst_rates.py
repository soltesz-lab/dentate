
import sys, os, gc, click, logging
import numpy as np
import dentate
from dentate import utils, spikedata
from neuroh5.io import read_population_ranges, read_population_names
from quantities import s, ms
import h5py

script_name = 'measure_inst_rates.py'
logger = utils.get_script_logger(script_name)

@click.command()
@click.option("--spike-events-path", '-p', required=True, type=click.Path())
@click.option("--spike-events-namespace", '-n', type=str, default='Spike Events')
@click.option("--include", '-i', type=str, multiple=True)
@click.option("--sigma", type=float, default=0.05)
@click.option("--sampling-period", type=float, default=1.0)
@click.option("--t-variable", type=str, default='t')
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--output-path", '-o', type=click.Path())
@click.option("--nprocs", type=int, default=1)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(spike_events_path, spike_events_namespace, include, sigma, sampling_period, t_variable, t_max, t_min, output_path, nprocs, verbose):

    if verbose:
        logger.setLevel(logging.INFO)
    
    if t_max is None:
        timeRange = None
    else:
        if t_min is None:
            timeRange = [0.0, t_max]
        else:
            timeRange = [t_min, t_max]

    if not include:
        population_names  = read_population_names(spike_events_path)
        for pop in population_names:
            include.append(pop)

    spkdata = spikedata.read_spike_events(spike_events_path, include, spike_events_namespace, timeVariable=t_variable, timeRange = timeRange, verbose = verbose)
    spkpoplst = spkdata['spkpoplst']
    spkindlst = spkdata['spkindlst']
    spktlst   = spkdata['spktlst']
    tmin      = spkdata['tmin']
    tmax      = spkdata['tmax']

    spike_file = h5py.File(spike_events_path,'r')
    output_file = h5py.File(output_path,'w')
    spike_file.copy('/H5Types',output_file)
    output_file.close()
    spike_file.close()
    
    for i, population in enumerate(include):
        
        spkts         = spktlst[i]
        spkinds       = spkindlst[i]
        spkdict       = spikedata.make_spike_dict(spkinds, spkts)

        spikedata.spike_inst_rates (population, spkdict, sampling_period=sampling_period*ms, sigma=sigma, timeRange=[tmin, tmax], nprocs=nprocs, saveData=output_path)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

