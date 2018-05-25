
import sys, os, gc, click, logging
import numpy as np
import dentate
from dentate import utils, spikedata
from neuroh5.io import read_population_ranges, read_population_names, read_cell_attributes
import h5py

script_name = 'measure_place_fields.py'
logger = utils.get_script_logger(script_name)

@click.command()
@click.option("--inst-rates-path", '-p', required=True, type=click.Path())
@click.option("--inst-rates-namespace", '-n', type=str, default='Instantaneous Rate')
@click.option("--include", '-i', type=str, multiple=True)
@click.option("--bin-size", type=float, default=150.0)
@click.option("--nstdev", type=float, default=1.5)
@click.option("--baseline-fraction", type=int)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(inst_rates_path, inst_rates_namespace, include, bin_size, nstdev, baseline_fraction, verbose):

    if verbose:
        logger.setLevel(logging.INFO)
    
    if not include:
        population_names  = read_population_names(inst_rates_path)
        for pop in population_names:
            include.append(pop)

    for i, population in enumerate(include):

        rate_inst_iter = read_cell_attributes(inst_rates_path, population, namespace=inst_rates_namespace)

        rate_inst_dict = dict(rate_inst_iter)

        spikedata.place_fields (population, bin_size, rate_inst_dict, nstdev, baseline_fraction=baseline_fraction, saveData=inst_rates_path)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

