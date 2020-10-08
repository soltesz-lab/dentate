import sys
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, read_cell_attribute_selection
import numpy as np
import pandas as pd
import click

def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None

script_name = 'read_tree_meas_selection.py'

@click.command()
@click.option("--population", '-p', type=str, default='GC')
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--forest-measurement-namespace", type=str, default='Tree Measurements')
@click.option("--attr-name", type=str, default='dendrite_length')
@click.option("--selection-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(population, forest_path, forest_measurement_namespace, attr_name, selection_path):

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    population_ranges = read_population_ranges(forest_path)[0]
    
    selection = []
    f = open(selection_path, 'r')
    for line in f.readlines():
        selection.append(int(line))
    f.close()

    columns = [attr_name]
    df_dict = {}
    it = read_cell_attribute_selection(forest_path, population, 
                                       namespace=forest_measurement_namespace, 
                                       selection=selection)

    for cell_gid, meas_dict in it:
        cell_attr = meas_dict[attr_name]
        df_dict[cell_gid] = [np.sum(cell_attr)]

    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=columns)
    df = df.reindex(selection)
    df.to_csv('tree.%s.%s.csv' % (attr_name, population))


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

