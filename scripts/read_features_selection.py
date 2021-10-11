import sys
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, read_cell_attribute_selection
import numpy as np
import pandas as pd
import click, logging


def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None

logging.basicConfig(level=logging.INFO)

script_name = 'read_features_selection.py'
logger = logging.getLogger(script_name)


@click.command()
@click.option("--population", '-p', type=str, default='GC')
@click.option("--features-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--features-namespace", type=str, required=True)
@click.option("--selection-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(population, features_path, features_namespace, selection_path):

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    population_ranges = read_population_ranges(features_path)[0]
    
    soma_coords = {}

    selection = []
    f = open(selection_path, 'r')
    for line in f.readlines():
        selection.append(int(line))
    f.close()

    columns = ['Field Width', 'X Offset', 'Y Offset']
    df_dict = {}
    it = read_cell_attribute_selection(features_path, population, 
                                       namespace=features_namespace, 
                                       selection=selection)

    for cell_gid, features_dict in it:
        cell_field_width = features_dict['Field Width'][0]
        cell_xoffset = features_dict['X Offset'][0]
        cell_yoffset = features_dict['Y Offset'][0]
        
        df_dict[cell_gid] = [cell_field_width, cell_xoffset, cell_yoffset]

        
    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=columns)
    df = df.reindex(selection)
    df.to_csv('features.%s.csv' % population)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

