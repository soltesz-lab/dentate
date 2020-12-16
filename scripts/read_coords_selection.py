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

script_name = 'read_coords_selection.py'

@click.command()
@click.option("--population", '-p', type=str, default='GC')
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Coordinates')
@click.option("--distances-namespace", type=str, default='Arc Distances')
@click.option("--selection-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(population, coords_path, coords_namespace, distances_namespace, selection_path):

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    population_ranges = read_population_ranges(coords_path)[0]
    
    soma_coords = {}

    selection = []
    f = open(selection_path, 'r')
    for line in f.readlines():
        selection.append(int(line))
    f.close()

    columns = ['U', 'V', 'L']
    df_dict = {}
    it = read_cell_attribute_selection(coords_path, population, 
                                       namespace=coords_namespace, 
                                       selection=selection)

    for cell_gid, coords_dict in it:
        cell_u = coords_dict['U Coordinate'][0]
        cell_v = coords_dict['V Coordinate'][0]
        cell_l = coords_dict['L Coordinate'][0]
        
        df_dict[cell_gid] = [cell_u, cell_v, cell_l]

    if distances_namespace is not None:
        columns.extend(['U Distance', 'V Distance'])
        it = read_cell_attribute_selection(coords_path, population, 
                                           namespace=distances_namespace, 
                                           selection=selection)
        for cell_gid, distances_dict in it:
            cell_ud = distances_dict['U Distance'][0]
            cell_vd = distances_dict['V Distance'][0]
        
            df_dict[cell_gid].extend([cell_ud, cell_vd])
        
    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=columns)
    df = df.reindex(selection)
    df.to_csv('coords.%s.csv' % population)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

