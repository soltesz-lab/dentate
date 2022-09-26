import sys
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, append_cell_attributes
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

script_name = 'write_features.py'
logger = logging.getLogger(script_name)


@click.command()
@click.option("--population", '-p', type=str, default='GC')
@click.option("--features-path", required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--features-namespace", type=str, required=True)
@click.option("--input-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
def main(population, features_path, features_namespace, input_path):

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    population_ranges = read_population_ranges(features_path)[0]

    df = None
    if rank == 0:
        df = pd.read_csv(input_path)
        df = df.reset_index()

    df = comm.bcast(df, root=0)
    
    fields = ['gid', 'Field Width', 'X Offset', 'Y Offset',
              'Num Fields', 'Peak Rate', 'Selectivity Type', 'Module ID']
    
    features_dict = {}
    for cell_gid, row in df.iterrows():

        if cell_gid % size != 0:
            continue
        
        cell_field_width = np.asarray([row['Field Width']], dtype=np.float32)
        cell_xoffset = np.asarray([row['X Offset']], dtype=np.float32)
        cell_yoffset = np.asarray([features_dict['Y Offset']], dtype=np.float32)
        cell_num_fields = np.asarray([features_dict['Num Fields']], dtype=np.uint8)
        cell_module_id = np.asarray([features_dict['Module ID']], dtype=np.int8)
        cell_peak_rate = np.asarray([features_dict['Peak Rate']], dtype=np.float32)
        cell_selectivity_type = np.asarray([features_dict['Selectivity Type']], dtype=np.uint8)
        
        features_dict[cell_gid] = {'Field Width': cell_field_width,
                                   'X Offset': cell_xoffset,
                                   'Y Offset': cell_yoffset,
                                   'Num Fields': cell_num_fields,
                                   'Module ID': cell_module_id,
                                   'Peak Rate': cell_peak_rate,
                                   'Selectivity Type': cell_selectivity_type,
                                   }

    
    append_cell_attributes(features_path, population, features_dict,
                           namespace=features_namespace, comm=comm)

        


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])

