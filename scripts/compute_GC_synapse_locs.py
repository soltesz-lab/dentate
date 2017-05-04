from specify_cells import *
from mpi4py import MPI
from neurotrees.io import NeurotreeGen
from neurotrees.io import append_cell_attributes
# import mkl

# mkl.set_num_threads(1)

comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0:
    print '%i ranks have been allocated' % comm.size
sys.stdout.flush()

neurotrees_dir = '/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/'

# forest_file = '122016_DGC_forest_test_copy.h5'
# neurotrees_dir = os.environ['PI_SCRATCH']+'/DGC_forest/hdf5/'
# neurotrees_dir = os.environ['PI_HOME']+'/'
# forest_file = 'DGC_forest_full.h5'
# forest_file = 'DGC_forest_syns_012717.h5'
forest_file = 'DGC_forest_syns_20170421.h5'

population = 'GC'
count = 0
start_time = time.time()
for gid, morph_dict in NeurotreeGen(MPI._addressof(comm), neurotrees_dir+forest_file, population, io_size=comm.size):
    local_time = time.time()
    # mismatched_section_dict = {}
    synapse_dict = {}
    if gid is not None:
        print  'Rank %i gid: %i' % (rank, gid)
        cell = DG_GC(neurotree_dict=morph_dict, gid=gid, full_spines=False)
        # this_mismatched_sections = cell.get_mismatched_neurotree_sections()
        # if this_mismatched_sections is not None:
        #    mismatched_section_dict[gid] = this_mismatched_sections
        synapse_dict[gid] = cell.export_neurotree_synapse_attributes()
        del cell
        print 'Rank %i took %i s to compute syn_locs for %s gid: %i' % (rank, time.time() - local_time, population, gid)
        count += 1
    else:
        print  'Rank %i gid is None' % rank
    # print 'Rank %i before append_cell_attributes' % rank
    append_cell_attributes(MPI._addressof(comm), neurotrees_dir+forest_file, population, synapse_dict,
                           namespace='Synapse_Attributes', io_size=min(comm.size, 256), chunk_size=100000,
                           value_chunk_size=2000000)
    sys.stdout.flush()
    del synapse_dict
    gc.collect()
# print 'Rank %i completed iterator' % rank

# len_mismatched_section_dict_fragments = comm.gather(len(mismatched_section_dict), root=0)
global_count = comm.gather(count, root=0)
if rank == 0:
    print 'target: %s, %i ranks took %i s to compute syn_locs for %i cells' % (population, comm.size,
                                                                                   time.time() - start_time,
                                                                                   np.sum(global_count))
    # print '%i morphologies have mismatched section indexes' % np.sum(len_mismatched_section_dict_fragments)
