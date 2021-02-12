import os
import numpy as np
import  mpi4py.MPI as MPI
mpicom = MPI.COMM_WORLD
from network_clamp import go as netclamp_go

cell_dt = np.dtype([('celltype', np.str_, 4), ('cell_id', np.int, 1), ('date', np.int, 1), ('time', np.int, 1)])
cell_ac = np.dtype([('celltype', np.str_, 4), ('cell_id', np.int, 1), ('N_trials', np.int, 1), ('idx', np.ndarray, 1)])

class NetclampGo():
    def __init__(self, dir_path, datesfilt, celltypfilt=None, cellidfilt=None, N_trials=None):
        rawlst = os.listdir(dir_path) 
        filtlist = []
        celltyprec = []

        if celltypfilt is None:
            for fil in rawlst:
                if fil.endswith('.yaml') and fil.startswith('network_clamp.optimize.'):
                    filtrunc = fil[23:-5].split('_')
                    if filtrunc[2] in datesfilt:
                        filtlist.append(tuple(filtrunc))
        else:
            for fil in rawlst:
                if fil.endswith('.yaml') and fil.startswith('network_clamp.optimize.'):
                    filtrunc = fil[23:-5].split('_')
                    if filtrunc[0] in celltypfilt and filtrunc[2] in datesfilt:
                        filtlist.append(tuple(filtrunc))
            
        filtlist.sort()
        N_fils = len(filtlist)
        self.raw_arr = np.empty(shape=(N_fils), dtype=cell_dt)
        for idx, fil in enumerate(filtlist):
            self.raw_arr[idx] = fil[0], int(fil[1]), int(fil[2]), int(fil[3])

#        uniq_celltypes = self.get_uniq_idx('celltype', inv=False)
#        N_uniq_celltypes = uniq_celltypes.size

        uniq_cellids = self.get_uniq_idx('cell_id', cnts=True)
        N_uniq_cellid = uniq_cellids[0].size
        trials_lst = [[] for i in range(N_uniq_cellid)]
        for idx, cell_id in enumerate(uniq_cellids[1]):
            trials_lst[cell_id].append(idx)

        N_trials_min = min(uniq_cellids[-1])
        N_trials = N_trials_min if N_trials is None else min(N_trials, N_trials_min) 
    
        all_idx = np.concatenate([lst[-N_trials:] for lst in trials_lst])

        self.get_go_yaml(all_idx, dir_path)
    #    self.get_my_chores(all_idx)

        print(self.go_yaml)

        self.set_comm_go_args()

    #    temp = self.go_yaml[0]
    #    netclamp_go(self.comm_go_args + ['-p', temp[0], '-g', temp[1], '--params-path', temp[2]])


    def get_go_yaml(self, all_idx, dir_path):
        self.go_yaml = np.empty(shape=all_idx.size, dtype='O')
        for idx, fil in enumerate(self.raw_arr[all_idx]):
            self.go_yaml[idx] = (fil[0], fil[1], '{!s}/{!s}{!s}_{:d}_{:08d}_{:06d}.yaml'.format(dir_path, 'network_clamp.optimize.', *fil)) 
        
    def get_my_chores(self, all_idx):
        N_chores = all_idx.size
        N_hosts = mpicom.Get_size()
        rank = mpicom.Get_rank()
        self.mychores = slice(rank, N_chores, N_hosts)
        
    def set_comm_go_args(self):
        DATASET_PREFIX='/Volumes/Work/SolteszLab/HDM/dentate/datasets/striped/dentate'
        self.comm_go_args = [
            '-c', '20201022_Network_Clamp_GC_Exc_Sat_SLN_IN_Izh.yaml',
            '--template-paths', 'templates',
            '-t',  9500,
            '--dt', 0.001,
            '--dataset-prefix', DATASET_PREFIX, 
            '--config-prefix', 'config',
            '--input-features-path', '{!s}/{!s}'.format(DATASET_PREFIX, '/Full_Scale_Control/DG_input_features_20200910_compressed.h5'), 
            '--input-features-namespaces', 'Place Selectivity',
            '--input-features-namespaces', 'Grid Selectivity',
            '--input-features-namespaces', 'Constant Selectivity',
            '--arena-id', 'A',
            '--trajectory-id', 'Diag',
            '--results-path', 'results/netclamp'
            ]
#            '--params-path config/20201105_Izhi_compiled.yaml
        
        
 #       netclamp_go(['--config-file', 'popo.yaml'])
#        k=0
#        for i, j in zip(uniq_cellids[0], uniq_cellids[2]):
#            print(i,j)
#            k+=1
#            if k%5 == 0:
#                print('\n')

        
    def get_uniq_idx(self, field, inv=True, idx=False, cnts=False):
        return np.unique(self.raw_arr[field], return_index=idx, return_inverse=inv, return_counts=cnts)



if __name__=='__main__':
    datesfilt = ('20210131', '20210201', '20210202')
    celltypfilt = ['HCC']
    dir_path = 'results/netclamp'
    NetclampGo(dir_path, datesfilt, celltypfilt=None)
#    NetclampGo(dir_path, datesfilt)
