import sys, time 
from mpi4py import MPI
import numpy as np
import itertools as it
import scipy as sp
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

class NetClampParam:
    def __init__(self, fils, prefix=None):
        self.N_fils = len(fils)
        self.fil_arr = np.empty(shape=(self.N_fils,2), dtype='O')
        for idx, fil in enumerate(fils):
            self.fil_arr[idx,0] = h5py.File(fil, 'r')
            self.fil_arr[idx,1] = np.array(self.fil_arr[idx,0]['network_clamp.optimize']['problem_ids']) 
        
        self.ref_point = self.fil_arr[0,0]['network_clamp.optimize']['{:d}'.format(self.fil_arr[0,1][0])]
        ref_features = np.array(self.ref_point['features'])
        self.N_cells = self.fil_arr[idx,1].shape[0]
        self.N_params, self.N_objectives, self.N_trials = ref_features['trial_objs'].shape
        self.param_dtype = np.array(self.ref_point['parameters']).dtype
        self.param_tup = []
        for parm in self.param_dtype.names:
            self.param_tup.append(parm.split('.')) 
        
        self.param_dim = len(self.param_tup) 
        self.param_bnds = np.empty(shape=(self.param_dim, 2)) 
        self.param_bnds[:,0] = np.array(self.fil_arr[0,0]['network_clamp.optimize']['parameter_spec']['lower'])
        self.param_bnds[:,1] = np.array(self.fil_arr[0,0]['network_clamp.optimize']['parameter_spec']['upper'])

        self.population = self.param_tup[0][0]
        self.target = np.array(self.fil_arr[0,0]['network_clamp.optimize']['metadata'])[0][0]

        ts = time.strftime("%Y%m%d_%H%M%S")
        prefix = '' if prefix is None else '{!s}_'.format(prefix)
        self.suffix = 'Combined' if self.N_fils > 1 else '{:08d}'.format(int(self.fil_arr[idx,0].filename.split('_')[-1][:-3]))
        save_ext = 'pdf'
        
        self.filnam = '{!s}{!s}_{!s}_{!s}.{!s}'.format(prefix, self.population, ts, self.suffix, save_ext) 

        self.get_best_arrs([self.target])

        self.pop_props = {
                        'AAC':  ('AxoAxonic', -65, -42),
                        'BC':   ('PV+ Basket', -62, -43.1),
                        'HC':   ('HIPP', -67, -28.1),
                        'HCC':  ('HICAP', -65, -36.6),
                        'IS':   ('IS', -69.7, -44.3),
                        'MOPP': ('MOPP', -75.6, -30),
                        'NGFC': ('NGFC', -75.6, -30),
                        'MC':   ('Mossy'),
                        }

        self.plot_best()

    def get_best_arrs(self, target):
        self.best_idx = np.empty(shape=(self.N_fils, self.N_cells, self.N_objectives, self.N_trials), dtype=np.int)
        self.best_arr = np.empty(shape=(self.N_fils, self.N_cells, self.N_objectives, self.N_trials))
        self.best_Vmean = np.empty(shape=(self.N_fils, self.N_cells, self.N_objectives, self.N_trials))
        self.best_prm = np.empty(shape=(self.N_fils, self.N_cells, self.N_objectives, self.N_trials), dtype=self.param_dtype)
        self.bestmean_idx = np.empty(shape=(self.N_fils, self.N_cells), dtype=np.int)
        self.bestmean_arr = np.empty(shape=(self.N_fils, self.N_cells))
        self.bestmean_V_mean = np.empty(shape=(self.N_fils, self.N_cells))
        self.bestmean_prm = np.empty(shape=(self.N_fils, self.N_cells), dtype=self.param_dtype)

        for fidx, filobj in enumerate(self.fil_arr[:,0]):
            for cidx, cell in enumerate(self.fil_arr[fidx, 1]):
                ref_cell = filobj['network_clamp.optimize']['{:d}'.format(cell)] 
                ref_features = ref_cell['features'] 
                for i in range(self.N_objectives):
                    np.argmin(np.abs(ref_features['trial_objs'][:, i ,:]-target[i]), axis=0, out=self.best_idx[fidx,cidx, i])
                for pidx, val in np.ndenumerate(self.best_idx[fidx,cidx]):
                    self.best_arr[fidx, cidx, pidx[0], pidx[1]] = ref_features['trial_objs'][val, pidx[0], pidx[1]]    
                    self.best_prm[fidx, cidx, pidx[0], pidx[1]] = ref_cell['parameters'][val]
                    self.best_Vmean[fidx, cidx, pidx[0], pidx[1]] = ref_features['mean_v'][val, pidx[1]] 

                self.bestmean_idx[fidx, cidx] = np.argmin(np.abs(ref_features['mean_rate']-target[0]))
                self.bestmean_arr[fidx, cidx] = ref_features['mean_rate'][self.bestmean_idx[fidx, cidx]]
                self.bestmean_prm[fidx, cidx] = ref_cell['parameters'][self.bestmean_idx[fidx, cidx]]

    def get_fig_axes(self):
        fig = plt.figure(figsize=(20,20), constrained_layout=True)
        fig.suptitle('{!s}\n{!s}'.format(self.pop_props[self.population][0], self.suffix))
        spec = gs.GridSpec(nrows=5, ncols=1, figure=fig)

        optobjspec = spec[0,0].subgridspec(self.N_objectives, self.N_cells)
        objspec = spec[1,0].subgridspec(1, 2)
        prmspec = spec[2,0].subgridspec(1, self.param_dim)
        optprmspec = spec[3,0].subgridspec(1, self.param_dim)
        corprmspec = spec[4,0].subgridspec(1, (self.param_dim**2-self.param_dim)//2)

        opt_axes = np.empty(shape=optobjspec.get_geometry(), dtype='O')
        obj_axes = np.empty(shape=self.N_objectives, dtype='O')
        fea_axes = np.empty(shape=1, dtype='O')
        prm_axes = np.empty(prmspec.get_geometry(), dtype='O')
        opr_axes = np.empty(optprmspec.get_geometry(), dtype='O')
        cpr_axes = np.empty(corprmspec.get_geometry(), dtype='O')

        for idx, ax in np.ndenumerate(opt_axes): 
            opt_axes[idx] = fig.add_subplot(optobjspec[idx])

        for idx, ax in np.ndenumerate(obj_axes): 
            obj_axes[idx] = fig.add_subplot(objspec[0,idx[0]])

        for idx, ax in np.ndenumerate(prm_axes): 
            prm_axes[idx] = fig.add_subplot(prmspec[idx])

        for idx, ax in np.ndenumerate(opr_axes): 
            opr_axes[idx] = fig.add_subplot(optprmspec[idx])

        for idx, ax in np.ndenumerate(cpr_axes): 
            cpr_axes[idx] = fig.add_subplot(corprmspec[idx])

        fea_axes[0] = fig.add_subplot(objspec[0, 1])

        return fig, opt_axes, obj_axes, fea_axes, prm_axes , opr_axes, cpr_axes 


    def plot_best(self):
        param_names = np.array(self.param_dtype.names)
        param_dim = self.param_dim 
        N_iter_arr = np.arange(self.N_params)

        prmcomb = np.array([i for i in it.combinations(range(param_dim), r=2)], dtype=np.int)
        N_prmcomb = prmcomb.shape[0]

        xcat = self.fil_arr[0,1]
        boxwidth = (xcat[-1]-xcat[0])/10
        plotshap = ['o', 'x', 'd', '*', '+']
        N_fil_cell_max = max(self.N_fils, self.N_cells)
        colors = plt.get_cmap('tab10').colors if N_fil_cell_max <=10 else mpl.cm.get_cmap('tab20').colors

        fig, opt_axes, obj_axes, fea_axes, prm_axes , opr_axes, cpr_axes = self.get_fig_axes() 


        for i in range(self.N_objectives): 
            ytop = self.target[i]*5 if self.target[i]<10 else self.target[i]*3
            for j in range(self.N_fils):
                obj_axes[i].scatter(xcat, self.bestmean_arr[j, :], marker=plotshap[j], color='k')
                obj_axes[i].axhline(self.target[0], color='k', ls='--')
                for tidx in range(self.N_trials):
                    obj_axes[i].scatter(xcat, self.best_arr[j,:, i,tidx], marker=plotshap[j], color=colors[tidx], facecolors='none', lw=0.5)

                    fea_axes[0].scatter(xcat, self.best_Vmean[j,:, i,tidx], marker=plotshap[j], color=colors[tidx], facecolors='none', lw=0.5)

            obj_axes[i].set_ylabel(r'Firing Rate [spikes/s]')

        fea_axes[0].set_ylabel(r'Mean voltage [mV]')
        fea_axes[0].axhline(self.pop_props[self.population][1], color='k', lw=0.5)
        fea_axes[0].axhline(self.pop_props[self.population][2], color='k', lw=0.5)

        for i, prm in enumerate(param_names): 
            prm_axes[0,i].axhline(self.param_bnds[i,0], color='k', ls='--')
            prm_axes[0,i].axhline(self.param_bnds[i,1], color='k', ls='--')
            prm_axes[0,i].set_ylabel(r'${!s}$ weight'.format(self.param_tup[i][3]))
            tit = self.param_tup[i][1][1:-1].replace("'", "") if self.param_tup[i][1].startswith('(') else self.param_tup[i][1]
            prm_axes[0,i].set_title(tit)

            opr_axes[0,i].axhline(self.param_bnds[i,0], color='k', ls='--')
            opr_axes[0,i].axhline(self.param_bnds[i,1], color='k', ls='--')
            opr_axes[0,i].set_ylabel(r'${!s}$ weight'.format(self.param_tup[i][3]))
            opr_axes[0,i].set_xlabel(r'Iteration Index')

            for cidx, cell in enumerate(xcat): 
                prm_axes[0,i].boxplot(np.ravel(self.best_prm[prm][:,cidx, 0,:]), vert=True, positions=[cell], showmeans=True, widths=boxwidth)

            for j in range(self.N_fils):
                prm_axes[0,i].scatter(xcat, self.bestmean_prm[prm][j, :], marker=plotshap[j], color='k')
#                plt.xticks(rotation = 45)
                for tidx in range(self.N_trials):
                    prm_axes[0,i].scatter(xcat, self.best_prm[prm][j,:, 0,tidx], marker=plotshap[j], color=colors[tidx], facecolors='none', lw=0.5)
                
                for cidx, cell in enumerate(xcat): 
                    ref_cell = self.fil_arr[j,0]['network_clamp.optimize']['{:d}'.format(cell)]
                    opr_axes[0,i].scatter(N_iter_arr, np.array(ref_cell['parameters'][prm]),  marker=plotshap[j], color=colors[cidx], facecolors='none', lw=0.01, s=1)



        opt_val_arr = np.empty(shape=(self.N_fils, self.N_cells, self.N_params, self.N_objectives, self.N_trials))
        for fidx, filobj in enumerate(self.fil_arr[:,0]):
            for cidx, cell in enumerate(self.fil_arr[fidx, 1]):
                ref_cell = filobj['network_clamp.optimize']['{:d}'.format(cell)] 
                opt_val_arr[fidx,cidx] = ref_cell['features']['trial_objs']
                ref_ax = opt_axes[0,cidx]
                ref_ax.axhline(self.target[0], color='k', ls='--')
                for tidx in range(self.N_trials):
                    ref_ax.scatter(N_iter_arr, opt_val_arr[fidx,cidx,:,0,tidx], marker=plotshap[fidx], color=colors[tidx], facecolors='none', lw=0.01, s=1)

                for combidx in range(N_prmcomb):
                    comb  = prmcomb[combidx, : ]
                    i, j = comb

                    cpr_axes[0,combidx].axvline(self.param_bnds[i,0], color='k', ls='--')
                    cpr_axes[0,combidx].axvline(self.param_bnds[i,1], color='k', ls='--')
                    cpr_axes[0,combidx].axhline(self.param_bnds[j,0], color='k', ls='--')
                    cpr_axes[0,combidx].axhline(self.param_bnds[j,1], color='k', ls='--')

                    titx = self.param_tup[i][1][1:-1].replace("'", "") if self.param_tup[i][1].startswith('(') else self.param_tup[i][1]
                    tity = self.param_tup[j][1][1:-1].replace("'", "") if self.param_tup[j][1].startswith('(') else self.param_tup[j][1]
                    cpr_axes[0,combidx].set_xlabel(r'${!s}$ {!s}'.format(self.param_tup[i][3], titx))
                    cpr_axes[0,combidx].set_ylabel(r'${!s}$ {!s}'.format(self.param_tup[j][3], tity))
                    cpr_axes[0,combidx].scatter(np.array(ref_cell['parameters'][param_names[i]]),  np.array(ref_cell['parameters'][param_names[j]]), marker=plotshap[fidx], color=colors[cidx], facecolors='none', lw=0.01, s=1)



        for cidx, cell in enumerate(self.fil_arr[0, 1]):
            ref_ax = opt_axes[0,cidx]
        #    ref_ax.plot(N_iter_arr, np.mean(opt_val_arr[:,cidx,:,0,:], axis=(0,-1)), color='y', lw=0.5)
            perc = np.quantile(opt_val_arr[:,cidx,:,0,:], [0.25,0.5, 0.75], axis=(0,-1), keepdims=False)
            ref_ax.plot(N_iter_arr, perc[1,:], color='y', lw=0.25, zorder=101)
            ref_ax.fill_between(N_iter_arr, y1=perc[2,:], y2=perc[0,:], color='y', alpha=0.5, zorder=100)
#            ref_ax.plot(N_iter_arr, np.min(opt_val_arr[:,cidx,:,0,:], axis=(0,-1)), color='y', lw=0.5, alpha=0.5)
#            ref_ax.plot(N_iter_arr, np.max(opt_val_arr[:,cidx,:,0,:], axis=(0,-1)), color='c', lw=0.5, alpha=0.5)
            ref_ax.set_title('{:d}'.format(int(cell)))
            ref_ax.set_ylim(top=ytop, bottom=0)

        opt_axes[0,0].set_ylabel(r'Firing Rate [spikes/s]')
        opt_axes[0,0].set_xlabel(r'Iteration Index')

#        fig.tight_layout()
        fig.savefig(self.filnam)


def distribute_chores(fil_list, Combined=True):
    N_fils = len(fil_list)

    mpicom = MPI.COMM_WORLD

    N_hosts = mpicom.Get_size()
    Rank = mpicom.Get_rank()
    
    my_chores = np.arange(Rank, N_fils, N_hosts)

    for idx in my_chores:
        if Rank == 0:
            if Combined:
                NetClampParam(fil_list[idx])
            else:
                for fil in fil_list[idx]:
                    NetClampParam([fil])
    

if __name__ == '__main__':

    popfils1 = [
      [ '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210304_004526_27137089.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210304_004526_36010476.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210304_004526_49937004.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210304_004526_53499406.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210304_014847_04893658.h5',],
      [ '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210304_004526_01503844.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210304_004527_28135771.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210304_004527_52357252.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210304_004527_74865768.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210304_004527_93454042.h5',],
      [ '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210304_004526_45419272.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210304_004527_15879716.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210304_004527_28682721.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210304_004527_53736785.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210304_004527_63599789.h5',],
      [ '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210304_004526_12260638.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210304_004526_71407528.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210304_004527_17609813.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210304_004527_33236209.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210304_004527_92055940.h5',],
      [ '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210304_004526_49627038.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210304_004527_04259860.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210304_004527_11745958.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210304_004527_75940072.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210304_004527_84013649.h5',],
      [ '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210304_004526_45373570.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210304_004527_29079471.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210304_004527_31571230.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210304_004527_68839073.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210304_004527_85763600.h5',],
      [ '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210304_004527_12740157.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210304_004527_93872787.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210304_004527_95844113.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210304_004527_96772370.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210304_004527_97895890.h5',],
    ] 

    popfils2 = [
      [ '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210305_033838_04893658.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210305_033838_27137089.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210305_033838_49937004.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210305_033838_53499406.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210305_033839_36010476.h5',],
]
    lif= [
      [ '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210305_033838_28135771.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210305_033838_52357252.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210305_033838_74865768.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210305_033838_93454042.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210305_033839_01503844.h5',],
      [ '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210305_033838_15879716.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210305_033838_28682721.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210305_033838_45419272.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210305_033838_63599789.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210305_033839_53736785.h5',],
      [ '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210305_033838_33236209.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210305_033839_12260638.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210305_033839_17609813.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210305_033839_71407528.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210305_033839_92055940.h5',],
      [ '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210305_033838_04259860.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210305_033838_11745958.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210305_033838_49627038.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210305_033838_75940072.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210305_033838_84013649.h5',],
      [ '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210305_033838_31571230.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210305_033838_68839073.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210305_033839_29079471.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210305_033839_45373570.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210305_033839_85763600.h5',],
      [ '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210305_033838_12740157.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210305_033838_93872787.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210305_033838_96772370.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210305_033839_95844113.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210305_033839_97895890.h5',],
    ]

    newHC= [
      [ '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210312_225530_15879716.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210312_225530_28682721.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210312_225530_45419272.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210312_225530_53736785.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210312_225530_63599789.h5',],
    ]
    firstMC= [
      [ '/Volumes/Work/SolteszLab/HDM/dentate/results/distgfs.network_clamp.MC_20210309_190038_78236474.h5',],
    ]
 
    newset = [
     #   ['/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210317_032406_27137089.h5',
     #   '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210317_032406_53499406.h5',
     #   '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210317_032407_04893658.h5',
     #   '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210317_032407_36010476.h5',
     #   '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210317_032407_49937004.h5',],
   #     ['/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210317_032406_01503844.h5',
   #     '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210317_032406_93454042.h5',
   #     '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210317_032407_28135771.h5',
   #     '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210317_032407_52357252.h5',
   #     '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210317_032407_74865768.h5',],
  #      ['/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210317_032407_15879716.h5',
  #      '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210317_032407_28682721.h5',
  #      '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210317_032407_45419272.h5',
  #      '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210317_032407_53736785.h5',
  #      '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210317_032407_63599789.h5',],
        ['/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210317_032406_12260638.h5',
#        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210317_032406_17609813.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210317_032407_33236209.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210317_032407_71407528.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210317_032407_92055940.h5',],
   #     ['/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210317_032406_04259860.h5',
   #     '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210317_032406_11745958.h5',
   #     '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210317_032407_49627038.h5',
   #     '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210317_032407_75940072.h5',
   #     '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210317_032407_84013649.h5',],
   #     ['/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210317_032406_85763600.h5',
   #     '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210317_032407_29079471.h5',
   #     '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210317_032407_31571230.h5',
   #     '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210317_032407_45373570.h5',
   #     '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210317_032407_68839073.h5',],
   #     ['/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210317_032406_12740157.h5',
   #     '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210317_032406_93872787.h5',
   #     '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210317_032407_95844113.h5',
   #     '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210317_032407_96772370.h5',
   #     '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210317_032407_97895890.h5',],
    ]


    newsetV = [
        ['/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210317_195313_04893658.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210317_195313_27137089.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210317_195313_36010476.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210317_195313_49937004.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.AAC_20210317_195313_53499406.h5',],
        ['/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210317_195312_52357252.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210317_195312_74865768.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210317_195313_01503844.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210317_195313_28135771.h5',],
    #    '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.BC_20210317_195313_93454042.h5',],
        ['/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210317_195312_15879716.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210317_195312_53736785.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210317_195313_28682721.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210317_195313_45419272.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HC_20210317_195313_63599789.h5',],
        ['/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210317_195312_12260638.h5',
#        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210317_195312_17609813.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210317_195312_33236209.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210317_195312_71407528.h5',],
 #       '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.HCC_20210317_195313_92055940.h5',],
        ['/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210317_195312_04259860.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210317_195313_11745958.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210317_195313_49627038.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210317_195313_75940072.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.IS_20210317_195313_84013649.h5',],
        ['/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210317_195312_29079471.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210317_195312_31571230.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210317_195313_45373570.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210317_195313_68839073.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.MOPP_20210317_195313_85763600.h5',],
        ['/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210317_195312_12740157.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210317_195312_95844113.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210317_195312_97895890.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210317_195313_93872787.h5',
        '/Volumes/Work/SolteszLab/HDM/dentate/results/netclamp/distgfs.network_clamp.NGFC_20210317_195313_96772370.h5',],
    ]

    distribute_chores(newsetV, Combined=True)
