"""
Tools for pulling individual neurons out of the dentate network simulation environment for single-cell tuning.
"""
__author__ = 'See AUTHORS.md'

import click, os, sys, time
from collections import defaultdict
from mpi4py import MPI
import numpy as np
import h5py
import click
from dentate.cells import get_biophys_cell, get_branch_order, get_dendrite_origin, get_distance_to_node, \
    init_biophysics, is_terminal, report_topology, modify_mech_param
from dentate.env import Env
from dentate.neuron_utils import h, configure_hoc_env
from dentate.synapses import config_biophys_cell_syns, init_syn_mech_attrs, modify_syn_param
from dentate.utils import viewitems, range, str, Context, list_find, basestring
from dentate.io_utils import set_h5py_attr

context = Context()


class QuickSim(object):
    """
    This class is used to organize and run simple single cell simulations. Stores references and metadata for current
    injection stimuli and vector recordings. Handles export of recorded simulation data to HDF5 files.
    """
    def __init__(self, tstop=400., cvode=True, daspk=False, dt=0.025, verbose=True):
        """

        :param tstop: float
        :param cvode: bool
        :param daspk: bool
        :param dt: float
        :param verbose: bool
        """
        self.recs = defaultdict(dict)  # dict: {rec_name: dict containing recording metadata }
        self.stims = defaultdict(dict) # dict: {stim_name: dict containing stimulation metadata }
        self.tstop = tstop
        h.load_file('stdrun.hoc')
        h.celsius = 35.0
        h.cao0_ca_ion = 1.3
        self.cvode_atol = 0.01  # 0.001
        self.daspk = daspk
        self._cvode = cvode
        self.cvode = cvode
        self.dt = dt
        self.verbose = verbose
        self.tvec = h.Vector()
        self.tvec.record(h._ref_t, self.dt)
        self.parameters = {}
        self.backup_state()

    def run(self, v_init=-65.):
        """

        :param v_init: float
        """
        start_time = time.time()
        h.tstop = self.tstop
        if self._cvode != self.cvode:
            self.cvode = self._cvode
        if not self._cvode:
            h.dt = self.dt
            h.steps_per_ms = int(1. / self.dt)
        h.v_init = v_init
        h.run()
        if self.verbose:
            print('Simulation runtime: %.2f s' % (time.time() - start_time))

    def backup_state(self):
        """
        Store backup of current state of simulation parameters: dt, tstop, cvode, daspk.
        """
        self._backup = {'dt': self.dt, 'tstop': self.tstop, 'cvode': self._cvode, 'daspk': self.daspk}

    def set_state(self, dt=None, tstop=None, cvode=None, daspk=None):
        """
        Convenience function for setting simulation parameters that are frequently modified.
        :param dt: float
        :param tstop: float
        :param cvode: bool
        :param daspk: bool
        """
        if dt is not None:
            self.dt = dt
        if tstop is not None:
            self.tstop = tstop
        if cvode is not None:
            self.cvode = cvode
        if daspk is not None:
            self.daspk = daspk

    def restore_state(self):
        """
        Restore state of simulation parameters from backup: dt, tstop, cvode, daspk.
        """
        self.set_state(**self._backup)

    def is_same_cell(self, cell1, cell2):
        """

        :param cell1: :class:'BiophysCell' or :class:'h.hocObject'
        :param cell2: :class:'BiophysCell' or :class:'h.hocObject'
        :return: bool
        """
        if cell1 == cell2:
            return True
        elif hasattr(cell1, 'tree'):
            return cell1.tree.root.sec.cell() == cell2
        else:
            raise RuntimeError('QuickSim: problem comparing cell objects: %s and %s' % (str(cell1), str(cell2)))

    def append_rec(self, cell, node, name=None, loc=None, param='_ref_v', object=None, ylabel='Vm', units='mV',
                   description=''):
        """

        :param cell: :class:'BiophysCell'
        :param node: :class:'SHocNode'
        :param name: str
        :param loc: float
        :param param: str
        :param object: :class:'HocObject'
        :param ylabel: str
        :param units: str
        :param description: str
        """
        if not self.is_same_cell(cell, node.sec.cell()):
            raise RuntimeError('QuickSim: append_rec: target cell does not match target node')
        if name is None:
            name = 'rec%i' % len(self.recs)
        elif name in self.recs:
            name = '%s%i' % (name, len(self.recs))
        self.recs[name]['cell'] = cell
        self.recs[name]['node'] = node
        self.recs[name]['ylabel'] = ylabel
        self.recs[name]['units'] = units
        self.recs[name]['vec'] = h.Vector()
        if object is None:
            if loc is None:
                loc = 0.5
            self.recs[name]['vec'].record(getattr(node.sec(loc), param), self.dt)
        else:
            if loc is None:
                try:
                    loc = object.get_segment().x
                except:
                    loc = 0.5  # if the object doesn't have a .get_segment() method, default to 0.5
            if param is None:
                self.recs[name]['vec'].record(object, self.dt)
            else:
                self.recs[name]['vec'].record(getattr(object, param), self.dt)
        self.recs[name]['loc'] = loc
        self.recs[name]['description'] = description

    def has_rec(self, name):
        """
        Report whether a recording exists with the provided name.
        :param name: str
        :return: bool
        """
        return name in self.recs

    def get_rec(self, name):
        """
        Return the rec_dict associated with the provided name.
        :param description: str
        :return: dict
        """
        if self.has_rec(name):
            return self.recs[name]
        else:
            raise KeyError('QuickSim: get_rec: cannot find recording with name: %s' % name)

    def modify_rec(self, name, node=None, loc=None, object=None, param='_ref_v', ylabel=None, units=None,
                   description=None, cell=None):
        """

        :param name: str
        :param node: class:'SHocNode'
        :param loc: float
        :param object: class:'HocObject'
        :param param: str
        :param ylabel: str
        :param units: str
        :param description: str
        :param cell: class'BiophysCell'
        """
        if not self.has_rec(name):
            raise KeyError('QuickSim: modify_rec: cannot find recording with name: %s' % name)
        if ylabel is not None:
            self.recs[name]['ylabel'] = ylabel
        if units is not None:
            self.recs[name]['units'] = units

        if cell is not None:
            if node is None:
                raise RuntimeError('QuickSim: modify_rec: cannot change target cell without specifying new target '
                                   'node')
            elif not self.is_same_cell(cell, node.sec.cell()):
                raise RuntimeError('QuickSim: modify_rec: target cell does not match target node')
            self.recs[name]['cell'] = cell

        if node is not None:
            if not self.is_same_cell(self.recs[name]['cell'], node.sec.cell()):
                raise RuntimeError('QuickSim: modify_rec: target cell does not match target node')
            self.recs[name]['node'] = node

        if loc is not None:
            self.recs[name]['loc'] = loc
        if object is None:
            self.recs[name]['vec'].record(getattr(self.recs[name]['node'].sec(self.recs[name]['loc']), param), self.dt)
        elif param is None:
            self.recs[name]['vec'].record(object, self.dt)
        else:
            self.recs[name]['vec'].record(getattr(object, param), self.dt)
        if description is not None:
            self.recs[name]['description'] = description

    def append_stim(self, cell, node, name=None, loc=0.5, amp=0., delay=0., dur=0., description='IClamp'):
        """

        :param cell: :class:'BiophysCell'
        :param node: :class:'SHocNode'
        :param name: str
        :param loc: float
        :param amp: float
        :param delay: float
        :param dur: float
        :param description: str
        """
        if not self.is_same_cell(cell, node.sec.cell()):
            raise RuntimeError('QuickSim: append_stim: target cell does not match target node')
        if name is None:
            name = 'stim%i' % len(self.stims)
        elif name in self.stims:
            name = '%s%i' % (name, len(self.stims))
        self.stims[name]['cell'] = cell
        self.stims[name]['node'] = node
        self.stims[name]['stim'] = h.IClamp(node.sec(loc))
        self.stims[name]['stim'].amp = amp
        self.stims[name]['stim'].delay = delay
        self.stims[name]['stim'].dur = dur
        self.stims[name]['vec'] = h.Vector()
        self.stims[name]['vec'].record(self.stims[name]['stim']._ref_i, self.dt)
        self.stims[name]['description'] = description

    def has_stim(self, name):
        """
        Report whether a stimulus exists with the provided name.
        :param name: str
        :return: bool
        """
        return name in self.stims

    def get_stim(self, name):
        """
        Return the stim_dict associated with the provided name.
        :param description: str
        :return: dict
        """
        if self.has_stim(name):
            return self.stims[name]
        else:
            raise KeyError('QuickSim: get_stim: cannot find stimulus with name: %s' % name)

    def modify_stim(self, name, node=None, loc=None, amp=None, delay=None, dur=None, description=None, cell=None):
        """

        :param name: str
        :param node: class:'SHocNode'
        :param loc: float
        :param amp: float
        :param delay: float
        :param dur: float
        :param description: str
        :param cell: :class:'BiophysCell'
        """
        if cell is not None:
            if node is None:
                raise RuntimeError('QuickSim: modify_stim: cannot change target cell without specifying new target '
                                   'node')
            elif not self.is_same_cell(cell, node.sec.cell()):
                raise RuntimeError('QuickSim: modify_stim: target cell does not match target node')
            self.stims[name]['cell'] = cell
        if not (node is None and loc is None):
            if node is not None:
                if not self.is_same_cell(self.stims[name]['cell'], node.sec.cell()):
                    raise RuntimeError('QuickSim: modify_stim: target cell does not match target node')
                self.stims[name]['node'] = node
            if loc is None:
                loc = self.stims[name]['stim'].get_segment().x
            self.stims[name]['stim'].loc(self.stims[name]['node'].sec(loc))

        if amp is not None:
            self.stims[name]['stim'].amp = amp
        if delay is not None:
            self.stims[name]['stim'].delay = delay
        if dur is not None:
            self.stims[name]['stim'].dur = dur
        if description is not None:
            self.stims[name]['description'] = description

    def plot(self, axes=None, show=True):
        """

        """
        import matplotlib.pyplot as plt
        from dentate.plot import clean_axes
        if len(self.recs) == 0:
            return
        if axes is None:
            fig, axes = plt.subplots()
        else:
            fig = axes.get_figure()
        for name, rec_dict in viewitems(self.recs):
            description = str(rec_dict['description'])
            axes.plot(self.tvec, rec_dict['vec'],
                      label='%s: %s(%.2f) %s' % (name, rec_dict['node'].name, rec_dict['loc'], description))
            axes.set_xlabel('Time (ms)')
            axes.set_ylabel('%s (%s)' % (rec_dict['ylabel'], rec_dict['units']))
        axes.legend(loc='best', frameon=False, framealpha=0.5)
        title = None
        if 'title' in self.parameters:
            title = self.parameters['title']
        if 'description' in self.parameters:
            if title is not None:
                title = title + '; ' + self.parameters['description']
            else:
                title = self.parameters['description']
        if title is not None:
            axes.set_title(title)
        clean_axes(axes)
        if show:
            fig.tight_layout()
            fig.show()
        else:
            return axes

    def export_to_file(self, file_path, append=True):
        """
        Exports simulated data and metadata to an HDF5 file. Arrays are saved as datasets and metadata is saved as
        attributes. Repeated simulations are stored in enumerated groups.
        :param file_path: str (path)
        :param append: bool
        """
        if append:
            io_type = 'a'
        else:
            io_type = 'w'
        with h5py.File(file_path, io_type) as f:
            if 'sim_output' not in f:
                f.create_group('sim_output')
                f['sim_output'].attrs['enumerated'] = True
            target = f['sim_output']
            simiter = len(target)
            if str(simiter) not in target:
                target.create_group(str(simiter))
            target[str(simiter)].create_dataset('time', compression='gzip', data=self.tvec)
            target[str(simiter)]['time'].attrs['dt'] = self.dt
            for parameter in self.parameters:
                set_h5py_attr(target[str(simiter)].attrs, parameter, self.parameters[parameter])
            if len(self.stims) > 0:
                target[str(simiter)].create_group('stims')
                for name, stim_dict in viewitems(self.stims):
                    stim = target[str(simiter)]['stims'].create_dataset(name, compression='gzip', data=stim_dict['vec'])
                    cell = stim_dict['cell']
                    stim.attrs['cell'] = cell.gid
                    node = stim_dict['node']
                    stim.attrs['index'] = node.index
                    set_h5py_attr(stim.attrs, 'type', node.type)
                    loc = stim_dict['stim'].get_segment().x
                    stim.attrs['loc'] = loc
                    distance = get_distance_to_node(cell, cell.tree.root, node, loc)
                    stim.attrs['soma_distance'] = distance
                    distance = get_distance_to_node(cell, get_dendrite_origin(cell, node), node, loc)
                    stim.attrs['branch_distance'] = distance
                    stim.attrs['amp'] = stim_dict['stim'].amp
                    stim.attrs['delay'] = stim_dict['stim'].delay
                    stim.attrs['dur'] = stim_dict['stim'].dur
                    set_h5py_attr(stim.attrs, 'description', stim_dict['description'])
            target[str(simiter)].create_group('recs')
            for name, rec_dict in viewitems(self.recs):
                rec = target[str(simiter)]['recs'].create_dataset(name, compression='gzip', data=rec_dict['vec'])
                cell = rec_dict['cell']
                rec.attrs['cell'] = cell.gid
                node = rec_dict['node']
                rec.attrs['index'] = node.index
                set_h5py_attr(rec.attrs, 'type', node.type)
                loc = rec_dict['loc']
                rec.attrs['loc'] = loc
                distance = get_distance_to_node(cell, cell.tree.root, node, loc)
                rec.attrs['soma_distance'] = distance
                distance = get_distance_to_node(cell, get_dendrite_origin(cell, node), node, loc)
                node_is_terminal = is_terminal(node)
                branch_order = get_branch_order(cell, node)
                rec.attrs['branch_distance'] = distance
                rec.attrs['is_terminal'] = node_is_terminal
                rec.attrs['branch_order'] = branch_order
                set_h5py_attr(rec.attrs, 'ylabel', rec_dict['ylabel'])
                set_h5py_attr(rec.attrs, 'units', rec_dict['units'])
                set_h5py_attr(rec.attrs, 'description', rec_dict['description'])

    def get_cvode(self):
        """

        :return bool
        """
        return bool(h.CVode().active())

    def set_cvode(self, state):
        """

        :param state: bool
        """
        if state:
            h.CVode().active(1)
            h.CVode().atol(self.cvode_atol)
            h.CVode().use_daspk(int(self.daspk))
            if self.daspk:
                #  Converts stop behavior to a warning when an initialization condition of IDA is not met
                eps = h.CVode().dae_init_dteps()
                h.CVode().dae_init_dteps(eps, 1)
        else:
            h.CVode().active(0)
        self._cvode = state

    cvode = property(get_cvode, set_cvode)


@click.command()
@click.option("--gid", required=True, type=int, default=0)
@click.option("--pop-name", required=True, type=str, default='GC')
@click.option("--config-file", required=True, type=str,
              default='Small_Scale_Control_LN_weights_Sat.yaml')
@click.option("--template-paths", type=str, default='../DGC/Mateos-Aparicio2014:../dentate/templates')
@click.option("--hoc-lib-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../dentate')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../dentate/datasets')
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='../dentate/config')
@click.option("--mech-file", required=True, type=str, default='20181205_DG_GC_excitability_mech.yaml')
@click.option("--load-edges", is_flag=True)
@click.option("--load-weights", is_flag=True)
@click.option("--correct-for-spines", is_flag=True)
@click.option('--verbose', '-v', is_flag=True)
def main(gid, pop_name, config_file, template_paths, hoc_lib_path, dataset_prefix, config_prefix, mech_file,
         load_edges, load_weights, correct_for_spines, verbose):
    """

    :param gid: int
    :param pop_name: str
    :param config_file: str; model configuration file name
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param hoc_lib_path: str; path to directory containing required hoc libraries
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param config_prefix: str; path to directory containing network and cell mechanism config files
    :param mech_file: str; cell mechanism config file name
    :param load_edges: bool; whether to attempt to load connections from a neuroh5 file
    :param load_weights: bool; whether to attempt to load connections from a neuroh5 file
    :param correct_for_spines: bool
    :param verbose: bool
    """
    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    env = Env(comm, config_file, template_paths, hoc_lib_path, dataset_prefix, config_prefix, verbose=verbose)
    configure_hoc_env(env)

    mech_file_path = config_prefix + '/' + mech_file
    cell = get_biophys_cell(env, pop_name=pop_name, gid=gid, load_edges=load_edges, load_weights=load_weights,
                            mech_file_path=mech_file_path)
    context.update(locals())
    
    init_biophysics(cell, reset_cable=True, correct_cm=correct_for_spines, correct_g_pas=correct_for_spines,
                    env=env, verbose=verbose)
    init_syn_mech_attrs(cell, env)
    config_biophys_cell_syns(env, gid, pop_name, insert=True, insert_netcons=True, insert_vecstims=True,
                             verbose=verbose)

    if verbose:
        report_topology(cell, env)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
