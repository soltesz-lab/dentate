import os
try:
    from mpi4py import MPI  # Must come before importing NEURON
except Exception:
    pass
from dentate.utils import *
from neuron import h

# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)

freq = 100  # Hz, frequency at which AC length constant will be computed
d_lambda = 0.1  # no segment will be longer than this fraction of the AC length constant
default_ordered_sec_types = ['soma', 'hillock', 'ais', 'axon', 'basal', 'trunk', 'apical', 'tuft', 'spine_neck',
                             'spine_head']
default_hoc_sec_lists = {'soma': 'somaidx', 'hillock': 'hilidx', 'ais': 'aisidx', 'axon': 'axonidx',
                         'basal': 'basalidx', 'apical': 'apicalidx', 'trunk': 'trunkidx', 'tuft': 'tuftidx'}


def hoc_results_to_python(hoc_results):
    results_dict = {}
    for i in range(0, int(hoc_results.count())):
        vect = hoc_results.o(i)
        gid = int(vect.x[0])
        pyvect = vect.to_python()
        results_dict[gid] = pyvect[1:]
    hoc_results.remove_all()
    return results_dict


def simulate(v_init, mainlength, prelength=0, cvode=True):
    """

    :param h:
    :param v_init:
    :param prelength:
    :param mainlength:
    :param cvode:
    """
    h.cvode_active(1 if cvode else 0)
    h.finitialize(v_init)
    h.tstop = prelength + mainlength
    h.fadvance()
    h.continuerun(h.tstop)


def mknetcon(pc, source, syn, weight=1, delay=0.1):
    """
    Creates a network connection from the provided source to the provided synaptic point process.
    :param pc: :class:'h.ParallelContext'
    :param source: int; source gid
    :param syn: synapse point process
    :param delay: float
    :param weight: float
    :return: :class:'h.NetCon'
    """
    nc = pc.gid_connect(source, syn)
    nc.weight[0] = weight
    nc.delay = delay
    return nc


def mknetcon_vecstim(syn, delay=0.1, weight=1, source=None):
    """
    Creates a VecStim object to drive the provided synaptic point process, 
    and a network connection from the VecStim source to the synapse target.
    :param syn: synapse point process
    :param delay: float
    :param weight: float
    :return: :class:'h.NetCon', :class:'h.VecStim'
    """
    vs = h.VecStim()
    nc = h.NetCon(vs, syn)
    nc.weight[0] = weight
    nc.delay = delay
    return nc, vs


def mkgap(env, cell, gid, secpos, secidx, sgid, dgid, w):
    """
    Create gap junctions
    :param pc:
    :param gjlist:
    :param gid:
    :param secidx:
    :param sgid:
    :param dgid:
    :param w:
    :return:
    """

    sec = list(cell.sections)[secidx]
    seg = sec(secpos)
    gj = h.ggap(seg)
    gj.g = w

    env.pc.source_var(seg._ref_v, sgid, sec=sec)
    env.pc.target_var(gj, gj._ref_vgap, dgid)

    env.gjlist.append(gj)
    return gj


def find_template(env, template_name, path=['templates'], template_file=None, root=0):
    """
    Finds and loads a template located in a directory within the given path list.
    :param env: :class:'Env'
    :param template_name: str; name of hoc template
    :param path: list of str; directories to look for hoc template
    :param template_file: str; file_name containing definition of hoc template
    :param root: int; MPI.COMM_WORLD.rank
    """
    pc = env.pc
    rank = int(pc.id())
    found = False
    foundv = h.Vector(1)
    template_path = ''
    if template_file is None:
        template_file = '%s.hoc' % template_name
    if pc is not None:
        pc.barrier()
    if (pc is None) or (int(pc.id()) == root):
        for template_dir in path:
            if template_file is None:
                template_path = '%s/%s.hoc' % (template_dir, template_name)
            else:
                template_path = '%s/%s' % (template_dir, template_file)
            found = os.path.isfile(template_path)
            if found and (rank == root):
                logger.info('Loaded %s from %s' % (template_name, template_path))
                break
        foundv.x[0] = 1 if found else 0
    if pc is not None:
        pc.barrier()
        pc.broadcast(foundv, root)
    if foundv.x[0] > 0.0:
        s = h.ref(template_path)
        if pc is not None:
            pc.broadcast(s, root)
        h.load_file(s)
    else:
        raise Exception('find_template: template %s not found: file %s; path is %s' %
                        (template_name, template_file, str(path)))


def configure_hoc_env(env):
    """

    :param env: :class:'Env'
    """
    h.load_file("stdrun.hoc")
    h.load_file("loadbal.hoc")
    h('objref pc, nc, nil')
    h('strdef dataset_path')
    if hasattr(env, 'dataset_path'):
        h.dataset_path = env.dataset_path if env.dataset_path is not None else ""
    h.pc = h.ParallelContext()
    env.pc = h.pc
    h.dt = env.dt
    h.tstop = env.tstop
    if 'celsius' in env.globals:
        h.celsius = env.globals['celsius']
    ## more accurate integration of synaptic discontinuities
    if hasattr(h, 'nrn_netrec_state_adjust'):
        h.nrn_netrec_state_adjust = 1
    ## sparse parallel transfer
    if hasattr(h, 'nrn_sparse_partrans'):
        h.nrn_sparse_partrans = 1


def make_rec(recid, population, gid, cell, sec=None, loc=None, ps=None, param='v', label=None, dt=h.dt, description=''):
    """
    Makes a recording vector for the specified quantity in the specified section and location.

    :param recid: integer
    :param population: str
    :param gid: integer
    :param cell: :class:'BiophysCell'
    :param sec: :class:'HocObject'
    :param loc: float
    :param ps: :class:'HocObject'
    :param param: str
    :param dt: float
    :param ylabel: str
    :param description: str
    """
    vec = h.Vector()
    name = 'rec%i' % recid
    if (sec is None) and (loc is None) and (ps is not None):
        hocobj = ps
    elif (sec is not None) and (loc is not None):
        hocobj = sec(loc)
    else:
        raise RuntimeError('make_rec: either sec and loc or ps must be specified')
    if label is None:
        label = param
    vec.record(getattr(hocobj, '_ref_%s' % param), dt)
    rec_dict = {'name': name,
                'gid': gid,
                'cell': cell,
                'population': population,
                'loc': loc,
                'sec': sec,
                'description': description,
                'vec': vec,
                'label': label
                }

    return rec_dict


# Code by Michael Hines from this discussion thread:
# https://www.neuron.yale.edu/phpBB/viewtopic.php?f=31&t=3628
def cx(env):
    """
    Estimates cell complexity. Uses the LoadBalance class.

    :param env: an instance of the `dentate.Env` class.
    """
    rank = int(env.pc.id())
    lb = h.LoadBalance()
    if os.path.isfile("mcomplex.dat"):
        lb.read_mcomplex()
    cxvec = h.Vector(len(env.gidset))
    for i, gid in enumerate(env.gidset):
        cxvec.x[i] = lb.cell_complexity(env.pc.gid2cell(gid))
    env.cxvec = cxvec
    return cxvec
