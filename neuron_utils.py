import os
from dentate.utils import *
try:
    from mpi4py import MPI  # Must come before importing NEURON
except Exception:
    pass
from neuron import h

# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)

freq = 100      # Hz, frequency at which AC length constant will be computed
d_lambda = 0.1  # no segment will be longer than this fraction of the AC length constant
default_ordered_sec_types = ['soma', 'hillock', 'ais', 'axon', 'basal', 'trunk', 'apical', 'tuft', 'spine_neck',
                             'spine_head']
default_hoc_sec_lists = {'soma': 'somaidx', 'hillock': 'hilidx', 'ais': 'aisidx', 'axon': 'axonidx',
                          'basal': 'basalidx', 'apical': 'apicalidx', 'trunk': 'trunkidx', 'tuft': 'tuftidx'}


def hoc_results_to_python(hoc_results):
    results_dict = {}
    for i in range(0, int(hoc_results.count())):
        vect   = hoc_results.o(i)
        gid    = int(vect.x[0])
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
    h.cvode_active (1 if cvode else 0)
    h.finitialize(v_init)
    h.tstop = prelength+mainlength
    h.fadvance()
    h.continuerun(h.tstop)


def mknetcon(pc, srcgid, dstgid, syn, weight, delay):
    """
    Creates a network connection from the provided source to the provided synaptic point process.
    :param pc: :class:'h.ParallelContext'
    :param srcgid: int; source gid
    :param dstgid: int; destination gid
    :param syn: synapse point process
    :param delay: float
    :param weight: float
    :return: :class:'h.NetCon'
    """
    assert pc.gid_exists(dstgid)
    nc = pc.gid_connect(srcgid, syn)
    nc.weight[0] = weight
    nc.delay = delay
    return nc


def mknetcon_vecstim(syn, delay=0.1, weight=1):
    """
    Creates a VecStim object to drive the provided synaptic point process, and a network connection from the VecStim
    source to the synapse target.
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
        raise Exception('find_template: template %s not found: file %s; path is %s' % (template_name, template_file, str(path)))


def configure_hoc_env(env):
    """

    :param env: :class:'Env'
    """
    h.load_file("nrngui.hoc")
    h.load_file("loadbal.hoc")
    h('objref pc, nc, nil')
    h('strdef datasetPath')
    if hasattr(env,'datasetPath'):
        h.datasetPath = env.datasetPath if env.datasetPath is not None else ""
    h.pc = h.ParallelContext()
    env.pc = h.pc
    h.dt = env.dt
    h.tstop = env.tstop


def make_rec(recid, population, gid, cell, sec, dt=h.dt, loc=None, param='v', description=''):
        """
        Makes a recording vector for the specified quantity in the specified section and location.

        :param recid: integer
        :param population: str
        :param gid: integer
        :param cell: :class:'BiophysCell'
        :param sec: :class:'HocObject'
        :param dt: float
        :param loc: float
        :param param: str
        :param ylabel: str
        :param description: str
        """
        vec = h.Vector()
        name = 'rec%i' % recid
        if loc is None:
           loc = 0.5
        vec.record(getattr(sec(loc), '_ref_%s' % param), dt)
        rec_dict = { 'name': name,
                     'gid': gid,
                     'cell': cell,
                     'population': population,
                     'loc': loc,
                     'sec': sec,
                     'description': description,
                     'vec': vec
                    }
                
        return rec_dict
