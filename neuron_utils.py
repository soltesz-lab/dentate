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


## Create gap junctions
def mkgap(env, gid, secpos, secidx, sgid, dgid, w):
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
    cell = env.pc.gid2cell(gid)

    sec = list(cell.sections)[secidx]
    seg = sec(secpos)
    gj = h.ggap(seg)

    env.pc.source_var(seg._ref_v, sgid, sec=sec)
    env.pc.target_var(gj, gj._ref_vgap, dgid)
    
    gj.g = w

    return gj

def configure_hoc_env(env):
    """

    :param env: :class:'Env'
    """
    h.load_file("nrngui.hoc")
    h.load_file("loadbal.hoc")
    h('objref pc, nc, nil')
    h('strdef datasetPath')
    h.datasetPath = env.datasetPath
    h.pc = h.ParallelContext()
    env.pc = h.pc
    ## polymorphic value template
    h.load_file(env.hoclibPath + '/templates/Value.hoc')
    ## randomstream template
    h.load_file(env.hoclibPath + '/templates/ranstream.hoc')
    ## stimulus cell template
    h.load_file(env.hoclibPath + '/templates/StimCell.hoc')
    h.xopen(env.hoclibPath + '/lib.hoc')
    h.dt = env.dt
    h.tstop = env.tstop

    h('objref templatePaths, templatePathValue')
    h.templatePaths = h.List()
    for path in env.templatePaths:
        h.templatePathValue = h.Value(1, path)
        h.templatePaths.append(h.templatePathValue)
