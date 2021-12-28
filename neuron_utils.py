import os, os.path
try:
    from mpi4py import MPI  # Must come before importing NEURON
except Exception:
    pass
from dentate.utils import *
from neuron import h
from scipy import interpolate

# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)

freq = 100  # Hz, frequency at which AC length constant will be computed
d_lambda = 0.1  # no segment will be longer than this fraction of the AC length constant
default_ordered_sec_types = ['soma', 'hillock', 'ais', 'axon', 'basal', 'trunk', 'apical', 'tuft', 'spine_neck',
                             'spine_head']
default_hoc_sec_lists = {'soma': 'somaidx', 'hillock': 'hilidx', 'ais': 'aisidx', 'axon': 'axonidx',
                         'basal': 'basalidx', 'apical': 'apicalidx', 'trunk': 'trunkidx', 'tuft': 'tuftidx'}
IzhiCellAttrs = namedtuple('IzhiCellAttrs', ['C', 'k', 'vr', 'vt', 'vpeak', 'a', 'b', 'c', 'd', 'celltype'])
default_izhi_cell_attrs_dict = {
    'RS': IzhiCellAttrs(C=1., k=0.7, vr=-65., vt=-50., vpeak=35., a=0.03, b=-2., c=-55., d=100., celltype=1),
    'IB': IzhiCellAttrs(C=1.5, k=1.2, vr=-75., vt=-45., vpeak=50., a=0.01, b=5., c=-56., d=130., celltype=2),
    'CH': IzhiCellAttrs(C=0.5, k=1.5, vr=-60., vt=-40., vpeak=25., a=0.03, b=1., c=-40., d=150., celltype=3),
    'LTS': IzhiCellAttrs(C=1.0, k=1.0, vr=-56., vt=-42., vpeak=40., a=0.03, b=8., c=-53., d=20., celltype=4),
    'FS': IzhiCellAttrs(C=0.2, k=1., vr=-55., vt=-40., vpeak=25., a=0.2, b=-2., c=-45., d=-55., celltype=5),
    'TC': IzhiCellAttrs(C=2.0, k=1.6, vr=-60., vt=-50., vpeak=35., a=0.01, b=15., c=-60., d=10., celltype=6),
    'RTN': IzhiCellAttrs(C=0.4, k=0.25, vr=-65., vt=-45., vpeak=0., a=0.015, b=10., c=-55., d=50., celltype=7)
}

PRconfig = namedtuple('PRconfig', ['pp', 'Ltotal', 'gc',
                                   'soma_gmax_Na', 
                                   'soma_gmax_K',
                                   'soma_g_pas',
                                   'dend_gmax_Ca',
                                   'dend_gmax_KCa',
                                   'dend_gmax_KAHP',
                                   'dend_g_pas',
                                   'dend_d_Caconc',
                                   'global_cm',
                                   'global_diam',
                                   'ic_constant',
                                   'cm_ratio',
                                   'e_pas',
                                   'V_rest',
                                   'V_threshold'])

HocCellInterface = namedtuple('HocCellInterface', ['sections', 'is_art', 'is_reduced', 'soma', 'hillock', 'ais', 'axon', 'basal', 'apical', 'all', 'state'])

def hoc_results_to_python(hoc_results):
    results_dict = {}
    for i in range(0, int(hoc_results.count())):
        vect = hoc_results.o(i)
        gid = int(vect.x[0])
        pyvect = vect.to_python()
        results_dict[gid] = pyvect[1:]
    hoc_results.remove_all()
    return results_dict


def simulate(v_init, mainlength, prelength=0, use_cvode=True):
    """

    :param h:
    :param v_init:
    :param prelength:
    :param mainlength:
    :param cvode:
    """
    h.cvode_active(1 if use_cvode else 0)
    h.cvode.use_fast_imem(1)
    h.cvode.cache_efficient(1)
    h.secondorder = 2
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


def find_template(env, template_name, path=['templates'], template_file=None, bcast_template=False, root=0):
    """
    Finds and loads a template located in a directory within the given path list.
    :param env: :class:'Env'
    :param template_name: str; name of hoc template
    :param path: list of str; directories to look for hoc template
    :param template_file: str; file_name containing definition of hoc template
    :param root: int; MPI.COMM_WORLD.rank
    """
    if env.comm is None:
        bcast_template = False
    rank = env.comm.rank if env.comm is not None else 0
    found = False
    template_path = ''
    if template_file is None:
        template_file = f'{template_name}.hoc'
    if bcast_template:
        env.comm.barrier()
    if (env.comm is None) or (not bcast_template) or (bcast_template and (rank == root)):
        for template_dir in path:
            if template_file is None:
                template_path = f'{template_dir}/{template_name}.hoc'
            else:
                template_path = f'{template_dir}/{template_file}'
            found = os.path.isfile(template_path)
            if found and (rank == 0):
                logger.info(f'Loaded {template_name} from {template_path}')
                break
    if bcast_template:
        found = env.comm.bcast(found, root=root)
        env.comm.barrier()
    if found:
        if bcast_template:
            template_path = env.comm.bcast(template_path, root=root)
            env.comm.barrier()
        h.load_file(template_path)
    else:
        raise Exception(f'find_template: template {template_name} not found: '
                        f'file {template_file}; path is {path}')


def configure_hoc_env(env, bcast_template=False):
    """

    :param env: :class:'Env'
    """
    h.load_file("stdrun.hoc")
    h.load_file("loadbal.hoc")
    for template_dir in env.template_paths:
        path = f"{template_dir}/rn.hoc"
        if os.path.exists(path):
            h.load_file(path)
    h.cvode.use_fast_imem(1)
    h.cvode.cache_efficient(1)
    h.secondorder = 2
    h('objref pc, nc, nil')
    h('strdef dataset_path')
    if hasattr(env, 'dataset_path'):
        h.dataset_path = env.dataset_path if env.dataset_path is not None else ""
    if env.use_coreneuron:
        from neuron import coreneuron
        coreneuron.enable = True
        coreneuron.verbose = 1 if env.verbose else 0
    h.pc = h.ParallelContext()
    h.pc.gid_clear()
    env.pc = h.pc
    h.dt = env.dt
    h.tstop = env.tstop
    env.t_vec = h.Vector()  # Spike time of all cells on this host
    env.id_vec = h.Vector()  # Ids of spike times on this host
    env.t_rec = h.Vector() # Timestamps of intracellular traces on this host
    if 'celsius' in env.globals:
        h.celsius = env.globals['celsius']
    ## more accurate integration of synaptic discontinuities
    if hasattr(h, 'nrn_netrec_state_adjust'):
        h.nrn_netrec_state_adjust = 1
    ## sparse parallel transfer
    if hasattr(h, 'nrn_sparse_partrans'):
        h.nrn_sparse_partrans = 1


def load_cell_template(env, pop_name, bcast_template=False):
    """
    :param pop_name: str
    """
    if pop_name in env.template_dict:
        return env.template_dict[pop_name]
    rank = env.comm.Get_rank()
    if not (pop_name in env.celltypes):
        raise KeyError(f'load_cell_templates: unrecognized cell population: {pop_name}')
    template_name = env.celltypes[pop_name]['template']
    if 'template file' in env.celltypes[pop_name]:
        template_file = env.celltypes[pop_name]['template file']
    else:
        template_file = None
    if not hasattr(h, template_name):
        find_template(env, template_name, template_file=template_file, path=env.template_paths, bcast_template=bcast_template)
    assert (hasattr(h, template_name))
    template_class = getattr(h, template_name)
    env.template_dict[pop_name] = template_class
    return template_class


def make_rec(recid, population, gid, cell, sec=None, loc=None, ps=None, param='v', label=None, dt=None, description=''):
    """
    Makes a recording vector for the specified quantity in the specified section and location.

    :param recid: str
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
    if (sec is None) and (loc is None) and (ps is not None):
        hocobj = ps
        seg = ps.get_segment()
        if seg is not None:
            loc = seg.x
            sec = seg.sec
            origin = list(cell.soma)[0]
            distance = h.distance(origin(0.5), seg)
            ri = h.ri(loc, sec=sec)
        else:
            distance = None
            ri = None
    elif (sec is not None) and (loc is not None):
        hocobj = sec(loc)
        if cell.soma.__class__.__name__.lower() == "section":
            origin = cell.soma
        else:
            origin = list(cell.soma)[0]
        h.distance(sec=origin)
        distance = h.distance(loc, sec=sec)
        ri = h.ri(loc, sec=sec)
    else:
        raise RuntimeError('make_rec: either sec and loc or ps must be specified')
    section_index = None
    if sec is not None:
        for i, this_section in enumerate(cell.sections):
            if this_section == sec:
                section_index = i
                break
    if label is None:
        label = param
    if dt is None:
        vec.record(getattr(hocobj, f'_ref_{param}'))
    else:
        vec.record(getattr(hocobj, f'_ref_{param}'), dt)
    rec_dict = {'name': recid,
                'gid': gid,
                'cell': cell,
                'population': population,
                'loc': loc,
                'section': section_index,
                'distance': distance,
                'ri': ri,
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
    cxvec = np.zeros((len(env.gidset),))
    for i, gid in enumerate(env.gidset):
        cxvec[i] = lb.cell_complexity(env.pc.gid2cell(gid))
    env.cxvec = cxvec
    return cxvec


def interplocs(sec, locs, return_interpolant=False):
    """Computes xyz coords of locations in a section whose topology & geometry are defined by pt3d data.
    Based on code by Ted Carnevale.
    """
    nn = sec.n3d()
    assert(nn > 1)

    xx = h.Vector(nn)
    yy = h.Vector(nn)
    zz = h.Vector(nn)
    dd = h.Vector(nn)
    ll = h.Vector(nn)
    
    for ii in range(0, nn):
        xx.x[ii] = sec.x3d(ii)
        yy.x[ii] = sec.y3d(ii)
        zz.x[ii] = sec.z3d(ii)
        dd.x[ii] = sec.diam
        ll.x[ii] = sec.arc3d(ii)
        
    ## normalize length
    ll.div(ll.x[nn - 1])

    xx = np.array(xx)
    yy = np.array(yy)
    zz = np.array(zz)
    dd = np.array(dd)
    ll = np.array(ll)

    u, indices = np.unique(ll, return_index=True)
    indices = np.asarray(indices)
    if len(u) < len(ll):
        ll = ll[indices]
        xx = xx[indices]
        yy = yy[indices]
        zz = zz[indices]
        dd = dd[indices]

    pch_x = interpolate.pchip(ll, xx)
    pch_y = interpolate.pchip(ll, yy)
    pch_z = interpolate.pchip(ll, zz)
    pch_diam = interpolate.pchip(ll, dd)

    if return_interpolant:
        return pch_x, pch_y, pch_z, pch_diam
    else:
        res = np.asarray([(pch_x(loc), pch_y(loc), pch_z(loc), pch_diam(loc)) for loc in locs], dtype=np.float32)
    return res


def calcRinp(cell, record_dt = 0.1, dt = 0.0125, celsius = 36., use_cvode=True):

    h.cvode.use_fast_imem(1)
    h.cvode.cache_efficient(1)
    h.secondorder = 2
    h.dt = dt

    if record_dt < dt:
        record_dt = dt

    # Enable variable time step solver
    if use_cvode:
        h.cvode.active(1)

    h.celsius = celsius
    h.tstop = 10000

    vec_t = h.Vector()
    vec_v = h.Vector()
    vec_t.record(h._ref_t, record_dt) # Time
    vec_v.record(list(cell.soma)[0](0.5)._ref_v, record_dt) # Voltage

    # Put an IClamp at the soma
    stim = h.IClamp(0.5, sec=list(cell.soma)[0])
    stim.delay = h.tstop / 2 # Stimulus start
    stim.dur = 1000. # Stimulus length
    stim.amp = 0.00005 # strength of current injection

    h.init()
    h.run()

    t = np.asarray(vec_t)
    v = np.asarray(vec_v)
    
    return (v[np.argwhere(t >= stim.delay+0.999*stim.dur)][0] - v[np.argwhere(t >= stim.delay-0.5*stim.dur)][0])/stim.amp
