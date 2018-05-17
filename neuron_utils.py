from dentate.utils import *
try:
    from mpi4py import MPI  # Must come before importing NEURON
except Exception:
    pass
from neuron import h


freq = 100      # Hz, frequency at which AC length constant will be computed
d_lambda = 0.1  # no segment will be longer than this fraction of the AC length constant


def lambda_f(sec, f=freq):
    """
    Calculates the AC length constant for the given section at the frequency f
    Used to determine the number of segments per hoc section to achieve the desired spatial and temporal resolution
    :param sec : :class:'h.Section'
    :param f : int
    :return : int
    """
    diam = np.mean([seg.diam for seg in sec])
    Ra = sec.Ra
    cm = np.mean([seg.cm for seg in sec])
    return 1e5*math.sqrt(diam/(4.*math.pi*f*Ra*cm))


def d_lambda_nseg(sec, lam=d_lambda, f=freq):
    """
    The AC length constant for this section and the user-defined fraction is used to determine the maximum size of each
    segment to achieve the d esired spatial and temporal resolution. This method returns the number of segments to set
    the nseg parameter for this section. For tapered cylindrical sections, the diam parameter will need to be
    reinitialized after nseg changes.
    :param sec : :class:'h.Section'
    :param lam : int
    :param f : int
    :return : int
    """
    L = sec.L
    return int((L/(lam*lambda_f(sec, f))+0.9)/2)*2+1


def hoc_results_to_python(hoc_results):
    results_dict = {}
    for i in xrange(0, int(hoc_results.count())):
        vect   = hoc_results.o(i)
        gid    = int(vect.x[0])
        pyvect = vect.to_python()
        results_dict[gid] = pyvect[1:]
    hoc_results.remove_all()
    return results_dict


def write_results(results, filepath, header):
    f = open(filepath,'w')
    f.write(header+'\n')
    for item in results:
        for (gid, vect) in item.iteritems():
            f.write (str(gid)+"\t")
            f.write (("\t".join(['{:0.3f}'.format(i) for i in vect])) + "\n")
    f.close()

    
def simulate(h, v_init, prelength, mainlength):
    h.cvode_active (1)
    h.finitialize(v_init)
    h.tstop = prelength+mainlength
    h.fadvance()
    h.continuerun(h.tstop)


def mknetcon(pc, srcgid, dstgid, syn, delay=0.1, weight=1):
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


def mkgap(pc, gjlist, gid, secidx, sgid, dgid, w):
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
    cell = pc.gid2cell(gid)
    
    ##printf ("host %d: gap junction: gid = %d branch = %d sec = %d coupling = %g sgid = %d dgid = %d\n", pc.id, gid, branch, sec, w, sgid, dgid)
    
    sec = cell.sections[secidx]
    seg = sec(0.5)
    gj = ggap(seg)
    pc.source_var(seg._ref_v, sgid)
    pc.target_var(gj, gj._ref_vgap, dgid)
    
    gjlist.append(gj)
    gj.g = w

