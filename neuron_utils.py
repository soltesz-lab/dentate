import itertools
from collections import defaultdict
import sys, os.path, string
import numpy as np
import os.path


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

    

## Adds a network connection to a single synapse point process
##    pc: ParallelContext
##    nclist: list of netcons
##    srcgid: source gid
##    dstgid: target gid
##    syn: synapse point process
def mknetcon(pc, nclist, srcgid, dstgid, syn, weight, delay):
    nc = pc.gid_connect(srcgid, syn)
    nc.weight[0] = weight
    nc.delay = delay
    nclist.append(nc)


## A variant of ParallelNetManager.nc_append that takes in a
## synaptic point process as an argument, as opposed to the index of a
## synapse in cell.synlist) 
def nc_appendsyn(pc, nclist, srcgid, dstgid, syn, weight, delay):
    ## target in this subset
    ## source may be on this or another machine
    assert (pc.gid_exists(dstgid))
    if (pc.gid_exists(dstgid)):
        cell = pc.gid2cell(dstgid)
	mknetcon(pc, nclist, srcgid, dstgid, syn, weight/1000.0, delay)



## A variant of ParallelNetManager.nc_append that 1) takes in a
## synaptic point process as an argument (as opposed to the index of a
## synapse in cell.synlist) and 2) chooses the synaptic
## weight from a predefined vector of synaptic weights for this
## connection type
def nc_appendsyn_wgtvector(pc, nclist, srcgid, dstgid, syn, weights, delay):
    assert (pc.gid_exists(dstgid))
    cell = pc.gid2cell(dstgid)
    widx = int(dstgid % weights.size())
    mknetcon(pc, nclist, srcgid, dstgid, syn, weights.x[widx]/1000.0, delay)



## Create gap junctions
def mkgap(pc, gjlist, gid, secidx, sgid, dgid, w):
    
    cell = pc.gid2cell(gid)
    
    ##printf ("host %d: gap junction: gid = %d branch = %d sec = %d coupling = %g sgid = %d dgid = %d\n", pc.id, gid, branch, sec, w, sgid, dgid)
    
    sec = cell.sections[secidx]
    seg = sec(0.5)
    gj = ggap(seg)
    pc.source_var(seg._ref_v, sgid)
    pc.target_var(gj, gj._ref_vgap, dgid)
    
    gjlist.append(gj)
    gj.g = w

