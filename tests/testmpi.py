from mpi4py import MPI
from neuron import h
pc = h.ParallelContext()
id = int(pc.id())
nhost = int(pc.nhost())
print "I am", id, "of", nhost

