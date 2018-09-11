from mpi4py import MPI
from neuron import h
from neuroh5 import io
root = 0
pc = h.ParallelContext()
id = int(pc.id())
nhost = int(pc.nhost())
print ("I am", id, "of", nhost)
v = h.Vector(1)
if id == root:
   v.x[0] = 17
pc.broadcast(v, root)
print v.x[0]

