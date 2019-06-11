import mpi4py
from mpi4py import MPI
from neuron import h

print(mpi4py.get_config())
print(mpi4py.get_include())

h.load_file("stdlib.hoc")
h.load_file("stdrun.hoc")
root = 0
pc = h.ParallelContext()
id = int(pc.id())
nhost = int(pc.nhost())
print("I am %i of %i" % (id, nhost))
v = h.Vector(1)
if id == root:
   v.x[0] = 17
pc.broadcast(v, root)
print(v.x[0])

