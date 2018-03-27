import matplotlib.pyplot as plt
import dentate
from dentate import stimulus

nodes,vert,smp = stimulus.generate_spatial_offsets(1000,scale_factor=6.0,maxit=40)
fig,ax = plt.subplots(figsize=(6,6))
# plot the domain
for s in smp:
    ax.plot(vert[s,0],vert[s,1],'k-')
ax.plot(nodes[:,0],nodes[:,1],'bo')
ax.set_aspect('equal')
fig.tight_layout()
plt.show()
