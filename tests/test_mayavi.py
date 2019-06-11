import numpy as np
from mayavi import mlab


x, y, z = np.random.random((3, 100))
data = x**2 + y**2 + z**2

pts = np.column_stack((x, y, z, data)).T
print(pts)
print(pts.shape)
src = mlab.pipeline.scalar_scatter(*pts)

field = mlab.pipeline.delaunay3d(src)

field.filter.offset = 999    # seems more reliable than the default
edges = mlab.pipeline.extract_edges(field)

mlab.pipeline.surface(edges, opacity=0.3, line_width=3)

fig = mlab.gcf()
fig.scene.camera.trait_set(parallel_projection=1)

mlab.show()
