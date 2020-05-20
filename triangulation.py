"""Triangulation-related routines."""

import numpy as np

def area_normals(simplices, points):
    """
    Compute the area and normal vectors of the surface triangles.
    The normal vectors are normalized. The area is always positive.
    """
    areas, fnormals = geomtools.areaNormals(points[simplices])
    return areas, fnormals


def free_boundary(simplices):
    """
    Returns the facets that are referenced only by simplex of the given triangulation.
    """

    ## Sort the facet indices in the triangulation
    simplices = np.sort(simplices, axis=1)
    facets = np.vstack((simplices[:, [0, 1, 2]], \
                        simplices[:, [0, 1, 3]], \
                        simplices[:, [0, 2, 3]], \
                        simplices[:, [1, 2, 3]]))

    ## Find unique facets                    
    ufacets, counts = np.unique(facets, return_counts=True, axis=0)

    ## Determine which facets are part of only one simplex
    bidxs = np.where(counts == 1)[0]

    if len(bidxs) == 0:
        raise RuntimeError("tri.free_boundary: unable to determine facets that belong only to one simplex")
    return ufacets[bidxs]


def edges(simplices):
    """
    Returns triangulation edges as a two-column matrix of vertex ids. 
    """
    return simplices.reshape((-1,2))


def edge_cosangles(simplices, points, edges, return_mask=False):
    """
    Returns the cos of the angles over all edges.
    Edges adjacent to only one element get cosangles = 1.0.
    If return_mask == True, a second return value is a boolean array
    with the edges that connect two faces.
    """
    # get connections of edges to faces
    conn = edges
    
    if conn.shape[1] > 2:
        raise RuntimeError("The surface is not a manifold")
    
    # get normals on all faces
    _,n = area_normals(simplices, points)
    
    # Flag edges that connect two faces
    connecting = (conn >= 0).sum(axis=-1) == 2
    
    # get adjacent facet normals for 2-connected edges
    n = n[conn[connecting]]
    
    # compute cosinus of angles over 2-connected edges
    cosa = np.dot(n[:,0], n[:,1])
    
    # Initialize cosangles to all 1. values
    cosangles = ones((conn.shape[0],))
    
    # Fill in the values for the 2-connected edges
    cosangles[connecting] = cosa
    
    # Clip to the -1...+1. range
    cosangles = cosangles.clip(min=-1.,max=1.)
    
    # Return results
    if return_mask:
        return cosangles, connecting
    else:
        return cosangles

    
def feature_edges(bnd, points, theta=60.):
    """
    Returns the feature edges of the surface.
    Feature edges are edges that are prominent features of the geometry.
    They are either border edges or edges where the normals on the two
    adjacent triangles differ more than a given angle.
    The non feature edges then represent edges on a rather smooth surface.

    - `bnd`: returned by free_boundary
    - `theta`: The angle by which the normals on adjacent triangles
      should differ in order for the edge to be marked as a feature.
    
    Returns a boolean array with shape (nedg,) where the feature angles
    are marked with True.
    """
    
    # Get the edge angles
    cosangles, connecting = edge_cosangles(edges, points, return_mask=True)
    
    # initialize all edges as features
    feature = ones((edges.shape[0],),dtype=bool)
    
    # unmark edges with small angle
    feature[connecting] = cosangles[connecting] <= cosd(angle)
    
    return features

