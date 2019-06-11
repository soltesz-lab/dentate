
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def init():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax

def points3d(ax,x,y,z,color='r'):
    ax.scatter(x,y,z,c=color)

def points2d(ax,x,y,color='r'):
    ax.scatter(x,y,c=color)

def plot_pc(ax, points,color='o'):
    ax.plot(points[:,0], points[:,1], points[:,2],color)

def plot_delaunay(ax, tri):
    points = tri.points
    simplices = tri.simplices
    ax.plot_trisurf(tri)
    #ax.plot(points[:,0], points[:,1], points[:,2],'o')

def plot_boundaries(ax, bnds,color='-b'):

    for bnd in bnds:
        p1 = bnd[0,:]
        p2 = bnd[1,:]
        p3 = bnd[2,:]

        c1 = np.transpose([p1, p2])
        c2 = np.transpose([p1, p3])
        c3 = np.transpose([p2, p3])

        ax.plot(c1[0], c1[1], c1[2], color)
        ax.plot(c2[0], c2[1], c2[2], color)
        ax.plot(c3[0], c3[1], c3[2], color)

def plot_tetrahedron(ax, points, simplices, num=100):

    triangles = points[simplices]
    triangles = triangles[0:num,:,:]
    for construct in triangles:
        p1 = construct[0,:]
        p2 = construct[1,:]
        p3 = construct[2,:]
        p4 = construct[3,:]

        c1 = np.transpose([p1, p2])
        c2 = np.transpose([p1, p3])
        c3 = np.transpose([p1, p4])
        c4 = np.transpose([p2, p3])
        c5 = np.transpose([p2, p4])
        c6 = np.transpose([p3, p4])

        ax.plot(c1[0], c1[1], c1[2], '-r')
        ax.plot(c2[0], c2[1], c2[2], '-r')
        ax.plot(c3[0], c3[1], c3[2], '-r')
        ax.plot(c4[0], c4[1], c4[2], '-r')
        ax.plot(c5[0], c5[1], c5[2], '-r')
        ax.plot(c6[0], c6[1], c6[2], '-r')

        


def show():
    plt.show()
