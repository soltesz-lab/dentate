
import numpy as np
import sys
from dentate.utils import *

sys.path.insert(0,'/home/dhadjia1/soltesz-lab/ca1/test')
import plot_helper as ph

class Cell(object):
    def __init__(self, x, y, z, r, parent):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.parent = parent

def plot_points(cell_info):
    colors = ['r','b','g']
    fig, ax = ph.init()
    for k in cell_info:
        xc, yc, zc = [], [], []
        color = colors[k-1]
        cells = cell_info[k]
        for cell in cells:
            x,y,z = cell.x, cell.y, cell.z
            xc.append(x), yc.append(y), zc.append(z)
        ph.points2d(ax, xc, yc,color=color)
    ph.show()

def plot_bounds(ub, lb, fig=None, ax=None, first=True, show=True):
    if first:
        fig, ax = ph.init()
    x1, x2 = ub[0], ub[0]
    y1, y2 = ub[1], lb[1]

    ax.plot([x1, x2], [y1, y2], 'g-')

    x1, x2 = lb[0], lb[0]
    y1, y2 = ub[1], lb[1]

    ax.plot([x1, x2], [y1, y2], 'g-')

    x1, x2 = ub[0], lb[0]
    y1, y2 = ub[1], ub[1]

    ax.plot([x1, x2], [y1, y2], 'g-')
 
    x1, x2 = ub[0], lb[0]
    y1, y2 = lb[1], lb[1]

    ax.plot([x1, x2], [y1, y2], 'g-')

    if show:
        ph.show()
    else:
        return fig, ax

def plot_pc_with_eigenvector(xy, eigenvalues, eigenvectors,fig=None, ax=None, first=True, show=True):
    if first:
        fig, ax = ph.init()
    ph.points2d(ax,xy[:,0],xy[:,1],color='r')
    zero = np.zeros(eigenvalues.shape[0])

    eigenvalues[0] = eigenvalues[0] ** 0.5
    eigenvalues[1] = eigenvalues[1] ** 0.5


    x1, x2 = zero[0], eigenvalues[0]*eigenvectors[0,0]
    y1, y2 = zero[1], eigenvalues[0]*eigenvectors[0,1]

    ax.plot([x1,x2],[y1,y2],'b-')

    x1, x2 = zero[0], eigenvalues[1]*eigenvectors[1,0]
    y1, y2 = zero[1], eigenvalues[1]*eigenvectors[1,1]
 
    ax.plot([x1, x2], [y1, y2],'b-')
    if show:
        ph.show()
    else:
        return fig, ax

def key2pc(cell_info, key):
    x,y,z = [], [], []
    for k, v in viewitems(cell_info):
        if key is None or k == int(key):
            for cell in v:
                x.append(cell.x), y.append(cell.y), z.append(cell.z)
    return x,y,z

def pca_bounding(cell_info, key=None):
    from sklearn.decomposition import PCA
    #from sklearn.preprocessing import normalize

    x,y,z = key2pc(cell_info, key)
    xy = np.array([x,y])
    xy = np.transpose(xy)
 
    pca = PCA(n_components=2)
    xy_transform = pca.fit_transform(xy)
    
    #for (eigenvalue, eigenvector) in zip(pca.explained_variance_, pca.components_):
    #    print eigenvalue, eigenvector 

    xy_center = xy - np.mean(xy,axis=0)
    return xy_center, pca.explained_variance_,pca.components_

def info2bounds(cell_info, key=None):
    x,y,z = key2pc(cell_info, key)
    xy = np.array([x,y])
    return bounds(xy,axis=1)

def bounds(xy, axis=0):
    return np.max(xy,axis=axis), np.min(xy,axis=axis)
    
def read_swc(fn):
    f = open('ex_gc.txt','r')
    cell_info = {}

    for line in f.readlines():
        line = line.strip('\n').split()
        if line[0] == '#':
            continue
        sid = int(line[1])
        x, y, z, r, p = float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6])
        if sid in cell_info.keys():
            cell_info[sid].append(Cell(x,y,z,r,p))
        else:
            cell_info[sid] = [Cell(x,y,z,r,p)]
    return cell_info


def run():
    fn = 'ex_gc.txt'
    cell_info = read_swc(fn)
    ncells = reduce(lambda x,y: x+y, [len(v) for _, v in viewitems(cell_info)])
    print('Number of points in morphology is %d' % (ncells))


    xy_center, eigenvalues, eigenvectors = pca_bounding(cell_info,key='3') # acquired directions of variance via pca
    ub, lb = bounds(xy_center,axis=0)
    fig, ax = plot_bounds(ub, lb, first=True, show=False)

    ncells = xy_center.shape[0]
    print('Number of points post transformation is %d' % (ncells))
    plot_pc_with_eigenvector(xy_center,eigenvalues, eigenvectors,fig=fig,ax=ax,first=False,show=True)


    


if __name__ == '__main__':
    run()
