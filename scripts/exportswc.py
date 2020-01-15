## Instantiates a hoc cell and exports its 3d points to SWC format

import sys
from neuron import h, gui
from collections import defaultdict

h.load_file("nrngui.hoc")
h.load_file("import3d.hoc")

h('objref nil')

def export_swc(sections=[("soma",1),("dend",4),("bas",3),("axon",7)]):
    swc_point_idx = 0
    swc_points = []
    swc_point_sec_dict = defaultdict(list)
    sec_dict = {}
    for section, sectype in sections:
        if hasattr(h, section):
            seclist = list(getattr(h, section))
            for secidx, sec in enumerate(seclist):
                if hasattr(sec, 'sec'):
                    sec = sec.sec
                n3d = sec.n3d()
                if n3d == 2:
                    x1 = sec.x3d(0)
                    y1 = sec.y3d(0)
                    z1 = sec.z3d(0)
                    d1 = sec.diam3d(0)
                    x2 = sec.x3d(1)
                    y2 = sec.y3d(1)
                    z2 = sec.z3d(1)
                    d2 = sec.diam3d(1)
                    mx = (x2 + x1) / 2.
                    my = (y2 + y1) / 2.
                    mz = (z2 + z1) / 2.
                    dd = d1 - (d1 - d2)/2.
                    sec.pt3dinsert(1, mx, my, mz, dd)
                    n3d = sec.n3d()
                L = sec.L
                for i in range(n3d):
                    x = sec.x3d(i)
                    y = sec.y3d(i)
                    z = sec.z3d(i)
                    d = sec.diam3d(i)
                    ll = sec.arc3d(i)
                    rad = d / 2.
                    loc = ll / L
                    first = True if i == 0 else False
                    swc_point = (swc_point_idx, sectype, x, y, z, rad, loc, sec, first)
                    swc_points.append(swc_point)
                    swc_point_sec_dict[sec.name()].append(swc_point)
                    swc_point_idx += 1
    for swc_point in swc_points:
        (swc_point_idx, sectype, x, y, z, rad, loc, sec, first) = swc_point
        parent_idx = -1
        if not first:
            parent_idx = swc_point_idx-1
        else:
            parent_seg = sec.parentseg()
            if parent_seg is not None:
                parent_x = parent_seg.x
                parent_sec = parent_seg.sec
                parent_points = swc_point_sec_dict[parent_sec.name()]
                for parent_point in parent_points:
                    (parent_point_idx, _, _, _, _, _, parent_point_loc, _, _) = parent_point
                    if parent_point_loc >= parent_x:
                        parent_idx = parent_point_idx
                        break
        print("%d %i %.04f %.04f %.04f %.04f %d" % (swc_point_idx, sectype, x, y, z, rad, parent_idx))
    
                    
h.load_file(sys.argv[1])
h.topology()
export_swc()



