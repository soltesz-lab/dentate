
import numpy as np
import networkx as nx
import sys

filename = sys.argv[1]
f = open(filename, 'r')

lines = []
for line in f.readlines():
    lines.append(line.strip('\n').split())
f.close()


graph = nx.DiGraph()
data_dict = {}

for line in lines:
    point_id, swc_type, x, y, z, radius, parent = line
    point_id = int(point_id)
    parent   = int(parent)
    swc_type = int(swc_type)
    x, y, z, radius  = float(x), float(y), float(z), float(radius)
    if swc_type == 7: continue

    data_dict[point_id] = [swc_type,x,y,z,radius,parent]
    if parent >= 0:
        graph.add_edge(parent, point_id)

outfilename = sys.argv[2]
f = open(outfilename, 'w')

ordered_nodes = list(nx.topological_sort(graph))
sorted_idx    = {}

for (i,node) in enumerate(ordered_nodes):
    sorted_idx[node] = i

for node in ordered_nodes:
    node_info = data_dict[node]
    if node_info[-1] == -1: 
        re_node = sorted_idx[node]
        re_parent = node_info[-1]
    else: re_node, re_parent = sorted_idx[node], sorted_idx[node_info[-1]]
    line = [re_node] + node_info[:-1] + [re_parent]
    f.write(' '.join(map(str, line))+'\n')
f.close()


