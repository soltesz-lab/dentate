import neuroml

def generate(root_dir, run_name, file_name):
    nml_doc = neuroml.NeuroMLDocument(id='Network_%s'%run_name)

    net = neuroml.Network(id="Network")
    nml_doc.networks.append(net)

    population_gids = {}

    cell_type = open('%s/celltype.dat'%root_dir, 'r')
    positions = open('%s/position.dat'%root_dir, 'r')

    for line in cell_type:
        words = line.split()
        if 'celltype' not in words[0]:
            id = words[0]
            start = int(words[3])
            end = int(words[4])
            gids = []
            for i in range(start, end+1):
                gids.append(i)
            population_gids[id] = gids

    xyz_locs = {}
    for line in positions:
        words = line.split()
        if 'cell' not in words[0]:
            xyz_locs[int(words[0])] = [words[1],words[2],words[3]]

    for id in population_gids.keys():
        gids = population_gids[id]
        if len(gids) > 0:
            pop = neuroml.Population(id="Pop_%s"%id, component=id, type="populationList")
            net.populations.append(pop)
            local_id = 0
            for gid in gids:
                inst = neuroml.Instance(id=local_id)
                pop.instances.append(inst)
                if xyz_locs.has_key(gid):
                    xyz = xyz_locs[gid]
                else:
                    xyz=[0,0,0]
                inst.location = neuroml.Location(x=xyz[0], y=xyz[1], z=xyz[2])
                local_id+=1

    import neuroml.writers as writers
    writers.NeuroMLWriter.write(nml_doc, '%s/%s'%(root_dir, file_name))

if __name__ == '__main__':
    generate('../results/none', 'none', 'network.nml')