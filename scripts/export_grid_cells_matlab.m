function export_grid_cells_matlab (gridmodfile, prefix)

m = load(gridmodfile);

for i = 1:numel(m.GridCellModules)

    module = m.GridCellModules{i};
    dir = sprintf('%s/%02d', prefix, i)
    
    for j = 1:numel(module)
        filename = sprintf('%s/GridCell_%04d.dat', dir, j)
        fid = fopen (filename,'w+');
        data = module{j};
        fprintf(fid, '%d %d\n', size(data,1), size(data,2));
        fclose(fid);
        save('-ascii', '-append', filename, 'data');
    end
    
end
