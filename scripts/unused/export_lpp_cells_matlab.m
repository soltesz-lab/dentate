function export_lpp_cells_matlab (lppmodfile, prefix)

m = load(lppmodfile);

for i = 1:numel(m.LPPCellModules)

    module = m.LPPCellModules{i};
    dir = sprintf('%s/%02d', prefix, i)
    
    for j = 1:numel(module)
        filename = sprintf('%s/LPPCell_%04d.dat', dir, j)
        fid = fopen (filename,'w+');
        data = module{j};
        fprintf(fid, '%d %d\n', size(data,1), size(data,2));
        fclose(fid);
        save('-ascii', '-append', filename, 'data');
    end
    
end
