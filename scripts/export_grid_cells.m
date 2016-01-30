#! /usr/bin/octave -qf
args = argv ();

m = load(args{1});

for i = 1:numel(m.GridCellModules)

    module = m.GridCellModules{i};
    dir = sprintf('%02d', i)
    mkdir(dir);
    
    for j = 1:numel(module)
        filename = sprintf('%s/GridCell_%04d.dat', dir, j)
        fid = fopen (filename,'w+');
        data = module{j};
        fprintf(fid, '%d %d\n', size(data,1), size(data,2));
        fclose(fid);
        save('-ascii', '-append', filename, 'data');
    end
    
end

