
m = load('GridCellModules.mat');
sz = numel(m.GridCellModules) * numel(m.GridCellModules{1})
indices = zeros(sz,4);

relk = 1
k = 1049660

for i = 1:numel(m.GridCellModules)

    i
    module = m.GridCellModules{i};
    
    for j = 1:numel(module)
        indices(relk,1) = k;
        indices(relk,2:4) = module{j}(1,:);
        k = k+1;
        relk = relk+1;
        
    end
    
end

filename='MPP.dat';
fid = fopen (filename,'w+');
fprintf(fid, '%d %d\n', size(indices,1), size(indices,2));
fclose(fid);
save('-ascii', '-append', 'MPP.dat', 'indices');


