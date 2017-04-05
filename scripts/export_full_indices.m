#! /usr/bin/octave -qf
args = argv ();

loc = load(args{1});
populations = {'GC';'MC';'HC';'BC';'AAC';'HCC';'NGFC';'IS';'MOPP';'MPP';'LPP'}

ax = 0;
for i = 1:numel(populations)
    
    name = populations{i}

    filename = sprintf('%s.dat', name);
    fid      = fopen (filename,'w+');
    data     = loc.soma_locations{i,1};
    datasize = size(data,1);
    ncol     = size(data,2);
    fprintf(fid, '%d %d\n', datasize, ncol+1);
    fclose(fid);
    inds = reshape(ax:(ax+datasize-1),[datasize,1]);
    newdata = ([inds data(:,1:ncol)]);
    size(newdata)
    save('-ascii', '-append', filename, 'newdata');
    ax = ax+datasize;
end

