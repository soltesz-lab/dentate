#! /usr/bin/octave -qf
args = argv ();

loc = load(args{1});

populations = {'GC';'MC';'HC';'BC';'AAC';'HCC';'NGFC'}

ax = 1;
for i = 1:numel(populations)

    name = populations{i}

    filename = sprintf('%s.dat', name);
    fid = fopen (filename,'w+');
    data = loc.locations{i,1};
    datasize = size(data,1);
    fprintf(fid, '%d %d\n', datasize, 1);
    fclose(fid);
    inds = reshape(ax:(ax+datasize-1),[datasize,1]);
    size(inds)
    save('-ascii', '-append', filename, 'inds');
    ax = ax+datasize;
end

