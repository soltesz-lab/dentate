#! /usr/bin/octave -qf
args = argv ();

m = load(args{1});

populations = {'GC';'MC';'HC';'BC';'AAC';'HCC';'NGFC'}

GCinds   = m.keep_indexes{1,1};
MCinds   = m.keep_indexes{2,1};
HCinds   = m.keep_indexes{3,1};
BCinds   = m.keep_indexes{4,1};
AACinds  = m.keep_indexes{5,1};
HCCinds  = m.keep_indexes{6,1};
NGFCinds = m.keep_indexes{7,1};

for i = 1:numel(populations)

    name = populations{i}

    filename = sprintf('%s.dat', name);
    fid = fopen (filename,'w+');
    data = eval(sprintf('%sinds', name));
    fprintf(fid, '%d %d\n', size(data,1), size(data,2));
    fclose(fid);
    save('-ascii', '-append', filename, sprintf('%sinds', name));
    
end

