#! /usr/bin/octave -qf
args = argv ();

projections = {'GCstim';'MCstim';'HCstim';'BCstim';'AACstim';'HCCstim';'NGFCstim'}

m = load(args{1});

GCstim   = m.stim_cells{1,1}
MCstim   = m.stim_cells{2,1};
HCstim   = m.stim_cells{3,1};
BCstim   = m.stim_cells{4,1};
AACstim  = m.stim_cells{5,1};
HCCstim  = m.stim_cells{6,1};
NGFCstim = m.stim_cells{7,1};

for i = 1:numel(projections)

    name = projections{i}

    filename = sprintf('%s.dat', name);
    fid = fopen (filename,'w+');
    data = eval(name);
    fprintf(fid, '%d %d\n', size(data,1), size(data,2));
    fclose(fid);
    save('-ascii', '-append', filename, name);
    
end

