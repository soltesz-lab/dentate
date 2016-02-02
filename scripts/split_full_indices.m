#! /usr/bin/octave -qf
args = argv ();

numforest=999;
forestsize=1000;
loc = load(args{1});
prefix = args{2};

populations = {'GC'}

ax = 1;
for i = 1:numel(populations)
    
    name = populations{i}

    for forestindex = 1:numforest

      filename = sprintf('%s/%d/%scoordinates.dat', prefix, forestindex, name)
      fid = fopen (filename,'w+');
      data = loc.locations{i,1}(ax:(ax+forestsize-1),:);
      datasize = size(data,1);
      fprintf(fid, '%d %d\n', datasize, 4);
      fclose(fid);
      inds = reshape(ax:(ax+datasize-1),[datasize,1]);
      newdata = ([inds data(:,1:3)]);
      size(newdata)
      save('-ascii', '-append', filename, 'newdata');
      ax = ax+datasize;

    end
end

