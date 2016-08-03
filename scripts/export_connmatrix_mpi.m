function export_connmatrix(name,m,msize,popsize,offset,batch_index,batch_size,output_dir)

  range = offset+(batch_index:batch_size:popsize);
  chunkSize=10000000;
  chunkStart=1; chunkEnd = chunkSize+chunkStart;
  filename = sprintf('%s/%s.%d.noheader.dat', output_dir, name, batch_index);
  fid = fopen (filename,'w');
  fclose(fid);
  while (chunkStart < msize)
    if chunkEnd < msize
      mchunk=m(chunkStart:chunkEnd,:);
    else
      mchunk=m(chunkStart:msize,:);
    end
    res = find(ismember(mchunk(:,2),range));
    data = double(mchunk(res,:));
    fprintf('export_connmatrix: name = %s offset = %d popsize = %d size(data) = %d batch_index = %d\n', name, offset, popsize, size(data,1), batch_index)
    if (not (isempty(data)))
      fid  = fopen (filename,'w+');
      for k = 1:size(data)
        fprintf(fid, '%d %d %f %f\n', data(k,1), data(k,2), data(k,3), data(k,4));
      end
      fclose(fid);
      %%save('-ascii', '-append', filename, 'data');
    end
    chunkStart = chunkEnd+1;
    chunkEnd = chunkSize+chunkStart;
  end
end
