function export_connmatrix(name,m,offset,batch_index,batch_size,output_dir)

    size(m)
    range = offset+(batch_index:batch_size:size(m,1));
    data = m(ismembc(m(:,2),range),:);
    filename = sprintf('%s/%s.%d.dat', output_dir, name, batch_index);

    if (not (isempty(data)))
      fid = fopen (filename,'w+');
      fprintf(fid, '%d %d\n', size(data,1), size(data,2));
      fclose(fid);
      save('-ascii', '-append', filename, 'data');
    end

end
