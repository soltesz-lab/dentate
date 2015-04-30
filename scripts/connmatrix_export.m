function connmatrix_export(name,m,batch_index,batch_size,output_dir)

    size(m)
    if (size(m,1) > 1e6)
      range = batch_index:batch_size:size(m,1);
      data = m(ismembc(m(:,2),range),:);
      filename = sprintf('%s/%s.%d.dat', output_dir, name, batch_index);
    else
      data = m;
      filename = sprintf('%s/%s.dat', output_dir, name);
    end

    if (not (isempty(data)))
      fid = fopen (filename,'w+');
      fprintf(fid, '%d %d\n', size(data,1), size(data,2));
      fclose(fid);
      save('-ascii', '-append', filename, 'data');
    end

end
