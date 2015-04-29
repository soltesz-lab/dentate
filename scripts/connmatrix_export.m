function connmatrix_export(name,m,batch_size,batch_index)

    if (size(m,1) > 1e6)

      range = batch_index:batch_size:size(m,1);
class(range)
class(m(:,2))
      data = m(ismembc(double(m(:,2)),double(range)),:);
      filename = sprintf('%s.%d.dat', name, batch_index);

    else

      data = m;
      filename = sprintf('%s.dat', name);

    end

    printf('data size is %d rows\n', size(data,1));
    fid = fopen (filename,'w+');
    fprintf(fid, '%d %d\n', size(data,1), size(data,2));
    fclose(fid);
    save('-ascii', '-append', filename, 'data');


end
