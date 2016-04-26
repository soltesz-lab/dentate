
function append_connmatrix(name,m,popsize,offset,batch_index,batch_size,output_dir)

    range = offset + batch_index:batch_size:popsize;
    data = m(ismember(m(:,2),range),:);
    fprintf('append_connmatrix: name = %s  size(m) = %d size(data) = %d batch_index = %d\n', name, size(m,1), size(data,1), batch_index)
    filename = sprintf('%s/%s.%d.noheader.dat', output_dir, name, batch_index);
    if (not (isempty(data)))
        %fid = fopen (filename,'w+');
        %fprintf(fid, '%d %d\n', size(data,1), size(data,2));
        %fclose(fid);
      save('-ascii', '-append', filename, 'data');
    end

end
