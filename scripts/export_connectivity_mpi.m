
output_dir=getenv('OUTPUT_DIR');
loc_input_file=getenv('LOC_INPUT_FILE');
syn_input_file=getenv('SYN_INPUT_FILE');
gj_input_file=getenv('GJ_INPUT_FILE');
batch_size=str2double(getenv('BATCH_SIZE'))
batch_index=str2double(getenv('SGE_TASK_ID'))


populations = {'GC';'MC';'HC';'BC';'AAC';'HCC';'NGFC'}
msize = size(populations,1)

tic;
loc = load(loc_input_file);
tic;

tic;
m = load(syn_input_file);
toc;

offset = 0;
for j = 1:msize
 for i = 1:msize
    name = sprintf('%sto%s',populations{i},populations{j})
    if (not (isempty(m.connection_M{i,j})))
      export_connmatrix(name,double(m.connection_M{i,j}),offset,batch_index,batch_size,output_dir);
    end
 end
 offset = offset+size(loc.locations{j,1},1)
end

if (batch_index == 1)

gjms = load(gj_input_file);
data = [];
for j = 1:size(populations,1)
 for i = 1:size(populations,1)
        name = sprintf('%sto%s',populations{i},populations{j})
        data = gjms.gap_junctions{i,j};
        if (not (isempty(data)))
          filename = sprintf('%s/gj%s.dat', output_dir, name);
          fid  = fopen (filename,'w+');
          fprintf(fid, '%d %d\n', size(data,1), size(data,2));
          fclose(fid);
          save('-ascii', '-append', filename, 'data');
        end        
  end
end

end

