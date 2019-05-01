

batch_size=str2num(getenv('BATCH_SIZE'))
batch_index=str2num(getenv('BATCH_INDEX'))
input_data_path=getenv('INPUT_DATA_PATH')
output_path=getenv('OUTPUT_PATH');						     

tend = 10000;
dt   = 0.5;
entry_size = (tend/dt)*4 + 8;

rbar=zeros(round(tend/dt), batch_size);
linear_place_file=fopen(input_data_path);

offset = ((batch_index-1)*batch_size)*entry_size;
fseek(linear_place_file, offset, -1);
for i = 1:batch_size
id = fread(linear_place_file, 1, 'uint32');
sz = fread(linear_place_file, 1, 'uint32');
rbar(:,i) = fread(linear_place_file, sz, 'float') * 1e-3;
end

fclose(linear_place_file);

[T,N] = size(rbar)
  
s = DG_SFromPSTHVarZ(rbar, 1);
spikes = DG_spike_gen(s,eye(N,N),1);

save(sprintf('%s/ca3_place_spikes_%d.mat',output_path,batch_index),'spikes');
