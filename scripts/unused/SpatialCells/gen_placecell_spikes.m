

batch_size=str2num(getenv('BATCH_SIZE'))
batch_index=str2num(getenv('BATCH_INDEX'))
input_data_path=getenv('INPUT_DATA_PATH')
output_path=getenv('OUTPUT_PATH');						     

linear_place_data=matfile(input_data_path);

rbar = permute(linear_place_data.place_rbar_modules((((batch_index-1)*batch_size)+1):(batch_index*batch_size),:),[2,1]) * 1e-2;

[T,N] = size(rbar)
  
s = DG_SFromPSTHVarZ(rbar, 1);
spikes = DG_spike_gen(s,eye(N,N),1);

save(sprintf('%s/place_spikes_%d.mat',output_path,batch_index),'spikes');
