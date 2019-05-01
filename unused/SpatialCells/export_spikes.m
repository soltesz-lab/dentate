
dt = 0.1;
batch_size=10;
batch_offset=0;

for batch=1:8
  load(sprintf('grid_spikes_%d.mat',batch));
  for i = 1:size(spikes,2)
    myspikes = find(spikes(:,i)) * dt;
    save('-ascii',sprintf('%d.dat',batch_offset+((i-1)*batch_size)+1),'myspikes');
  end

  batch_offset=batch_offset+size(spikes,2)*batch_size;
end
