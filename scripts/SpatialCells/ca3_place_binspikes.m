
dt = 0.5
binfile = fopen ('ca3_placespikes.bin', 'w');
idxfile = fopen ('ca3_placespikeindex.bin', 'w');

fwrite(binfile, [0 4], 'int32'); % 4 = type double
fwrite(idxfile, [0 5], 'int32'); % 5 = type int

pos = 0;
numitems = 0;
for i=1:512
  spikes = load(sprintf('ca3_place_spikes_%d.mat',i));
  ssz = size(spikes.spikes)
  for j = 1:(ssz(2))
      times  = find(spikes.spikes(:,j)) * dt;
      sz = size(times);
      fwrite(binfile, times, 'double');
      fwrite(idxfile, [pos], 'int');
      pos    = pos+sz(1);
      numitems = numitems+1;
  end

end

fseek(binfile, 0);
fwrite(binfile, [pos], 'int32');
fclose(binfile);

fseek(idxfile, 0);
fwrite(idxfile, [numitems], 'int32');
fclose(idxfile);
