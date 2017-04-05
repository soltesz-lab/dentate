
dt = 0.5
#binfile = fopen ("placespikes.bin", "w");
#idxfile = fopen ("placespikeindex.bin", "w");
binfile = fopen ("lec3_placespikes.bin", "w");
idxfile = fopen ("lec3_placespikeindex.bin", "w");

fwrite(binfile, [0 4], 'int32'); % 4 = type double
fwrite(idxfile, [0 5], 'int32'); % 5 = type int

pos = 0;
numitems = 0;
for i=1:100
  spikes = load(sprintf('lec3_place_spikes_%d.mat',i));
  for j = 1:(size(spikes.spikes)(2))
      times  = find(spikes.spikes(:,j)) * dt;
      sz     = size(times)(1);
      fwrite(binfile, times, 'double');
      fwrite(idxfile, [pos], 'int');
      pos    = pos+sz;
      numitems = numitems+1;
  endfor

endfor

fseek(binfile, 0);
fwrite(binfile, [pos], 'int32');
fclose(binfile);

fseek(idxfile, 0);
fwrite(idxfile, [numitems], 'int32');
fclose(idxfile);
