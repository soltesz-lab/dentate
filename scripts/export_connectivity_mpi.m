
output_dir=getenv('OUTPUT_DIR');
syn_input_file=getenv('SYN_INPUT_FILE');
gj_input_file=getenv('GJ_INPUT_FILE');
batch_size=str2double(getenv('BATCH_SIZE'))
batch_index=str2double(getenv('SGE_TASK_ID'))

projection_names = {'GCtoGC';'MCtoGC';'HCtoGC';'BCtoGC';'AACtoGC';
		    'HCCtoGC';'NGFCtoGC';'GCtoMC';'MCtoMC';'HCtoMC';
		    'BCtoMC';'AACtoMC';'HCCtoMC';'GCtoHC';'MCtoHC';
		    'HCtoHC';'BCtoHC';'GCtoBC';'MCtoBC';'HCtoBC';'BCtoBC';
		    'HCCtoBC';'NGFCtoBC';'GCtoAAC';'MCtoAAC';'HCtoAAC';'NGFCtoAAC';
		    'GCtoHCC';'MCtoHCC';'BCtoHCC';'HCCtoHCC';'NGFCtoHCC';'HCtoNGFC';'NGFCtoNGFC'}

projections = [ 1 1; 2 1; 3 1; 4 1; 5 1; 6 1; 7 1; % GC
		1 2; 2 2; 3 2; 4 2; 5 2; 6 2; 1 3; % MC
	        2 3; 3 3; 4 3; % HC
		1 4; 2 4; 3 4; 4 4; 6 4; 7 4; % BC
	        1 5; 2 5; 3 5; 7 5; % GCC
		1 6; 2 6; 4 6; 6 6; 7 6;  % AAC
		3 7; 7 7 ]; % NGFC

tic;
m = load(syn_input_file);
toc;

for i = 1:size(projections,1)

    name = projection_names{i}
    r = projections(i,1); c = projections(i,2);
    if (not (isempty(m.connection_M{r,c})))
      export_connmatrix(name,double(m.connection_M{r,c}),batch_index,batch_size,output_dir);
    end
end

gjprojection_names = {'HCtoHC';'BCtoBC';'HCCtoHCC';'NGFCtoNGFC'};
gjprojections = [ 3 3; 4 4; 6 6; 7 7; ];

if (batch_index == 1)

gjms = load(gj_input_file);
data = [];
for i = 1:size(gjprojections,1)

        name = gjprojection_names{i}
        r = gjprojections(i,1); c = gjprojections(i,2);
        
        filename = sprintf('%s/gj%s.dat', output_dir, name);
        fid  = fopen (filename,'w+');
        data = gjms.gap_junctions{r,c};
        if (not (isempty(data)))
          fprintf(fid, '%d %d\n', size(data,1), size(data,2));
          fclose(fid);
          save('-ascii', '-append', filename, 'data');
        end        
end

end

