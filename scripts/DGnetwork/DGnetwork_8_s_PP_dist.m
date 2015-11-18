
function DGnetwork_8_s_PP_GC_dist(directory,indices)

%% Number of PP synapses onto a GC

Nsyns_PP_GC = 1200;

% Synapse sizes values and corresponding probabilities; adapted from
% Trommald and Hulleberg 1997
PD_PP_GC = [ 0.005 0.0;
             0.01 0.0754716981;
             0.02 0.2547169811;
             0.03 0.2169811321;
             0.04 0.1320754717;
             0.05 0.0849056604;
             0.06 0.0754716981;
             0.07 0.0283018868;
             0.08 0.0283018868;
             0.09 0.0283018868;
             0.1  0.0188679245;
             0.12 0.0188679245;
             0.14 0.0188679245;
             0.16 0.0094339623;
             0.18 0.0094339623;
           ];


% Cumulative distribution
CD_PP_GC = cumsum(PD_PP_GC(:,2));

for i = indices

    R = rand(Nsyns_PP_GC,1); % random trials
    
    %% Counts the number of elements of X that fall in the 
    %% cumulative distribution; PP_GC_idx are the bin indices

    [n, idx_PP_GC] = histc(R, CD_PP_GC);

    %% Determines the actual synaptic sizes
    dist_PP_GC = PD_PP_GC(idx_PP_GC,1);

    save(sprintf('%s/DGC_PP_synapses_%06d.dat',directory,i), ...
         'Nsyns_PP_GC','-ascii');
    save(sprintf('%s/DGC_PP_synapses_%06d.dat',directory,i), ...
         'dist_PP_GC','-ascii','-append');
    
end



