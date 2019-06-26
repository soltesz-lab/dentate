
function DGnetwork_8_s_PP_dist(directory,epilepsy,slice,indices)

if str2double(slice) == 0 && str2double(epilepsy) == 0
  load(sprintf('%s/Locations.mat',directory));
elseif str2double(slice) == 1 && str2double(epilepsy) == 0
  load(sprintf('%s/Slice_Locations.mat',directory));
  load(sprintf('%s/Slice_Indexes.mat',directory));
elseif str2double(slice) == 0 && str2double(epilepsy) == 1
  load(sprintf('%s/Locations.mat',directory));
elseif str2double(slice) == 1 && str2double(epilepsy) == 1
  load(sprintf('%s/Slice_Locations.mat',directory));
  load(sprintf('%s/Slice_Indexes.mat',directory));
end

%% Number of PP synapses onto a GC
%% 1200 MPP synapses; 1500 LPP synapses
%% (Almeida and Lisman, JNeurosci 2009)

Nsyns_MPP_GC = 1200;
Nsyns_LPP_GC = 1500;

%% Scaling factor used to obtain conductance from spine size
f_PP_GC = 0.001;

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

weights_MPP_GC = cell(length(locations),1);
weights_LPP_GC = cell(length(locations),1);

for i = 1:length(locations)
    
    RM = rand(Nsyns_MPP_GC,1); % random trials
    RL = rand(Nsyns_LPP_GC,1); % random trials
    
    %% Counts the number of elements of X that fall in the 
    %% cumulative distribution; idx_?PP_GC are the bin indices

    [n, idx_MPP_GC] = histc(RM, CD_PP_GC);
    [m, idx_LPP_GC] = histc(RL, CD_PP_GC);

    %% Determines the synaptic conductances based on sizes
    weights_MPP_GC{i} = f_PP_GC * PD_PP_GC(idx_MPP_GC,1);
    weights_LPP_GC{i} = f_PP_GC * PD_PP_GC(idx_LPP_GC,1);
    
end
    
save(sprintf('%s/MPPsynweights.mat',directory),'weights_MPP_GC','-v6');
save(sprintf('%s/LPPsynweights.mat',directory),'weights_LPP_GC','-v6');
