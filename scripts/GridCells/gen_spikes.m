
% Assumptions:
% 
% - Grid cells: organized in 4-5 modules, extending over 50% of the dorsoventral MEC extent
% - By extrapolation, possibly around 10 modules in the entire MEC
% - Discrete logarithmic increment of grid cell scale: 40, 50, 70, 100 cm
% - Comodular organization of grid orientation and scale: -20, -10, 10, 20 degrees
%   (Stensola et al., Nature 2012)
% - Approximately 58,000 MEC neurons in the rat, comprised of 38% ovoid stellate cells, 29% polygonal stellate cells and 17% pyramidal cells. 
%   (Number estimates of neuronal phenotypes in layer II of the medial entorhinal cortex of rat and mouse, Gatome et al., Neuroscience 2010)
% 
% - We assume 10 modules with 3800 grid cells per module, for a total of 38000 grid cells.
% - Each module has its own scale and orientation; within a module, grid cells have different phases.
% - This contrasts with model by de Almeida and Lisman which has 10,000 grid cells.
%   (de Almeida and Lisman, JNeurosci 2009)


batch_size=str2num(getenv('BATCH_SIZE'))
batch_index=str2num(getenv('BATCH_INDEX'))
						     
load('grid_data.mat');
size(grid_rbar)

rbar = grid_rbar(:,(((batch_index-1)*batch_size)+1):(batch_index*batch_size)) * 1e-3;
[T,N] = size(rbar)
  
s = DG_SFromPSTHVarZ(rbar, 1);
spikes = DG_spike_gen(s,eye(N,N),1);

save(sprintf('grid_spikes_%d.mat',batch_index),'spikes');
