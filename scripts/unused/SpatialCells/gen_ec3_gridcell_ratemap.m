%
% Grid cell rate map generator
%
% Assumptions:
% 
% - Grid cells: organized in 4-5 modules, extending over 50% of the dorsoventral MEC extent
% - By extrapolation, possibly around 10 modules in the entire MEC
% - Discrete logarithmic increment of grid cell scale: 40, 50, 70, 100 cm
% - Comodular organization of grid orientation and scale: -20, -10, 10, 20 degrees
%   (Stensola et al., Nature 2012)
% - Approximately 130,000 MEC LIII neurons in the rat
% - assuming the same ratios of LII, 38% ovoid stellate cells, 29% polygonal stellate cells and 17% pyramidal cells (i.e. 67% of MEC cells project to HC)
%   (Number estimates of neuronal phenotypes in layer II of the medial entorhinal cortex of rat and mouse, Gatome et al., Neuroscience 2010)
% 
% - We assume 10 modules with 8710 grid cells per module, for a total of 87100 grid cells.
% - Each module has its own scale and orientation; within a module, grid cells have different phases.
% - This contrasts with model by de Almeida and Lisman which has 10,000 grid cells.
%   (de Almeida and Lisman, JNeurosci 2009)


W = 200.0; % linear track length, cm
H = 200.0; % 
N = 87100;
M = 10; % number of grid cell modules
lambda_range = [40.0, 400.0];
grid_unit = 25;


seed = 25;

[X,Y,lambda,theta,xoff,yoff] = init_network(W, H, M, N, lambda_range, 1, grid_unit, seed);
ratemap  = grid_ratemap(X,Y,lambda,theta,xoff,yoff);

grid_data.W = W;
grid_data.H = 0;
grid_data.M = M;
grid_data.N = N;
grid_data.X = X;
grid_data.Y = Y;
grid_data.lambda = lambda;
grid_data.theta = theta;
grid_data.xoff = xoff;
grid_data.yoff = yoff;

save('-v7','ec3_grid_ratemap.mat','ratemap');
save('-v7','ec3_grid_data.mat','grid_data');


