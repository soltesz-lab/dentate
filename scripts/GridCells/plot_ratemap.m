%
% Linear track grid cell activity generator
%
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


W = 200.0; % linear track length, cm
H = 200.0; % 
N = 38000 * 0.01; % fraction of active grid cells
M = 10; % number of grid cell modules

grid_unit = 1;

[X,Y,lambda,theta,xoff,yoff] = init_network(W, H, M, N, grid_unit);
ratemap1  = grid_ratemap(X,Y,lambda,theta,xoff,yoff);

m1_1 = reshape(ratemap1(1,:,:),[200 200]);
m50_1 = reshape(ratemap1(50,:,:),[200 200]);
m380_1 = reshape(ratemap1(380,:,:),[200 200]);

grid_unit = 25;

[X,Y,lambda,theta,xoff,yoff] = init_network(W, H, M, N, grid_unit);
ratemap25  = grid_ratemap(X,Y,lambda,theta,xoff,yoff);

m1_25 = reshape(ratemap25(1,:,:),[40 40]);
m50_25 = reshape(ratemap25(50,:,:),[40 40]);
m380_25 = reshape(ratemap25(380,:,:),[40 40]);






