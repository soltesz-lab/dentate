
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


W = 200.0; % box dimensions, cm
H = 200.0;
N = 38000 * 0.04; % fraction of active grid cells
M = 10; % number of grid cell modules
grid_unit = 36;

savegrid  = 1;

[X,Y,lambda,theta,xoff,yoff] = init_network(W, H, M, N, grid_unit);
ratemap  = grid_ratemap(X,Y,lambda,theta,xoff,yoff);

dt = 0.01; % ms
tend = 150; % ms
[Xpos,Ypos] = random_walk(tend, dt);

size_len = sqrt(grid_unit);
npts  = round(W/size_len);
Xgrid = round(Xpos/npts);
Ygrid = round(Ypos/npts);

T = size(Xgrid,1);
rbar  = zeros(T,N);

for t = 1:T
    rbar(t,:) = ratemap(:,Xgrid(t),Ygrid(t));
end
size(rbar)

grid_data.W = W;
grid_data.H = H;
grid_data.M = M;
grid_data.N = N;
grid_data.Xpos = Xpos;
grid_data.Ypos = Ypos;
grid_data.Xgrid = Xgrid;
grid_data.Ygrid = Ygrid;
grid_data.rbar = rbar;

save("grid_data.mat","grid_data");


