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
N = 38000 * 0.04; % fraction of active grid cells
M = 10; % number of grid cell modules
grid_unit = 25;

savegrid  = 1;

[X,Y,lambda,theta,xoff,yoff] = init_network(W, H, M, N, grid_unit);
ratemap  = grid_ratemap(X,Y,lambda,theta,xoff,yoff);

dt = 0.1; % ms

tend = 2000; % ms
[Xpos,Ypos] = linear_walk(W, tend, dt);


size_len = sqrt(grid_unit);
Xgrid = round(Xpos/size_len)+1;
Ygrid = round(Ypos/size_len)+1;

maxXgrid = W / size_len;
maxYgrid = H / size_len

T = size(Xgrid,1)
grid_rbar  = zeros(T,N);
border_rbar  = zeros(T,N);

for t = 1:T
    grid_rbar(t,:) = ratemap(:,Xgrid(t),Ygrid(t));
    if (Xgrid(t) < 3) || (Xgrid(t) > (maxXgrid - 3))
        border_rbar(t,:) = ratemap(:,Xgrid(t),Ygrid(t));
    else
        border_rbar(t,:) = zeros(1,N);
    end
end
size(grid_rbar)

grid_data.W = W;
grid_data.H = 0;
grid_data.M = M;
grid_data.N = N;
grid_data.Xpos = Xpos;
grid_data.Ypos = Ypos;
grid_data.Xgrid = Xgrid;
grid_data.Ygrid = Ygrid;

save('-v7','linear_grid_data.mat','grid_data','grid_rbar','border_rbar');


