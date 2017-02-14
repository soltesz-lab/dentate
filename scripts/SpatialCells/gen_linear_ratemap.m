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
N = 250; % active grid cells per module
M = 152; % number of grid cell modules
grid_unit = 25;

data_path=getenv('DATA_PATH')

load(sprintf('%s/grid_data.mat',data_path));
load(sprintf('%s/grid_ratemap.mat',data_path));


dt = 0.5; % ms

tend = 10000; % ms
[Xpos,Ypos] = linear_walk(W, tend, dt);


size_len = sqrt(grid_unit);
Xgrid = round(Xpos/size_len)+1;
Ygrid = round(Ypos/size_len)+1;

maxXgrid = W / size_len;
maxYgrid = H / size_len

T = size(Xgrid,1)
grid_rbar_modules=zeros(N*M,T);
for m = 1:M
    s = (m-1)*N + 1
    e = m*N
    grid_rbar  = zeros(N,T);
    for t = 1:T
      t
      grid_rbar(:,t) = ratemap(s:e,Xgrid(t),Ygrid(t));
    end
    grid_rbar_modules(s:e,:) = grid_rbar;
    clear grid_rbar
end

save('-binary',sprintf('%s/linear_grid_data.mat',data_path),'grid_rbar_modules');

grid_data.W = W;
grid_data.H = 0;
grid_data.M = M;
grid_data.N = N;
grid_data.Xpos = Xpos;
grid_data.Ypos = Ypos;
grid_data.Xgrid = Xgrid;
grid_data.Ygrid = Ygrid;



