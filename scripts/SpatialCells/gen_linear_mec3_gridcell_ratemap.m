%
% Linear track grid cell activity generator
%


W = 200.0; % linear track length, cm
H = 200.0; % 
N = 249; % active grid cells per module
M = 350; % number of grid cell modules
grid_unit = 25;

data_path=getenv('DATA_PATH')

load(sprintf('%s/ec3_grid_data.mat',data_path));
load(sprintf('%s/ec3_grid_ratemap.mat',data_path));

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
    m
    s = (m-1)*N + 1;
    e = m*N;
    if e > size(ratemap,1)
        e = size(ratemap,1);
    end
    l = e-s+1;
    grid_rbar  = zeros(l,T);
    for t = 1:T
      grid_rbar(:,t) = ratemap(s:e,Xgrid(t),Ygrid(t));
    end
    grid_rbar_modules(s:e,:) = grid_rbar;
    clear grid_rbar
end

save('-binary',sprintf('%s/linear_mec3_grid_data.oct',data_path),'grid_rbar_modules');



