%
% Linear track place cell activity generator
%
W = 200.0; % linear track length, cm
H = 200.0; % 
N = 200; % active place cells per module
M = 402; % number of place cell modules
grid_unit = 25;

data_path=getenv('DATA_PATH')

load(sprintf('%s/ec3_place_data.mat',data_path));
load(sprintf('%s/ec3_place_ratemap.mat',data_path));


dt = 0.5; % ms

tend = 10000; % ms
[Xpos,Ypos] = linear_walk(W, tend, dt);


size_len = sqrt(grid_unit);
Xgrid = round(Xpos/size_len)+1;
Ygrid = round(Ypos/size_len)+1;

maxXgrid = W / size_len;
maxYgrid = H / size_len

T = size(Xgrid,1)
place_rbar_modules=zeros(N*M,T);
for m = 1:M
    m
    s = (m-1)*N + 1;
    e = m*N;
    if e > size(ratemap,1)
        e = size(ratemap,1);
    end
    l = e-s+1;
    place_rbar  = zeros(N,T);
    for t = 1:T
      place_rbar(:,t) = ratemap(s:e,Xgrid(t),Ygrid(t));
    end
    place_rbar_modules(s:e,:) = place_rbar;
    clear place_rbar
end

save('-hdf5',sprintf('%s/linear_lec3_place_data.h5',data_path),'place_rbar_modules');

