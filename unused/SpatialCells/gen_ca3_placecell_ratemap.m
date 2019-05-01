%
% Place cell rate map generator
%

data_path=getenv('DATA_PATH');

W = 200.0; % linear track length, cm
H = 200.0; % 
N = 204700; % Approximately reflecting the CA3 principal cells in the rat
M = 1;
lambda_range = [W*2, W*2];
grid_unit = 25;

seed = 25;

[X,Y,lambda,theta,xoff,yoff] = init_network(W, H, M, N, lambda_range, 4, grid_unit, seed);
ratemap  = place_ratemap(X,Y,lambda,theta,xoff,yoff);

place_data.W = W;
place_data.H = 0;
place_data.M = M;
place_data.N = N;
place_data.X = X;
place_data.Y = Y;
place_data.lambda = lambda;
place_data.theta = theta;
place_data.xoff = xoff;
place_data.yoff = yoff;

save('-binary',sprintf('%s/ca3_placecell_ratemap.bin',data_path),'ratemap');
save('-binary','-append',sprintf('%s/ca3_placecell_ratemap.bin',data_path),'place_data');


