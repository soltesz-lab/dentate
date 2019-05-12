%% Initializes the grid cell network.
%% W: width
%% H: height
%% M: number of modules
%% N: number of grid cells 
function [X,Y,lambda,rot,xoff,yoff] = init_network(W, H, M, N, lambda_range, r_off_scale, grid_unit, seed)
        
  rand ('seed', seed);


  [X, Y] = make_grid(W, H, N, grid_unit);
  xsz = size(X);
  sz = xsz(2:end);
  xx = sz(1);
  yy = sz(2);
  
  
  lambda_modules = linspace(lambda_range(1), lambda_range(2), M)
  rot_modules    = uniform(0, pi/3, M)
            
  lambda = zeros(N,1);  %% The grid length matrix
  rot    = zeros(N,1);  %% The rotation matrix
  xoff   = zeros(N,1);  %% x offset matrix
  yoff   = zeros(N,1);  %% y offset matrix

  
  binEdges = linspace(0, N+1, M+1);
  [h which] = histc(1:N,binEdges);

  for i=1:N

    cur_module = which(i);

    cur_lambda = lambda_modules(cur_module);
    lambda(i)  = cur_lambda;
    rot(i)     = rot_modules(cur_module);
                
    r_off      = r_off_scale * cur_lambda * sqrt(rand());
    phi_off    = uniform(-pi, pi, 1);

    xoff(i)    = r_off * cos(phi_off); 
    yoff(i)    = r_off * sin(phi_off);

  end
