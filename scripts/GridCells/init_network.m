%% Initializes the grid cell network.
%% W: width
%% H: height
%% M: number of modules
%% N: number of grid cells 
function [X,Y,lambda,rot,xoff,yoff] = init_network(W, H, M, N, grid_unit)
        
  lambda_range = [40.0, 400.0];

  [X, Y] = make_grid(W, H, N, grid_unit);
  xsz = size(X);
  sz = xsz(2:end);
  xx = sz(1);
  yy = sz(2);

  lambda_modules = linspace(lambda_range(1), lambda_range(2), M)
  rot_modules    = uniform(0, 2.0*pi/3, M)
            
  lambda = zeros([N,xx,yy]);  %% The grid length matrix
  rot    = zeros([N,xx,yy]);  %% The rotation matrix
  xoff   = zeros([N,xx,yy]);  %% x offset matrix
  yoff   = zeros([N,xx,yy]);  %% y offset matrix

  binEdges = linspace(0, N+1, M+1);
  [h which] = histc(1:N,binEdges);

  for i=1:N

    cur_module = which(i);

    cur_lambda = lambda_modules(cur_module);
    lambda(i,:,:)   = cur_lambda;
    rot(i,:,:) = rot_modules(cur_module);
                
    cur_xoff    = uniform(-cur_lambda, cur_lambda, 1);
    xoff(i,:,:) = cur_xoff;
                
    y_range = sqrt(cur_lambda^2 - cur_xoff^2);
    yoff(i,:,:) = uniform(-y_range, y_range, 1);

  end
