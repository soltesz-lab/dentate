%% Initializes the grid cell network.
%% W: width
%% H: height
%% M: number of modules
%% N: number of grid cells 
function [l,rot,xoff,yoff] = init_network(W, H, M, N)
        
  min_grid_size = 0.0001;
  len_range = [.1, .9];

  [X, Y] = calc_grid(W, H, N);
  sz = size(X)(2:end);
  xx = sz(1);
  yy = sz(2);

  l_modules    = uniform(len_range(1), len_range(2), M);
  rot_modules  = uniform(0, 2.0*pi/3, M);
            
  l    = zeros([N,xx,yy]);  %% The grid length matrix
  rot  = zeros([N,xx,yy]);  %% The rotation matrix
  xoff = zeros([N,xx,yy]);  %% x offset matrix
  yoff = zeros([N,xx,yy]);  %% y offset matrix
            
  for i=1:N 

    cur_module = mod(i,M) + 1;

    cur_l      = l_modules(cur_module);
    l(i,:,:)   = cur_l;
    rot(i,:,:) = rot_modules(cur_module);
                
    cur_xoff    = uniform(-cur_l, cur_l, 1);
    xoff(i,:,:) = cur_xoff;
                
    y_range = sqrt(cur_l^2 - cur_xoff^2);
    yoff(i,:,:) = uniform(-y_range, y_range, 1);

  end
