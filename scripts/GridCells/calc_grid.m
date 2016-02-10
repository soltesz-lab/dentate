%% Compute the spatial grid
function [X,Y] = calc_grid(W, H, N)
        
  min_grid_size = 0.001;

  size_len = sqrt(min_grid_size)
  mesh_npts = round(W/size_len)
        
  x = linspace(-1.0*W/2, 1.0*W/2, mesh_npts);
  y = linspace(-1.0*H/2, 1.0*H/2, mesh_npts);

  [X_tmp,Y_tmp] = meshgrid(x, y);

  X = zeros([N,mesh_npts,mesh_npts]);
  Y = zeros([N,mesh_npts,mesh_npts]);
        
  for i=1:N
    X(i,:,:) = X_tmp;
    Y(i,:,:) = Y_tmp;
  end
    
end
