
function [uv_points] =  nearest_uv(knn,index_nn,u_params,v_params)

uv_points = zeros(size(index_nn,1),2);
for p = 1:size(index_nn,1)
    p    
    i = randsample(knn,1);
    % Find u and v coordinates from closest points
    u_bin_nn    = ceil(index_nn(p,i)/v_params(1,3));
    u_nn        = u_params(1,1) + (u_bin_nn - 1) * ((u_params(1,2)-u_params(1,1))/(u_params(1,3)-1));
    v_bin_nn    = index_nn(p,i) - ((u_bin_nn - 1) * v_params(1,3));
    v_nn        = v_params(1,1) + (v_bin_nn - 1) * ((v_params(1,2)-v_params(1,1))/(v_params(1,3)-1));
    
    uv_points(p,1) = u_nn;
    uv_points(p,2) = v_nn;
    
    %[x,y,z] = layer_eq_point(l_nn,u_nn,v_nn);
    %soma_xyz_points(p,1) = x; % calculate new x,y,z
    % coordinates consistent with u,v coordinates
    %soma_xyz_points(p,2) = y;
    %soma_xyz_points(p,3) = z;
end
