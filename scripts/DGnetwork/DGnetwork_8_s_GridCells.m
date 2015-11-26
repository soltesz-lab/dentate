
%% MEC grid cells
%% Assumptions:
%% 
%%
%% - Grid cells: organized in 4-5 modules, extending over 50% of the dorsoventral MEC extent
%% - By extrapolation, possibly around 10 modules in the entire MEC
%% - Discrete logarithmic increment of grid cell scale: 40, 50, 70, 100 cm
%% - Comodular organization of grid orientation and scale: -20, -10, 10, 20 degrees
%%   (Stensola et al., Nature 2012)
%%
%% - Approximately 58,000 MEC neurons in the rat, comprised of 38% ovoid stellate cells, 29% polygonal stellate cells and 17% pyramidal cells. 
%%   (Number estimates of neuronal phenotypes in layer II of the medial entorhinal cortex of rat and mouse, Gatome et al., Neuroscience 2010)
%%
%% 10,000 grid cells
%%   (Almeida and Lisman, JNeurosci 2009)
%%
%%

%% We assume 10 modules with 1000 grid cells per module
%% Each module has its own scale and orientation; within a module,
%% grid cells have different phases

N_GridCellModules = 10;
N_GridCellsPerModule = 1000;

N_Gridcells = N_GridCellModules * N_GridCellsPerModule;


X = cell(4,1);
Y = cell(4,1);
Z = cell(4,1);
for sublayer = 1:4
    if sublayer > 1
        [x_i,y_i,z_i]   = layer_eq_ML(sublayer-2);
        switch sublayer
          case 4
            [x_o,y_o,z_o]   = layer_eq_ML(4);
          otherwise
            [x_o,y_o,z_o]   = layer_eq_ML(sublayer-1);
        end
        X{sublayer}               = [x_o;x_i];
        Y{sublayer}               = [y_o;y_i];
        Z{sublayer}               = [z_o;z_i];
        [~,S{sublayer}] = alphavol([X{sublayer}(:),Y{sublayer}(:),Z{sublayer}(:)],150);
    elseif sublayer == 1
        [x_i,y_i,z_i]  	= layer_eq_GCL(-1.95);
        [x_m,y_m,z_m]   = layer_eq_GCL(-1.0);
        [x_o,y_o,z_o]   = layer_eq_GCL(0);
        X{sublayer}               = [x_o;x_m;x_i];
        Y{sublayer}               = [y_o;y_m;y_i];
        Z{sublayer}               = [z_o;z_m;z_i];
        [~,S{sublayer}] = alphavol([X{sublayer}(:),Y{sublayer}(:),Z{sublayer}(:)],120);          
    end
end

X_g         = [X{4};X{3}];
Y_g         = [Y{4};Y{3}];
Z_g         = [Z{4};Z{3}];

M = [X_g(:),Y_g(:),Z_g(:)];

% Draw line to estimate septotemporal extent
[x,y,z]  = layer_eq_line(0,1000000);
u        = linspace(pi*1/100,pi*98/100,1000000);
sums      = zeros(length(x)-1,1);
for i = 1:length(x)-1
    d = pdist2([x(i),y(i),z(i)],[x(i+1),y(i+1),z(i+1)]);
    switch i
      case 1
        sums(i,1) = d;
      otherwise
        sums(i,1) = d + sums(i-1,1);
    end
end
stsums = [0;sums];

% Determine where splits should be and corresponding u coordinate
split_num       = 10;
num_splits      = split_num - 1;
splits          = zeros(num_splits,1);
end_septal      = pi*1/100;
end_temporal    = pi*98/100;
v_min           = pi*-23/100;
v_max           = pi*142.5/100;
u_var           = transpose(linspace(end_septal,end_temporal,10000));
for split = 1:num_splits
    total_length    = sums(end,1);
    current_split   = total_length * split/split_num;
    split_point     = find(sums(:,1)>current_split,1,'first');
    splits(split,1)  = u_var(split_point,1);
end


clearvars sums x y z

GridModuleSlices  = cell(N_GridCellModules,1);
GridModuleDists   = cell(N_GridCellModules,1);
GridModulePlanes  = cell(N_GridCellModules,1);

%% Determine grid cell module sections
for gridModule = 1:N_GridCellModules

    percent = 0.375 + 0.025*gridModule;
    width = 500;

    % Get septotemporal center
    [~,center_index]                = min(abs(stsums - (percent*max(stsums))));
    u_center                        = u(center_index);
    
    % Get points to make up plane
    plane_pts    = zeros(3,3);

    % Get longitudinal points
    [x_pt1,y_pt1,z_pt1]             = layer_eq_point(0,u_center-0.001,59.75*pi/100);
    [x_center,y_center,z_center]    = layer_eq_point(0,u_center,59.75*pi/100);
    [x_pt2,y_pt2,z_pt2]             = layer_eq_point(0,u_center+0.001,59.75*pi/100);
    plane_pts(1,:)                  = [x_center,y_center,z_center];
    direction                       = [x_pt2-x_pt1,y_pt2-y_pt1,z_pt2-z_pt1];
    unit_direction                  = direction/norm(direction);
    full_direction                  = unit_direction*width/2;
    plane_center1                   = [x_center - full_direction(1), y_center - full_direction(2), z_center + full_direction(3)];
    plane_center2                   = [x_center + full_direction(1), y_center + full_direction(2), z_center + full_direction(3)];
    
    % Get transverse points
    [x_pt1,y_pt1,z_pt1]             = layer_eq_point(0,u_center,59.75*pi/100-1);
    plane_pts(2,:)                  = [x_pt1,y_pt1,z_pt1];
    
    % Get layer points
    [x_pt1,y_pt1,z_pt1]             = layer_eq_point(-1,u_center,59.75*pi/100);
    plane_pts(3,:)                  = [x_pt1,y_pt1,z_pt1];
    
    % Get plane information
    vec1  = plane_pts(3,:) - plane_pts(1,:);
    vec2  = plane_pts(2,:) - plane_pts(1,:);
    plane = cross(vec1,vec2);
    start_d = sum(plane.*plane_center1);
    stop_d  = sum(plane.*plane_center2);

    GridModuleDists{gridModule} = [start_d stop_d];
    GridModulePlanes{gridModule} = plane;
    GridModuleSlices{gridModule} = M(find(M(:,1:3)*transpose(plane) + start_d < 0 & M(:,1:3)*transpose(plane) + stop_d > 0),:);

end


%% MOML section

end_septal      = pi*-1.6/100;
end_temporal    = pi*101/100;
layer_min       = 1;
layer_max       = 3;  

% Cycle through each grid module section
for j = 1:N_GridCellModules
        
    tic
    switch j
        %% TODO: split points
      case 1
        split1 = end_septal;
        split2 = splits(1,1);
      case split_num
        split1 = splits(end,1);
        split2 = end_temporal;
      otherwise
        split1 = splits(j-1,1);
        split2 = splits(j,1);
    end        

    % Determine if needs to lie within boundary and determine previous pts
    [x_o,y_o,z_o]   = layer_eq_ML_split(3,split1,split2);
    boundary        = [x_o,y_o,z_o];

                
    %% Place each grid cell connection point
    for cell_num = 1:N_GridCellsPerModule
        while true
            % Choose a possible point
            random_uvl = zeros(1,3);
            random_uvl(:,1) = split1 + (split2 - split1)*rand(1,1);
            random_uvl(:,2) = v_min + (v_max-v_min)*rand(1,1);
            random_uvl(:,3) = layer_min + (layer_max-layer_min)*rand(1,1);
            [x,y,z] = layer_eq(random_uvl(:,1),random_uvl(:,2),random_uvl(:,3));
            pt = [x,y,z];
            
            % Limit cells tested to those within 10 micron y value  
            if ~isempty(prev_pts)
                distance_pts    = prev_pts(prev_pts(:,2) > (pt(2) - 10) & prev_pts(:,2) < (pt(2) + 10),:);
                if ~isempty(distance_pts)
                    d_otherpts   = pdist2(double(distance_pts(:,1:3)),pt,'euclidean','Smallest',1);
                else
                    d_otherpts   = 10000;
                end                                
            else
                d_otherpts   = 10000;                              
            end
            
            % If it doesn't overlap with other placed cells
            if d_otherpts > (20)
                
                % Test if possible point reaches outside dentate volume
                if ~isempty(boundary)
                    d_boundary = pdist2(double(boundary),pt,'euclidean','Smallest',1);
                else
                    d_boundary = 10000;
                end
                
                if d_boundary > 10
                    
                    % Write out location information
                    locations_bysection{section}{j}{type}(cell_num,:)  = [pt random_uvl];
                    prev_pts = [prev_pts;[pt random_uvl]];
                    break
                end
            end
        end
        toc
    end        
end
