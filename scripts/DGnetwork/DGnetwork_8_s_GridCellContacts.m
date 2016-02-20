function GridCellModules = DGnetwork_8_s_GridCellContacts(directory)

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
N_GridCellsPerModule = 3800;
N_GridCellLongExtent = 1000;

N_GridCells = N_GridCellModules * N_GridCellsPerModule;
GridCell_percent_start = 0.1;
GridCell_percent_end = 0.9;
GridCell_percent_step = (GridCell_percent_end - GridCell_percent_start) / N_GridCells;

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

clearvars sums x y z

GridCellModules = cell(N_GridCellModules,1);
gridCellIndex = 0;

%% Determine grid cell module sections
for gridModule = 1:N_GridCellModules
    
    GridCellModules{gridModule} = cell(N_GridCellsPerModule,1);

    
    for gridCell = 1:N_GridCellsPerModule

        gridCell
        
        %% Generate new set of grid points every 40 cells
        if (mod(gridCell-1,40) == 0)

            X = cell(4,1);
            Y = cell(4,1);
            Z = cell(4,1);
            for sublayer = 1:4
                if sublayer > 1
                    [x_i,y_i,z_i]   = layer_eq_ML_poisson(sublayer-2);
                    [x_m,y_m,z_m]   = layer_eq_ML_poisson(sublayer-1.5);
                    switch sublayer
                      case 4
                        [x_o,y_o,z_o]   = layer_eq_ML_poisson(4);
                      otherwise
                        [x_o,y_o,z_o]   = layer_eq_ML_poisson(sublayer-1);
                    end
                    X{sublayer}     = [x_o;x_m;x_i];
                    Y{sublayer}     = [y_o;y_m;y_i];
                    Z{sublayer}     = [z_o;z_m;z_i];
                elseif sublayer == 1
                    [x_i,y_i,z_i]  	= layer_eq_GCL_poisson(-1.95);
                    [x_m,y_m,z_m]   = layer_eq_GCL_poisson(-1.0);
                    [x_o,y_o,z_o]   = layer_eq_GCL_poisson(0);
                    X{sublayer}     = [x_o;x_m;x_i];
                    Y{sublayer}     = [y_o;y_m;y_i];
                    Z{sublayer}     = [z_o;z_m;z_i];
                end
            end
            
            X_g = [X{2};X{3}];
            Y_g = [Y{2};Y{3}];
            Z_g = [Z{2};Z{3}];
            
            M = [X_g(:),Y_g(:),Z_g(:)];
            
        end
            
        percent = GridCell_percent_start + GridCell_percent_step*gridCellIndex
        width = N_GridCellLongExtent;
        
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
        
        GridCellModules{gridModule}{gridCell} = M(find(M(:,1:3)*transpose(plane) + start_d < 0 & M(:,1:3)*transpose(plane) + stop_d > 0),:);
        gridCellIndex = gridCellIndex + 1;
    end
end

save(sprintf('%s/GridCellModules.mat',directory),'GridCellModules');
