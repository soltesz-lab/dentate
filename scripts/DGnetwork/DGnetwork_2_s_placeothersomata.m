% Place non-granule cell somata

%function DGnetwork_1_s_findsomata()

% Set random number generator for 
rng(1)

%% Define number and distribution of cell types
num_types       = 9;
num_cells       = cell(num_types,1);
soma_diam       = zeros(num_types,1);
soma_length     = zeros(num_types,1);
soma_border     = zeros(num_types,1);
distribution    = cell(num_types,1);

%1 Granule cells
num_cells{1}    = [0;1000000;0;0;1000000];                                        %Total;Hilus;GCL;IML;MOML
soma_diam(1)    = 0;
soma_length(1)  = 0;
septotemporal   = [3.85;6.16;6.46;6.8;6.2;5.85;5.43;5.76;5.12;5.08;2.35;0.15];
distribution{1} = septotemporal/sum(septotemporal(:,1));

%2 Mossy
num_cells{2}    = [30000;0;0;0;30000];                                        %Total;Hilus;GCL;IML;MOML
soma_diam(2)    = 20;
soma_length(2)  = 20;
septotemporal   = [20.8;38.2;67.6;75.4;51.2;48.6;84.1;68.5;145.7;385.0];
distribution{2} = septotemporal/sum(septotemporal(:,1));

%3 HIPP
num_cells{3}    = [9000;0;0;0;9000];
soma_diam(3)    = 10;
soma_length(3)  = 20;
septotemporal   = [16.2;35.3;38.1;31.8;36.1;38.8;39.6;55.0;73.1;141.5];
distribution{3} = septotemporal/sum(septotemporal(:,1));

%4 PVBC
num_cells{4}    = [1700;1700;400;0;3800];
soma_diam(4)    = 15;
soma_length(4)  = 20;
septotemporal   = [9.5;16.5;19.6;17.5;16.8;15.7;15.8;17.5;21.2;40.9];
distribution{4} = septotemporal/sum(septotemporal(:,1));

%5 AA 
num_cells{5}    = [200;200;50;0;450];
soma_diam(5)    = 15;
soma_length(5)  = 20;
septotemporal   = [9.5;16.5;19.6;17.5;16.8;15.7;15.8;17.5;21.2;40.9];
distribution{5} = septotemporal/sum(septotemporal(:,1));

%6 HICAP/CCK+ Basket Cells
num_cells{6}    = [1150;250;0;0;1400];
soma_diam(6)    = 15;
soma_length(6)  = 20;
septotemporal   = [9.1;16.3;13.1;11.3;9.3;8.3;8.3;9.9;9.8;6.1];
distribution{6} = septotemporal/sum(septotemporal(:,1));

%7 NGFC
num_cells{7}    = [0;0;0;5000;5000];
soma_diam(7)    = 15;
soma_length(7)  = 20;
septotemporal   = [9.5;16.5;19.6;17.5;16.8;15.7;15.8;17.5;21.2;40.9];
distribution{7} = septotemporal/sum(septotemporal(:,1));

%8 MEC
% 58000 neurons in MEC estimated by Gatome et al., of those 38000
% are dentate-projection stellate cells.
% Number estimates of neuronal phenotypes in layer II of the medial entorhinal cortex of rat and mouse.
% Neuroscience. 2010 Sep 29;170(1):156-65. 
num_cells{8}    = [0;0;0;38000;38000]; %Hilus;GCL;IML;MOML;Total
soma_diam(8)    = 15;
soma_length(8)  = 20;
septotemporal   = [9.5;16.5;19.6;17.5;16.8;15.7;15.8;17.5;21.2;40.9];
distribution{8} = septotemporal/sum(septotemporal(:,1));

%9 LEC
num_cells{9}    = [0;0;0;34000;34000]; %Hilus;GCL;IML;MOML;Total
soma_diam(9)    = 15;
soma_length(9)  = 20;
septotemporal   = [9.5;16.5;19.6;17.5;16.8;15.7;15.8;17.5;21.2;40.9];
distribution{9} = septotemporal/sum(septotemporal(:,1));

% Determine how far neighboring somata must be
for type = 1:num_types
    soma_border(type) = sqrt((soma_length(type)/2)^2 + (soma_diam(type)/2)^2);
end

% Preallocate somata locations
locations_bysection = cell(length(num_cells{1})-1,1);
for section = 1:length(locations_bysection)
    locations_bysection{section} = cell(length(distribution{2}),1);
    for j = 1:length(distribution{2})
        locations_bysection{section}{j} = cell(num_types,1);
    end
end


%% Determine splits

% Determine total length
[x1,y1,z1]  = layer_eq_GCL_line(0);
sums        = zeros(length(x1)-1,1);
for i = 1:length(x1)-1
    d = pdist2([x1(i),y1(i),z1(i)],[x1(i+1),y1(i+1),z1(i+1)]);
    switch i
        case 1
            sums(i,1) = d;
        otherwise
            sums(i,1) = d + sums(i-1,1);
    end
end

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

% Keep track of granule cells close to the edge of the layer
load('./Outputs/GC_locations')
GC_locations = GC_locations_all;
edgeGCs   = GC_locations(GC_locations(:,6)<(-1.95*9/10) | GC_locations(:,6)>(-1.95/10),:);
edgeGCs_split = cell(split_num,1);
for j = 1:split_num
    switch j
        case 1 
            locations_bysection{2}{j}{1}    = GC_locations(GC_locations(:,4)>end_septal  & GC_locations(:,4)<splits(j),:);
            edgeGCs_split{j}                = edgeGCs(edgeGCs(:,4)>end_septal  & edgeGCs(:,4)<splits(j),1:6);
        case split_num
            locations_bysection{2}{j}{1}    = GC_locations(GC_locations(:,4)>splits(j-1) & GC_locations(:,4)<end_temporal,:);
            edgeGCs_split{j}                = edgeGCs(edgeGCs(:,4)>splits(j-1) & edgeGCs(:,4)<end_temporal,1:6);
        otherwise
            locations_bysection{2}{j}{1}    = GC_locations(GC_locations(:,4)>splits(j-1) & GC_locations(:,4)<splits(j),:);
            edgeGCs_split{j}                = edgeGCs(edgeGCs(:,4)>splits(j-1) & edgeGCs(:,4)<splits(j),1:6);
    end
end

%% Distribute All Other Cells
for section = 1:length(num_cells{1}-1)
    % Define parametric surface parameters
    switch section
        case 1
            end_septal      = pi*1/100;
            end_temporal    = pi*98/100;
            layer_min       = -3.95;
            layer_max       = -1.95;
        case 2
            end_septal      = pi*1/100;
            end_temporal    = pi*98/100;
            layer_min       = -1.95;
            layer_max       = 0;           
        case 3
            end_septal      = pi*-1.6/100;
            end_temporal    = pi*101/100;
            layer_min       = 0;
            layer_max       = 1;  
        case 4
            end_septal      = pi*-1.6/100;
            end_temporal    = pi*101/100;
            layer_min       = 1;
            layer_max       = 3;  
    end
     

    % Cycle through each septotemporal section
    for j = 1:split_num
        tic
        switch j
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

        % Determine if needs to lie within boundary and determine previous somata
        switch section
            case 1
                [x_o,y_o,z_o]   = layer_eq_GCL_point(-3.95,split1,split2);
                boundary        = [x_o,y_o,z_o];
            case 4
                [x_o,y_o,z_o]   = layer_eq_ML_point(3,split1,split2);
                boundary        = [x_o,y_o,z_o];
            otherwise
                boundary = [];
        end        

        % Combine previously placed somata
        switch j 
            case 1
                switch section
                    case 1
                        prev_somata = vertcat(edgeGCs_split{j:j+1});
                    case 2
                        prev_somata = vertcat(edgeGCs_split{j:j+1},locations_bysection{section-1}{j}{:},locations_bysection{section-1}{j+1}{:});
                    case 3
                        prev_somata = vertcat(edgeGCs_split{j:j+1},locations_bysection{section-1}{j}{2:end},locations_bysection{section-1}{j+1}{2:end});
                    case 4
                        prev_somata = vertcat(locations_bysection{section-1}{j}{2:end},locations_bysection{section-1}{j+1}{2:end});
                end
            case 10
                switch section
                    case 1
                        prev_somata = vertcat(edgeGCs_split{j-1:j});
                    case 2
                        prev_somata = vertcat(edgeGCs_split{j-1:j},locations_bysection{section-1}{j-1}{:},locations_bysection{section-1}{j}{:});
                    case 3
                        prev_somata = vertcat(edgeGCs_split{j-1:j},locations_bysection{section-1}{j-1}{2:end},locations_bysection{section-1}{j}{2:end});
                    case 4
                        prev_somata = vertcat(locations_bysection{section-1}{j-1}{2:end},locations_bysection{section-1}{j}{2:end});                
                end
            otherwise
                switch section
                    case 1
                        prev_somata = vertcat(edgeGCs_split{j-1:j+1});
                    case 2
                        prev_somata = vertcat(edgeGCs_split{j-1:j+1},locations_bysection{section-1}{j-1}{:},locations_bysection{section-1}{j+1}{:});
                    case 3
                        prev_somata = vertcat(edgeGCs_split{j-1:j+1},locations_bysection{section-1}{j-1}{2:end},locations_bysection{section-1}{j+1}{2:end});
                    case 4
                        prev_somata = vertcat(locations_bysection{section-1}{j-1}{2:end},locations_bysection{section-1}{j+1}{2:end});                  
                end
        end

                
        % Cycle through each type
        for type = 2:num_types
            type
            % If there are cells in the layer for this type
            if num_cells{type}(section) > 0
                
                % Determine the number of cells to be distributed in this section
                switch j
                    case 10
                        total_cells = num_cells{type}(section) - sum(round(num_cells{type}(section) * distribution{type}(1:(j-1),1)));
                    otherwise
                        total_cells = round(num_cells{type}(section) * distribution{type}(j,1));
                end
                
                locations_bysection{section}{j}{type} = zeros(total_cells,6,'double');
                                        
                % Place each cell
                for cell_num = 1:total_cells
                    while true
                        % Choose a possible point
                        random_uvl = zeros(1,3);
                        random_uvl(:,1) = split1 + (split2 - split1)*rand(1,1);
                        random_uvl(:,2) = v_min + (v_max-v_min)*rand(1,1);
                        random_uvl(:,3) = layer_min + (layer_max-layer_min)*rand(1,1);
                        [x,y,z] = layer_eq(random_uvl(:,1),random_uvl(:,2),random_uvl(:,3));
                        pt = [x,y,z];

                        % Limit cells tested to those within 10 micron y value  
                        if ~isempty(prev_somata)
                            distance_pts    = prev_somata(prev_somata(:,2) > (pt(2) - 10) & prev_somata(:,2) < (pt(2) + 10),:);
                            if ~isempty(distance_pts)
                                d_othersomata   = pdist2(double(distance_pts(:,1:3)),pt,'euclidean','Smallest',1);
                            else
                                d_othersomata   = 10000;
                            end                                
                        else
                            d_othersomata   = 10000;                              
                        end

                        % If it doesn't overlap with other placed cells
                        if d_othersomata > (20)

                            % Test if possible somata reaches outside dentate volume
                            if ~isempty(boundary)
                                d_boundary = pdist2(double(boundary),pt,'euclidean','Smallest',1);
                            else
                                d_boundary = 10000;
                            end

                            if d_boundary > 10

                                % Write out location information
                                locations_bysection{section}{j}{type}(cell_num,:)  = [pt random_uvl];
                                prev_somata = [prev_somata;[pt random_uvl]];
                                break
                            end
                        end
                    end
                end
            end    
        end
    toc
    end        
end

%% Combine cells for each type
locations = cell(num_types,1);
for i = 1:length(locations)
    locations{i} = zeros(num_cells{i}(end,1),6);
    counter = 1;
    cells = 0;
    for section = 1:4
        for j = 1:10
            if ~isempty(locations_bysection{section}{j}{i})
                cells = size(locations_bysection{section}{j}{i}(:,:),1);
                locations{i}(counter:(counter+cells-1),:) = locations_bysection{section}{j}{i}(:,1:6);
                counter = counter + cells;
            end
        end
    end
end

%% Save files
save('Outputs/Locations.mat','locations','-v7.3');