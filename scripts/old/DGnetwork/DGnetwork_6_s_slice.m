
%function DGnetwork_6_s_slice(percent,epilepsy,width,directory)

percent = '50';
epilepsy = '1';
width = '500';
trees_directory = '/som/iraikov';
full_scale_directory = '/som/iraikov/dentate/Full_Scale_Control';
output_directory = '/som/iraikov/dentate/Slice_50_500_Control';
output_directory = '/som/iraikov/dentate/Slice_50_500_Epileptic';

rng(1)

% Draw line to estimate septotemporal extent
[x,y,z]  = layer_eq_line(0,1000000);
u          = linspace(pi*1/100,pi*98/100,1000000);
sums        = zeros(length(x)-1,1);
for i = 1:length(x)-1
    d = pdist2([x(i),y(i),z(i)],[x(i+1),y(i+1),z(i+1)]);
    switch i
        case 1
            sums(i,1) = d;
        otherwise
            sums(i,1) = d + sums(i-1,1);
    end
end
sums = [0;sums];

% Get septotemporal center
[~,center_index]                = min(abs(sums - (str2double(percent)/100*max(sums))));
u_center                        = u(center_index);
clearvars sums x y z

% Get points to make up plane
plane_pts    = zeros(3,3);
% Get longitudinal points
[x_pt1,y_pt1,z_pt1]             = layer_eq_point(0,u_center-0.001,59.75*pi/100);
[x_center,y_center,z_center]    = layer_eq_point(0,u_center,59.75*pi/100);
[x_pt2,y_pt2,z_pt2]             = layer_eq_point(0,u_center+0.001,59.75*pi/100);
plane_pts(1,:)                  = [x_center,y_center,z_center];
direction                       = [x_pt2-x_pt1,y_pt2-y_pt1,z_pt2-z_pt1];
unit_direction                  = direction/norm(direction);
full_direction                  = unit_direction*str2double(width)/2;
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


% Load in full-scale locations and connections
load(sprintf('%s/Locations.mat',full_scale_directory));
load(sprintf('%s/Syn_Connections.mat',full_scale_directory));
load(sprintf('%s/GJ_Connections.mat',full_scale_directory));
if str2double(epilepsy) == 0
    connection_M{1,1}   = [];
    hilus_percent       = 1;
else  
    hilus_percent       = 0.2;
end

% Keep only those cells that are centered within the slice for each cell type
new_locations   = cell(size(locations,1),1);
keep_indexes    = cell(size(locations,1),1);
for i = 1:size(locations,1)
    keep_indexes{i}     = find(locations{i}(:,1:3)*transpose(plane) + start_d < 0 & locations{i}(:,1:3)*transpose(plane) + stop_d > 0);
    new_locations{i}    = zeros(size(keep_indexes{i},1),6);
    new_locations{i}    = locations{i}(keep_indexes{i},:);    
    switch i 
        case 1
        case {2,3}
            if str2double(epilepsy) == 1
                keep_cells          = transpose(randperm(length(keep_indexes{i}),round(hilus_percent*length(keep_indexes{i}))));
                keep_indexes{i}     = keep_indexes{i} + size(vertcat(locations{1:i-1}),1);
                keep_indexes{i}     = keep_indexes{i}(keep_cells);
                new_locations{i}    = locations{i}(keep_cells,:);
            else 
                keep_indexes{i}     = keep_indexes{i} + size(vertcat(locations{1:i-1}),1);
            end
        otherwise
            keep_indexes{i}     = keep_indexes{i} + size(vertcat(locations{1:i-1}),1);
    end
end
all_keep_indexes = vertcat(keep_indexes{:});

% Only keep connections where pre- and postsynaptic cells are both within slice
new_connection_M    = cell(size(locations,1),size(locations,1));
new_gap_junctions   = cell(size(locations,1),size(locations,1));
for pre = 1:size(locations,1)
    for post = 1:size(locations,1)
        if ~isempty(connection_M{pre,post})
            all_connections             = double(connection_M{pre,post});
            new_connection_M{pre,post}  = all_connections(ismembc(all_connections(:,1),all_keep_indexes) & ismembc(all_connections(:,2),all_keep_indexes),:);
        end
        if ~isempty(gap_junctions{pre,post})
            all_gap_junctions           = double(gap_junctions{pre,post});
            new_gap_junctions{pre,post} = all_gap_junctions(ismembc(all_gap_junctions(:,1),all_keep_indexes) & ismembc(all_gap_junctions(:,2),all_keep_indexes),:);
        end
    end
end

% First load in granule cells with cell bodies in the layer
tree_subset_size    = 1000;
GC_indices  = keep_indexes{1};
GC_files    = ceil(GC_indices/tree_subset_size);
cut_trees   = cell(size(GC_indices,1),1);
counter     = 1;
for i = min(GC_files):max(GC_files);
    if any(GC_files == i)
        trees   = load_tree(sprintf('%s/Trees_Tapered/%i.mtr',trees_directory,i));
        cut_trees(counter:(counter+size(find(GC_files==i),1)-1)) = trees((GC_indices(GC_files == i,1)-tree_subset_size*(i-1)));
        counter = counter + size(find(GC_files==i),1);
    end
end

% Remove the trees without dendrites from slice
new_cut_trees   = cell(size(cut_trees,1),1);
trees_to_remove = zeros(size(cut_trees,1),1);
for i = 1:size(cut_trees,1)
    out_pts = ~([cut_trees{i}.X cut_trees{i}.Y cut_trees{i}.Z]*transpose(plane) + start_d < 0 & [cut_trees{i}.X cut_trees{i}.Y cut_trees{i}.Z]*transpose(plane) + stop_d > 0);
    if any(out_pts)
        ipar                = ipar_tree(cut_trees{i});
        [rows,~]            = find(ismember(ipar,find(out_pts)));
        delete_indices      = unique(rows);
        new_cut_trees{i}    = delete_tree(cut_trees{i},delete_indices);
        if length(new_cut_trees{i}.X) < 2
            trees_to_remove(i,1) = 1;
        end
    else
        new_cut_trees{i}    = cut_trees{i};
    end
end
trees_to_remove = find(trees_to_remove);
new_cut_trees(trees_to_remove)  = [];
new_locations{1}(trees_to_remove,:)  = [];
for i = 1:length(new_connection_M)
    for j = 1:length(new_connection_M)
        [rows,~]                            = find(ismember(new_connection_M{i,j},all_keep_indexes(trees_to_remove)));
        delete_indices                      = unique(rows);
        new_connection_M{i,j}(delete_indices,:)  = [];
    end
end
keep_indexes{1}(trees_to_remove) = [];

% Reset variables before saving
locations       = new_locations;
connection_M    = new_connection_M;
gap_junctions   = new_gap_junctions;
tree            = new_cut_trees; 

% Save outputs
save(sprintf('%s/Slice_Locations.mat',output_directory),'locations','-v6');
save(sprintf('%s/Slice_Indexes.mat',output_directory),'keep_indexes','-v6');
save(sprintf('%s/Slice_Trees.mtr',output_directory),'tree','-v7.3');
save(sprintf('%s/Slice_Syn_Connections.mat',output_directory),'connection_M','-v6');
save(sprintf('%s/Slice_Gap_Junctions.mat',output_directory),'gap_junctions','-v6');

