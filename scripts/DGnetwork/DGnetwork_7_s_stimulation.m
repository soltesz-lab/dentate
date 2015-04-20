
%function DGnetwork_9_s_stimulation(slice,epilepsy,input,total_processors)

epilepsy = '1';
slice = '1';
e_distance = 200;
if str2double(slice) == 0 && str2double(epilepsy) == 0
    load('Outputs/Locations.mat');
    load('Outputs/Syn_Connections.mat');
    load('Outputs/GJ_Connections.mat');
elseif str2double(slice) == 1 && str2double(epilepsy) == 0
    load('./Outputs/Slice_50_300_Control/Slice_Locations.mat');
    load('./Outputs/Slice_50_300_Control/Slice_Indexes.mat');
    %tree = load_tree('Outputs/Slice_Trees.mtr');
elseif str2double(slice) == 0 && str2double(epilepsy) == 1
    load('./Outputs/Epileptic/Locations.mat');
elseif str2double(slice) == 1 && str2double(epilepsy) == 1
    load('./Outputs/Slice_50_300_Epileptic/Slice_Locations.mat');
    load('./Outputs/Slice_50_300_Epileptic/Slice_Indexes.mat');
end
%input_pt = locations{2}(130,:);

stim_cells = cell(length(locations),1);
for i = 1:length(locations)
    distances = sqrt((locations{i}(:,1)-input_pt(1)).^2 + (locations{i}(:,2)-input_pt(2)).^2 + (locations{i}(:,3)-input_pt(3)).^2);
    stim_cells{i} = keep_indexes{i}(distances < e_distance);
end
%{
for i = 1:length(locations)
    counter = 1;
    if i ==1
        stim_cells{i} = zeros(size(locations{i},1),1);
        for j = 1:size(locations{i},1)
            distances = sqrt((tree{j}.X-input_pt(1)).^2 + (tree{j}.Y-input_pt(2)).^2 + (tree{j}.Z-input_pt(3)).^2);
            if any(distances < e_distance)
                stim_cells{i}(j,1) = j;
            end
        end
        stim_cells{i}(stim_cells{i}==0) = [];
    else
        distances = sqrt((locations{i}(:,1)-input_pt(1)).^2 + (locations{i}(:,2)-input_pt(2)).^2 + (locations{i}(:,3)-input_pt(3)).^2);
        stim_cells{i} = find(distances < e_distance);
    end
end
%}