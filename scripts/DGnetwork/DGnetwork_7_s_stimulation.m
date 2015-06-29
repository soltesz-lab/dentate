
%function DGnetwork_7_s_stimulation(directory,slice,epilepsy,input,total_processors)

epilepsy = '1';
slice = '1';
e_distance = 200;

directory = '/som/iraikov/dentate/Slice_50_500_Control';
directory = '/som/iraikov/dentate/Slice_50_500_Epileptic';

if str2double(slice) == 0 && str2double(epilepsy) == 0
  load(sprintf('%s/Locations.mat',directory));
  load(sprintf('%s/Syn_Connections.mat',directory));
  load(sprintf('%s/GJ_Connections.mat',directory));
elseif str2double(slice) == 1 && str2double(epilepsy) == 0
  load(sprintf('%s/Slice_Locations.mat',directory));
  load(sprintf('%s/Slice_Indexes.mat',directory));
elseif str2double(slice) == 0 && str2double(epilepsy) == 1
  load(sprintf('%s/Locations.mat',directory));
elseif str2double(slice) == 1 && str2double(epilepsy) == 1
  load(sprintf('%s/Slice_Locations.mat',directory));
  load(sprintf('%s/Slice_Indexes.mat',directory));
end

input_pt = locations{2}(130,:);

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

save(sprintf('%s/Stimulation.mat',directory),'stim_cells','-v6');
