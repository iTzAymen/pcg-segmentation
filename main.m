% Specify the folders containing the .wav files
% folderList = {'PhysioNet/training/training-a', 'PhysioNet/training/training-b', 'PhysioNet/training/training-c', 'PhysioNet/training/training-d', 'PhysioNet/training/training-e', 'PhysioNet/training/training-f'};
folderList = {'PASCAL/abnormal', 'PASCAL/normal'};
% Loop through each folder
for j = 1:length(folderList)
    % Get a list of all .wav files in the folder
    folderPath = folderList{j};
    fileList = dir(fullfile(folderPath, '*.wav'));
    for i = 1:length(fileList)
        % Get the file name
        [~, fileName, ~] = fileparts(fileList(i).name);
        assigned_states = getLabels(folderPath, fileName);
        writematrix(assigned_states, [folderPath '/' fileName '.csv']);
    end
end