function [assigned_states] = getLabels(folderPath,recordName)
    load('Springer_B_matrix.mat');
    load('Springer_pi_vector.mat');
    load('Springer_total_obs_distribution.mat');
    
    %% Load data and resample data
    springer_options   = default_Springer_HSMM_options;
    springer_options.use_mex = 1;
    [PCG,Fs1] = audioread([folderPath '/' recordName '.wav']);  % load data
                
    % resample to 1000 Hz
    PCG_resampled = resample(PCG,springer_options.audio_Fs,Fs1); % resample to springer_options.audio_Fs (1000 Hz)
    % filter the signal between 25 to 400 Hz
    PCG_resampled = butterworth_low_pass_filter(PCG_resampled,2,400,springer_options.audio_Fs, false);
    PCG_resampled = butterworth_high_pass_filter(PCG_resampled,2,25,springer_options.audio_Fs);
    % remove spikes
    PCG_resampled = schmidt_spike_removal(PCG_resampled,springer_options.audio_Fs);
    
    %% Running runSpringerSegmentationAlgorithm.m to obtain the assigned_states
    assigned_states = runSpringerSegmentationAlgorithm(PCG_resampled,... 
                    springer_options.audio_Fs,... 
                    Springer_B_matrix, Springer_pi_vector,...
                    Springer_total_obs_distribution, false);
end

