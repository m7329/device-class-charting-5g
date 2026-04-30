% wrapper function to run SM_fasta.m for specified labels saved in JSON file and save the results to the scratch directory


closedSetDir = '/home/bsc26f18/DeviceClassCharting/device-classification-5g/Closed_set_RFFI';
scratchDir   = '/scratch/bsc26f18/sammon_scratch';
addpath(closedSetDir);
addpath(fullfile(closedSetDir, 'matlab_files'));

input_mat  = fullfile(scratchDir, ['d_cov.mat']);
output_mat = fullfile(scratchDir, ['mappedX.mat']);
run_SM_fasta_wrapper(input_mat, output_mat);
fprintf('Done: %s -> %s\n', input_mat, output_mat);