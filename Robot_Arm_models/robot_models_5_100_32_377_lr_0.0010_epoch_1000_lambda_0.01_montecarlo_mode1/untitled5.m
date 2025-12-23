%% Batch rename variables & save as NEW files (originals preserved)
clear; clc;

folder_path = pwd;                     % current folder
pattern = 'networks_*.mat';            % your files
new_suffix = '_NNSS';                  % optional comment to ADD

files = dir(fullfile(folder_path, pattern));
assert(~isempty(files), 'No .mat files found.');

for k = 1:numel(files)

    old_name = files(k).name;
    old_path = fullfile(files(k).folder, old_name);

    fprintf('\nProcessing: %s\n', old_name);

    % -------- load everything --------
    A = load(old_path);

    % -------- rename variables if they exist --------
    if isfield(A,'net_sertbas')
        A.net_NNSS = A.net_sertbas;
        A = rmfield(A,'net_sertbas');
        fprintf('  net_sertbas -> net_NNSS\n');
    end

    if isfield(A,'loss_values_sertbas')
        A.loss_values_NNSS = A.loss_values_sertbas;
        A = rmfield(A,'loss_values_sertbas');
        fprintf('  loss_values_sertbas -> loss_values_NNSS\n');
    end

    if isfield(A,'val_loss_values_sertbas')
        A.val_loss_values_NNSS = A.val_loss_values_sertbas;
        A = rmfield(A,'val_loss_values_sertbas');
        fprintf('  val_loss_values_sertbas -> val_loss_values_NNSS\n');
    end

    % -------- generate NEW filename --------
    [~, base, ext] = fileparts(old_name);

    % append new optional comment BEFORE .mat
    new_name = [base, new_suffix, ext];
    new_path = fullfile(files(k).folder, new_name);

    % -------- save NEW file --------
    save(new_path, '-struct', 'A');

    fprintf('  Saved NEW file: %s\n', new_name);
end

fprintf('\nDONE: Originals untouched, renamed copies created.\n');
