%% Stable-by-Design NN-SS / LPV (Research Code) — TRAINING ONLY
% =========================================================================
% This script test and plots:
%   (1) NN-SS / NNSS model
%   (2) SIMBa baseline
%   (3) Classical subspace baselines: SSREGEST / N4SID / SSEST
%
% Expected workspace variables BEFORE running:
%   Data_test : [N x (ny + nu + ny)] columns: y(k), u(k), y(k+1)
% =========================================================================


% Define the parameters used in training
number_of_states = 5;
number_of_inputs = 1;
number_of_outputs = 1;

numSamples = 100;
batchSize = 32;
learnRate = 0.001;
numEpochs = 1000;
totalTrajectories = 377;
Ts = 0.005;
% Ts = 1228.8;
optional_comment = 'montecarlo';
normalization ='zscore';
mode= 1;
lambda = 0.01;

test_data = Data_test;

if ~isempty(optional_comment)
    comment_part = sprintf('_%s', strrep(optional_comment, ' ', '_'));
else
    comment_part = '';
end

folder_to_load = sprintf('robot_models_%d_%d_%d_%d_lr_%.4f_epoch_%d_lambda_%.2f%s_mode%d', ...
    number_of_states, numSamples, batchSize, totalTrajectories, learnRate, numEpochs, lambda, comment_part, mode);

if exist(folder_to_load, 'dir') ~= 7
    error('The folder "%s" does not exist.', folder_to_load);
end

mat_files = dir(fullfile(folder_to_load, sprintf('*_mode%d.mat', mode)));
if isempty(mat_files)
    error('No model files for mode %d found in "%s".', mode, folder_to_load);
end

model_path = fullfile(folder_to_load, mat_files(1).name);
load(model_path);

num_models = length(mat_files);
rmse_results = zeros(num_models, number_of_outputs,5);
simulation_results = zeros(num_models, size(test_data,1), number_of_outputs,5);

for i = 1:num_models
    model_filename = fullfile(folder_to_load, mat_files(i).name);
    fprintf('Loading model from: %s\n', model_filename);

    nets = load(model_filename);

    net_NNSS = nets.net_NNSS;
    map_x0_net = nets.map_x0_net;
    sys_simba = nets.sys_simba;
    sys_ssregest = nets.sys_ssregest;
    sys_n4sid = nets.sys_n4sid;
    sys_ssest = nets.sys_ssest;
    stats = nets.stats;
    number_of_states = nets.number_of_states;
    mode = nets.mode;

    loss_values_NNSS = nets.loss_values_NNSS;
    val_loss_values_NNSS = nets.val_loss_values_NNSS;
    loss_values_simba = nets.loss_values_simba;
    val_loss_values_simba = nets.val_loss_values_simba;

    % --------- NNSS LOSS PLOTS ---------
    figure('Name', sprintf('NNSS Loss Curves - Model %d', i), 'NumberTitle', 'off');
    subplot(2,1,1);
    plot(loss_values_NNSS, 'b-', 'LineWidth', 1);
    title('NNSS - Training Loss');
    xlabel('Iteration'); ylabel('Loss'); grid on;

    subplot(2,1,2);
    plot(val_loss_values_NNSS, 'r-', 'LineWidth', 1.5);
    title('NNSS - Validation RMSE');
    xlabel('Epoch'); ylabel('RMSE'); grid on;

    % --------- SIMBA LOSS PLOTS ---------
    figure('Name', sprintf('SIMBA Loss Curves - Model %d', i), 'NumberTitle', 'off');
    subplot(2,1,1);
    plot(loss_values_simba, 'b-', 'LineWidth', 1);
    title('SIMBA - Training Loss');
    xlabel('Iteration'); ylabel('Loss'); grid on;

    subplot(2,1,2);
    plot(val_loss_values_simba, 'r-', 'LineWidth', 1.5);
    title('SIMBA - Validation RMSE');
    xlabel('Epoch'); ylabel('RMSE'); grid on;

     y_inference = test_data(:, 1:number_of_outputs); % Actual output (y_k)
    u_inference = test_data(:,number_of_outputs+1 : number_of_outputs + number_of_inputs); % Input (u_k)
    t_inference = 0:Ts:(length(u_inference) - 1) * Ts;

    switch lower(normalization)
        case 'zscore'
            u_prediction_norm = (u_inference - stats.U.mu) ./ stats.U.sigma;
            y_prediction_norm = (y_inference - stats.Yk.mu) ./ stats.Yk.sigma;

        case 'minmax'
            u_prediction_norm = (u_inference - stats.U.min) ./ (stats.U.max - stats.U.min);
            y_prediction_norm = (y_inference - stats.Yk.min) ./ (stats.Yk.max - stats.Yk.min);

        otherwise
            error('Unsupported normalization method. Use "zscore" or "minmax".');
    end

   

    %% **Simulation Mode for SIMBa & subspace methods**
    U_test = test_data(:, number_of_outputs+1:number_of_outputs+number_of_inputs);
    t_sim = 0:Ts:(size(U_test,1) - 1) * Ts;

    switch lower(normalization)
        case 'zscore'
            U_test_norm = (U_test - stats.U.mu) ./ stats.U.sigma;
        case 'minmax'
            U_test_norm = (U_test - stats.U.min) ./ (stats.U.max - stats.U.min);
        otherwise
            error('Unsupported normalization method. Use "zscore" or "minmax".');
    end

    dlY0 = dlarray(y_prediction_norm(1,:));
    X0_simba = pinv(sys_simba.C) * (dlY0' - sys_simba.D * u_prediction_norm(1,:)');

    [Y_Simba_norm, ~, ~] = lsim(sys_simba, U_test_norm, t_sim, X0_simba);

    switch lower(normalization)
        case 'zscore'
            Y_Simba = Y_Simba_norm .* stats.Yk.sigma + stats.Yk.mu;
        case 'minmax'
            Y_Simba = Y_Simba_norm .* (stats.Yk.max - stats.Yk.min) + stats.Yk.min;
        otherwise
            error('Unsupported normalization method. Use "zscore" or "minmax".');
    end

    % Normalize inputs for test simulation (baselines)
    switch lower(normalization)
        case 'zscore'
            U_test_norms = (test_data(:, number_of_outputs+1 : number_of_outputs+number_of_inputs) - stats.U.mu) ./ stats.U.sigma;
            y_prediction_norm = (test_data(:, 1:number_of_outputs) - stats.Yk.mu) ./ stats.Yk.sigma;
        case 'minmax'
            U_test_norms = (test_data(:, number_of_outputs+1 : number_of_outputs+number_of_inputs) - stats.U.min) ./ (stats.U.max - stats.U.min);
            y_prediction_norm = (test_data(:, 1:number_of_outputs) - stats.Yk.min) ./ (stats.Yk.max - stats.Yk.min);
        otherwise
            error('Unsupported normalization method. Use "zscore" or "minmax".');
    end

    initial_data = iddata(y_prediction_norm(1,:), U_test_norms(1,:), Ts);
    ss_data_test = iddata([], U_test_norms, Ts);

    x0_ssregest   = findstates(sys_ssregest, initial_data);
    Y_ssregest_norm = sim(sys_ssregest, ss_data_test, x0_ssregest);

    x0_n4sid      = findstates(sys_n4sid, initial_data);
    Y_n4sid_norm  = sim(sys_n4sid, ss_data_test, x0_n4sid);

    x0_ssest      = findstates(sys_ssest, initial_data);
    Y_ssest_norm  = sim(sys_ssest, ss_data_test, x0_ssest);

    switch lower(normalization)
        case 'zscore'
            Y_ssregest = Y_ssregest_norm.OutputData .* stats.Yk.sigma + stats.Yk.mu;
            Y_n4sid    = Y_n4sid_norm.OutputData    .* stats.Yk.sigma + stats.Yk.mu;
            Y_ssest    = Y_ssest_norm.OutputData    .* stats.Yk.sigma + stats.Yk.mu;
        case 'minmax'
            Y_ssregest = Y_ssregest_norm.OutputData .* (stats.Yk.max - stats.Yk.min) + stats.Yk.min;
            Y_n4sid    = Y_n4sid_norm.OutputData    .* (stats.Yk.max - stats.Yk.min) + stats.Yk.min;
            Y_ssest    = Y_ssest_norm.OutputData    .* (stats.Yk.max - stats.Yk.min) + stats.Yk.min;
    end

    %% **Simulation Mode for NNSS with Mode Support**
    A_matrix_array = zeros(number_of_states, number_of_states, size(U_test,1));
    X_state_array  = zeros(number_of_states, size(U_test,1));

    dlY0 = dlarray(y_prediction_norm(1,:)', 'CB');
    dlX0_NNSS = forward(map_x0_net, dlY0);

    Y_NNSS = zeros(size(U_test,1), number_of_outputs);

    switch mode
        case 1
            input_feat = dlX0_NNSS;
        case 2
            input_feat = [dlX0_NNSS; dlarray(u_prediction_norm(1,:)', 'CB')];
        case 3
            input_feat = [dlX0_NNSS; dlarray(y_prediction_norm(1,:)', 'CB')];
        case 4
            input_feat = [dlX0_NNSS; dlarray(u_prediction_norm(1,:)', 'CB'); dlarray(y_prediction_norm(1,:)', 'CB')];
    end

    input_feat = dlarray(input_feat, 'CB');
    ss_parameters = forward(net_NNSS, input_feat);
    [~, ~, C_matrix, dlYPred, dlXPred] = intermediateStateSpace_simulation(ss_parameters, dlX0_NNSS, ...
        u_prediction_norm(1,:)', number_of_states, number_of_inputs, number_of_outputs); 

    switch lower(normalization)
        case 'zscore'
            Y_NNSS(1,:) = double(dlYPred)' .* stats.Yk.sigma + stats.Yk.mu;
        case 'minmax'
            Y_NNSS(1,:) = double(dlYPred)' .* (stats.Yk.max - stats.Yk.min) + stats.Yk.min;
        otherwise
            error('Unsupported normalization method. Use "zscore" or "minmax".');
    end

    dlX0_NNSS = dlarray(dlXPred, 'CB');

    for k = 2:size(U_test,1)
        switch mode
            case 1
                input_feat = dlX0_NNSS;
            case 2
                input_feat = [dlX0_NNSS; dlarray(u_prediction_norm(k,:)', 'CB')];
            case 3
                input_feat = [dlX0_NNSS; dlarray(y_prediction_norm(k-1,:)', 'CB')];
            case 4
                input_feat = [dlX0_NNSS; dlarray(u_prediction_norm(k,:)', 'CB'); dlarray(y_prediction_norm(k-1,:)', 'CB')];
        end
        input_feat = dlarray(input_feat, 'CB');
        ss_parameters = forward(net_NNSS, input_feat);
        [A_matrix, ~, ~, dlYPred, dlXPred] = intermediateStateSpace_simulation(ss_parameters, dlX0_NNSS, ...
            u_prediction_norm(k,:)', number_of_states, number_of_inputs, number_of_outputs);

        A_matrix_array(:,:,k) = extractdata(A_matrix);
        X_state_array(:,k)    = double(dlXPred);

        switch lower(normalization)
            case 'zscore'
                Y_NNSS(k,:) = double(dlYPred)' .* stats.Yk.sigma + stats.Yk.mu;
            case 'minmax'
                Y_NNSS(k,:) = double(dlYPred)' .* (stats.Yk.max - stats.Yk.min) + stats.Yk.min;
        end

        dlX0_NNSS = dlarray(dlXPred, 'CB');
    end

    %% **Store RMSE Results**
    for output_idx = 1:number_of_outputs
        rmse_results(i, output_idx, 1) = sqrt(mean((Y_NNSS(:, output_idx)    - y_inference(:, output_idx)).^2));
        rmse_results(i, output_idx, 2) = sqrt(mean((Y_Simba(:, output_idx)   - y_inference(:, output_idx)).^2));
        rmse_results(i, output_idx, 3) = sqrt(mean((Y_ssregest(:, output_idx)- y_inference(:, output_idx)).^2));
        rmse_results(i, output_idx, 4) = sqrt(mean((Y_n4sid(:, output_idx)   - y_inference(:, output_idx)).^2));
        rmse_results(i, output_idx, 5) = sqrt(mean((Y_ssest(:, output_idx)   - y_inference(:, output_idx)).^2));
    end

    %% **Store Simulation Results**
    for t = 1:size(y_inference,1)
        for o = 1:number_of_outputs
            simulation_results(i, t, o, 1) = Y_NNSS(t, o);
            simulation_results(i, t, o, 2) = Y_Simba(t, o);
            simulation_results(i, t, o, 3) = Y_ssregest(t, o);
            simulation_results(i, t, o, 4) = Y_n4sid(t, o);
            simulation_results(i, t, o, 5) = Y_ssest(t, o);
        end
    end

    fprintf('Results stored for model %d\n', i);
end

assignin('base', 'rmse_results', rmse_results);
assignin('base', 'simulation_results', simulation_results);

%% RMSE Box Plots and Simulation Results

plot_rmse_box_adaptive(rmse_results);

plot_best_seed_simulations_from_rmse(simulation_results, y_inference, rmse_results);

plot_best_seed_states_from_rmse(folder_to_load, mat_files, rmse_results, test_data, Ts, normalization);



%% FUNCTIONS

function [A_matrix, B_matrix, C_matrix, Z_response, Z_state] = intermediateStateSpace_simulation(X1, dlX, U, number_of_states, number_of_inputs, number_of_outputs)

    mbs=1;
    gamma = 1;
    epsilon = exp(-10);

    W = reshape(X1(1:4*number_of_states^2, :), 2*number_of_states, 2*number_of_states, mbs);
    V = reshape(X1(4*number_of_states^2+1:5*number_of_states^2, :), number_of_states, number_of_states, mbs);

    S = pagemtimes(W, pagetranspose(W)) + epsilon * eye(2 * number_of_states);

    S11 = S(1:number_of_states, 1:number_of_states, :);
    S12 = S(1:number_of_states, number_of_states+1:end, :);
    S22 = S(number_of_states+1:end, number_of_states+1:end, :);

    inv_term = 0.5 * (S11/(gamma^2) + S22) + V - pagetranspose(V);
    inv_term_cell = num2cell(inv_term, [1 2]);

    a = cellfun(@(inv_term_cell) pinv(inv_term_cell), inv_term_cell,'UniformOutput',false);
    inv_term_inv = cat(3, a{:});

    A_matrix = pagemtimes(S12,inv_term_inv);

    B_matrix = reshape(X1(5*number_of_states^2 + 1:5*number_of_states^2 + number_of_states * number_of_inputs, :), ...
        number_of_states, number_of_inputs, mbs);

    C_matrix = reshape(X1(5*number_of_states^2 + number_of_states * number_of_inputs+1 : ...
        5*number_of_states^2 + number_of_states*number_of_inputs + number_of_outputs *number_of_states, :), ...
        number_of_outputs, number_of_states, mbs);

    X_State = reshape(dlX, number_of_states, 1, []);
    U_Vector = reshape(U, size(U,1), 1, []);

    AX = pagemtimes(A_matrix, X_State);
    BU = pagemtimes(B_matrix, U_Vector);

    Z_state = AX + BU;
    Z_response = pagemtimes(C_matrix, X_State);
end

function plot_rmse_box_adaptive(rmse_results)

    assert(~isempty(rmse_results), 'rmse_values is empty.');
    assert(isnumeric(rmse_results), 'rmse_values must be numeric.');

    nd = ndims(rmse_results);

    if nd == 3
        [S,O,M] = size(rmse_results); 
    elseif nd == 2
        [S,M] = size(rmse_results);
        rmse_results = reshape(rmse_results, S, 1, M);
    else
        error('rmse_values must be SxOxM or SxM.');
    end

    assert(M==5, '3rd dim (methods) must be 5: (NN-SS,SIMBa,SSREGEST,N4SID,SSEST).');

    model_names = {'NN-SS','SIMBa','SSREGEST','N4SID','SSEST'};
    iSSNN = 1; iSIMB = 2; iOTH = 3:5;

    % ---- reduce over outputs: average per seed per method ----
    rm = squeeze(mean(rmse_results, 2, 'omitnan'));
    if isvector(rm)
        rm = reshape(rm, [], M);
    end

    rSSNN = rm(:, iSSNN); rSSNN = rSSNN(~isnan(rSSNN));
    rSIMB = rm(:, iSIMB); rSIMB = rSIMB(~isnan(rSIMB));

    othersMean = nan(1, numel(iOTH));
    for j = 1:numel(iOTH)
        othersMean(j) = mean(rm(:, iOTH(j)), 'omitnan');
    end

    % ---- figure style ----
    colors = lines(numel(model_names));
    dx  = 0.18;
    jit = 0.06;
    spanDiam = 0.08;

    figure('Name','RMSE','NumberTitle','off'); hold on; grid on;

    x = 1;

    % Draw NN-SS boxplot alone
    prevBoxes = findobj(gca,'Tag','Box');
    boxplot(rSSNN, 'Positions', x-dx, 'Widths', 0.18, 'Symbol','');
    set(findobj(gca,'Tag','Median'),'Color',[0 0 0],'LineWidth',1.2);
    newBoxes = setdiff(findobj(gca,'Tag','Box'), prevBoxes);
    if ~isempty(newBoxes)
        % pick the newest box (there should be 1)
        [~,idx] = max(arrayfun(@(h) mean(get(h,'XData')), newBoxes));
        b1 = newBoxes(idx);
        patch(get(b1,'XData'), get(b1,'YData'), colors(iSSNN,:), ...
              'FaceAlpha',0.15,'EdgeColor',colors(iSSNN,:), 'LineWidth',1.2);
    end

    % Draw SIMBa boxplot alone
    prevBoxes = findobj(gca,'Tag','Box');
    boxplot(rSIMB, 'Positions', x+dx, 'Widths', 0.18, 'Symbol','');
    set(findobj(gca,'Tag','Median'),'Color',[0 0 0],'LineWidth',1.2);
    newBoxes = setdiff(findobj(gca,'Tag','Box'), prevBoxes);
    if ~isempty(newBoxes)
        [~,idx] = max(arrayfun(@(h) mean(get(h,'XData')), newBoxes));
        b2 = newBoxes(idx);
        patch(get(b2,'XData'), get(b2,'YData'), colors(iSIMB,:), ...
              'FaceAlpha',0.15,'EdgeColor',colors(iSIMB,:), 'LineWidth',1.2);
    end

    % ---- seed scatters ----
    rng(0);
    s1 = scatter((x-dx) + jit*(rand(size(rSSNN))-0.5), rSSNN, 42, colors(iSSNN,:), 'filled', 'MarkerFaceAlpha',0.9);
    s2 = scatter((x+dx) + jit*(rand(size(rSIMB))-0.5), rSIMB, 42, colors(iSIMB,:), 'filled', 'MarkerFaceAlpha',0.9);

    % ---- diamonds for other methods ----
    xd = x + linspace(-spanDiam, spanDiam, numel(iOTH));
    dH = gobjects(1,numel(iOTH));
    for j = 1:numel(iOTH)
        dH(j) = scatter(xd(j), othersMean(j), 80, colors(iOTH(j),:), 'filled', 'd');
    end

    % ---- axes formatting ----
    xticks(1);
    xticklabels("RMSE");
    xlim([0.4 1.6]);

    allY = [rSSNN(:); rSIMB(:); othersMean(:)];
    if all(isnan(allY)), allY = 1; end
    ylim([0, max(allY)*1.10 + eps]);

    ylabel('RMSE');
    legend([s1 s2 dH], {'NN-SS','SIMBa','SSREGEST','N4SID','SSEST'}, ...
        'Location','northoutside','Orientation','horizontal');

    hold off;

    % ---- best seed ----
    [vS, iS] = min(rm(:,iSSNN), [], 'omitnan');
    [vB, iB] = min(rm(:,iSIMB), [], 'omitnan');
    fprintf('\n[Best seed (avg over outputs)]\n');
    fprintf('  NN-SS : seed #%d (RMSE=%.6f)\n', iS, vS);
    fprintf('  SIMBa : seed #%d (RMSE=%.6f)\n', iB, vB);

end

function [bestSeedIdx, bestRMSE] = plot_best_seed_simulations_from_rmse(simulation_results, y_true, rmse_results)

    % ---------------- sanity ----------------
    assert(ndims(simulation_results)==4, 'simulation_results must be SxTxOxM');
    [S,T,O,M] = size(simulation_results);
    assert(M==5, 'simulation_results 4th dim must be 5 methods');
    assert(size(y_true,1)==T, 'T mismatch: y_true must be TxO');
    assert(size(y_true,2)==O, 'O mismatch: y_true must be TxO');

    % rmse_results -> force to S x O x M
    nd = ndims(rmse_results);
    if nd==2
        assert(size(rmse_results,1)==S, 'rmse_results S mismatch');
        assert(size(rmse_results,2)==M, 'rmse_results M mismatch');
        rmse_results = reshape(rmse_results, S, 1, M);
    elseif nd==3
        assert(size(rmse_results,1)==S, 'rmse_results S mismatch');
        assert(size(rmse_results,3)==M, 'rmse_results M mismatch');
    else
        error('rmse_results must be SxOxM or SxM.');
    end

    method_names  = {'NN-SS','SIMBa','SSREGEST','N4SID','SSEST'};
    method_colors = {'r','b','m','g','c'};

    % ---------------- pick best seeds using rmse_results ----------------
    rmse_seed_method = squeeze(mean(rmse_results, 2, 'omitnan')); % SxM
    if isvector(rmse_seed_method)
        rmse_seed_method = reshape(rmse_seed_method, S, M);
    end

    bestSeedIdx = zeros(1,M);
    bestRMSE    = zeros(1,M);
    for m = 1:M
        [bestRMSE(m), bestSeedIdx(m)] = min(rmse_seed_method(:,m), [], 'omitnan');
    end

    fprintf('\n[Best seeds (selected by rmse_results, avg over outputs)]\n');
    for m = 1:M
        fprintf('  %-8s : seed #%d | RMSE = %.6f\n', method_names{m}, bestSeedIdx(m), bestRMSE(m));
    end

    % ---------------- collect best simulations ----------------
    % Force each Ybest{m} to be TxO (not 1xTxO etc.)
    Ybest = cell(1,M);
    for m = 1:M
        Ym = squeeze(simulation_results(bestSeedIdx(m),:,:,m)); % could be TxO or T or 1xT
        if isvector(Ym)
            Ym = Ym(:);              % Tx1
        end
        if size(Ym,1) ~= T && size(Ym,2) == T
            Ym = Ym.';               % transpose if it's 1xT
        end
        % if O==1 and still ends up Tx1, OK. If TxO, OK.
        Ybest{m} = Ym;
    end

    % ---------------- plot ----------------
    figure('Name','Best-Seed Simulations vs Actual (seed chosen by RMSE code)','NumberTitle','off');
    idx = (1:T).';

    for o = 1:O
        subplot(O,1,o); hold on; grid on;

        % Actual (force Tx1)
        yplot = y_true(:,o);
        yplot = yplot(:);
        hActual = plot(idx, yplot, 'k', 'LineWidth', 2, 'DisplayName','Actual');

        % Methods
        hM = gobjects(1,M);
        for m = 1:M
            yhat = Ybest{m};
            % pick output o robustly
            if size(yhat,2) >= o
                yhat_o = yhat(:,o);
            else
                % if somehow yhat is Tx1 but o>1, skip
                yhat_o = nan(T,1);
            end
            yhat_o = yhat_o(:); % force Tx1

            p = plot(idx, yhat_o, method_colors{m}, ...
                'LineWidth', 2, ...
                'DisplayName', sprintf('%s (seed %d)', method_names{m}, bestSeedIdx(m)));

            % plot might return multiple handles -> keep the first one
            hM(m) = p(1);
        end

        ylabel(sprintf('Output %d', o));
        if o==O, xlabel('Sample Index'); end

        % legend only once (prevents >50 warning)
        if o==1
            legend([hActual hM], 'Location','northoutside','Orientation','horizontal');
        else
            legend off;
        end

        ax = gca;
        box(ax,'on');
        ax.LineWidth = 0.5;

        hold off;
    end
end

function [bestSeedNNSS, Xtraj_best] = plot_best_seed_states_from_rmse(folder_to_load, mat_files, rmse_results, test_data, Ts, normalization)
% Plot NN-SS state evolution for the best seed selected by rmse_results
%
% Inputs:
%   folder_to_load : model folder
%   mat_files      : dir(...) output (list of .mat files)
%   rmse_results   : S x O x M (or S x M), M=5, method1 = NN-SS
%   test_data      : [y u ...] same format you used in inference
%   Ts             : sample time
%   normalization  : 'zscore' or 'minmax'
%
% Output:
%   bestSeedNNSS   : selected seed index (file index)
%   Xtraj_best     : nStates x T state trajectory (x(k+1) you stored)

    % --- rmse -> pick best NN-SS seed (method 1) ---
    S = numel(mat_files);
    assert(size(rmse_results,1)==S, 'rmse_results seed count must match mat_files.');

    nd = ndims(rmse_results);
    if nd==2
        rmse_seed_method = rmse_results; % SxM
    else
        rmse_seed_method = squeeze(mean(rmse_results,2,'omitnan')); % SxM
    end
    if isvector(rmse_seed_method), rmse_seed_method = reshape(rmse_seed_method,S,[]); end
    assert(size(rmse_seed_method,2) >= 1, 'rmse_results must include NN-SS in method 1');

    [~, bestSeedNNSS] = min(rmse_seed_method(:,1), [], 'omitnan');
    fprintf('\n[NN-SS best seed for STATE plot] seed #%d\n', bestSeedNNSS);

    % --- load that best model ---
    best_file = fullfile(folder_to_load, mat_files(bestSeedNNSS).name);
    nets = load(best_file);

    net_NNSS      = nets.net_NNSS;
    map_x0_net    = nets.map_x0_net;
    stats         = nets.stats;
    nStates       = nets.number_of_states;
    mode          = nets.mode;

    % --- infer dimensions from your test_data ---
    % assumes test_data = [y u ...] and you know nOut/nIn from stats sizes
    nOut = size(stats.Yk.mu,2);
    nIn  = size(stats.U.mu,2);

    y_inf = test_data(:,1:nOut);
    u_inf = test_data(:,nOut+1:nOut+nIn);

    % --- normalize ---
    switch lower(normalization)
        case 'zscore'
            yN = (y_inf - stats.Yk.mu) ./ stats.Yk.sigma;
            uN = (u_inf - stats.U.mu ) ./ stats.U.sigma;
        case 'minmax'
            yN = (y_inf - stats.Yk.min) ./ (stats.Yk.max - stats.Yk.min);
            uN = (u_inf - stats.U.min ) ./ (stats.U.max - stats.U.min);
        otherwise
            error('Unsupported normalization: %s', normalization);
    end

    T = size(test_data,1);

    % --- simulate NN-SS and collect states ---
    Xtraj_best = zeros(nStates, T);

    dlY0 = dlarray(yN(1,:)', 'CB');
    dlX  = forward(map_x0_net, dlY0);

    if mode==3 || mode==4
        y_prev = dlY0;
    end

    for k = 1:T
        switch mode
            case 1
                net_in = dlX;
            case 2
                net_in = [dlX; dlarray(uN(k,:)', 'CB')];
            case 3
                net_in = [dlX; y_prev];
            case 4
                net_in = [dlX; dlarray(uN(k,:)', 'CB'); y_prev];
            otherwise
                error('Unsupported mode: %d', mode);
        end

        ss_params = forward(net_NNSS, dlarray(net_in,'CB'));
        [~,~,~, ~, dlXnext] = intermediateStateSpace_simulation( ...
            ss_params, dlX, uN(k,:)', nStates, nIn, nOut);

        Xtraj_best(:,k) = double(dlXnext(:));

        dlX = dlarray(dlXnext,'CB');
        if mode==3 || mode==4
            % y_prev should be the *normalized* output; you can use last yN(k,:) or predicted output.
            % Here, we use measured yN(k,:) for stability with your earlier code style.
            y_prev = dlarray(yN(k,:)', 'CB');
        end
    end

    % --- plot ---
    figure('Name', sprintf('NN-SS State Evolution (Best Seed=%d)', bestSeedNNSS), 'NumberTitle','off');
    hold on; grid on;
    idx = 1:T;
    for i = 1:nStates
        plot(idx, Xtraj_best(i,:), 'LineWidth', 2, 'DisplayName', sprintf('x_{%d}', i));
    end
    xlabel('Sample Index'); ylabel('State Value');
    legend('Location','northoutside','Orientation','horizontal');

    ax = gca;
    box(ax,'on'); ax.LineWidth = 0.5;
    hold off;
end




