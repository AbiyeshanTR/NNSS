%% Stable-by-Design NN-SS / LPV (Research Code) — TRAINING ONLY
% =========================================================================
% This script trains and saves:
%   (1) NN-SS / NNSS model
%   (2) SIMBa baseline
%   (3) Classical subspace baselines: SSREGEST / N4SID / SSEST
%
% Expected workspace variables BEFORE running:
%   - Data_train / Data_val : [N x (ny + nu + ny)] columns: y(k), u(k), y(k+1)
% =========================================================================

%% PARAMS
% Model dimensions
number_of_states = 4;
number_of_inputs = 5;
number_of_outputs = 3;

% Monte Carlo seeds
rng_values = [2];

% Optimization hyperparameters
batchSize = 1;
learnRate = 0.001;
numEpochs = 1000;

% Sampling time
Ts = 1228.8;

% Data windowing
numSamples = 100;
stride = 1;

% Normalization + early stopping
normalization = 'zscore';
patience = 400;

% Feature mode selection for scheduling:
% 1: x only
% 2: x + u
% 3: x + y
% 4: x + u + y
mode = 1;

% Loss weight for state-consistency regularizer
lambda =  0.01;

%% ========================================================================
%% DATA PREPARATION (Sliding windows / overlapping trajectories)


[U_norm, Yk_norm, stats] = Trajectory_maker(Data_train, numSamples, stride, normalization, ...
    number_of_inputs, number_of_outputs);

Data_val_norm = [(Data_val(:,1:number_of_outputs)-stats.Yk.mu) ./ stats.Yk.sigma,(Data_val(:,number_of_outputs+1:number_of_outputs + number_of_inputs)-stats.U.mu) ./ stats.U.sigma];


totalTrajectories = size(Yk_norm,1);

%% ========================================================================
%% EXPERIMENT FOLDER (Reproducible naming)
optional_comment = 'montecarlodene';

if ~isempty(optional_comment)
    comment_part = sprintf('_%s', strrep(optional_comment, ' ', '_'));
else
    comment_part = '';
end
% this code line shows how the created folders are named
folder_name = sprintf('powerplant_models_%d_%d_%d_%d_lr_%.4f_epoch_%d_lambda_%.2f%s_mode%d', ...
    number_of_states, numSamples, batchSize, totalTrajectories, learnRate, numEpochs, lambda, comment_part, mode);

if ~exist(folder_name, 'dir')
    mkdir(folder_name);
end

%% ========================================================================
%% TRAIN (Monte Carlo over RNG seeds)
for rng_val = rng_values
    fprintf('Starting training with rng = %d\n', rng_val);

    %% --------------------------------------------------------------------
    %% NN-SS (NNSS): Initialize RNG + mode-dependent input size
    rng(rng_val);

    switch mode
        case 1
            inputSize = number_of_states;
        case 2
            inputSize = number_of_states + number_of_inputs;
        case 3
            inputSize = number_of_states + number_of_outputs;
        case 4
            inputSize = number_of_states + number_of_inputs + number_of_outputs;
        otherwise
            error('Invalid mode selected.');
    end

    %% --------------------------------------------------------------------
    %% NNSS NETWORK (State-dependent SS parameter generator)
    param_dim = 5*number_of_states^2 + number_of_states*number_of_inputs + number_of_outputs*number_of_states;
%% Modify the Neural Network layers as you wish, what you see below is just an example
% randomly initialize neural networks or use Bayesian.m initialization for selecting number of layers, neurons per layer etc.

    layers = [
        featureInputLayer(inputSize, 'Name', 'xk_input')
        fullyConnectedLayer(120)
        sigmoidLayer
        fullyConnectedLayer(120)
        sigmoidLayer
        fullyConnectedLayer(240)
        sigmoidLayer
        fullyConnectedLayer(240)
        sigmoidLayer
        fullyConnectedLayer(240)
        sigmoidLayer
        fullyConnectedLayer(120)
        sigmoidLayer
        fullyConnectedLayer(120)
        sigmoidLayer
        fullyConnectedLayer(param_dim, 'Name', 'param_output')
    ];

    %% --------------------------------------------------------------------
    %% x0 MAPPER NETWORK (Encoder)
    layers_x0 = [
        featureInputLayer(number_of_outputs, 'Name', 'input_yk0')
        fullyConnectedLayer(64)
        fullyConnectedLayer(64)
        fullyConnectedLayer(64)
        fullyConnectedLayer(64)
        fullyConnectedLayer(number_of_states, 'Name', 'x0_output')
    ];

    net_NNSS   = dlnetwork(layerGraph(layers));
    map_x0_net = dlnetwork(layerGraph(layers_x0));

    

    %% --------------------------------------------------------------------
    %% Optimizer state (Adam) + early stopping (patience-with-window)
    numBatches = floor(totalTrajectories / batchSize);

    avgGrad = [];
    avgSqGrad = [];
    avgGrad_map = [];
    avgSqGrad_map = [];

    epsilon = 1e-8;
    beta1 = 0.9;
    beta2 = 0.999;
% %5 improvement required before resetting the patience
    min_delta_rel = 0.05;

    restart_metric = Inf;
    epochs_since_restart = 0;
    stopped = false;

    restart_net_NNSS    = net_NNSS;
    restart_map_x0_net  = map_x0_net;

    window_best_metric       = Inf;
    window_best_net_NNSS     = net_NNSS;
    window_best_map_x0_net   = map_x0_net;

    bestValLoss = Inf;

    loss_values_NNSS     = nan(1,numEpochs);
    val_loss_values_NNSS = nan(1,numEpochs);

    %% --------------------------------------------------------------------
    %% Live plots (training and validation)
    figure;
    hold on;
    plotHandle = plot(nan, 'b-', 'LineWidth', 1.5);
    xlabel('Epoch'); ylabel('Average Loss');
    title('NNSS Live Training Loss');
    grid on;
    hold off

    figure;
    hold on;
    valPlot = plot(nan, 'r--', 'LineWidth', 1.5);
    xlabel('Epoch'); ylabel('Loss');
    legend('Validation Loss');
    title('Validation Loss NNSS');
    grid on;

    %% --------------------------------------------------------------------
    %% NNSS TRAINING LOOP
    for epoch = 1:numEpochs

        fprintf('Epoch %d/%d\n', epoch, numEpochs);
        epochLoss = 0;

        shuff_indexes = randperm(totalTrajectories);
        Yk_train_shuffled = Yk_norm(shuff_indexes, :);
        U_train_shuffled  = U_norm(shuff_indexes, :);

        for batchIdx = 1:numBatches
            startIdx = (batchIdx - 1) * batchSize + 1;
            endIdx   = min(batchIdx * batchSize, totalTrajectories); 

            dlU_batch = zeros(numSamples, number_of_inputs,  batchSize);
            dlY_batch = zeros(numSamples, number_of_outputs, batchSize);

            for j = 1:batchSize
                for i = 1:number_of_inputs
                    dlU_batch(:, i, j) = U_train_shuffled{startIdx + j - 1, i};
                end
                for i = 1:number_of_outputs
                    dlY_batch(:, i, j) = Yk_train_shuffled{startIdx + j - 1, i};
                end
            end

            dlU_batch = dlarray(dlU_batch);
            dlY_batch = dlarray(dlY_batch);

            [~, ~, ~, gradients, loss] = dlfeval(@gradFcn, net_NNSS, map_x0_net, ...
                dlY_batch, dlU_batch, number_of_states, number_of_inputs, number_of_outputs, mode, lambda);

            [net_NNSS, avgGrad, avgSqGrad] = adamupdate(net_NNSS, gradients{1}, ...
                avgGrad, avgSqGrad, epoch, learnRate, beta1, beta2, epsilon);

            [map_x0_net, avgGrad_map, avgSqGrad_map] = adamupdate(map_x0_net, gradients{2}, ...
                avgGrad_map, avgSqGrad_map, epoch, learnRate, beta1, beta2, epsilon);

            loss = extractdata(loss);
            epochLoss = epochLoss + loss;

            fprintf('Batch %d/%d, Loss: %.4f\n', batchIdx, numBatches, loss);
        end

        %% --------------------------- Validation (NNSS) --------------------
        avgEpochLoss = epochLoss / numBatches;
        loss_values_NNSS(1,epoch) = avgEpochLoss;

        fprintf('Average Loss for Epoch %d: %.4f\n', epoch, avgEpochLoss);
        set(plotHandle, 'XData', 1:epoch, 'YData', loss_values_NNSS(1:epoch));
        drawnow;

        y_val_norm = Data_val_norm(:,1:number_of_outputs);
        u_val_norm = Data_val_norm(:,number_of_outputs+1:number_of_outputs+number_of_inputs);

        dlY0 = dlarray(y_val_norm(1,:)', 'CB');
        dlX0_NNSS = forward(map_x0_net, dlY0);
        x_current = dlX0_NNSS;

        Y_NNSS = zeros(size(Data_val_norm,1), number_of_outputs);

        if mode == 3 || mode == 4
            y_prev = dlY0;
        end

        for k = 1:size(Data_val_norm,1)

            switch mode
                case 1
                    net_input = x_current;
                case 2
                    u_current = dlarray(u_val_norm(k,:)', 'CB');
                    net_input = [x_current; u_current];
                case 3
                    net_input = [x_current; y_prev];
                case 4
                    u_current = dlarray(u_val_norm(k,:)', 'CB');
                    net_input = [x_current; u_current; y_prev];
            end

            ss_parameters = forward(net_NNSS, net_input);
            [~, ~, C_matrix, dlYPred, dlXPred] = ...
                intermediateStateSpace_simulation(ss_parameters, x_current, u_val_norm(k,:)', ...
                number_of_states, number_of_inputs, number_of_outputs); 

            Y_NNSS(k,:) = double(dlYPred)';

            x_current = dlarray(dlXPred, 'CB');
            if mode == 3 || mode == 4
                y_prev = dlarray(dlYPred, 'CB');
            end
        end

        val_loss = l2loss(dlarray(Y_NNSS), y_val_norm, NormalizationFactor="none", Dataformat='CB');
        val_loss = val_loss / (length(y_val_norm) * number_of_outputs);
        val_loss = sqrt(val_loss);
        val_loss_values_NNSS(epoch) = val_loss;

        fprintf('Validation Loss for Epoch %d: %.4f\n', epoch, val_loss);

        set(valPlot, 'XData', 1:epoch, 'YData', val_loss_values_NNSS(1:epoch));
        drawnow;

        %% ------------------- Early stopping (patience-with-window) --------
        if isinf(restart_metric)
            restart_metric = val_loss; epochs_since_restart = 0;
            restart_net_NNSS = net_NNSS; restart_map_x0_net = map_x0_net;
            window_best_metric = val_loss;
            window_best_net_NNSS = net_NNSS; window_best_map_x0_net = map_x0_net;
        else
            rel_impr = (restart_metric - val_loss) / max(restart_metric, eps);
            if rel_impr >= min_delta_rel
                restart_metric = val_loss; epochs_since_restart = 0;
                restart_net_NNSS = net_NNSS; restart_map_x0_net = map_x0_net;
                window_best_metric = val_loss;
                window_best_net_NNSS = net_NNSS; window_best_map_x0_net = map_x0_net;
            else
                epochs_since_restart = epochs_since_restart + 1;
                if val_loss < window_best_metric
                    window_best_metric = val_loss;
                    window_best_net_NNSS = net_NNSS; window_best_map_x0_net = map_x0_net;
                end
                if epochs_since_restart >= patience
                    if window_best_metric < restart_metric
                        best_net_NNSS = window_best_net_NNSS;
                        best_map_x0_net  = window_best_map_x0_net;
                        bestValLoss      = window_best_metric;
                    else
                        best_net_NNSS = restart_net_NNSS;
                        best_map_x0_net  = restart_map_x0_net;
                        bestValLoss      = restart_metric;
                    end
                    stopped = true; break;
                end
            end
        end

    end

    %% --------------------------------------------------------------------
    %% Finalize best NNSS nets after training (if not stopped by patience)
    if ~stopped
        if window_best_metric < restart_metric
            best_net_NNSS = window_best_net_NNSS;
            best_map_x0_net  = window_best_map_x0_net;
            bestValLoss      = window_best_metric;
        else
            best_net_NNSS = restart_net_NNSS;
            best_map_x0_net  = restart_map_x0_net;
            bestValLoss      = restart_metric;
        end
    end
    net_NNSS   = best_net_NNSS;
    map_x0_net = best_map_x0_net;

    %% ====================================================================
    %% BASELINES: SUBSPACE METHODS (train on normalized training data)
    switch lower(normalization)
        case 'zscore'
            Yk_norms = (Data_train(:, 1:number_of_outputs)-stats.Yk.mu)./stats.Yk.sigma;
            U_norms  = (Data_train(:, number_of_outputs+1:number_of_outputs+number_of_inputs)-stats.U.mu)./stats.U.sigma;
        case 'minmax'
            Yk_norms = (Data_train(:, 1:number_of_outputs)- stats.Yk.min)./(stats.Yk.max - stats.Yk.min);
            U_norms  = (Data_train(:, number_of_outputs+1:number_of_outputs+number_of_inputs)- stats.U.min)./(stats.U.max - stats.U.min);
    end

    ss_data_train = iddata(Yk_norms, U_norms, Ts);

    sys_ssregest = ssregest(ss_data_train, number_of_states, 'DisturbanceModel', 'none','Display','off','focus','simulation');
    sys_n4sid    = n4sid   (ss_data_train, number_of_states,'DisturbanceModel', 'none', 'Display', 'off','focus','simulation');
    sys_ssest    = ssest   (ss_data_train, number_of_states,'DisturbanceModel', 'none', 'Display', 'off','Ts',Ts,'focus','simulation');

    %% ====================================================================
    %% BASELINE: SIMBa (initialize from N4SID A -> fit (W,V) then train)
    A_target = sys_n4sid.A;
    n        = size(A_target,1);
    gamma    = 1;
    epsilon  = exp(-10); 

    [W_opt, V_opt, info] = fit_WV_to_A(A_target, n); 

    SIMBa_layer = SIMBa_Layer(n, number_of_inputs, number_of_outputs, 'SIMBa', rng_val);
    SIMBa_layer.W = dlarray(W_opt);
    SIMBa_layer.V = dlarray(V_opt);
    SIMBa_layer.B = dlarray(sys_n4sid.B);
    SIMBa_layer.C = dlarray(sys_n4sid.C);
    SIMBa_layer.D = dlarray(sys_n4sid.D);

    layers = [
        featureInputLayer(n + number_of_inputs)
        SIMBa_layer
    ];
    net_Simba = dlnetwork(layers);

    %% --------------------------------------------------------------------
    %% SIMBa training setup
    numBatches = floor(totalTrajectories / batchSize);
    velocity = []; 
    avgGrad = [];
    avgSqGrad = [];
    epsilon = 1e-8;
    beta1 = 0.9;
    beta2 = 0.999;

    bestValLoss = inf;
    epochsWithoutImprovement = 0;

    loss_values_simba = nan(1, numEpochs);
    val_loss_values_simba = nan(1, numEpochs);

    figure;
    hold on;
    plotHandle = plot(nan, 'b-', 'LineWidth', 1.5);
    xlabel('Epoch'); ylabel('Average Loss');
    title('SIMBa Live Training Loss');
    grid on;

    figure;
    hold on;
    valPlot = plot(nan, 'r--', 'LineWidth', 1.5);
    xlabel('Epoch'); ylabel('Loss');
    legend('Validation Loss');
    title('Validation Loss SIMBa');
    grid on;

    
    %% SIMBa TRAINING LOOP
    for epoch = 1:numEpochs
        fprintf('Epoch %d/%d\n', epoch, numEpochs);
        epochLoss = 0;

        shuff_indexes = randperm(totalTrajectories);
        U_norm_shuffled  = U_norm(shuff_indexes, :);
        Yk_norm_shuffled = Yk_norm(shuff_indexes, :);

        for batchIdx = 1:numBatches
            startIdx = (batchIdx - 1) * batchSize + 1;
            endIdx   = min(batchIdx * batchSize, totalTrajectories); 

            dlX_batch_data = zeros(numSamples, number_of_inputs,  batchSize);
            dlY_batch_data = zeros(numSamples, number_of_outputs, batchSize);

            for j = 1:batchSize
                for i = 1:number_of_inputs
                    dlX_batch_data(:, i, j) = U_norm_shuffled{startIdx + j - 1, i};
                end
                for i = 1:number_of_outputs
                    dlY_batch_data(:, i, j) = Yk_norm_shuffled{startIdx + j - 1, i};
                end
            end

            [gradients, loss] = dlfeval(@modelGradients, net_Simba, dlX_batch_data, dlY_batch_data, batchSize);

            if epoch == 1 && batchIdx == 1
                avgGrad   = dlupdate(@zerosLike, gradients);
                avgSqGrad = dlupdate(@zerosLike, gradients);
            end

            [net_Simba, avgGrad, avgSqGrad] = adamupdate(net_Simba, gradients, avgGrad, avgSqGrad, ...
                epoch, learnRate, beta1, beta2, epsilon);

            fprintf('Batch %d/%d, Loss: %.4f\n', batchIdx, numBatches, loss);
            loss = extractdata(loss);
            epochLoss = epochLoss + loss;
        end

        %% --------------------------- Validation (SIMBa) -------------------
        avgEpochLoss = epochLoss / numBatches;
        loss_values_simba(1, epoch) = avgEpochLoss;

        fprintf('Average Loss for Epoch %d: %.4f\n', epoch, avgEpochLoss);
        set(plotHandle, 'XData', 1:epoch, 'YData', loss_values_simba(1:epoch));
        drawnow;

        y_val_norm = Data_val_norm(:,1:number_of_outputs);
        u_val_norm = Data_val_norm(:,number_of_outputs+1:number_of_outputs+number_of_inputs);

        A_pred = computeSchurStableA(net_Simba.Learnables.Value{1,1}, net_Simba.Learnables.Value{2,1}, 1);
        A_pred = extractdata(A_pred);
        eigenvalues = eig(A_pred); %#ok<NASGU>
        B_pred = extractdata(net_Simba.Learnables.Value{3,1});
        C_pred = extractdata(net_Simba.Learnables.Value{4,1});
        D_pred = extractdata(net_Simba.Learnables.Value{5,1});

        sys_simba = ss(A_pred, B_pred, C_pred, D_pred, Ts);

        dlY0 = dlarray(y_val_norm(1,:));
        X0_simba = pinv(sys_simba.C) * (dlY0' - sys_simba.D * u_val_norm(1,:)');

        t_sim = 0:Ts:(size(u_val_norm,1) - 1) * Ts;
        [Y_Simba, ~, ~] = lsim(sys_simba, u_val_norm, t_sim, X0_simba);

        val_loss = l2loss(dlarray(Y_Simba),  Data_val_norm(:,1:number_of_outputs), ...
            NormalizationFactor="none", Dataformat='CB');

        val_loss = val_loss / (length(y_val_norm) * number_of_outputs);
        val_loss = sqrt(val_loss);
        val_loss_values_simba(epoch) = val_loss;

        fprintf('Validation Loss for Epoch %d: %.4f\n', epoch, val_loss);

        set(valPlot, 'XData', 1:epoch, 'YData', val_loss_values_simba(1:epoch));
        drawnow;

        if val_loss < bestValLoss
            bestValLoss = val_loss;
            epochsWithoutImprovement = 0;
            best_net_Simba = net_Simba;
            fprintf('New best validation loss: %.4f\n', bestValLoss);
        else
            epochsWithoutImprovement = epochsWithoutImprovement + 1;
            fprintf('No improvement for %d epochs.\n', epochsWithoutImprovement);
        end

        if epochsWithoutImprovement >= patience
            fprintf('Validation loss did not improve for %d epochs. Stopping early at epoch %d.\n', patience, epoch);
            break;
        end
    end

    %% --------------------------------------------------------------------
    %% Finalize best SIMBa model
    net_Simba = best_net_Simba;
    A_pred = computeSchurStableA(net_Simba.Learnables.Value{1,1}, net_Simba.Learnables.Value{2,1}, 1);
    A_pred = extractdata(A_pred);
    eigenvalues = eig(A_pred); 
    B_pred = extractdata(net_Simba.Learnables.Value{3,1});
    C_pred = extractdata(net_Simba.Learnables.Value{4,1});
    D_pred = extractdata(net_Simba.Learnables.Value{5,1});
    sys_simba = ss(A_pred, B_pred, C_pred, D_pred, Ts);

    %% ====================================================================
    %% SAVE (Per RNG seed)
    filename = fullfile(folder_name, sprintf('networks_%d_%d_%d_%d_rng(%d)_lr_%.4f_epoch_%d_lambda_%.2f_mode%d.mat', ...
        number_of_states, numSamples, batchSize, totalTrajectories, rng_val, learnRate, numEpochs, lambda, mode));

    save(filename, 'net_NNSS', 'map_x0_net', ...
        'sys_simba', 'sys_ssregest', 'sys_n4sid', 'sys_ssest', ...
        'number_of_states', 'stats', 'mode', ...
        'loss_values_simba', 'loss_values_NNSS', ...
        'val_loss_values_simba', 'val_loss_values_NNSS', ...
        'numEpochs');

    disp(['Saved model to: ', filename]);
end

disp(['All models are stored in folder: ', folder_name]);

%% FUNCTIONS
function [U_norm, Yk_norm, stats] = Trajectory_maker(Data, numSamples, stride, normalization, number_of_inputs, number_of_outputs)

    totalPoints  = size(Data, 1);
    maxStartIdx  = totalPoints - numSamples + 1;
    startIndexes = 1:stride:maxStartIdx;
    numTrajectories = length(startIndexes);

    U_traj  = cell(numTrajectories, number_of_inputs);
    Yk_traj = cell(numTrajectories, number_of_outputs);
    Yk1_traj = cell(numTrajectories, number_of_outputs);

    U_norm  = cell(numTrajectories, number_of_inputs);
    Yk_norm = cell(numTrajectories, number_of_outputs);
    Yk1_norm = cell(numTrajectories, number_of_outputs);

    switch lower(normalization)
        case 'zscore'
            stats.U.mu    = mean(Data(:,number_of_outputs+1:number_of_outputs+number_of_inputs));
            stats.U.sigma = std (Data(:,number_of_outputs+1:number_of_outputs+number_of_inputs));
            stats.Yk.mu   = mean(Data(:,1:number_of_outputs));
            stats.Yk.sigma= std (Data(:,1:number_of_outputs));
            stats.Yk1.mu  = mean(Data(:,number_of_outputs+number_of_inputs+1:end));
            stats.Yk1.sigma=std(Data(:,number_of_outputs+number_of_inputs+1:end));
        case 'minmax'
            stats.U.min = min(Data(:,number_of_outputs+1:number_of_outputs+number_of_inputs));
            stats.U.max = max(Data(:,number_of_outputs+1:number_of_outputs+number_of_inputs));
            stats.Yk.min = min(Data(:,1:number_of_outputs));
            stats.Yk.max = max(Data(:,1:number_of_outputs));
            stats.Yk1.min = min(Data(:,number_of_outputs+number_of_inputs+1:end));
            stats.Yk1.max = max(Data(:,number_of_outputs+number_of_inputs+1:end));
    end

    for traj = 1:numTrajectories
        idx = startIndexes(traj);

        yk  = Data(idx:idx+numSamples-1, 1:number_of_outputs);
        uk  = Data(idx:idx+numSamples-1, number_of_outputs+1:number_of_outputs+number_of_inputs);
        yk1 = Data(idx:idx+numSamples-1, number_of_outputs+number_of_inputs+1:end);

        switch lower(normalization)
            case 'zscore'
                uk_norm  = (uk  - stats.U.mu) ./ stats.U.sigma;
                yk_norm  = (yk  - stats.Yk.mu) ./ stats.Yk.sigma;
                yk1_norm = (yk1 - stats.Yk.mu) ./ stats.Yk.sigma;
            case 'minmax'
                uk_norm  = (uk  - stats.U.min) ./ (stats.U.max - stats.U.min);
                yk_norm  = (yk  - stats.Yk.min) ./ (stats.Yk.max - stats.Yk.min);
                yk1_norm = (yk1 - stats.Yk1.min) ./ (stats.Yk1.max - stats.Yk1.min);
        end

        for j = 1:number_of_inputs
            U_traj{traj, j} = uk(:, j);
            U_norm{traj, j} = uk_norm(:, j);
        end
        for j = 1:number_of_outputs
            Yk_traj{traj, j}  = yk(:, j);
            Yk_norm{traj, j}  = yk_norm(:, j);
            Yk1_traj{traj, j} = yk1(:, j);
            Yk1_norm{traj, j} = yk1_norm(:, j);
        end
    end
end



function [dlYPred, dlXPred, dlX2_pred, gradients, loss] = gradFcn(net, map_x0_net, dlY, dlU, number_of_states, number_of_inputs, number_of_outputs, mode, lambda)

    num_time_steps = size(dlU, 1);
    batch_size     = size(dlU, 3);

    dlYPred   = dlarray(zeros(num_time_steps, number_of_outputs, batch_size));
    dlXPred   = dlarray(zeros(number_of_states, num_time_steps + 1, batch_size));
    dlX2_pred = dlarray(zeros(number_of_states, num_time_steps, batch_size));

    if mode == 3 || mode == 4
        y_prev = dlarray(reshape(dlY(1,:,:), number_of_outputs, batch_size), 'CB');
        t_start = 2;
        yk_initials = dlY(2,:,:);
        yk_initials = dlarray(reshape(yk_initials, number_of_outputs , batch_size ), 'CB');
        x0_mapped = forward(map_x0_net, yk_initials);
    else
        t_start = 1;
        yk_initials = dlY(1,:,:);
        yk_initials = dlarray(reshape(yk_initials, number_of_outputs , batch_size ), 'CB');
        x0_mapped = forward(map_x0_net, yk_initials);
    end

    dlXPred(:, 1, :) = x0_mapped;
    x_current = x0_mapped;

    for t = t_start:num_time_steps
        u_current = dlarray(reshape(dlU(t,:,:), number_of_inputs, batch_size), 'CB');
        x_current = dlarray(squeeze(x_current), 'CB');

        switch mode
            case 1
                net_input = x_current;
            case 2
                net_input = [x_current; u_current];
            case 3
                net_input = [x_current; y_prev];
            case 4
                net_input = [x_current; u_current; y_prev];
            otherwise
                error('Invalid mode selected.');
        end

        dl_SS_Params = forward(net, net_input);
        [A_matrix, B_matrix, C_matrix, x_next, y_next] = ...
            intermediateStateSpace(dl_SS_Params, x_current, u_current, number_of_states, number_of_inputs, number_of_outputs, batch_size); %#ok<ASGLU>

        dlXPred(:, t+1, :) = x_next;

        y_next_reshaped = reshape(y_next, 1, number_of_outputs, batch_size);
        for j = 1:batch_size
            dlYPred(t, :, j) = y_next_reshaped(:, :, j);
        end

        if mode == 1 || mode == 2
            x2 = forward(map_x0_net, dlarray(reshape(dlY(t,:,:), number_of_outputs, batch_size), 'CB'));
            dlX2_pred(:, t, :) = x2;
        elseif mode == 3 || mode == 4
            if t >= 3
                x2 = forward(map_x0_net, dlarray(reshape(dlY(t,:,:), number_of_outputs, batch_size), 'CB'));
                dlX2_pred(:, t, :) = x2;
            end
        end

        x_current = x_next;

        if mode == 3 || mode == 4
            y_prev = dlarray(reshape(y_next, number_of_outputs, batch_size), 'CB');
        end
    end

    dlX2_pred = permute(dlX2_pred, [2, 1, 3]);
    dlXPred   = permute(dlXPred,   [2, 1, 3]);

    switch mode
        case {1, 2}
            loss_state  = computeBatchLoss(dlX2_pred(2:end, :, :),   dlXPred(2:end-1, :, :));
            loss_output = computeBatchLoss(dlYPred,                  dlY(1:end,:,:));
        case {3, 4}
            loss_state  = computeBatchLoss(dlX2_pred(3:end, :, :),   dlXPred(3:end-1, :, :));
            loss_output = computeBatchLoss(dlYPred(2:end,:,:),       dlY(2:end,:,:));
    end

    loss = loss_output + lambda * loss_state;

    gradients_net    = dlgradient(loss, net.Learnables);
    gradients_map_x0 = dlgradient(loss, map_x0_net.Learnables);
    gradients = {gradients_net, gradients_map_x0};
end

function [A_matrix, B_matrix, C_matrix, Z_state, Z_response] = intermediateStateSpace(X1, dlX, U, number_of_states, number_of_inputs, number_of_outputs, mbs)

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

    a = cellfun(@(inv_term_cell) pinv(inv_term_cell), inv_term_cell, 'UniformOutput', false);
    inv_term_inv = cat(3, a{:});

    A_matrix = pagemtimes(S12, inv_term_inv);

    B_matrix = reshape(X1(5*number_of_states^2 + 1:5*number_of_states^2 + number_of_states * number_of_inputs, :), ...
        number_of_states, number_of_inputs, mbs);

    C_matrix = reshape(X1(5*number_of_states^2 + number_of_states * number_of_inputs + 1: ...
        5*number_of_states^2 + number_of_states*number_of_inputs + number_of_outputs*number_of_states, :), ...
        number_of_outputs, number_of_states, mbs);

    X_State = reshape(dlX, number_of_states, 1, []);
    U_Vector = reshape(U, size(U,1), 1, []);

    AX = pagemtimes(A_matrix, X_State);
    BU = pagemtimes(B_matrix, U_Vector);

    Z_state = AX + BU;
    Z_response = pagemtimes(C_matrix, X_State);
end

function totalLoss = computeBatchLoss(predicted, targets)
    l2LossValue = l2loss(predicted, targets, NormalizationFactor="none", DataFormat="SCB");

    numSamples  = size(predicted, 1);
    numChannels = size(predicted, 2);
    numBatch    = size(predicted, 3);

    totalLoss = l2LossValue / (numSamples * numChannels * numBatch);
end

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

function [gradients, loss] = modelGradients(net, X, Y, batch_size )

    dlYPred = dlarray(zeros(size(Y, 1), size(Y, 2), size(Y, 3)));

    x_current = pinv(net.Learnables.Value{4, 1}) * (reshape(Y(1,:,:), size(Y, 2), batch_size ) - ...
        net.Learnables.Value{5, 1} * reshape(X(1,:,:), size(X, 2), batch_size ));

    for i = 1:size(X, 1)
        int = dlarray([x_current; reshape(X(i, :, :), size(X, 2), batch_size )], 'CB');
        Z = forward(net, int);

        x_current = Z(1:size(x_current, 1), :);
        y_current = Z(size(x_current, 1) + 1:end, :);

        for j = 1:size(y_current, 2)
            dlYPred(i, :, j) = y_current(:, j);
        end
    end

    totalLoss = computeBatchLoss(dlYPred, Y);
    loss = totalLoss;

    gradients = dlgradient(loss, net.Learnables);
end

function A = computeSchurStableA(W, V, gamma)
    epsilon_tilde = -1;
    epsilon = exp(epsilon_tilde);

    S = W' * W + epsilon * eye(size(W, 1));

    numStates = size(V, 1);
    S11 = S(1:numStates, 1:numStates);
    S12 = S(1:numStates, numStates+1:end);
    S22 = S(numStates+1:end, numStates+1:end);

    MatToInvert = 0.5 * ((S11 + S22) / gamma^2) + V - V';
    invMat = pinv(MatToInvert);

    A = S12 * invMat;
end

function z = zerosLike(x)
    z = zeros(size(x), 'like', x);
end

function [W_opt, V_opt, info] = fit_WV_to_A(A_target, n, gamma)

    if nargin < 3, gamma = 1; end
    assert(all(size(A_target) == [n n]), 'A_target must be n x n');

    idxW = (2*n)*(2*n);

    pack   = @(W,V) [W(:); V(:)];
    unpack = @(z) deal( reshape(z(1:idxW), 2*n, 2*n), ...
                        reshape(z(idxW+1:end), n, n) );

    obj = @(z) ofun(z, A_target, n, gamma, unpack);

    W0 = 1*(eye(2*n) + randn(2*n, 2*n));
    V0 = 1* randn(n, n);

    z0 = pack(W0, V0);

    opts = optimoptions('fmincon', ...
        'Algorithm','sqp', ...
        'Display','iter', ...
        'MaxIterations', 500, ...
        'MaxFunctionEvaluations', 5e4);

    [z_opt, fval, exitflag, output] = fmincon(obj, z0, [],[],[],[], [],[], [], opts);

    [W_opt, V_opt] = unpack(z_opt);
    info.fval = fval; info.exitflag = exitflag; info.output = output;
end

function f = ofun(z, A_target, n, gamma, unpack)
    [W, V] = unpack(z);
    Ahat = computeSchurStableA(W, V, gamma);
    R = Ahat - A_target;
    f = 0.5 * sum(R(:).^2);
end
