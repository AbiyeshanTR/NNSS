%% Stable-by-Design NN-SS / LPV (Research Code) — Bayesian ONLY
% =========================================================================
% This script finds optimal initialization parameters for NN-SS 
% by making 5 epoch long trainings and see the results on validation data:
%
% Expected workspace variables BEFORE running:
%   - Data_train / Data_val : [N x (ny + nu + ny)] columns: y(k), u(k), y(k+1)
% =========================================================================

% Define optimization variables
optimVars = [
    optimizableVariable('NNSS_numLayers', [1 8], 'Type', 'integer')
    optimizableVariable('NNSS_numNeurons', [16 512], 'Type', 'integer')
    % optimizableVariable('NNSS_activation', {'tanh', 'sigmoid', 'relu'}, 'Type', 'categorical')
    optimizableVariable('x0_numLayers', [1 4], 'Type', 'integer')
    optimizableVariable('x0_numNeurons', [8 128], 'Type', 'integer')
    optimizableVariable('lambda', [1e-3, 1], 'Transform','log')
    

    ];

% Call bayesopt with anonymous function that injects your data
results = bayesopt(@(opt) trainAndValidate(opt, Data_train, Data_val), ...
    optimVars, ...
    'MaxObjectiveEvaluations', 100, ...
    'UseParallel', false, ...
    'AcquisitionFunctionName', 'expected-improvement-plus');


function results = trainAndValidate(opt, Data_train, Data_val)
%% Hyperparameters from opt
lambda = opt.lambda;

numLayers_NNSS = opt.NNSS_numLayers;
numNeurons_NNSS = opt.NNSS_numNeurons;
% activation_NNSS = char(opt.NNSS_activation);
activation_NNSS = char('sigmoid');

numLayers_x0 = opt.x0_numLayers;
numNeurons_x0 = opt.x0_numNeurons;

%% Assume these are constant or passed
number_of_states = 5;
number_of_inputs = 5;
number_of_outputs = 3;
numSamples = 100;
batchSize = 1;
learnRate = 0.001;
numEpochs = 5;
stride = 1;
mode = 1;
normalization = 'zscore';



% Scheduling input size based on mode
switch mode
    case 1; inputSize = number_of_states;
    case 2; inputSize = number_of_states + number_of_inputs;
    case 3; inputSize = number_of_states + number_of_outputs;
    case 4; inputSize = number_of_states + number_of_inputs + number_of_outputs;
    otherwise; error('Invalid mode');
end

% Parameter dimension (A, B, C matrices)
param_dim = 5*number_of_states^2 + number_of_states*number_of_inputs + number_of_outputs*number_of_states;

%% Process training data

[U_norm, Yk_norm, stats] = Trajectory_maker(Data_train, numSamples, stride, normalization, ...
    number_of_inputs, number_of_outputs);

Data_val_norm = [(Data_val(:,1:number_of_outputs)-stats.Yk.mu) ./ stats.Yk.sigma,(Data_val(:,number_of_outputs+1:number_of_outputs + number_of_inputs)-stats.U.mu) ./ stats.U.sigma];


totalTrajectories = size(Yk_norm,1);
numBatches = floor(totalTrajectories / batchSize);

%% Build net_NNSS dynamically
layers_NNSS = [featureInputLayer(inputSize, 'Name', 'input')];
for i = 1:numLayers_NNSS
    layers_NNSS = [layers_NNSS, fullyConnectedLayer(numNeurons_NNSS, 'Name', ['fc' num2str(i)])];
    switch activation_NNSS
        case 'relu'; layers_NNSS = [layers_NNSS, reluLayer('Name', ['relu' num2str(i)])];
        case 'tanh'; layers_NNSS = [layers_NNSS, tanhLayer('Name', ['tanh' num2str(i)])];
        case 'sigmoid'; layers_NNSS = [layers_NNSS, sigmoidLayer('Name', ['sigmoid' num2str(i)])];
    end
end
layers_NNSS = [layers_NNSS, fullyConnectedLayer(param_dim, 'Name', 'output')];
net_NNSS = dlnetwork(layerGraph(layers_NNSS));

%% Build map_x0_net (linear only)
layers_x0 = [featureInputLayer(number_of_outputs, 'Name', 'input')];
for i = 1:numLayers_x0
    layers_x0 = [layers_x0, fullyConnectedLayer(numNeurons_x0, 'Name', ['fcx0' num2str(i)])];
end
layers_x0 = [layers_x0, fullyConnectedLayer(number_of_states, 'Name', 'x0_output')];
map_x0_net = dlnetwork(layerGraph(layers_x0));

%% Train for a few epochs
for epoch = 1:numEpochs
    loss_total = 0;
    for batchIdx = 1:numBatches
        idx_start = (batchIdx-1)*batchSize + 1;
        idx_end = idx_start + batchSize - 1;

        dlU_batch = zeros(numSamples, number_of_inputs, batchSize);
        dlY_batch = zeros(numSamples, number_of_outputs, batchSize);

        for j = 1:batchSize
            for i = 1:number_of_inputs
                dlU_batch(:, i, j) = U_norm{idx_start + j - 1, i};
            end
            for i = 1:number_of_outputs
                dlY_batch(:, i, j) = Yk_norm{idx_start + j - 1, i};
            end
        end

        dlU_batch = dlarray(dlU_batch);
        dlY_batch = dlarray(dlY_batch);

        [~, ~, ~, gradients, loss] = dlfeval(@gradFcn, net_NNSS, map_x0_net, dlY_batch, dlU_batch, number_of_states, number_of_inputs, number_of_outputs, mode, lambda);

        net_NNSS = adamupdate(net_NNSS, gradients{1}, [], [], epoch, learnRate);
        map_x0_net = adamupdate(map_x0_net, gradients{2}, [], [], epoch, learnRate);

        loss_total = loss_total + extractdata(loss);
    end
end

%% Validation on Data_val
y_val = Data_val(:,1:number_of_outputs);
u_val = Data_val(:,number_of_outputs+1:number_of_outputs+number_of_inputs);

switch normalization
    case 'zscore'
        y_val_norm = (y_val - stats.Yk.mu) ./ stats.Yk.sigma;
        u_val_norm = (u_val - stats.U.mu) ./ stats.U.sigma;
    case 'minmax'
        y_val_norm = (y_val - stats.Yk.min) ./ (stats.Yk.max - stats.Yk.min);
        u_val_norm = (u_val - stats.U.min) ./ (stats.U.max - stats.U.min);
end

% Use only first point for simulation
dlY0 = dlarray(y_val_norm(1,:)', 'CB');
dlX0 = forward(map_x0_net, dlY0);
x_current = dlX0;

Y_hat = zeros(size(y_val));

for k = 1:size(u_val_norm,1)
    switch mode
        case 1
            net_input = x_current;
        case 2
            u_current = dlarray(u_val_norm(k,:)', 'CB');
            net_input = [x_current; u_current];
        case 3
            y_prev = dlarray(y_val_norm(max(k-1,1),:)', 'CB');
            net_input = [x_current; y_prev];
        case 4
            u_current = dlarray(u_val_norm(k,:)', 'CB');
            y_prev = dlarray(y_val_norm(max(k-1,1),:)', 'CB');
            net_input = [x_current; u_current; y_prev];
    end
    ss_params = forward(net_NNSS, net_input);
    [~, ~, ~, dlY_pred, dlX_next] = intermediateStateSpace_simulation(ss_params, x_current, u_val_norm(k,:)', number_of_states, number_of_inputs, number_of_outputs);

    y_pred = extractdata(dlY_pred)';
    switch normalization
        case 'zscore'; Y_hat(k,:) = y_pred .* stats.Yk.sigma + stats.Yk.mu;
        case 'minmax'; Y_hat(k,:) = y_pred .* (stats.Yk.max - stats.Yk.min) + stats.Yk.min;
    end

    x_current = dlarray(dlX_next, 'CB');
end

% RMSE validation loss
val_loss = sqrt(mean((y_pred - y_val_norm).^2, 'all'));
results = val_loss;
end
%% Functions

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
batch_size = size(dlU, 3);

dlYPred = dlarray(zeros(num_time_steps, number_of_outputs, batch_size));
dlXPred = dlarray(zeros(number_of_states, num_time_steps + 1, batch_size));
dlX2_pred = dlarray(zeros(number_of_states, num_time_steps, batch_size));


% For mode 3 or 4, use dlY(1,:,:) as y_prev only once
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

    % Prepare input based on selected mode
    switch mode
        case 1  % x only
            net_input = x_current;
        case 2  % x + u
            net_input = [x_current; u_current];
        case 3  % x + y
            net_input = [x_current; y_prev];
        case 4  % x + u + y
            net_input = [x_current; u_current; y_prev];
        otherwise
            error('Invalid mode selected.');
    end




    dl_SS_Params = forward(net, net_input);
    [A_matrix, B_matrix, C_matrix, x_next, y_next] = ...
        intermediateStateSpace(dl_SS_Params, x_current, u_current, number_of_states, number_of_inputs, number_of_outputs, batch_size);

    dlXPred(:, t+1, :) = x_next;

    y_next_reshaped = reshape(y_next, 1, number_of_outputs, batch_size);
    for j = 1:batch_size
        dlYPred(t, :, j) = y_next_reshaped(:, :, j);
    end

    if mode == 1 || mode == 2
        % In modes 1 and 2, start mapping from t=1, but exclude t=1 from loss later
        x2 = forward(map_x0_net, dlarray(reshape(dlY(t,:,:), number_of_outputs, batch_size), 'CB'));
        dlX2_pred(:, t, :) = x2;
    elseif mode == 3 || mode == 4
        % In modes 3 and 4, start mapping from t=3
        if t >= 3
            x2 = forward(map_x0_net, dlarray(reshape(dlY(t,:,:), number_of_outputs, batch_size), 'CB'));
            dlX2_pred(:, t, :) = x2;
        end
    end




    % Update x(k)
    x_current = x_next;

    % Update y_prev for next time step (only in modes 3 or 4)
    if mode == 3 || mode == 4
        y_prev = dlarray(reshape(y_next, number_of_outputs, batch_size), 'CB');
    end
end


dlX2_pred = permute(dlX2_pred, [2, 1, 3]);       % [time, state, batch]
dlXPred = permute(dlXPred, [2, 1, 3]);

switch mode
    case {1, 2}
        loss_state = computeBatchLoss(dlX2_pred(2:end, :, :), dlXPred(2:end-1, :, :));
        loss_output = computeBatchLoss(dlYPred, dlY(1:end,:,:));
    case {3, 4}
        loss_state = computeBatchLoss(dlX2_pred(3:end, :, :), dlXPred(3:end-1, :, :));
        loss_output = computeBatchLoss(dlYPred(2:end,:,:), dlY(2:end,:,:));
end

loss = loss_output + lambda * loss_state;

gradients_net = dlgradient(loss, net.Learnables);
gradients_map_x0 = dlgradient(loss, map_x0_net.Learnables);
gradients = {gradients_net, gradients_map_x0};

end



function [A_matrix, B_matrix, C_matrix, Z_state,Z_response] = intermediateStateSpace(X1, dlX, U, number_of_states,number_of_inputs,number_of_outputs,mbs)

% Paperin schur denemesi

gamma = 1;  % Desired spectral bound
epsilon = exp(-10);  % Small positive constant


W = reshape(X1(1:4*number_of_states^2, :), 2*number_of_states, 2*number_of_states, mbs);
V = reshape(X1(4*number_of_states^2+1:5*number_of_states^2, :), number_of_states, number_of_states, mbs);

S = pagemtimes(W,  pagetranspose(W)) + epsilon * eye(2 * number_of_states);


S11 = S(1:number_of_states, 1:number_of_states, :);
S12 = S(1:number_of_states, number_of_states+1:end, :);
S22 = S(number_of_states+1:end, number_of_states+1:end, :);


% A_matrix = zeros(number_of_states, number_of_states, mbs, 'like', X1);
inv_term = 0.5 * (S11/(gamma^2) + S22) + V - pagetranspose(V);
inv_term_cell = num2cell(inv_term, [1 2]);

a = cellfun(@(inv_term_cell) pinv(inv_term_cell),inv_term_cell,'UniformOutput',false);
inv_term_inv = cat(3, a{:});

A_matrix = pagemtimes(S12,inv_term_inv);


B_matrix = reshape(X1(5*number_of_states^2 + 1:5*number_of_states^2 + number_of_states * number_of_inputs, :), number_of_states, number_of_inputs, mbs);
C_matrix = reshape(X1(5*number_of_states^2 + number_of_states * number_of_inputs+1 :5*number_of_states^2 + number_of_states*number_of_inputs + number_of_outputs *number_of_states, :),number_of_outputs,number_of_states,mbs);



X_State = reshape(dlX, number_of_states, 1, []);
U_Vector = reshape(U, size(U,1), 1, []);

% AX + BU calculation
AX = pagemtimes(A_matrix, X_State);
BU = pagemtimes(B_matrix, U_Vector);

Z_state = AX + BU;
Z_response = pagemtimes(C_matrix, X_State);
end
function totalLoss = computeBatchLoss(predicted, targets)

l2LossValue = l2loss(predicted, targets,NormalizationFactor="none",DataFormat="SCB");

% Normalize by number of trajectories (|Z|) and timesteps (l_s)
numSamples = size(predicted, 1);  % |Z| = Number of trajectories
numChannels = size(predicted, 2);
numBatch = size(predicted, 3);  % l_s = Number of timesteps per trajectory

% SIMBa's Loss: Normalized sum of squared errors
totalLoss = l2LossValue / (numSamples * numChannels * numBatch);

end
function [A_matrix, B_matrix,C_matrix, Z_response,Z_state] = intermediateStateSpace_simulation(X1, dlX, U, number_of_states,number_of_inputs,number_of_outputs)

mbs=1;
gamma = 1;  % Desired spectral bound
epsilon = exp(-10);  % Small positive constant


W = reshape(X1(1:4*number_of_states^2, :), 2*number_of_states, 2*number_of_states, mbs);
V = reshape(X1(4*number_of_states^2+1:5*number_of_states^2, :), number_of_states, number_of_states, mbs);

S = pagemtimes(W,  pagetranspose(W)) + epsilon * eye(2 * number_of_states);


S11 = S(1:number_of_states, 1:number_of_states, :);
S12 = S(1:number_of_states, number_of_states+1:end, :);
S22 = S(number_of_states+1:end, number_of_states+1:end, :);


% A_matrix = zeros(number_of_states, number_of_states, mbs, 'like', X1);
inv_term = 0.5 * (S11/(gamma^2) + S22) + V - pagetranspose(V);
inv_term_cell = num2cell(inv_term, [1 2]);

a = cellfun(@(inv_term_cell) pinv(inv_term_cell),inv_term_cell,'UniformOutput',false);
inv_term_inv = cat(3, a{:});

A_matrix = pagemtimes(S12,inv_term_inv);

B_matrix = reshape(X1(5*number_of_states^2 + 1:5*number_of_states^2 + number_of_states * number_of_inputs, :), number_of_states, number_of_inputs, mbs);
C_matrix = reshape(X1(5*number_of_states^2 + number_of_states * number_of_inputs+1 :5*number_of_states^2 + number_of_states*number_of_inputs + number_of_outputs *number_of_states, :),number_of_outputs,number_of_states,mbs);



X_State = reshape(dlX, number_of_states, 1, []);
U_Vector = reshape(U, size(U,1), 1, []);

% AX + BU calculation
AX = pagemtimes(A_matrix, X_State);
BU = pagemtimes(B_matrix, U_Vector);

Z_state = AX + BU;
Z_response = pagemtimes(C_matrix, X_State);





end

