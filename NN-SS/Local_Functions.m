%% LOCAL FUNCTIONS
% ========================================================================

function [U_traj, Yk_traj, Yk1_traj, U_norm, Yk_norm, Yk1_norm,stats] = processStateSpaceData_SIMBa(Data, numSamples, stride, normalization, number_of_inputs, number_of_outputs)

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

function [U_traj, Yk_traj, Yk1_traj, U_norm, Yk_norm, Yk1_norm] = processStateSpaceData_With_Stats(Data, numSamples, stride, stats, number_of_inputs, number_of_outputs)

    totalPoints = size(Data, 1);
    maxStartIdx = totalPoints - numSamples+1;
    startIndexes = 1:stride:maxStartIdx;
    numTrajectories = length(startIndexes);

    U_traj  = cell(numTrajectories, number_of_inputs);
    Yk_traj = cell(numTrajectories, number_of_outputs);
    Yk1_traj = cell(numTrajectories, number_of_outputs);

    U_norm  = cell(numTrajectories, number_of_inputs);
    Yk_norm = cell(numTrajectories, number_of_outputs);
    Yk1_norm = cell(numTrajectories, number_of_outputs);

    for traj = 1:numTrajectories
        idx = startIndexes(traj);

        yk  = Data(idx:idx+numSamples-1, 1:number_of_outputs);
        uk  = Data(idx:idx+numSamples-1, number_of_outputs+1:number_of_outputs+number_of_inputs);
        yk1 = Data(idx:idx+numSamples-1, number_of_outputs+number_of_inputs+1:end);

        uk_norm  = (uk  - stats.U.mu)  ./ stats.U.sigma;
        yk_norm  = (yk  - stats.Yk.mu) ./ stats.Yk.sigma;
        yk1_norm = (yk1 - stats.Yk.mu) ./ stats.Yk.sigma;

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