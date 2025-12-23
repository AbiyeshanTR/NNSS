classdef SIMBa_Layer < nnet.layer.Layer
    properties (Learnable)
        W
        V
        B
        C
        D
    end

    methods
        function layer = SIMBa_Layer(nx, nu, ny, name,seed)
            layer.Name = name;
            layer.Description = "State-space layer with learnable A, B, C, D";
            rng(seed)
            % Initialize learnable parameters
            layer.W = randn(2 * nx, 2 * nx) * sqrt(2 / (2 * nx));
            layer.V = randn(nx, nx) * sqrt(2 / nx);
            layer.B = randn(nx, nu) * sqrt(2 / nu);
            layer.C = randn(ny, nx) * sqrt(2 / nx);
            layer.D = randn(ny, nu) * sqrt(2 / nu);



        end


        function Z = predict(layer, int)
            nx = size(layer.B, 1);
            X = int(1:nx,:);
            U = int(nx+1:end,:);

            A = computeSchurStableA(layer.W, layer.V, 1);

            X_plus1 = pagemtimes(A,X) + pagemtimes(layer.B, U);
            Y = pagemtimes(layer.C,X) + pagemtimes(layer.D, U);

            Z = [X_plus1;Y];
            
        end

    end
end

function A = computeSchurStableA(W, V, gamma)
    epsilon_tilde = -10;
    epsilon = exp(epsilon_tilde);

    % Calculate S
    S = W' * W + epsilon * eye(size(W, 1));

    % Extract submatrices
    numStates = size(V, 1);
    S11 = S(1:numStates, 1:numStates);
    S12 = S(1:numStates, numStates+1:end);
    S22 = S(numStates+1:end, numStates+1:end);

    % Calculate the matrix to be inverted
    MatToInvert = 0.5 * ((S11 + S22) / gamma^2) + V - V';
    invMat = pinv(MatToInvert);
%     % Calculate A
     A = S12 * invMat;
 end




