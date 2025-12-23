% first load the powerplant.dat using "import data" and 
% in numeric array format
% Load the data from 'powerplant.dat'
% data = load('powerplant.dat'); 
data = powerplant;


% Assuming the first 5 columns are inputs (U) and the next 3 columns are outputs (Y)
U = data(:, 2:6);  % Inputs (columns 1 to 5)
Y = data(:, 7:9);  % Outputs (columns 6 to 8)


Data_train = [Y(1:100, :)   U(1:100, :)   Y(2:101, :)];
Data_val   = [Y(101:149, :) U(101:149, :) Y(102:150, :)];
Data_test  = [Y(150:199, :) U(150:199, :) Y(151:200, :)];

%% ========================== TRAINING DATA ===============================
figure;

% Plot for Inputs
subplot(2, 1, 1);
hold on;
plot(Data_train(:, 4), 'r', 'DisplayName', 'uk1');
plot(Data_train(:, 5), 'g', 'DisplayName', 'uk2');
plot(Data_train(:, 6), 'b', 'DisplayName', 'uk3');
plot(Data_train(:, 7), 'm', 'DisplayName', 'uk4');
plot(Data_train(:, 8), 'c', 'DisplayName', 'uk5');
xlabel('Time (s)');
ylabel('Input values');
title('Training Inputs');
legend;

% Plot for Outputs
subplot(2, 1, 2);
hold on;
plot(Data_train(:, 1), 'r', 'DisplayName', 'yk1');
plot(Data_train(:, 2), 'g', 'DisplayName', 'yk2');
plot(Data_train(:, 3), 'b', 'DisplayName', 'yk3');
xlabel('Time (s)');
ylabel('Output values');
title('Training Outputs');
legend;

%% ========================== VALIDATION DATA ==============================
figure;

% Plot for Inputs
subplot(2, 1, 1);
hold on;
plot(Data_val(:, 4), 'r', 'DisplayName', 'uk1');
plot(Data_val(:, 5), 'g', 'DisplayName', 'uk2');
plot(Data_val(:, 6), 'b', 'DisplayName', 'uk3');
plot(Data_val(:, 7), 'm', 'DisplayName', 'uk4');
plot(Data_val(:, 8), 'c', 'DisplayName', 'uk5');
xlabel('Time (s)');
ylabel('Input values');
title('Validation Inputs');
legend;

% Plot for Outputs
subplot(2, 1, 2);
hold on;
plot(Data_val(:, 1), 'r', 'DisplayName', 'yk1');
plot(Data_val(:, 2), 'g', 'DisplayName', 'yk2');
plot(Data_val(:, 3), 'b', 'DisplayName', 'yk3');
xlabel('Time (s)');
ylabel('Output values');
title('Validation Outputs');
legend;

%% ========================== TEST DATA ===================================
figure;

% Plot for Inputs
subplot(2, 1, 1);
hold on;
plot(Data_test(:, 4), 'r', 'DisplayName', 'uk1');
plot(Data_test(:, 5), 'g', 'DisplayName', 'uk2');
plot(Data_test(:, 6), 'b', 'DisplayName', 'uk3');
plot(Data_test(:, 7), 'm', 'DisplayName', 'uk4');
plot(Data_test(:, 8), 'c', 'DisplayName', 'uk5');
xlabel('Time (s)');
ylabel('Input values');
title('Testing Inputs');
legend;

% Plot for Outputs
subplot(2, 1, 2);
hold on;
plot(Data_test(:, 1), 'r', 'DisplayName', 'yk1');
plot(Data_test(:, 2), 'g', 'DisplayName', 'yk2');
plot(Data_test(:, 3), 'b', 'DisplayName', 'yk3');
xlabel('Time (s)');
ylabel('Output values');
title('Testing Outputs');
legend;

