% Load twotankdata
load('twotankdata.mat'); % Assuming it loads variables `u` and `y`

% Create the Data matrix
% Format: [x1, x2, u, x1_next, x2_next]
Data = zeros(length(y)-1, 3); % Preallocate
for k = 1:length(y)-1
    Data(k, :) = [y(k, 1), u(k),y(k+1, 1)]; % Use y for x2 and x2_next, x1 and x1_next are zero
end

% Split into train and test datasets
train_size = floor(0.5 * size(Data, 1)); % 50% for training
Data_train = Data(1:train_size, :);
Data_train = [Data_train ; Data_train(end,:)];
Data_test = Data(train_size+1:end, :);
Data_val =  Data(train_size+1:train_size+500, :);

% Visualization for training
figure;
subplot(2, 1, 1);
plot(Data_train(:, 2)); % x2 (Tank 2 water level)
xlabel('Time (s)');
ylabel('x2(k)');
title('Input Signal (Training)');

subplot(2, 1, 2);
plot(Data_train(:, 3)); % u (Input signal)
ylabel('u(k)');
title('Water Level in Tank 2 (Training)');

% Visualization for validation
figure;
subplot(2, 1, 1);
plot(Data_val(:, 2)); % x2(k) (Tank 2 water level)
xlabel('Time (s)');
ylabel('x2(k)');
title('Input Signal (Validation)');

subplot(2, 1, 2);
plot(Data_val(:, 3)); % u(k) (Input signal)
xlabel('Time (s)');
ylabel('u(k)');
title('Water Level in Tank 2 (Validation)');



% Visualization for testing
figure;
subplot(2, 1, 1);
plot(Data_test(:, 2)); % x2 (Tank 2 water level)
xlabel('Time (s)');
ylabel('x2(k)');
title('Input Signal (Testing)');

subplot(2, 1, 2);
plot(Data_test(:, 3)); % u (Input signal)
ylabel('u(k)');
title('Water Level in Tank 2 (Testing)');
