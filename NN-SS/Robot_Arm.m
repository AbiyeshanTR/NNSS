% Load robotic arm data
load('robotarmdata.mat'); % Assuming it loads variables ue, ye, uv3, yv3, uv1, yv1

downsample_factor = 10; % Adjust the downsampling rate as needed

%% Prepare training dataset (from ye, ue)
Data = zeros(length(ye)-1, 3); % Preallocate
for k = 1:length(ye)-1
    Data(k, :) = [ ye(k, 1), ue(k),  ye(k+1, 1)];
end
Data_train = downsample(Data, downsample_factor);
Data_train(end-3:end,:) = [];

%% Prepare validation dataset (from yv1, uv1)
Data = zeros(length(yv1)-1, 3); % Preallocate
for k = 1:length(yv1)-1
    Data(k, :) = [ yv1(k, 1), uv1(k),  yv1(k+1, 1)];
end
Data_val = downsample(Data, downsample_factor);
Data_val(end-3:end,:) = [];

%% Prepare testing dataset (from yv3, uv3)
Data = zeros(length(yv3)-1, 3); % Preallocate
for k = 1:length(yv3)-1
    Data(k, :) = [ yv3(k, 1), uv3(k),  yv3(k+1, 1)];
end
Data_test = downsample(Data, downsample_factor);
Data_test(end-3:end,:) = [];

%% Visualization for training
figure;
subplot(2, 1, 1);
plot(Data_train(:, 2));
xlabel('Time (s)');
ylabel('u');
title('Training Set - Input Signal');

subplot(2, 1, 2);
plot(Data_train(:, 3));
ylabel('y');
title('Training Set - Output Signal');

%% Visualization for validation
figure;
subplot(2, 1, 1);
plot(Data_val(:, 2));
xlabel('Time (s)');
ylabel('u');
title('Validation Set - Input Signal');

subplot(2, 1, 2);
plot(Data_val(:, 3));
ylabel('y');
title('Validation Set - Output Signal');

%% Visualization for testing
figure;
subplot(2, 1, 1);
plot(Data_test(:, 2));
xlabel('Time (s)');
ylabel('u');
title('Testing Set - Input Signal');

subplot(2, 1, 2);
plot(Data_test(:, 3));
ylabel('y');
title('Testing Set - Output Signal');