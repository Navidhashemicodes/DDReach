function  [Net, performance] = Learning_unit(Input, Output, dim , tr_ratio, val_ratio, tst_ratio, use_GPU, epochs, activation_type)


% Assuming Input and Output are Outputour input and output data matrices

% Create a neural network with ReLU activation and GPU acceleration
net = feedforwardnet(dim(2:end-1), 'trainlm');
for i=1:length(dim)-2
    net.layers{i}.transferFcn = activation_type; % Set ReLU activation for the first hidden layer
end

% % Set the input size based on your data
% net.inputs{1}.size = dim(1);

% Set normalization to "none"
net.input.processFcns = {};
net.output.processFcns = {};


% Divide the data into training, validation, and testing sets (adjust as needed)
net.divideFcn = 'dividerand'; % Divide data randomly
net.divideMode = 'sample'; % Divide the data at random
net.divideParam.trainRatio = tr_ratio;
net.divideParam.valRatio = val_ratio;
net.divideParam.testRatio = tst_ratio;

% Choose a training function (Levenberg-Marquardt backpropagation)
net.trainFcn = 'trainlm';

% Enable GPU acceleration if available
if gpuDeviceCount > 0
    net.trainParam.useGPU = use_GPU;
end

% Train the neural network
net.trainParam.epochs = epochs; % Set the number of epochs
net.trainParam.showWindow = true; % Show training progress window

% Train the network with your data Input and Output
[net, ~] = train(net, Input, Output);

% Make predictions using the trained network
predictions = net(Input);

% Evaluate the performance of the network (optional)
performance = perform(net, Output, predictions);


num_weights = size(net.LW,1);
W = cell(1, num_weights);
B  = cell(1, num_weights);

% Extract input weights
W{1} = net.IW{1, 1};
B{1} = net.b{1};
for i = 2:num_weights
    W{i} = net.LW{i, i-1};
    B{i} = net.b{i};
end

for i = 1:length(dim)-2
    L = cell(dim(i+1),1);
    L(:) = {activation_type};
    Net.layers{i} = L ;
end

Net.weights = W;
Net.biases = B;


end
