%%=========================================================================
%%  Load the SDI DataSet(Dermatology)
%%=========================================================================

% Load the SDI skins dataset.
clear();
load A2
load B
N = 3; % N is the total number of output neurons i.e total number of diseases
Y = eye(N);
for i=1:size(B,1)
    train_y(i,:) = Y(B(i),:);
end

% Reshape the images back into 2D images.
% 'train_x' begin as 2D matrices with one image per row.
%   train_x  [ - x - ]
% Also, rescale the pixel values from 0 - 255 to 0 - 1. 

train_x = double(reshape(A2',56,56,762))/255;
train_y = train_y';

% To display an example image. Note that the 
% images need to be transposed in order to be oriented properly.
colormap gray;
imagesc(train_x(:, :,50)');
axis square;

%%-------------------------------------------------------------------------
%%  Define and Train the CNN
%%-------------------------------------------------------------------------

% Define the structure of our CNN.
% Our CNN will have six layers: an input layer, 2 convolution layers,
% 2 pooling layers, and an output layer
%
% Each layer is defined by a structure with the following fields
%
sdi.layers = {
    struct('type', 'input') %input layer
    struct('type', 'conv', 'outputmaps',6, 'ks', 5) %convolution layer
    struct('type', 'pool', 'scale', 2) %pooling layer
    struct('type', 'conv', 'outputmaps',12, 'ks', 5) %convolution layer
    struct('type', 'pool', 'scale', 2) %pooling layer
};
rand('state', 0)
ptms.alpha = 1;
ptms.batchsize =6;
ptms.numepochs =50;

% Create all of the parameters for the network and randomly initialize

sdi = sdi_initial(sdi, train_x, train_y);
fprintf('Training the CNN_modal...\n');

startTime = tic();

% Train the CNN using the training data.
sdi = sdi_train(sdi, train_x, train_y, ptms);

%To save the training set remove the comment below
%save ("--location to save--","--filename.mat--");


fprintf('...Done. Training took %.2f seconds\n', toc(startTime));

%%=========================================================================
%%  Test the SDI on the test set
%%=========================================================================

fprintf('Evaluating test set...\n');

% Evaluate the trained CNN over the test samples.
[er, bad] = sdi_test(sdi, train_x, train_y);

% Calculate the number of correctly classified examples.
numRight = size(train_y, 2) - numel(bad);

fprintf('Accuracy: %.2f%%\n', numRight / size(train_y, 2) * 100); 

% Plot mean squared error over the course of the training.
figure(1); 
plot(sdi.rL);
title('Mean Squared Error');
xlabel('Training Batch');
ylabel('Mean Squared Error');

% Verify the accuracy is at least 88%.
assert(er < 0.12, 'Too big error');
