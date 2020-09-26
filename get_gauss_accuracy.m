% Digit recognition evaluation
%
% Author: M.W. Mak (Sept. 2015)

clear; close all;

dataType = 'noisy';                         % Type of data, can be 'clean' or 'noisy'         
covType = 'full';                       % Type of covariance matrix, 'full' or 'diagonal'

% Load training and test data into memory
trnfile = sprintf('../data/%s_train_digits.mat',dataType);
tstfile = sprintf('../data/%s_test_digits.mat',dataType);
load(trnfile);                              % Load data structure trainData
load(tstfile);                              % Load data structure testData

% Extract 1000 samples (vectors) from each class in trainData{}
%trainData = extract_data(trainData, 1000);

fprintf('Start evaluating %s digit data using Gaussian classifier with %s cov matrix\n',dataType,covType);

% Train a Gaussian density function for each digit and store the PDF parameters
% in the structure array GModel
GModel = train_gauss_model(trainData, covType);

% For each test pattern (testData{k}(t,:)), present it to the classifier to find
% the most likely class (label). Then, compare the the true label to see if
% the classification decision is correct. Sum all the correct classification counts
% to estimate the overall accuracy.
totalTest = 0;
nCorrect = 0;
for k = 1:length(testData),
    nTest = size(testData{k},1);
    fprintf('Evaluating %d samples of digit %d\n', nTest, k-1);
    totalTest = totalTest + nTest;
    label = zeros(1,nTest);
    for t = 1:nTest,
        label(t) = gauss_classification(GModel, testData{k}(t,:));
    end
    nCorrect = nCorrect + length(find(label==k-1));
end

acc = 100*nCorrect/totalTest;
fprintf('Accuracy = %.2f\n',acc);