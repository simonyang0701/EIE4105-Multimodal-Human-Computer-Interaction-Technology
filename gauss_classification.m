% Predict the digit of a test image based on the Gaussian digit-classifier
% Input:
%   GModel           - A Struct array containing the mean and cov matrix of each digit
%                      See train_gauss_model.m for details
%   x                - A test image in D-dim row vector
% Output:
%   label            - Predicted class label of x, '0' - '9'
%   loglikelh        - K-dim array containing the log-likelihood for each class 
%  
% Example:
%   load '../data/noisy_test_digits.mat';
%   [label, loglikelh] = gauss_classification(GModel, testData{1}(1,:));
%
% Author: M.W. Mak (Sept. 2015)

function [label, loglikelh] = gauss_classification(GModel, x)
nClasses = length(GModel);
loglikelh = zeros(1,nClasses);         % log-Likelihood, log p(x|mu,Sigma)

% Compute log-likelihood of x for each class
for k = 1:nClasses,
    loglikelh(k) = logGaussian(x', GModel(k).mu', GModel(k).Sigma, GModel(k).const);
end

% Find the predicted class (implement the argmax operator)
[~, label] = max(loglikelh);
label = label - 1;              % Adjust for offset (index in Matlab starts from 1, but our
                                % digits start from 0)