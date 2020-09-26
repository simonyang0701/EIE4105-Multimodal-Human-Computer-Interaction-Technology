% Predict the digit of a test image based on the GMM digit classifier
% Input:
%   GMMmodel         - Struct array containing GMM structures with the following fields:
%                       w     : Cell array of mixture weights
%                       mu    : Cell array of 1xD mean vectors 
%                       Sigma : Cell array of DxD cov matrices
%                       const : Cell array of const terms independent of x
%                      GMMmodel(1) contains the GMM object of digit '0'
%   x                - A test image in D-dim row vector
% Output:
%   label            - Predicted class label of x, '0' - '9'
%   loglikelh        - K-dim array containing the log-likelihood for each class 
%  
% Example:
%   load '../data/noisy_test_digits.mat';
%   [label, loglikelh] = gmm_classification(GMMmodel, testData{1}(1,:));
% Author: M.W. Mak (Sept. 2015)

function [label, loglikelh] = gmm_classification(GMMmodel, x)
nClasses = length(GMMmodel);
loglikelh = zeros(1,nClasses);         % log-Likelihood, log p(x|mu,Sigma)

% Compute log-likelihood of x for each class. Note that to avoid having exp(z)=0 and
% therefore log(y) = -inf, a constant alpha is added to the exponent term, which are
% then cancelled when compute the loglikelihood.
nMix = length(GMMmodel(1).w);
for k = 1:nClasses,
    % Implement the equation in Step 10 of the lab sheet here
    % Put your code here
    
end

% Find the predicted class (implement the argmax operator)
[~, label] = max(loglikelh);
label = label - 1;              % Adjust for offset