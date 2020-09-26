% Fit one Gaussian density function to each digit and
% use Bayes rule to construct a digit classifier
% Input:
%   trainData        - 10x1 cell array containing training data. trainData{1}
%                      contains a number of '0', each is stored in a row.
%   covType          - Type of covariance matrix, could be 'full' or 'diagonal'
% Output:
%   GModel           - Array of structure with the following fields:
%                       mu    : 1xD mean vector
%                       Sigma : DxD covariance matrix
%                       const : const term independent of x
%                      GModel(1) contains the mean and cov matrix of digit '0'
% Example:
%   load '../data/noisy_train_digits.mat';
%   GModel = train_gauss_model(trainData,'diagonal');
%
% Author: M.W. Mak (Sept. 2015)

function GModel = train_gauss_model(trainData, covType)
nClasses = length(trainData);
D = size(trainData{1},2);

GModel = struct([]);
for k = 1:nClasses,
    GModel(k).mu = mean(trainData{k},1);
    Sigma = cov(trainData{k},1);
    if strcmp(covType, 'full'),
        GModel(k).Sigma = Sigma;
    else
        GModel(k).Sigma = diag(diag(Sigma));    % Diagonal cov
    end
    GModel(k).const = -(D/2)*log(2*pi) - 0.5*logDet(GModel(k).Sigma);
end    
    

