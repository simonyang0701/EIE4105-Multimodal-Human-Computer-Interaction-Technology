% Fit a GMM density function to each digit and
% use Bayes rule to construct a digit classifier
% Input:
%   trainData        - 10x1 cell array containing training data. trainData{1}
%                      contains a number of '0', each is stored in a row.
%   nMix             - No. of mixtures
% Output:
%   GMMmodel         - Struct array containing GMM structures with the following fields:
%                       w     : Cell array of mixture weights
%                       mu    : Cell array of 1xD mean vectors 
%                       Sigma : Cell array of DxD cov matrices
%                       const : Cell array of const terms independent of x
%                      GMMmodel(1) contains the GMM object of digit '0'
% Example:
%   load '../data/noisy_train_digits.mat';
%   GMMmodel = train_gmm_model(trainData, 4);
%
% Author: M.W. Mak (Sept. 2015)

function GMMmodel = train_gmm_model(trainData, nMix)

% If statistical Toolbox is not available, use the GMM in netlab
if exist('gmdistribution','class'),
    GMMmodel = train_stattb_gmm(trainData, nMix);
else
    GMMmodel = train_netlab_gmm(trainData, nMix);
end


%% Private functions

function GMMmodel = train_stattb_gmm(trainData, nMix)
nClasses = length(trainData);
D = size(trainData{1}, 2);
GMMmodel = struct([]);
options = statset('Display','iter','MaxIter',20);
for k = 1:nClasses,
    fprintf('Training digit %d\n', k-1);
    gmm = gmdistribution.fit(trainData{k}, nMix, 'CovType','diagonal', 'Options',options);
    mu = cell(nMix,1);
    Sigma = cell(nMix,1);
    w = cell(nMix,1);
    const = cell(nMix,1);
    for j = 1:nMix,
        mu{j} = gmm.mu(j,:);
        Sigma{j} = diag(gmm.Sigma(1,:,j));
        w{j} = gmm.PComponents(j);
        const{j} = -(D/2)*log(2*pi) - 0.5*logDet(Sigma{j});
    end
    GMMmodel(k).mu = mu;
    GMMmodel(k).Sigma = Sigma;
    GMMmodel(k).w = w;
    GMMmodel(k).const = const;
end    

function GMMmodel = train_netlab_gmm(trainData, nMix)
addpath('../netlab');                    % Use GMM in netlab
nClasses = length(trainData);
D = size(trainData{1}, 2);
GMMmodel = struct([]);
options = foptions; 
options(14) = 20;           % No. of iterations
options(1) = 1;             % Display message
for k = 1:nClasses,
    X = trainData{k};
    fprintf('Training digit %d\n', k-1);
    mix = gmm(D,nMix,'diag');       % Only work for diagonal covariance
    mix = gmminit(mix, X, options); 
    mix = gmmem(mix, X, options);
    mu = cell(nMix,1);
    Sigma = cell(nMix,1);
    w = cell(nMix,1);
    const = cell(nMix,1);
    for j = 1:nMix,
        mu{j} = mix.centres(j,:);
        Sigma{j} = diag(mix.covars(j,:));
        w{j} = mix.priors(j);
        const{j} = -(D/2)*log(2*pi) - 0.5*logDet(Sigma{j});
    end
    GMMmodel(k).mu = mu;
    GMMmodel(k).Sigma = Sigma;
    GMMmodel(k).w = w;
    GMMmodel(k).const = const;
end
return;
