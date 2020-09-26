%=======================================================================
% function lg = logGaussian(x, mu, Sigma, const)
% Return the log of Gaussian given data x (col. vec), mean mu, and cov Sigma
% Input:
%   x            - Dx1 col vector 
%   mu           - Dx1 mean vector
%   Sigma        - DxD cov. matrix
%   const        - Constant term independent of x for speeding up computation
%                  const = -(D/2)*log(2*pi) - 0.5*logDet(Sigma)
% Output:
%   lg        - log of Gaussian likelihood
%
% Author: M.W. Mak (Sept. 2015)
%=======================================================================
function lg = logGaussian(x, mu, Sigma, const)

temp = x - mu;
lg = const - 0.5*((temp'/Sigma)*temp);