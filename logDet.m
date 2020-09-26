%=======================================================================
% function d = logDet(A)
% Return the log of determinate of a matrix
%
% Author: M.W. Mak (Sept. 2015)
%=======================================================================
function d = logDet(A)
L = chol(A);
d = 2*sum(log(diag(L)));