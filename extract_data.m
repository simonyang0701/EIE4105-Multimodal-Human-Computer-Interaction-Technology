function X = extract_data(X, nSamples)
% Extract nSamples from each class in X
for k = 1:length(X),
    %P = randperm(size(X{k},1));
    P = 1:size(X{k},1);
    X{k} = X{k}(P(1:nSamples),:);
end
