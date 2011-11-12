%% Initialize Checks
nHidden = 16;
nInput = 4;
W = randn(nHidden, 4);
X = randn(4, 2);
modelScoreMatchingFunc = @(W, X) icaScoreMatching (W, X, nHidden, nInput);

%% Check Xg
[loss, grad] = smCheckXgLoss(X(:), W(:), X, modelScoreMatchingFunc);
numgrad = computeNumericalGradient( @(p) smCheckXgLoss(p, W, X, modelScoreMatchingFunc), X(:));

numgrad=numgrad(:); grad = grad(:);

% Use this to visually compare the gradients side by side
disp([numgrad grad]); 

% Compare numerically computed gradients wit the ones obtained from backpropagation
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); % Should be small. In our implementation, these values are
            % usually less than 1e-9.

%% Check Xg2
% [smloss, paramgrad, energy, Xg, Xg2, Xg3] = rbmScoreMatching(modelParams, X);
numgrad = zeros(numel(X),1);
grad = zeros(numel(X),1);
for Xidx = 1:numel(X)
    params = X(Xidx);
    [loss, grad(Xidx)] = smCheckXg2Loss(params, W, X, modelScoreMatchingFunc, Xidx);
    numgrad(Xidx) = computeNumericalGradient( @(p) smCheckXg2Loss(p, W, X, modelScoreMatchingFunc, Xidx), params);
end

numgrad=numgrad(:); grad = grad(:);

disp([numgrad grad]); 

diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); 
            
%% Check Param Gradient

[loss, grad] = icaScoreMatching(W, X, nHidden, nInput);
numgrad = computeNumericalGradient( @(p) icaScoreMatching(p, X, nHidden, nInput), W(:));

disp([numgrad grad]); 

% Compare numerically computed gradients with the ones obtained from backpropagation
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); % Should be small. In our implementation, these values are
            % usually less than 1e-9.

