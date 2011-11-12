function [smloss, grad, energy, Xg, Xg2] = ...
    icaScoreMatching(W, X, nHidden, nInput)

% Save Original Weights
oW = reshape(W, nHidden, nInput);

% Force onto unit ball
W = l2row(oW);

lambda  = 0;      % weight decay
epsilon = 1;
smReg   = 0;      % Regularized score matching?

F  = W*X;
absF = sqrt(epsilon + (W*X).^2);
energy = 0;
%% Compute Energy (only if interested, for checking gradients)
if nargout > 2
    energy = energy + sum(absF);
end

%% Compute del E/del X
oX = F ./ absF;
Xg = W' * oX;

%% Score Matching Loss
smloss = 0.5*sum(sum(Xg.^2));

%% Compute del^2 E/del X_i ^2 
oX2 = (1 - oX.^2) ./ absF;
Xg2 = (W.^2)' * oX2;

smloss = smloss + sum(sum(-Xg2, 1)) + smReg * sum(sum(Xg2.^2)) ;

%% Weight Regularization , note that we sum over examples so * size(X,2)
smloss = smloss + 0.5 * lambda * size(X,2) * sum(W(:).^2);

%% Scale smloss to be nicer
smloss = smloss / size(X, 2);

%% Done if we are not interested in gradients
if nargout <= 1
    return
end

%% Compute Gradient for 0.5*sumsmCheck(sum(Xg.^2))
if nargout > 1 
    Wgrad = oX * Xg';
    
    Xg2a = W * Xg;
    Xg2a = Xg2a .* oX2;
    
    Wgrad = Wgrad + Xg2a * X';
end

%% Compute Gradient for  sum(sum(-Xg2, 1)) + smReg * sum(sum(Xg2.^2)) 
if nargout > 1
    
    vhg2 = oX2 * (- 1 + 2 * smReg * Xg2)';
    Wgrad = Wgrad + 2 * W .* vhg2;
    
    Xg3a = (W.^2) * ( - 1 + 2 * smReg * Xg2);
    Xg3a = Xg3a .* (-3 * oX .* (1 - oX.^2) ./ (absF.^2));    % back prop
    
    Wgrad = Wgrad + Xg3a * X';
end


%% Weight Regularization
Wgrad = Wgrad + lambda * size(X,2) * W;

%% Compress for returning
Wgrad = l2rowg(oW, W, Wgrad);

grad = [Wgrad(:)] / size(X, 2);

end
