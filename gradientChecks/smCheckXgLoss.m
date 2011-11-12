function [loss, paramgrad] = smCheckXgLoss(params, W, X, modelScoreFunc)

[sml, pg, energy, paramgrad, Xg2] = modelScoreFunc(W, reshape(params, size(X)));
loss = sum(energy);
paramgrad = paramgrad(:);

end

