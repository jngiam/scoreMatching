function [loss, paramgrad] = smCheckXg2Loss(params, W, X, modelScoreFunc, Xidx)

X(Xidx) = params;

[sml, pg, energy, Xg, paramgrad] = modelScoreFunc(W, X);

loss = Xg(Xidx);
paramgrad = paramgrad(Xidx);

end

