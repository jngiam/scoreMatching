%% Startup (addpaths)
startup

%% Clear
clear ; close all ; clc ;

%% Load Data
%  You can obtain patches.mat from 
%  http://cs.stanford.edu/~jngiam/data/patches.mat

fprintf('Loading Data\n');

%  Loads a variable data (size 256x50000)
load patches.mat

%  Reduce dataset size for faster training
data = data(:, 1:20000);

%% PCA Whitening
fprintf('\nPCA Whitening\n');

% Remove DC
data = bsxfun(@minus, data, mean(data, 1));

% Remove the "mean" patch
data = bsxfun(@minus, data, mean(data, 2));

% Compute Covariance Matrix and Eigenstuff
cov = data * data' / size(data, 2);
[E,D] = eig(cov);
d = diag(D);

% Sort eigenvalues in descending order
[dsort, idx] = sort(d, 'descend');

% PCA Whitening (and pick top 99% of eigenvalues)
dsum = cumsum(dsort);
dcutoff = find(dsum > 0.99 * dsum(end), 1);
E = E(:, idx(1:dcutoff));
d = d(idx(1:dcutoff));
V = diag(1./sqrt(d+1e-6)) * E';

%% Whiten the data
whiteData = V * data;

%% Run the optimization with minFunc (ICA)
fprintf('\nTraining ICA (w/ Score Matching)\n\n');
nHidden = 400; nInput = size(whiteData, 1);
W = randn(nHidden, nInput); 
options.Method  = 'lbfgs';
options.maxIter = 100;	    % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

tic
[optW, cost] = minFunc( @icaScoreMatching, ...
                        W(:), options, whiteData, ...
                        nHidden, nInput);
toc

%% Display Results
optW = reshape(optW, nHidden, nInput);
displayData(optW * V);

fprintf('ICA Training Completed.\n');
fprintf('Press Enter to Continue.\n\n');
pause
