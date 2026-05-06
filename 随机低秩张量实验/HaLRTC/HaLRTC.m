%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADM algorithm: tensor completion
% paper: Tensor completion for estimating missing values in visual data
% date: 05-22-2011
% min_X: \sum_i \alpha_i \|X_{i(i)}\|_*
% s.t.:  X_\Omega = T_\Omega
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X, errList, history] = HaLRTC(T, Omega, alpha, beta, maxIter, epsilon, X, Xtrue)

if nargin < 7 || isempty(X)
    X = T;
    X(logical(1-Omega)) = mean(T(Omega));
end

errList = zeros(maxIter, 1);
dim = size(T);
Y = cell(ndims(T), 1);
M = Y;

normT = norm(T(:));
for i = 1:ndims(T)
    Y{i} = X;
    M{i} = zeros(dim);
end

Msum = zeros(dim);
Ysum = zeros(dim);

% History recording
record_history = (nargout >= 3);
if record_history
    elapsed_arr = zeros(maxIter, 1);
    has_xtrue = (nargin >= 8 && ~isempty(Xtrue));
    if has_xtrue
        mse_arr = zeros(maxIter, 1);
        rmse_arr = zeros(maxIter, 1);
        psnr_arr = zeros(maxIter, 1);
        rse_arr = zeros(maxIter, 1);
        mae_arr = zeros(maxIter, 1);
        maxP = double(max(Xtrue(:)));
        if maxP <= 1
            maxP = 1;
        else
            maxP = 255;
        end
    end
end

start_time = tic;

for k = 1:maxIter
    if mod(k, 20) == 0
        fprintf('HaLRTC: iterations = %d   difference=%f\n', k, errList(k-1));
    end
    beta = beta * 1.05;

    % update Y
    Msum = 0*Msum;
    Ysum = 0*Ysum;
    for i = 1:ndims(T)
        Y{i} = Fold(Pro2TraceNorm(Unfold(X-M{i}/beta, dim, i), alpha(i)/beta), dim, i);
        Msum = Msum + M{i};
        Ysum = Ysum + Y{i};
    end

    % update X
    lastX = X;
    X = (Msum + beta*Ysum) / (ndims(T)*beta);
    X(Omega) = T(Omega);

    % update M
    for i = 1:ndims(T)
        M{i} = M{i} + beta*(Y{i} - X);
    end

    % compute the error
    errList(k) = norm(X(:)-lastX(:)) / normT;

    if record_history
        elapsed_arr(k) = toc(start_time);
        if has_xtrue
            diff_v = Xtrue(:) - X(:);
            mse_arr(k) = mean(diff_v .^ 2);
            rmse_arr(k) = sqrt(mse_arr(k));
            psnr_arr(k) = 10 * log10(maxP^2 / max(mse_arr(k), eps));
            rse_arr(k) = norm(diff_v) / max(norm(Xtrue(:)), eps);
            mae_arr(k) = mean(abs(diff_v));
        end
    end

    if errList(k) < epsilon
        break;
    end
end

errList = errList(1:k);
fprintf('HaLRTC ends: total iterations = %d   difference=%f\n\n', k, errList(k));

if record_history
    elapsed_arr = elapsed_arr(1:k);
    if has_xtrue
        mse_arr = mse_arr(1:k);
        rmse_arr = rmse_arr(1:k);
        psnr_arr = psnr_arr(1:k);
        rse_arr = rse_arr(1:k);
        mae_arr = mae_arr(1:k);
        history = table((1:k)', elapsed_arr, errList, mse_arr, rmse_arr, psnr_arr, rse_arr, mae_arr, ...
            'VariableNames', {'iteration', 'elapsed_time_seconds', 'relative_change', ...
            'MSE', 'RMSE', 'PSNR', 'RSE', 'MAE'});
    else
        history = table((1:k)', elapsed_arr, errList, ...
            'VariableNames', {'iteration', 'elapsed_time_seconds', 'relative_change'});
    end
end
end
