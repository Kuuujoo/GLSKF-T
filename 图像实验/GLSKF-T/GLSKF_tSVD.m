function [Xori, Rtensor_final, Mtensor_final, psnr, history] = GLSKF_tSVD(I, Omega, lengthscaleU, lengthscaleR, varianceU, varianceR, tapering_range, d_MaternU, d_MaternR, rg, rho, gamma, maxiter, K0, epsilon)
% GLSKF_tSVD 图像修复算法主函数
% 全局项只在观测位置拟合，局部项使用观测位置残差拟合。

    N = size(I);
    n1 = N(1); n2 = N(2); n3 = N(3);
    maxP = double(max(I(:)));

    Omega = logical(Omega);
    pos_obs = find(Omega == 1);
    pos_miss = find(Omega == 0);
    num_obser = sum(Omega(:));
    if num_obser == 0
        error('Omega 中没有观测位置。');
    end

    train_matrix = double(I) .* double(Omega);
    train_mean = sum(train_matrix(:)) / num_obser;
    Isubmean = double(I) - train_mean;
    T = Isubmean .* double(Omega);

    hyper_Ku = cell(2, 1);
    hyper_Ku{1} = [log(lengthscaleU(1)), log(varianceU(1))];
    hyper_Ku{2} = [log(lengthscaleU(2)), log(varianceU(2))];

    hyper_Kr = cell(2, 1);
    hyper_Kr{1} = [log(lengthscaleR(1)), log(varianceR(1)), log(tapering_range)];
    hyper_Kr{2} = [log(lengthscaleR(2)), log(varianceR(2)), log(tapering_range)];

    x = 1:n1;
    K1 = matern(d_MaternU, hyper_Ku{1}, x);
    K1 = K1 + 1e-6 * eye(n1);
    invK1 = inv(K1);
    TaperM = wendland(hyper_Kr{1}(3), x);
    Kr1 = sparse(matern(d_MaternR, hyper_Kr{1}(1:2), x) .* TaperM);

    x = 1:n2;
    K2 = matern(d_MaternU, hyper_Ku{2}, x);
    K2 = K2 + 1e-6 * eye(n2);
    invK2 = inv(K2);
    TaperM = wendland(hyper_Kr{2}(3), x);
    Kr2 = sparse(matern(d_MaternR, hyper_Kr{2}(1:2), x) .* TaperM);

    Kr3 = sparse(eye(n3));

    X = T;
    X(pos_miss) = sum(T(:)) / num_obser;

    U = 0.1 * randn(n1, rg, n3);
    V = 0.1 * randn(n2, rg, n3);
    S = 0.1 * randn(rg, rg, n3);
    M = tproduct(tproduct(U, S), ttranspose(V));

    Rtensor = zeros(N);
    z = Rtensor(:);

    train_norm = max(norm(T(:)), eps);
    X_last = X;
    psnrf = zeros(maxiter, 1);
    psnr_global = zeros(maxiter, 1);
    mse_hist = zeros(maxiter, 1);
    rmse_hist = zeros(maxiter, 1);
    elapsed_hist = zeros(maxiter, 1);
    tol_hist = zeros(maxiter, 1);
    iter = 0;
    run_timer = tic;

    while true
        iter = iter + 1;

        Gtensor = X - Rtensor;
        [U, V, S] = masked(Gtensor, double(Omega), U, V, S, invK1, invK2, rho, 10, 1e-5);
        M = tproduct(tproduct(U, S), ttranspose(V));

        X_pred = M + Rtensor;
        X = zeros(N);
        X(pos_obs) = Isubmean(pos_obs);
        X(pos_miss) = X_pred(pos_miss);

        Xglobal = min(max(X + train_mean, 0), maxP);
        mse_global = mean((double(I(:)) - double(Xglobal(:))) .^ 2);
        psnr_global(iter) = 10 * log10(maxP^2 / max(mse_global, eps));

        if iter >= K0
            Ltensor = X - M;
            Ltensor_mask = Ltensor .* double(Omega);
            [z(pos_obs), ~] = cg_local(gamma, Kr3, Kr2, Kr1, pos_obs, Ltensor_mask, z(pos_obs), 100);

            Rvector = kroneckerMVM(Kr3, Kr2, Kr1, z, n1, n2, n3);
            Rtensor = reshape(Rvector, N);

            R_unfold3 = unfold(Rtensor, 3);
            idx = sum(abs(R_unfold3), 1) > 0;
            if sum(idx) > 1
                R_obs = R_unfold3(:, idx);
                Kr3 = sparse(cov(R_obs'));
            end
        else
            Rtensor = zeros(size(Rtensor));
        end

        X_pred = M + Rtensor;
        X = zeros(N);
        X(pos_obs) = Isubmean(pos_obs);
        X(pos_miss) = X_pred(pos_miss);

        Xori = min(max(X + train_mean, 0), maxP);
        [mse_hist(iter), rmse_hist(iter), psnrf(iter)] = tensor_metrics(I, Xori, maxP);
        elapsed_hist(iter) = toc(run_timer);

        fprintf('Epoch = %d, GlobalPSNR = %.6f, PSNR = %.6f\n', ...
            iter, psnr_global(iter), psnrf(iter));

        tol = norm(X(:) - X_last(:)) / train_norm;
        tol_hist(iter) = tol;
        X_last = X;

        if (tol < epsilon) || (iter >= maxiter)
            break;
        end
    end

    Xori = min(max(Xori, 0), maxP);
    Rtensor_final = Rtensor + train_mean;
    Mtensor_final = M + train_mean;
    psnr = psnrf(iter);
    history = table((1:iter)', elapsed_hist(1:iter), mse_hist(1:iter), ...
        rmse_hist(1:iter), psnrf(1:iter), psnr_global(1:iter), tol_hist(1:iter), ...
        'VariableNames', {'iteration', 'elapsed_time_seconds', 'MSE', ...
        'RMSE', 'PSNR', 'global_PSNR', 'relative_change'});
end

function [mse_val, rmse_val, psnr_val] = tensor_metrics(I, X, maxP)
    diff = double(I) - double(X);
    mse_val = mean(diff(:) .^ 2);
    rmse_val = sqrt(mse_val);
    psnr_val = 10 * log10(maxP^2 / max(mse_val, eps));
end
