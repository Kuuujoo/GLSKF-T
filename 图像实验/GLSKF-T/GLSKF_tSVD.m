function [Xori, Rtensor_final, Mtensor_final, psnr] = GLSKF_tSVD(I, Omega, lengthscaleU, lengthscaleR, varianceU, varianceR, tapering_range, d_MaternU, d_MaternR, rg, rho, gamma, maxiter, K0, epsilon)
% GLSKF_tSVD 图像修复算法主函数
% 全局项只在观测位置拟合，使用 masked 更新 t-SVD 因子。
% 每轮外循环中全局项做 3 次内循环，让 U、V、S 更新更充分。
% 局部项仍然使用观测位置的残差进行拟合。
%
% 输入：
%   I               原始图像张量
%   Omega           观测掩码
%   lengthscaleU    全局分量长度尺度
%   lengthscaleR    局部分量长度尺度
%   varianceU       全局分量方差
%   varianceR       局部分量方差
%   tapering_range  锥化范围
%   d_MaternU       全局 Matern 平滑度参数
%   d_MaternR       局部 Matern 平滑度参数
%   rg              全局分量 tubal-rank
%   rho             全局正则化参数
%   gamma           局部正则化参数
%   maxiter         最大迭代次数
%   K0              开始更新局部分量的迭代轮次
%   epsilon         收敛阈值
%
% 输出：
%   Xori            修复后的图像张量
%   Rtensor_final   局部分量
%   Mtensor_final   全局分量
%   psnr            最后一轮 PSNR

    % 获取张量维度
    N = size(I);
    n1 = N(1); n2 = N(2); n3 = N(3);
    maxP = double(max(I(:)));
    
    % 处理观测掩码
    Omega = logical(Omega);
    pos_obs = find(Omega == 1);
    pos_miss = find(Omega == 0);
    num_obser = sum(Omega(:));
    
    % 数据预处理
    train_matrix = double(I) .* double(Omega);
    train_mean = sum(train_matrix(:)) / num_obser;
    Isubmean = double(I) - train_mean;
    T = Isubmean .* double(Omega);
    
    % 设置超参数
    hyper_Ku = cell(2, 1);
    hyper_Ku{1} = [log(lengthscaleU(1)), log(varianceU(1))];
    hyper_Ku{2} = [log(lengthscaleU(2)), log(varianceU(2))];
    
    hyper_Kr = cell(2, 1);
    hyper_Kr{1} = [log(lengthscaleR(1)), log(varianceR(1)), log(tapering_range)];
    hyper_Kr{2} = [log(lengthscaleR(2)), log(varianceR(2)), log(tapering_range)];
    
    % 初始化协方差矩阵
    % 第一维，高度
    x = 1:n1;
    K1 = matern(d_MaternU, hyper_Ku{1}, x);
    K1 = K1 + 1e-6 * eye(n1);
    invK1 = inv(K1);
    TaperM = wendland(hyper_Kr{1}(3), x);
    Kr1 = sparse(matern(d_MaternR, hyper_Kr{1}(1:2), x) .* TaperM);
    
    % 第二维，宽度
    x = 1:n2;
    K2 = matern(d_MaternU, hyper_Ku{2}, x);
    K2 = K2 + 1e-6 * eye(n2);
    invK2 = inv(K2);
    TaperM = wendland(hyper_Kr{2}(3), x);
    Kr2 = sparse(matern(d_MaternR, hyper_Kr{2}(1:2), x) .* TaperM);
    
    % 第三维，RGB 通道
    Kr3 = sparse(eye(n3));
    
    % 初始化 X 张量
    X = T;
    X(pos_miss) = sum(T(:)) / num_obser;
    
    % 初始化 t-SVD 因子张量
    U = 0.1 * randn(n1, rg, n3);
    V = 0.1 * randn(n2, rg, n3);
    S = 0.1 * randn(rg, rg, n3);
    
    % 初始化全局分量
    M = tproduct(tproduct(U, S), ttranspose(V));
    
    % 初始化局部分量
    Rtensor = zeros(N);
    z = Rtensor(:);
    
    % 初始化迭代变量
    train_norm = norm(T(:));
    X_last = X;
    psnrf = zeros(maxiter, 1);
    iter = 0;
    
    % 主迭代循环
    while true
        iter = iter + 1;
        
        % 全局项更新，只在观测位置拟合
        
            Gtensor = X - Rtensor;
            [U, V, S] = masked(Gtensor, double(Omega), U, V, S, invK1, invK2, rho, 10, 1e-5);
            M = tproduct(tproduct(U, S), ttranspose(V));
        
        
        % 掩码投影，观测位置强制为真值，缺失位置使用预测值
        X_pred = M + Rtensor;
        X = zeros(N);
        X(pos_obs) = Isubmean(pos_obs);
        X(pos_miss) = X_pred(pos_miss);
        
        % 更新局部分量
        if iter >= K0
            Ltensor = X - M;
            Ltensor_mask = Ltensor .* double(Omega);
            
            mask_flat = Omega(:);
            pos_obs_local = find(mask_flat == 1);
            
            [z(pos_obs_local), ~] = cg_local(gamma, Kr3, Kr2, Kr1, pos_obs_local, Ltensor_mask, z(pos_obs_local), 100);
            
            Rvector = kroneckerMVM(Kr3, Kr2, Kr1, z, n1, n2, n3);
            Rtensor = reshape(Rvector, N);
            
            % 根据当前局部分量更新第三维协方差
            R_unfold3 = unfold(Rtensor, 3);
            idx = sum(abs(R_unfold3), 1) > 0;
            if sum(idx) > 1
                R_obs = R_unfold3(:, idx);
                Kr3 = cov(R_obs');
            end
        else
            Rtensor = zeros(size(Rtensor));
        end
        
        % 再次掩码投影
        X_pred = M + Rtensor;
        X = zeros(N);
        X(pos_obs) = Isubmean(pos_obs);
        X(pos_miss) = X_pred(pos_miss);
        
        % 恢复到原始图像值域
        Xori = X + train_mean;
        Xrecovery = max(0, Xori);
        Xrecovery = min(maxP, Xrecovery);
        
        % 计算 PSNR
        mse = 0;
        for c = 1:3
            mse_c = norm(double(I(:, :, c)) - Xrecovery(:, :, c), 'fro')^2 / (n1 * n2);
            mse = mse + mse_c;
        end
        mse = mse / 3;
        psnrf(iter) = 10 * log10(maxP^2 / mse);
        
        % 显示迭代进度
        fprintf('Epoch = %d, PSNR = %.6f\n', iter, psnrf(iter));
        
        % 收敛判断
        tol = norm(X(:) - X_last(:)) / train_norm;
        X_last = X;
        
        if (tol < epsilon) || (iter >= maxiter)
            break;
        end
    end
    
    % 返回最终结果
    Xori = max(0, Xori);
    Xori = min(maxP, Xori);
    Rtensor_final = Rtensor + train_mean;
    Mtensor_final = M + train_mean;
    psnr = psnrf(iter);
end

