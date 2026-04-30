function [Xori, Rtensor_final, Mtensor_final, psnr] = GLSKF_tSVD(I, Omega, lengthscaleU, lengthscaleR, varianceU, varianceR, tapering_range, d_MaternU, d_MaternR, rg,rho, gamma, maxiter, K0, epsilon)
% GLSKF_tSVD算法主函数 (轻量修改版：时域掩码投影 + 频域无掩码更新)
% I: 原始图像张量
% Omega: 观测掩码
% lengthscaleU: 全局分量长度尺度参数
% lengthscaleR: 局部分量长度尺度参数  
% varianceU: 全局分量方差参数
% varianceR: 局部分量方差参数
% tapering_range: 锥化范围
% d_MaternU, d_MaternR: Matérn函数平滑度参数
% rg: 全局分量的tubal-rank
% rho, gamma: 正则化参数
% maxiter: 最大迭代次数
% K0: 开始更新局部分量的迭代次数
% epsilon: 收敛阈值

% 返回值：
% Xori: 恢复的原始图像
% Rtensor_final: 局部分量
% Mtensor_final: 全局分量
% psnr: 最终PSNR值

    % 获取张量维度
    N = size(I);
    n1 = N(1); n2 = N(2); n3 = N(3);
    maxP = double(max(I(:)));
    
    % 处理掩码
    Omega = logical(Omega);
    pos_obs = find(Omega == 1);
    pos_miss = find(Omega == 0);
    num_obser = sum(Omega(:));
    
    % 预处理数据
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
    % 第一个维度（高度）
    x = 1:n1;
    K1 = matern(d_MaternU, hyper_Ku{1}, x);
    K1 = K1 + 1e-6 * eye(n1); % 添加正则化确保正定
    invK1 = inv(K1);
    TaperM = wendland(hyper_Kr{1}(3), x);
    Kr1 = sparse(matern(d_MaternR, hyper_Kr{1}(1:2), x) .* TaperM);
    
    % 第二个维度（宽度）
    x = 1:n2;
    K2 = matern(d_MaternU, hyper_Ku{2}, x);
    K2 = K2 + 1e-6 * eye(n2); % 添加正则化确保正定
    invK2 = inv(K2);
    TaperM = wendland(hyper_Kr{2}(3), x);
    Kr2 = sparse(matern(d_MaternR, hyper_Kr{2}(1:2), x) .* TaperM);
    
    % 第三个维度（RGB通道）
    Kr3 = sparse(eye(n3));
    
    % 初始化X张量
    X = T;
    X(pos_miss) = sum(T(:)) / num_obser;
    
    % 初始化t-SVD分解的因子张量（真实空间）
    U = 0.1 * randn(n1, rg, n3);
    V = 0.1 * randn(n2, rg, n3);
    S = 0.1 * randn(rg, rg, n3);
    
    % 计算初始M张量
    M = tproduct(tproduct(U, S), ttranspose(V));
    
    % 初始化R张量
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
        
        for inner = 1:3
            % 计算G张量 (G = X - R)
            Gtensor = X - Rtensor;
            
            % 对第3维做FFT，进入频域
            Gtensor_f = fft(Gtensor, [], 3);
            U_f = fft(U, [], 3);
            V_f = fft(V, [], 3);
            S_f = fft(S, [], 3);
            
            % 只更新前半谱，后半谱通过共轭对称获得
            half = floor(n3/2) + 1;
            for k = 1:half
                % 获取第k个频率切片
                Gk = Gtensor_f(:, :, k);
                Uk = U_f(:, :, k);
                Vk = V_f(:, :, k);
                Sk = S_f(:, :, k);
                
                % 更新Uk（频域完整拟合，无掩码）
                [Uk_new] = tsvd_U(Gk, Uk, Vk, Sk, invK1, rho, n3);
                U_f(:, :, k) = Uk_new;
                
                % 更新Vk
                [Vk_new] = tsvd_V(Gk, Uk_new, Vk, Sk, invK2, rho, n3);
                V_f(:, :, k) = Vk_new;
                
                % 更新S
                if k == 1
                    % DC分量：使用零初值
                    Sk_new = tsvd_S(Gk, Uk_new, Vk_new);
                else
                    % 其他频率：使用上一轮的S作为初值
                    Sk_prev = S_f(:, :, k);
                    Sk_new = tsvd_S(Gk, Uk_new, Vk_new, Sk_prev);
                end
                S_f(:, :, k) = Sk_new;
                
                % 强制频域共轭对称
                if k > 1 && k < n3
                    kk = n3 - k + 2;
                    U_f(:, :, kk) = conj(U_f(:, :, k));
                    V_f(:, :, kk) = conj(V_f(:, :, k));
                    S_f(:, :, kk) = conj(S_f(:, :, k));
                end
            end
            
            % 返回时域
            U = ifft(U_f, [], 3, 'symmetric');
            V = ifft(V_f, [], 3, 'symmetric');
            S = ifft(S_f, [], 3, 'symmetric');
            
            % 只在内循环第3遍重构M
            if inner == 3
                M = tproduct(tproduct(U, S), ttranspose(V));
            end
        end
        
        % 时域掩码投影
        % M更新后立即投影到观测约束
        X_pred = M + Rtensor;
        X = zeros(N);
        X(pos_obs) = Isubmean(pos_obs);  % 观测位置强制为真值
        X(pos_miss) = X_pred(pos_miss);  % 缺失位置用预测
        
        % 更新局部分量R
        if iter >= K0
            % 计算L张量 (L = X - M)
            Ltensor = X - M;
            Ltensor_mask = Ltensor .* double(Omega);
            
            % 创建掩码和观测位置
            mask_flat = Omega(:);
            pos_obs_local = find(mask_flat == 1);
            
            % 使用共轭梯度法求解局部分量
            [z(pos_obs_local), ~] = cg_local(gamma, Kr3, Kr2, Kr1, pos_obs_local, Ltensor_mask, z(pos_obs_local), 100);
            
            % 应用Kronecker矩阵乘法
            Rvector = kroneckerMVM(Kr3, Kr2, Kr1, z, n1, n2, n3);
            Rtensor = reshape(Rvector, N);
            
            % 更新第三维协方差矩阵
            R_unfold3 = unfold(Rtensor, 3);
            idx = sum(abs(R_unfold3), 1) > 0;
            if sum(idx) > 1
                R_obs = R_unfold3(:, idx);
                Kr3 = cov(R_obs');
            end
        else
            Rtensor = zeros(size(Rtensor));
        end
        
        % 再次时域投影
        X_pred = M + Rtensor;
        X = zeros(N);
        X(pos_obs) = Isubmean(pos_obs);  % 观测位置强制为真值
        X(pos_miss) = X_pred(pos_miss);  % 缺失位置用预测
        
        % 恢复原始图像
        Xori = X + train_mean;
        Xrecovery = max(0, Xori);
        Xrecovery = min(maxP, Xrecovery);
        
        % 计算PSNR
        mse = 0;
        for c = 1:3
            mse_c = norm(double(I(:,:,c)) - Xrecovery(:,:,c), 'fro')^2 / (n1 * n2);
            mse = mse + mse_c;
        end
        mse = mse / 3;
        psnrf(iter) = 10 * log10(maxP^2 / mse);
        
        % 显示进度
        fprintf('Epoch = %d, PSNR = %.6f\n', iter, psnrf(iter));

        % 检查收敛性
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