function [Xori, Rtensor_final, Mtensor_final, psnr_hist] = GLSKF_tSVD(I, Omega, lengthscaleU, lengthscaleR, varianceU, varianceR, tapering_range, d_MaternU, d_MaternR, rg, rho, gamma, maxiter, K0, epsilon)
% GLSKF_tSVD  视频修复算法主函数
% 全局项：频域逐切片更新（tsvd_U/V/S），传入前对 G 乘掩码，确保缺失位置不参与拟合。
% 局部项：空间维度使用 Matern x Wendland 锥化核，时间维度 Kr3 由 eye 初始化后经验更新。
%
% 输入：
%   I               原始视频张量 (h x w x n3)，uint8
%   Omega           观测掩码 (h x w x n3)，logical
%   lengthscaleU    全局分量长度尺度，1x2
%   lengthscaleR    局部分量长度尺度，1x2
%   varianceU       全局分量方差，1x2
%   varianceR       局部分量方差，1x2
%   tapering_range  空间锥化范围
%   d_MaternU       全局 Matern 平滑度参数
%   d_MaternR       局部 Matern 平滑度参数
%   rg              tubal-rank
%   rho             全局正则化参数
%   gamma           局部正则化参数
%   maxiter         最大迭代次数
%   K0              开始更新局部分量的迭代轮次
%   epsilon         收敛阈值
%
% 输出：
%   Xori            修复后视频张量
%   Rtensor_final   局部分量（含均值）
%   Mtensor_final   全局分量（含均值）
%   psnr_hist       每轮 PSNR 记录

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

% 数据预处理
train_matrix = double(I) .* double(Omega);
train_mean = sum(train_matrix(:)) / num_obser;
Isubmean = double(I) - train_mean;
T = Isubmean .* double(Omega);

%% 协方差矩阵初始化
% 第一维（高度）
x = 1:n1;
K1 = matern(d_MaternU, [log(lengthscaleU(1)), log(varianceU(1))], x);
K1 = K1 + 1e-6 * eye(n1);
invK1 = inv(K1);
TaperM1 = wendland(log(tapering_range), x);
Kr1 = sparse(matern(d_MaternR, [log(lengthscaleR(1)), log(varianceR(1))], x) .* TaperM1);

% 第二维（宽度）
x = 1:n2;
K2 = matern(d_MaternU, [log(lengthscaleU(2)), log(varianceU(2))], x);
K2 = K2 + 1e-6 * eye(n2);
invK2 = inv(K2);
TaperM2 = wendland(log(tapering_range), x);
Kr2 = sparse(matern(d_MaternR, [log(lengthscaleR(2)), log(varianceR(2))], x) .* TaperM2);

% 第三维（时间帧）：identity 初始化，迭代中经验更新，无锥化无正则
Kr3 = sparse(eye(n3));

%% 变量初始化
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
iter = 0;

%% 主迭代循环
while true
    iter = iter + 1;

    %% 全局项更新
    
    Gtensor = X - Rtensor;
    [U, V, S] = masked(Gtensor, double(Omega), U, V, S, invK1, invK2, rho, 10, 1e-5);
    M = tproduct(tproduct(U, S), ttranspose(V));
 
   

    % 掩码投影：观测位置强制为真值，缺失位置用预测
    X_pred = M + Rtensor;
    X = zeros(N);
    X(pos_obs) = Isubmean(pos_obs);
    X(pos_miss) = X_pred(pos_miss);

    %% 局部项更新
    if iter >= K0
        Ltensor = X - M;
        Ltensor_mask = Ltensor .* double(Omega);

        [z(pos_obs), ~] = cg_local(gamma, Kr3, Kr2, Kr1, pos_obs, Ltensor_mask, z(pos_obs), 100);

        Rvector = kroneckerMVM(Kr3, Kr2, Kr1, z, n1, n2, n3);
        Rtensor = reshape(Rvector, N);

        R_unfold3 = unfold(Rtensor, 3);
        active_cols = sum(abs(R_unfold3), 1) > 0;
        if sum(active_cols) > 1
            R_obs = double(R_unfold3(:, active_cols));
            Kr3 = sparse(cov(R_obs'));
        end
    else
        Rtensor = zeros(N);
    end

    % 再次掩码投影
    X_pred = M + Rtensor;
    X = zeros(N);
    X(pos_obs) = Isubmean(pos_obs);
    X(pos_miss) = X_pred(pos_miss);

    % 恢复原始值域并计算 PSNR
    Xori = min(max(X + train_mean, 0), maxP);
    mse = mean((double(I(:)) - double(Xori(:))) .^ 2);
    psnrf(iter) = 10 * log10(maxP^2 / max(mse, eps));
    fprintf('Epoch = %d, PSNR = %.6f\n', iter, psnrf(iter));

    % 收敛判断
    tol = norm(X(:) - X_last(:)) / train_norm;
    X_last = X;
    if (tol < epsilon) || (iter >= maxiter)
        break;
    end
end

Xori = min(max(Xori, 0), maxP);
Rtensor_final = Rtensor + train_mean;
Mtensor_final = M + train_mean;
psnr_hist = psnrf(1:iter);
end
