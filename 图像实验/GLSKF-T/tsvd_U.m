function Uk_new = tsvd_U(Gk, Uk, Vk, Sk, invK1, rho, n3)
% 更新t-SVD的U因子
% Gk: 当前频率切片的观测数据
% Uk: 当前的U因子（真实空间）
% Vk: 当前的V因子（真实空间）
% Sk: 当前的S矩阵
% invK1: 第一维的逆协方差矩阵
% rho: 正则化参数
% n3: 第三维大小

    [n1, rg] = size(Uk);
    
    % 构建右端项：b = G * V * S^H
    b = Gk * Vk * (Sk');
    
    % 使用共轭梯度法求解
    uk_vec = Uk(:);
    [uk_new, ~] = cg_tsvd(Vk, Sk, invK1, rho/n3, b, uk_vec, 100, n1, rg);
    
    % 重塑为矩阵
    Uk_new = reshape(uk_new, n1, rg);
end