function Vk_new = tsvd_V(Gk, Uk, Vk, Sk, invK2, rho, n3)
% 更新t-SVD的V因子
% Gk: 当前频率切片的观测数据
% Uk: 当前的U因子（真实空间）
% Vk: 当前的V因子（真实空间）
% Sk: 当前的S矩阵
% invK2: 第二维的逆协方差矩阵
% rho: 正则化参数
% n3: 第三维大小

    [n2, rg] = size(Vk);
    
    % 构建右端项：b = G^H * U * S
    b = Gk' * Uk * Sk;
    
    % 使用共轭梯度法求解
    vk_vec = Vk(:);
    [vk_new, ~] = cg_tsvd(Uk, Sk, invK2, rho/n3, b, vk_vec, 100, n2, rg);
    
    % 重塑为矩阵
    Vk_new = reshape(vk_new, n2, rg);
end