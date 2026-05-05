function Sk_new = tsvd_S(Gk, Uk_tilde, Vk_tilde, Sprev)
% 更新t-SVD的S因子
% Gk: 当前频率切片的观测数据
% Uk_tilde: 白化空间中的U因子
% Vk_tilde: 白化空间中的V因子
% Sprev: 上一轮的S

    [~, r] = size(Uk_tilde);
      
    % 构建右端项：b = vec(Ũ^H * G * Ṽ)
    temp_b = Uk_tilde' * Gk * Vk_tilde;
    b_vec = temp_b(:);
    
    % 初值选择
    if nargin < 4 || isempty(Sprev)
        s = zeros(r * r, 1);
    else
        s = Sprev(:);
    end
    
    % CG迭代求解
    % 算子：A(S) = vec(Ũ^H * (Ũ * S * Ṽ^H) * Ṽ)
    max_iter = max(50, 5 * r);
    
    % 计算初始残差：r = b - A*s
    S_mat = reshape(s, r, r);
    temp = Uk_tilde * S_mat * Vk_tilde';
    As_mat = Uk_tilde' * temp * Vk_tilde;
    As = As_mat(:);
    r_vec = b_vec - As;
    p = r_vec;
    rsold = r_vec' * r_vec;
    
    % 检查初始残差
    if sqrt(rsold) < 1e-10
        Sk_new = reshape(s, r, r);
        return;
    end
    
    % CG主循环
    for i = 1:max_iter
        % 计算 Ap = A*p
        S_mat = reshape(p, r, r);
        temp = Uk_tilde * S_mat * Vk_tilde';
        Ap_mat = Uk_tilde' * temp * Vk_tilde;
        Ap = Ap_mat(:);
        
        pAp = p' * Ap;
        if abs(pAp) < 1e-14
            break;
        end
        
        alpha = rsold / pAp;
        s = s + alpha * p;
        r_vec = r_vec - alpha * Ap;
        rsnew = r_vec' * r_vec;
        
        if sqrt(abs(rsnew)) < 1e-7
            break;
        end
        
        beta = rsnew / rsold;
        p = r_vec + beta * p;
        rsold = rsnew;
    end
    
    % 重塑为矩阵
    Sk_new = reshape(s, r, r);
end