function [x, approxE] = cg_tsvd(A_factor, B_factor, invK, rho, b, x0, max_iter, n, r)
% t-SVD因子更新的共轭梯度求解器
% A_factor: 另一个因子矩阵（V或U）
% B_factor: S矩阵
% invK: 逆协方差矩阵
% rho: 正则化参数（已除以n3）
% b: 右端项
% x0: 初始值（真实空间）
% max_iter: 最大迭代次数
% n, r: 维度参数

    b_vec = b(:);
    x = x0;
    
    % 计算初始残差
    Ax = Ap_operatorT(x, A_factor, B_factor, invK, rho, n, r);
    r_vec = b_vec - Ax;
    p = r_vec;
    rsold = r_vec' * r_vec;
    approxE = zeros(max_iter, 1);
    
    % 检查初始残差
    if sqrt(rsold) < 1e-10
        approxE(1) = sqrt(rsold);
        return;
    end
    
    for i = 1:max_iter
        Ap = Ap_operatorT(p, A_factor, B_factor, invK, rho, n, r);
        pAp = p' * Ap;
        
        % 数值保护
        if abs(pAp) < 1e-14
            break;
        end
        
        alpha = rsold / pAp;
        x = x + alpha * p;
        r_vec = r_vec - alpha * Ap;
        rsnew = r_vec' * r_vec;
        approxE(i) = sqrt(abs(rsnew));
        
        if approxE(i) < 1e-6
            break;
        end
        
        p = r_vec + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end