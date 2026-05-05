function [z, approxE] = cg_local(gamma, Kd, Kt, Ks, pos_obs, l, z0, max_iter)
% 局部分量的共轭梯度求解
% gamma: 正则化参数
% Kd, Kt, Ks: 三个协方差矩阵
% pos_obs: 观测位置索引
% l: 观测数据张量
% z0: 初始值
% max_iter: 最大迭代次数

% z: 求解结果
% approxE: 收敛误差

    [d1, d2, d3] = size(l);
    Y_flat = l(:);
    Y_obs = Y_flat(pos_obs);
    z = z0;
    
    Ax = Ap_operatorL(z, pos_obs, Kd, Kt, Ks, gamma, d1, d2, d3);
    r = Y_obs - Ax;
    p = r;
    rsold = r' * r;
    approxE = zeros(max_iter, 1);
    
    for i = 1:max_iter
        Ap = Ap_operatorL(p, pos_obs, Kd, Kt, Ks, gamma, d1, d2, d3);
        alpha = rsold / (p' * Ap);
        z = z + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        approxE(i) = sqrt(rsnew);
        
        if approxE(i) < 1e-4
            break;
        end
        
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end