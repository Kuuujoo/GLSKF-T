function K = matern(d, loghyper, x)
% Matérn协方差函数
% d: 平滑度参数 (1, 3, 5, 7)
% loghyper: 超参数 [log(长度尺度l), log(信号标准差 sf)]
% x: 点向量
% K: 协方差矩阵

    l = exp(loghyper(1));
    sf = exp(2 * loghyper(2));
    
    % 距离矩阵
    dist_sq = ((x(:) - x(:)') / l).^2;
    t = sqrt(d * dist_sq);
    
    % 根据d值选择函数形式
    if d == 1
        f_t = ones(size(t));
    elseif d == 3
        f_t = 1 + t;
    elseif d == 5
        f_t = 1 + t .* (1 + t / 3);
    elseif d == 7
        f_t = 1 + t .* (1 + t .* (6 + t) / 15);
    end
    
    % Matern协方差
    K = sf * f_t .* exp(-t);
end