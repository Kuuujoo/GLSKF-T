function k = bohman(loghyper, x)
% Bohman锥化函数
% loghyper: 超参数 [log(range)]
% x: 输入点向量

    range = exp(loghyper(1));
    dis = abs(x(:) - x(:)');
    r = min(dis / range, 1); %实现截断
    % k: 锥化协方差矩阵
    k = (1 - r) .* cos(pi * r) + sin(pi * r) / pi;
    k(k < 1e-16) = 0;
    k(isnan(k)) = 0;
end