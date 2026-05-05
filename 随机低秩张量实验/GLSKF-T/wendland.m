function k = wendland(loghyper, x)
% Wendland C2 锥化函数
% loghyper: [log(range)]
% x: 输入点向量

    range = exp(loghyper(1));
    dis = abs(x(:) - x(:)');
    r = min(dis / range, 1);            % 归一化距离，截断到 [0,1]

    k = (1 - r).^4 .* (1 + 4*r);        % Wendland C2
    k(r >= 1) = 0;
    k(k < 1e-16) = 0;
    k(isnan(k)) = 0;
end
