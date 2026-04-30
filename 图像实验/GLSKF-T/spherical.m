function k = spherical(loghyper, x)
% Spherical 锥化函数
    range = exp(loghyper(1));
    dis = abs(x(:) - x(:)');
    r = min(dis / range, 1);

    k = 1 - 1.5*r + 0.5*r.^3;
    k(r >= 1) = 0;
    k(k < 1e-16) = 0;
    k(isnan(k)) = 0;
end
