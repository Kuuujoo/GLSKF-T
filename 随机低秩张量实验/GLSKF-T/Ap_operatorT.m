function Ax = Ap_operatorT(x, A_factor, B_factor, invK, rho, n, r)
% t-SVD因子更新的算子函数
% x: 输入向量
% A_factor: 另一个因子矩阵（V或U）
% B_factor: S矩阵
% invK: 逆协方差矩阵
% rho: 正则化参数
% n, r: 维度参数

    % 重塑为矩阵
    X = reshape(x, n, r);
    
    % 数据项：X * B_factor * A_factor^H 然后再 * A_factor * conj(B_factor^H)
    temp = X * B_factor * A_factor';
    Ax1 = temp * A_factor * B_factor';
    
    % 正则项：rho * invK * X
    Ax2 = rho * (invK * X);
    
    % 合并并向量化
    Ax = (Ax1 + Ax2);
    Ax = Ax(:);
end