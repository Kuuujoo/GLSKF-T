function At = ttranspose(A)
% t-transpose 张量转置
% A: n1 × n2 × n3 张量
% At: n2 × n1 × n3 张量（转置后）

    [n1, n2, n3] = size(A);
    At = zeros(n2, n1, n3);
    
    % 第一个切片直接转置
    At(:, :, 1) = A(:, :, 1)';
    
    % 其余切片转置并循环
    for k = 2:n3
        At(:, :, k) = A(:, :, n3-k+2)';
    end
end