function C = tproduct(A, B)
% t-product张量乘法
% A: n1 × r × n3 张量
% B: r × n2 × n3 张量
% C: n1 × n2 × n3 张量

    [n1, r1, n3] = size(A);
    [r2, n2, ~] = size(B);
    
    % 对第3维做FFT
    A_f = fft(A, [], 3);
    B_f = fft(B, [], 3);
    
    % 初始化结果张量
    C_f = zeros(n1, n2, n3);
    
    % 逐切片矩阵乘法
    for k = 1:n3
        C_f(:, :, k) = A_f(:, :, k) * B_f(:, :, k);
    end
    
    % 返回时域
    C = ifft(C_f, [], 3, 'symmetric');
end