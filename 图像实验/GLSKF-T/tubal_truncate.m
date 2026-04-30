function S_truncated = tubal_truncate(S, r)
% 对奇异管张量S进行tubal-rank截断
% S: 奇异管张量
% r: 目标tubal-rank
% S_truncated: 截断后的奇异管张量

    [r_curr, ~, n3] = size(S);
    r_new = min(r, r_curr);
    
    % 对第3维做FFT
    S_f = fft(S, [], 3);
    
    % 逐切片截断
    for k = 1:n3
        Sk = S_f(:, :, k);
        % 保留前r_new个奇异值
        if r_new < r_curr
            Sk(r_new+1:end, :) = 0;
            Sk(:, r_new+1:end) = 0;
        end
        S_f(:, :, k) = Sk;
    end
    
    % 返回时域
    S_truncated = ifft(S_f, [], 3, 'symmetric');
end