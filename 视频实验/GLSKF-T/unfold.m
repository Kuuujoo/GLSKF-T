function mat = unfold(tensor, mode)
% 张量展开函数
% tensor: 输入张量
% mode: 展开模式
% mat: 展开后的矩阵
    
    % 获取张量维度
    sz = size(tensor);
    n = length(sz);
    
    % 重新排列维度，将mode维度放在第一位
    perm = [mode, 1:mode-1, mode+1:n];
    tensor_p = permute(tensor, perm);
    
    % 展开为矩阵
    mat = reshape(tensor_p, sz(mode), []);
end