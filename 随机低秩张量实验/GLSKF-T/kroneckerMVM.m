function result = kroneckerMVM(K3, K2, K1, vec, d1, d2, d3)
% Kronecker矩阵-向量乘法
% K3, K2, K1: 三个矩阵
% vec: 向量
% d1, d2, d3: 三个维度

    %把vec向量变成3维张量
    tensor = reshape(vec, d1, d2, d3);
    
    % K1乘法
    temp1 = reshape(tensor, d1, []);
    temp1 = K1 * temp1;
    temp1 = reshape(temp1, d1, d2, d3);
    
    % K2乘法
    temp2 = permute(temp1, [2, 1, 3]);
    temp2 = reshape(temp2, d2, []);
    temp2 = K2 * temp2;
    temp2 = reshape(temp2, d2, d1, d3);
    temp2 = permute(temp2, [2, 1, 3]);
    
    % K3乘法
    temp3 = permute(temp2, [3, 1, 2]);
    temp3 = reshape(temp3, d3, []);
    temp3 = K3 * temp3;
    temp3 = reshape(temp3, d3, d1, d2);
    temp3 = permute(temp3, [2, 3, 1]);
    
    % 转换为向量
    result = temp3(:);
end