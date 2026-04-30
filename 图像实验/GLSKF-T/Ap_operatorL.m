function A3 = Ap_operatorL(vec, pos_obs, Kd, Kt, Ks, gamma, d1, d2, d3)
% 局部分量的算子函数
% vec: 输入向量
% pos_obs: 观测位置索引
% Kd, Kt, Ks: 三个协方差矩阵
% gamma: 正则化参数
% d1, d2, d3: 三个维度
% A3: 结果

    x = zeros(d1 * d2 * d3, 1);
    x(pos_obs) = vec;
    Ap1 = kroneckerMVM(Kd, Kt, Ks, x, d1, d2, d3);
    A3 = Ap1(pos_obs) + gamma * vec;
end