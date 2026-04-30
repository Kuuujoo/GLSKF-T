%% GLSKF-tSVD 批量图像修复脚本
clear; clc;

%% 参数设置区域 
addpath GLSKF-T
% 图像列表
image_names = {'Airplane', 'House256', 'House512', 'Peppers', 'Tree', 'Sailboat', 'Female'};

% 缺失率设置
missing_rate = 0.95;  % 90% 缺失

% 设置随机种子
rng(920);

% GLSKF-tSVD 固定参数
lengthscaleU = ones(1, 2) * 30;  % 全局Matérn长度尺度
varianceU = ones(1, 2);          % 全局方差
lengthscaleR = ones(1, 2) * 5;   % 局部Matérn长度尺度
varianceR = ones(1, 2);          % 局部方差
tapering_range = 30;             % 锥化范围
d_MaternU = 3;                   % 全局Matérn平滑度
d_MaternR = 3;                   % 局部Matérn平滑度
rg = 10;                         % 全局tubal-rank
maxiter = 30;                    % 最大迭代次数
K0 = 25;                         % 开始更新局部分量的迭代
epsilon = 1e-4;                  % 收敛阈值

% 参数网格 - 需要搜索的参数
rho_list = [1, 5, 10, 15, 20];
gamma_list = [1, 5, 10, 15, 20];

%% 创建结果保存目录
result_dir = sprintf('results/GLSKF-T/GLSKF-T%.0f%%', missing_rate*100);
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

%% 批量处理图像
results = struct();
summary_data = cell(length(image_names), 5);

for idx = 1:length(image_names)
    img_name = image_names{idx};
    fprintf('\n========================================\n');
    fprintf('处理图像: %s\n', img_name);
    fprintf('========================================\n\n');
    
    % 读取原始图像
    img_path = sprintf('./data/%s.tiff', img_name);
    I = imread(img_path);
    [h, w, c] = size(I);
    
    fprintf('图像尺寸: %d x %d x %d\n', h, w, c);
    fprintf('缺失率: %.1f%%\n\n', missing_rate*100);
    
    % 生成缺失掩码
    Omega = rand(h, w, c) > missing_rate;
    
    observed_pixels = sum(Omega(:));
    total_pixels = numel(I);
    fprintf('观测像素: %d / %d\n\n', observed_pixels, total_pixels);
    
    % 参数搜索
    best_psnr = 0;
    best_params = struct();
    best_result = struct();
    
    fprintf('开始参数网格搜索...\n');
    fprintf('rho候选: [%s]\n', sprintf('%d ', rho_list));
    fprintf('gamma候选: [%s]\n', sprintf('%d ', gamma_list));
    fprintf('总共需要测试: %d 组参数\n\n', length(rho_list) * length(gamma_list));
    
    param_count = 0;
    total_params = length(rho_list) * length(gamma_list);
    
    % 遍历所有参数组合
    for rho_idx = 1:length(rho_list)
        for gamma_idx = 1:length(gamma_list)
            rho = rho_list(rho_idx);
            gamma = gamma_list(gamma_idx);
            param_count = param_count + 1;
            
            fprintf('\n[%d/%d] 测试参数: rho=%d, gamma=%d\n', param_count, total_params, rho, gamma);
            fprintf('--------------------------------------------------\n');
            
            % 调用GLSKF_tSVD算法
            tic;
            [Xori, Rtensor, Mtensor, ~] = GLSKF_tSVD(I, Omega, lengthscaleU, lengthscaleR, ...
                                                     varianceU, varianceR, tapering_range, ...
                                                     d_MaternU, d_MaternR, rg, rho, gamma, ...
                                                     maxiter, K0, epsilon);
            elapsed_time = toc;
            
            % 计算 PSNR - 分通道计算后取平均
            psnrC1 = 10 * log10(255^2 / (norm(double(I(:,:,1)) - Xori(:,:,1), 'fro')^2 / (h*w)));
            psnrC2 = 10 * log10(255^2 / (norm(double(I(:,:,2)) - Xori(:,:,2), 'fro')^2 / (h*w)));
            psnrC3 = 10 * log10(255^2 / (norm(double(I(:,:,3)) - Xori(:,:,3), 'fro')^2 / (h*w)));
            psnr_value = (psnrC1 + psnrC2 + psnrC3) / 3;
            
            % 计算 SSIM - 分通道计算后取平均
            ssimC1 = ssim(uint8(Xori(:,:,1)), I(:,:,1));
            ssimC2 = ssim(uint8(Xori(:,:,2)), I(:,:,2));
            ssimC3 = ssim(uint8(Xori(:,:,3)), I(:,:,3));
            ssim_value = (ssimC1 + ssimC2 + ssimC3) / 3;
            
            fprintf('PSNR: %.4f dB (C1:%.2f, C2:%.2f, C3:%.2f)\n', psnr_value, psnrC1, psnrC2, psnrC3);
            fprintf('SSIM: %.4f (C1:%.4f, C2:%.4f, C3:%.4f)\n', ssim_value, ssimC1, ssimC2, ssimC3);
            fprintf('运行时间: %.2f 秒\n', elapsed_time);
            
            % 更新最佳结果
            if psnr_value > best_psnr
                best_psnr = psnr_value;
                best_params.rho = rho;
                best_params.gamma = gamma;
                best_result.Xori = Xori;
                best_result.Rtensor = Rtensor;
                best_result.Mtensor = Mtensor;
                best_result.psnr = psnr_value;
                best_result.ssim = ssim_value;
                best_result.time = elapsed_time;
                fprintf('*** 发现更好的参数! PSNR提升至 %.4f dB ***\n', best_psnr);
            end
        end
    end
    
    % 保存最佳结果
    fprintf('\n========================================\n');
    fprintf('最佳参数: rho=%d, gamma=%d\n', best_params.rho, best_params.gamma);
    fprintf('最佳PSNR: %.4f dB\n', best_result.psnr);
    fprintf('最佳SSIM: %.4f\n', best_result.ssim);
    fprintf('========================================\n');
    
    results.(img_name).original = I;
    results.(img_name).observed = uint8(double(I) .* double(Omega));
    results.(img_name).recovered = uint8(best_result.Xori);
    results.(img_name).Mtensor = uint8(best_result.Mtensor);
    results.(img_name).Rtensor = uint8(abs(best_result.Rtensor));
    results.(img_name).psnr = best_result.psnr;
    results.(img_name).ssim = best_result.ssim;
    results.(img_name).best_rho = best_params.rho;
    results.(img_name).best_gamma = best_params.gamma;
    results.(img_name).time = best_result.time;
    
    % 保存修复图像
    imwrite(uint8(best_result.Xori), sprintf('%s/%s_recovered.png', result_dir, img_name));
    
    % 保存到汇总表格
    summary_data{idx, 1} = img_name;
    summary_data{idx, 2} = best_result.psnr;
    summary_data{idx, 3} = best_result.ssim;
    summary_data{idx, 4} = best_params.rho;
    summary_data{idx, 5} = best_params.gamma;
end

fprintf('\n所有图像处理完成！\n');

%% 可视化和保存结果 
fields = fieldnames(results);

for idx = 1:length(fields)
    img_name = fields{idx};
    data = results.(img_name);
    
    % 对比图
    figure('Position', [100, 100, 1500, 500]);
    
    subplot(1, 3, 1);
    imshow(data.original);
    title('原始图像', 'FontSize', 14, 'FontName', 'SimHei');
    axis off;
    
    subplot(1, 3, 2);
    imshow(data.observed);
    title(sprintf('观测数据 (%.0f%% 缺失)', missing_rate*100), 'FontSize', 14, 'FontName', 'SimHei');
    axis off;
    
    subplot(1, 3, 3);
    imshow(data.recovered);
    title(sprintf('GLSKF-T 修复\nPSNR=%.2fdB, SSIM=%.4f\nrho=%d, gamma=%d', ...
                  data.psnr, data.ssim, data.best_rho, data.best_gamma), ...
          'FontSize', 14, 'FontName', 'SimHei');
    axis off;
    
    % 保存对比图
    saveas(gcf, sprintf('%s/%s_对比.png', result_dir, img_name));
    
    % 分量分解图
    figure('Position', [100, 100, 1200, 400]);
    
    subplot(1, 3, 1);
    imshow(data.recovered);
    title('恢复图像', 'FontSize', 12, 'FontName', 'SimHei');
    axis off;
    
    subplot(1, 3, 2);
    imshow(data.Mtensor);
    title('全局分量 M (t-SVD)', 'FontSize', 12, 'FontName', 'SimHei');
    axis off;
    
    subplot(1, 3, 3);
    imshow(data.Rtensor);
    title('局部分量 R (t-SVD)', 'FontSize', 12, 'FontName', 'SimHei');
    axis off;
    
    % 保存分量图
    saveas(gcf, sprintf('%s/%s_分量分解.png', result_dir, img_name));
    
    close all;
end

%% 保存结果到 Excel
% 创建表格
summary_table = cell2table(summary_data, ...
    'VariableNames', {'图像名称', 'PSNR_dB', 'SSIM', '最佳rho', '最佳gamma'});

% 保存为 Excel
excel_path = sprintf('%s/实验总结.xlsx', result_dir);
writetable(summary_table, excel_path);

fprintf('\n结果已保存\n');
fprintf('最佳参数已记录在Excel文件中\n');