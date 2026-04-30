%% SPC 批量图像修复脚本
clear; clc;

%% 参数设置区域
addpath SPC
% 图像列表
image_names = {'Airplane', 'House256', 'House512', 'Peppers', 'Tree', 'Sailboat', 'Female'};

% 缺失率设置
missing_rates = [0.8, 0.9, 0.95];

% 设置随机种子
rng(920);

% SPC 算法参数
TVQV_list = {'tv', 'qv'};  % 测试两种模式
rho_tv = [0.5 0.5 0];      % TV 平滑参数
rho_qv = [0.5 0.5 0];      % QV 平滑参数
K       = 10;              % 每次迭代更新的成分数量
SNR     = 25;              % 误差界
nu      = 0.01;            % R 更新阈值
maxiter = 1000;            % 最大迭代次数
tol     = 1e-5;            % 收敛容差
out_im  = 0;               % 是否输出过程图像 (1:是, 0:否)

%% 按缺失率批量处理图像
for rate_idx = 1:length(missing_rates)
    missing_rate = missing_rates(rate_idx);
    result_dir = sprintf('results/SPC/SPC%.0f%%', missing_rate*100);
    if ~exist(result_dir, 'dir')
        mkdir(result_dir);
    end

    results = struct();
    summary_data = cell(length(image_names), 5);

    fprintf('\n##################################################\n');
    fprintf('开始处理缺失率 %.1f%%\n', missing_rate*100);
    fprintf('##################################################\n');

    for idx = 1:length(image_names)
        img_name = image_names{idx};
        fprintf('\n========================================\n');
        fprintf('处理图像: %s\n', img_name);
        fprintf('========================================\n\n');
        
        % 读取原始图像
        img_path = sprintf('./data/%s.tiff', img_name);
        X0 = imread(img_path);
        [h, w, c] = size(X0);
        
        fprintf('图像尺寸: %d x %d x %d\n', h, w, c);
        fprintf('缺失率: %.1f%%\n\n', missing_rate*100);
        
        % 生成缺失掩码
        mask = rand(h, w) > missing_rate;
        
        % 创建观测数据
        T = zeros(h, w, c);
        Q = zeros(h, w, c, 'logical');
        for k = 1:c
            T(:,:,k) = double(X0(:,:,k)) .* mask;
            Q(:,:,k) = mask;
        end
        
        observed_pixels = sum(Q(:));
        total_pixels = numel(X0);
        fprintf('观测像素: %d / %d\n\n', observed_pixels, total_pixels);
        
        % 测试两种模式
        best_psnr = 0;
        best_mode = '';
        best_result = struct();
        
        for mode_idx = 1:length(TVQV_list)
            TVQV = TVQV_list{mode_idx};
            
            % 选择对应的 rho 参数
            if strcmp(TVQV, 'tv')
                rho = rho_tv;
            else
                rho = rho_qv;
            end
            
            fprintf('测试模式: %s\n', upper(TVQV));
            fprintf('------------------------------------------\n');
            
            % 运行 SPC 算法
            tic;
            [X, Z, G, U, histo, histo_R] = SPC(T, Q, TVQV, rho, K, SNR, nu, maxiter, tol, out_im);
            elapsed_time = toc;
            
            % 计算 PSNR
            psnrC1 = 10 * log10(255^2 / (norm(double(X0(:,:,1)) - X(:,:,1), 'fro')^2 / (h*w)));
            psnrC2 = 10 * log10(255^2 / (norm(double(X0(:,:,2)) - X(:,:,2), 'fro')^2 / (h*w)));
            psnrC3 = 10 * log10(255^2 / (norm(double(X0(:,:,3)) - X(:,:,3), 'fro')^2 / (h*w)));
            psnr_value = (psnrC1 + psnrC2 + psnrC3) / 3;
            
            % 计算 SSIM
            ssimC1 = ssim(uint8(X(:,:,1)), X0(:,:,1));
            ssimC2 = ssim(uint8(X(:,:,2)), X0(:,:,2));
            ssimC3 = ssim(uint8(X(:,:,3)), X0(:,:,3));
            ssim_value = (ssimC1 + ssimC2 + ssimC3) / 3;
            
            fprintf('PSNR: %.4f dB (C1:%.2f, C2:%.2f, C3:%.2f)\n', psnr_value, psnrC1, psnrC2, psnrC3);
            fprintf('SSIM: %.4f (C1:%.4f, C2:%.4f, C3:%.4f)\n', ssim_value, ssimC1, ssimC2, ssimC3);
            fprintf('运行时间: %.2f 秒\n\n', elapsed_time);
            
            % 更新最佳结果
            if psnr_value > best_psnr
                best_psnr = psnr_value;
                best_mode = TVQV;
                best_result.X = X;
                best_result.Z = Z;
                best_result.G = G;
                best_result.U = U;
                best_result.histo = histo;
                best_result.histo_R = histo_R;
                best_result.psnr = psnr_value;
                best_result.ssim = ssim_value;
                best_result.time = elapsed_time;
                fprintf('*** 当前最佳模式: %s ***\n\n', upper(TVQV));
            end
        end
        
        % 保存最佳结果
        fprintf('========================================\n');
        fprintf('最佳模式: %s\n', upper(best_mode));
        fprintf('最佳PSNR: %.4f dB\n', best_result.psnr);
        fprintf('最佳SSIM: %.4f\n', best_result.ssim);
        fprintf('最佳时间: %.2f 秒\n', best_result.time);
        fprintf('========================================\n');
        
        results.(img_name).original = X0;
        results.(img_name).observed = uint8(T);
        results.(img_name).recovered = uint8(best_result.X);
        results.(img_name).psnr = best_result.psnr;
        results.(img_name).ssim = best_result.ssim;
        results.(img_name).histo = best_result.histo;
        results.(img_name).histo_R = best_result.histo_R;
        results.(img_name).G = best_result.G;
        results.(img_name).best_mode = best_mode;
        results.(img_name).time = best_result.time;
        
        % 保存修复图像
        imwrite(uint8(best_result.X), sprintf('%s/%s_修复后.png', result_dir, img_name));
        
        % 保存到汇总表格
        summary_data{idx, 1} = img_name;
        summary_data{idx, 2} = best_result.psnr;
        summary_data{idx, 3} = best_result.ssim;
        summary_data{idx, 4} = upper(best_mode);
        summary_data{idx, 5} = best_result.time;
    end

    fprintf('\n当前缺失率下所有图像处理完成！\n');

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
        title(sprintf('SPC 修复 (%s)\nPSNR=%.2fdB, SSIM=%.4f\n时间=%.2fs', ...
            upper(data.best_mode), data.psnr, data.ssim, data.time), ...
            'FontSize', 14, 'FontName', 'SimHei');
        axis off;
        
        % 保存对比图
        saveas(gcf, sprintf('%s/%s_对比.png', result_dir, img_name));
        
        % 收敛曲线
        figure('Position', [100, 100, 1000, 600]);
        plot(1:length(data.histo), data.histo, 'b-', 'LineWidth', 2);
        xlabel('迭代次数', 'FontSize', 12, 'FontName', 'SimHei');
        ylabel('PSNR (dB)', 'FontSize', 12, 'FontName', 'SimHei');
        title(sprintf('%s - SPC 收敛曲线 (%s)', img_name, upper(data.best_mode)), ...
              'FontSize', 14, 'FontName', 'SimHei');
        grid on;
            
        % 保存收敛曲线
        saveas(gcf, sprintf('%s/%s_收敛曲线.png', result_dir, img_name));
       
        close all;
    end

    %% 保存结果到 Excel
    summary_table = cell2table(summary_data, ...
        'VariableNames', {'图像名称', 'PSNR_dB', 'SSIM', '最佳模式', '运行时间_s'});

    excel_path = sprintf('%s/实验总结.xlsx', result_dir);
    writetable(summary_table, excel_path);

    fprintf('\n缺失率 %.1f%% 的结果已保存到 %s\n', missing_rate*100, excel_path);
end

fprintf('\n所有缺失率处理完成！\n');
