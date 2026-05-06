%% HaLRTC 图像实验批量脚本
% 使用原作者 HaLRTC 算法进行彩色图像修复
clear; clc; close all;

%% 路径设置
base_dir = fileparts(mfilename('fullpath'));
if isempty(base_dir)
    base_dir = pwd;
end
addpath(fullfile(base_dir, 'HaLRTC'));

%% 参数设置
seed = 920;
missing_rates = [0.8, 0.9, 0.95];
rho_candidates = [1e-4, 1e-5];
maxIter = 500;
epsilon = 1e-5;
alpha = [1, 1, 1e-3];
alpha = alpha / sum(alpha);

%% 图像列表（与现有图像实验保持一致）
image_names = {'Airplane', 'House256', 'House512', 'Peppers', 'Tree', 'Sailboat', 'Female'};

%% 结果根目录
result_root = fullfile(base_dir, 'results', 'HaLRTC');
if ~exist(result_root, 'dir')
    mkdir(result_root);
end

%% 批量处理
for mi = 1:numel(missing_rates)
    missing_rate = missing_rates(mi);
    miss_tag = sprintf('miss%d', round(missing_rate * 100));

    rate_dir = fullfile(result_root, miss_tag);
    if ~exist(rate_dir, 'dir')
        mkdir(rate_dir);
    end

    % 该缺失率下的汇总表格
    summary_cols = {'图像名称', 'PSNR_dB', 'SSIM', '运行时间', 'MSE', 'RMSE', '最佳rho'};
    summary_data = cell(numel(image_names), numel(summary_cols));

    for idx = 1:numel(image_names)
        img_name = image_names{idx};
        img_path = fullfile(base_dir, 'data', [img_name, '.tiff']);

        fprintf('\n========================================\n');
        fprintf('Image: %s\n', img_name);
        fprintf('Missing rate: %.2f\n', missing_rate);
        fprintf('========================================\n');

        %% 读取图像并生成缺失掩码
        rng(seed);
        I = imread(img_path);
        I = double(I);
        [h, w, c] = size(I);
        Omega = rand(h, w, c) > missing_rate;
        T = I .* Omega;
        maxP = 255;

        fprintf('Size: %d x %d x %d\n', h, w, c);
        fprintf('Observed pixels: %d / %d\n', sum(Omega(:)), numel(Omega));

        %% 对每个 rho 运行 HaLRTC
        best_psnr = -inf;
        best = struct();

        for ri = 1:numel(rho_candidates)
            rho = rho_candidates(ri);
            status = 'ok';
            error_msg = '';

            fprintf('Running rho = %.0e ...\n', rho);

            try
                tic;
                X_init = T;
                X_init(~Omega) = mean(T(Omega));
                [Xhat, errList, history] = HaLRTC(T, Omega, alpha, rho, maxIter, epsilon, X_init, I);
                elapsed_time = toc;

                Xhat = max(0, min(255, Xhat));
                Xhat_uint8 = uint8(Xhat);

                mse_val = mean((I(:) - Xhat(:)) .^ 2);
                rmse_val = sqrt(mse_val);
                psnr_val = 10 * log10(maxP^2 / max(mse_val, eps));
                ssim_val = calc_ssim(uint8(I), Xhat_uint8);

                fprintf('  PSNR: %.6f dB\n', psnr_val);
                fprintf('  MSE: %.10f\n', mse_val);
                fprintf('  RMSE: %.10f\n', rmse_val);
                fprintf('  SSIM: %.6f\n', ssim_val);
                fprintf('  Time: %.2f s\n', elapsed_time);
                fprintf('  Iterations: %d\n', length(errList));
            catch ME
                status = 'error';
                error_msg = getReport(ME, 'extended', 'hyperlinks', 'off');
                warning('HaLRTC failed for %s rho=%.0e: %s', img_name, rho, ME.message);

                Xhat = T;
                Xhat_uint8 = uint8(T);
                elapsed_time = 0;
                mse_val = NaN;
                rmse_val = NaN;
                psnr_val = NaN;
                ssim_val = NaN;
                history = table();
                errList = [];
            end

            % 保存当前 rho 的结果
            run_result.rho = rho;
            run_result.Xhat = Xhat;
            run_result.errList = errList;
            run_result.history = history;
            run_result.mse = mse_val;
            run_result.rmse = rmse_val;
            run_result.psnr = psnr_val;
            run_result.ssim = ssim_val;
            run_result.time = elapsed_time;
            run_result.status = status;
            run_result.error = error_msg;

            % 按 PSNR 选择最佳 rho
            if strcmp(status, 'ok') && isfinite(psnr_val) && psnr_val > best_psnr
                best_psnr = psnr_val;
                best = run_result;
                best.rho = rho;
            end
        end

        % 若全部失败，使用第一个
        if isempty(fieldnames(best))
            best = run_result;
            best_psnr = NaN;
        end

        fprintf('Best rho = %.0e, PSNR = %.6f dB\n', best.rho, best_psnr);

        %% 保存恢复图像
        recovered_uint8 = uint8(max(0, min(255, best.Xhat)));
        recovered_path = fullfile(rate_dir, [img_name, '_修复后.png']);
        imwrite(recovered_uint8, recovered_path);

        %% 保存 metrics.txt
        metrics_path = fullfile(rate_dir, [img_name, '_metrics.txt']);
        fid = fopen(metrics_path, 'w');
        fprintf(fid, '图像名称=%s\n', img_name);
        fprintf(fid, '缺失率=%.2f\n', missing_rate);
        fprintf(fid, 'seed=%d\n', seed);
        fprintf(fid, '方法=HaLRTC\n');
        fprintf(fid, 'alpha=[%s]\n', num2str(alpha, '%.6f '));
        fprintf(fid, 'rho候选=[%s]\n', num2str(rho_candidates, '%.0e '));
        fprintf(fid, '最佳rho=%.0e\n', best.rho);
        fprintf(fid, 'maxIter=%d\n', maxIter);
        fprintf(fid, 'epsilon=%.0e\n', epsilon);
        fprintf(fid, 'PSNR=%.6f\n', best.psnr);
        fprintf(fid, 'SSIM=%.6f\n', best.ssim);
        fprintf(fid, 'MSE=%.10f\n', best.mse);
        fprintf(fid, 'RMSE=%.10f\n', best.rmse);
        fprintf(fid, '运行时间=%.6f\n', best.time);
        fprintf(fid, '状态=%s\n', best.status);
        if ~isempty(best.error)
            fprintf(fid, '错误=%s\n', best.error);
        end
        fclose(fid);

        %% 保存 result.mat
        recovered = best.Xhat;
        best_errList = best.errList;
        best_history = best.history;
        best_psnr_val = best.psnr;
        best_ssim_val = best.ssim;
        best_mse_val = best.mse;
        best_rmse_val = best.rmse;
        best_time_val = best.time;
        best_rho_val = best.rho;
        best_status_val = best.status;

        summary_table = table({img_name}, missing_rate, best_psnr_val, best_ssim_val, best_mse_val, best_rmse_val, best_rho_val, best_time_val, ...
            'VariableNames', {'图像名称', '缺失率', 'PSNR', 'SSIM', 'MSE', 'RMSE', '最佳rho', '运行时间'});

        result_path = fullfile(rate_dir, [img_name, '_result.mat']);
        save(result_path, '-v7.3', ...
            'img_name', 'I', 'Omega', 'T', ...
            'alpha', 'rho_candidates', 'best', 'maxIter', 'epsilon', ...
            'missing_rate', 'seed', 'summary_table', ...
            'recovered', 'best_errList', 'best_history', ...
            'best_psnr_val', 'best_ssim_val', 'best_mse_val', 'best_rmse_val', ...
            'best_time_val', 'best_rho_val', 'best_status_val');

        %% 汇入该缺失率的总结表
        summary_data(idx, :) = {img_name, best.psnr, best.ssim, best.time, best.mse, best.rmse, best.rho};

        fprintf('Saved results for %s at %s\n', img_name, miss_tag);
    end

    %% 保存该缺失率的实验总结
    summary_table = cell2table(summary_data, 'VariableNames', summary_cols);
    xlsx_path = fullfile(rate_dir, '实验总结.xlsx');
    writetable(summary_table, xlsx_path);

    fprintf('\nFinished missing rate %.2f. Summary saved to %s\n', missing_rate, xlsx_path);
end

fprintf('\n========================================\n');
fprintf('All image experiments finished.\n');
fprintf('========================================\n');

%% 辅助函数：SSIM
function ssim_val = calc_ssim(img1, img2)
    ssim_val = NaN;
    try
        if exist('ssim', 'file') == 2
            ssim_val = ssim(img1, img2);
        end
    catch
        ssim_val = NaN;
    end
end
