%% GLSKF-tSVD 图像批量实验脚本
clear; clc; close all;

%% 路径设置
base_dir = fileparts(mfilename('fullpath'));
if isempty(base_dir)
    base_dir = pwd;
end
addpath(fullfile(base_dir, 'GLSKF-T'));

%% 参数设置
image_names = {'Peppers', 'Tree', 'Sailboat', 'Female'};
%image_names = {'Airplane', 'House256', 'House512', 'Peppers', 'Tree', 'Sailboat', 'Female'};
missing_rates = [0.8, 0.9, 0.95];
seed = 920;

% GLSKF-tSVD 固定参数
lengthscaleU = ones(1, 2) * 30;
varianceU = ones(1, 2);
lengthscaleR = ones(1, 2) * 5;
varianceR = ones(1, 2);
tapering_range = 30;
d_MaternU = 3;
d_MaternR = 3;
epsilon = 1e-4;

% 参数网格
K0_vals = [5,7,10];
maxiter_vals = 25;
rg_vals = 10;
rho_vals = [1, 5, 10, 15, 20];
gamma_vals = [1, 5, 10, 15, 20];

n_params = numel(K0_vals) * numel(maxiter_vals) * numel(rg_vals) * numel(rho_vals) * numel(gamma_vals);

%% 结果根目录
result_root = fullfile(base_dir, 'results', 'GLSKF-T');
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

    summary_cols = {'图像名称', 'PSNR_dB', 'SSIM', 'MSE', 'RMSE', '运行时间', ...
        '最佳rg', '最佳K0', '最佳maxiter', '最佳rho', '最佳gamma'};
    summary_data = cell(numel(image_names), numel(summary_cols));

    for idx = 1:numel(image_names)
        img_name = image_names{idx};
        img_path = fullfile(base_dir, 'data', [img_name, '.tiff']);

        %% 读取图像并生成缺失掩码
        rng(seed);
        I = imread(img_path);
        [h, w, c] = size(I);
        Omega = rand(h, w, c) > missing_rate;

        fprintf('\n========================================\n');
        fprintf('Image: %s, Missing rate: %.2f\n', img_name, missing_rate);
        fprintf('Size: %d x %d x %d, Observed: %d / %d\n', h, w, c, sum(Omega(:)), numel(Omega));
        fprintf('========================================\n');

        %% 参数搜索
        best = struct('psnr', -inf, 'ssim', NaN, 'mse', NaN, 'rmse', NaN, ...
            'case_id', 0, 'rg', NaN, 'K0', NaN, 'maxiter', NaN, 'rho', NaN, ...
            'gamma', NaN, 'time', NaN, 'Xori', [], 'Rtensor', [], 'Mtensor', [], ...
            'history', table());
        case_id = 0;

        for K0_idx = 1:numel(K0_vals)
            K0_value = K0_vals(K0_idx);
            for maxiter_idx = 1:numel(maxiter_vals)
                maxiter_value = maxiter_vals(maxiter_idx);
                for rg_idx = 1:numel(rg_vals)
                    rg_value = rg_vals(rg_idx);
                    for rho_idx = 1:numel(rho_vals)
                        rho_value = rho_vals(rho_idx);
                        for gamma_idx = 1:numel(gamma_vals)
                            gamma_value = gamma_vals(gamma_idx);
                            case_id = case_id + 1;

                            param_str = sprintf('rg=%g, K0=%g, maxiter=%g, rho=%g, gamma=%g', ...
                                rg_value, K0_value, maxiter_value, rho_value, gamma_value);
                            fprintf('[%d/%d] %s | %s\n', case_id, n_params, img_name, param_str);

                            try
                                rng(seed);
                                tic;
                                [Xori, Rtensor, Mtensor, psnr_val, history] = GLSKF_tSVD( ...
                                    I, Omega, lengthscaleU, lengthscaleR, varianceU, varianceR, ...
                                    tapering_range, d_MaternU, d_MaternR, rg_value, rho_value, gamma_value, ...
                                    maxiter_value, K0_value, epsilon);
                                elapsed = toc;

                                [mse_val, rmse_val, psnr_val] = calc_metrics(I, Xori);
                                ssim_val = calc_mean_ssim(I, Xori);

                                fprintf('  PSNR: %.6f dB, SSIM: %.6f, Time: %.2f s\n', psnr_val, ssim_val, elapsed);

                                if psnr_val > best.psnr
                                    best.psnr = psnr_val;
                                    best.ssim = ssim_val;
                                    best.mse = mse_val;
                                    best.rmse = rmse_val;
                                    best.case_id = case_id;
                                    best.rg = rg_value;
                                    best.K0 = K0_value;
                                    best.maxiter = maxiter_value;
                                    best.rho = rho_value;
                                    best.gamma = gamma_value;
                                    best.time = elapsed;
                                    best.Xori = Xori;
                                    best.Rtensor = Rtensor;
                                    best.Mtensor = Mtensor;
                                    best.history = history;
                                end
                            catch ME
                                fprintf('  ERROR: %s\n', ME.message);
                            end
                        end
                    end
                end
            end
        end

        fprintf('Best: case=%d, PSNR=%.6f dB, rg=%g, K0=%g, rho=%g, gamma=%g\n', ...
            best.case_id, best.psnr, best.rg, best.K0, best.rho, best.gamma);

        %% 保存结果（HaLRTC 风格：每个图片3个文件）
        recovered_uint8 = uint8(max(0, min(255, best.Xori)));
        imwrite(recovered_uint8, fullfile(rate_dir, [img_name, '_修复后.png']));

        % metrics.txt
        fid = fopen(fullfile(rate_dir, [img_name, '_metrics.txt']), 'w');
        fprintf(fid, '图像名称=%s\n', img_name);
        fprintf(fid, '缺失率=%.2f\n', missing_rate);
        fprintf(fid, 'seed=%d\n', seed);
        fprintf(fid, '方法=GLSKF-tSVD\n');
        fprintf(fid, 'PSNR=%.6f\n', best.psnr);
        fprintf(fid, 'SSIM=%.6f\n', best.ssim);
        fprintf(fid, 'MSE=%.10f\n', best.mse);
        fprintf(fid, 'RMSE=%.10f\n', best.rmse);
        fprintf(fid, '最佳rg=%g\n', best.rg);
        fprintf(fid, '最佳K0=%g\n', best.K0);
        fprintf(fid, '最佳maxiter=%g\n', best.maxiter);
        fprintf(fid, '最佳rho=%g\n', best.rho);
        fprintf(fid, '最佳gamma=%g\n', best.gamma);
        fprintf(fid, '运行时间=%.6f\n', best.time);
        fclose(fid);

        % result.mat
        recovered = best.Xori;
        best_psnr_val = best.psnr;
        best_ssim_val = best.ssim;
        best_mse_val = best.mse;
        best_rmse_val = best.rmse;
        best_history = best.history;
        save(fullfile(rate_dir, [img_name, '_result.mat']), '-v7.3', ...
            'img_name', 'I', 'Omega', 'missing_rate', 'seed', ...
            'recovered', 'best', 'best_psnr_val', 'best_ssim_val', ...
            'best_mse_val', 'best_rmse_val', 'best_history');

        % 汇总
        summary_data(idx, :) = {img_name, best.psnr, best.ssim, best.mse, ...
            best.rmse, best.time, best.rg, best.K0, best.maxiter, best.rho, best.gamma};
    end

    %% 保存该缺失率的实验总结
    summary_table = cell2table(summary_data, 'VariableNames', summary_cols);
    writetable(summary_table, fullfile(rate_dir, '实验总结.xlsx'));
    fprintf('\nFinished miss%d. Summary saved to %s\n', round(missing_rate * 100), rate_dir);
end

fprintf('\n========================================\n');
fprintf('All GLSKF-tSVD image experiments finished.\n');
fprintf('========================================\n');

%% 辅助函数
function [mse_value, rmse_value, psnr_value] = calc_metrics(I, X)
    diff = double(I) - double(X);
    mse_value = mean(diff(:) .^ 2);
    rmse_value = sqrt(mse_value);
    psnr_value = 10 * log10(255^2 / max(mse_value, eps));
end

function ssim_value = calc_mean_ssim(I, X)
    ssim_value = NaN;
    try
        channel_count = size(I, 3);
        vals = zeros(channel_count, 1);
        for ch = 1:channel_count
            vals(ch) = ssim(uint8(X(:, :, ch)), I(:, :, ch));
        end
        ssim_value = mean(vals);
    catch
        ssim_value = NaN;
    end
end
