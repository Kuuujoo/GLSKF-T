%% GLSKF-tSVD 图像批量实验脚本
clear; clc; close all;

%% 参数设置
base_dir = fileparts(mfilename('fullpath'));
if isempty(base_dir)
    base_dir = pwd;
end
addpath(fullfile(base_dir, 'GLSKF-T'));

% 图像列表
image_names = {'Airplane', 'House256', 'House512', 'Peppers', 'Tree', 'Sailboat', 'Female'};

% 缺失率设置
missing_rates = [0.8,0.9,0.95];

% 设置随机种子
seed = 920;

% GLSKF-tSVD 固定参数
lengthscaleU = ones(1, 2) * 30;
varianceU = ones(1, 2);
lengthscaleR = ones(1, 2) * 5;
varianceR = ones(1, 2);
tapering_range =30;
d_MaternU = 3;
d_MaternR = 3;
epsilon = 1e-4;

% 参数网格
K0 = [10];
maxiter = 25;
rg = 10;
rho = [1,5,10,15,20];
gamma =[1,5,10,15,20];

n_params = numel(K0) * numel(maxiter) * numel(rg) * numel(rho) * numel(gamma);
total_cases = numel(missing_rates) * numel(image_names) * n_params;
case_id = 0;

%% 全局最优结果
best_all = struct('psnr', -inf, 'ssim', NaN, 'mse', NaN, 'rmse', NaN, ...
    'image', '', 'missing_rate', NaN, 'case_id', 0, 'rg', NaN, 'K0', NaN, ...
    'maxiter', NaN, 'rho', NaN, 'gamma', NaN, 'time', NaN);

%% 批量处理
for miss_idx = 1:numel(missing_rates)
    missing_rate = missing_rates(miss_idx);

    for idx = 1:numel(image_names)
        img_name = image_names{idx};

        img_path = fullfile(base_dir, 'data', [img_name, '.tiff']);
        result_dir = fullfile(base_dir, 'results', 'GLSKF-T', ...
            [img_name, '_miss', strrep(num2str(missing_rate * 100), '.', ''), ...
            '_K0_5_maxiter25_rg7_rho_gamma_grid']);
        history_dir = fullfile(result_dir, 'iteration_logs');

        if ~exist(result_dir, 'dir')
            mkdir(result_dir);
        end
        if ~exist(history_dir, 'dir')
            mkdir(history_dir);
        end

        %% 读取图像并生成缺失掩码
        rng(seed);
        I = imread(img_path);
        [h, w, c] = size(I);
        Omega = rand(h, w, c) > missing_rate;
        observed = uint8(double(I) .* double(Omega));

        imwrite(I, fullfile(result_dir, [img_name, '_original.png']));
        imwrite(observed, fullfile(result_dir, [img_name, '_observed.png']));
        save(fullfile(result_dir, 'mask.mat'), 'Omega', 'seed', 'missing_rate', '-v7.3');

        diary(fullfile(result_dir, 'console_output.txt'));
        diary_cleanup = onCleanup(@() diary('off')); %#ok<NASGU>

        fprintf('\n========================================\n');
        fprintf('Image: %s\n', img_name);
        fprintf('Size: %d x %d x %d\n', h, w, c);
        fprintf('Missing rate: %.2f\n', missing_rate);
        fprintf('Seed: %d\n', seed);
        fprintf('Observed pixels: %d / %d\n', sum(Omega(:)), numel(Omega));
        fprintf('========================================\n\n');

        summary_cols = {'case_id', 'dataset', 'method', 'variant', 'missing_rate', ...
            'seed', 'rg', 'K0', 'maxiter', 'rho', 'gamma', 'final_PSNR', ...
            'final_MSE', 'final_RMSE', 'final_SSIM', 'best_iter_PSNR', ...
            'best_iter', 'best_global_PSNR', 'elapsed_time_seconds', ...
            'convergence_status', 'parameter_settings', 'history_file', 'error_message'};
        summary_data = cell(0, numel(summary_cols));
        summary_path = fullfile(result_dir, 'parameter_summary.csv');

        best = struct('psnr', -inf, 'ssim', NaN, 'mse', NaN, 'rmse', NaN, ...
            'case_id', 0, 'rg', NaN, 'K0', NaN, 'maxiter', NaN, 'rho', NaN, ...
            'gamma', NaN, 'time', NaN, 'Xori', [], 'Rtensor', [], 'Mtensor', [], ...
            'history', table());

        fprintf('Start parameter search...\n');
        fprintf('Total cases for this image: %d\n\n', n_params);

        for K0_idx = 1:numel(K0)
            K0_value = K0(K0_idx);
            for maxiter_idx = 1:numel(maxiter)
                maxiter_value = maxiter(maxiter_idx);
                for rg_idx = 1:numel(rg)
                    rg_value = rg(rg_idx);
                    for rho_idx = 1:numel(rho)
                        rho_value = rho(rho_idx);
                        for gamma_idx = 1:numel(gamma)
                            gamma_value = gamma(gamma_idx);
                            case_id = case_id + 1;

                            history_file = fullfile(history_dir, sprintf('case_%04d_history.csv', case_id));
                            parameter_settings = sprintf('rg=%g,K0=%g,maxiter=%g,rho=%g,gamma=%g', ...
                                rg_value, K0_value, maxiter_value, rho_value, gamma_value);

                            fprintf('\n[%d/%d] %s | missing %.2f | %s\n', ...
                                case_id, total_cases, img_name, missing_rate, parameter_settings);
                            fprintf('--------------------------------------------------\n');

                            elapsed_time = NaN;
                            psnr_value = NaN;
                            mse_value = NaN;
                            rmse_value = NaN;
                            ssim_value = NaN;
                            best_iter_psnr = NaN;
                            best_iter = NaN;
                            best_global_psnr = NaN;
                            convergence_status = "failed";
                            error_message = "";

                            try
                                rng(seed);
                                tic;
                                [Xori, Rtensor, Mtensor, psnr_value, history] = GLSKF_tSVD( ...
                                    I, Omega, lengthscaleU, lengthscaleR, varianceU, varianceR, ...
                                    tapering_range, d_MaternU, d_MaternR, rg_value, rho_value, gamma_value, ...
                                    maxiter_value, K0_value, epsilon);
                                elapsed_time = toc;

                                [mse_value, rmse_value, psnr_value] = calc_metrics(I, Xori);
                                ssim_value = calc_mean_ssim(I, Xori);
                                [best_iter_psnr, best_iter] = max(history.PSNR);
                                best_global_psnr = max(history.global_PSNR);
                                convergence_status = "completed";

                                history.dataset = repmat({img_name}, height(history), 1);
                                history.method = repmat({'GLSKF-tSVD'}, height(history), 1);
                                history.variant = repmat({'observed_global_update'}, height(history), 1);
                                history.missing_rate = repmat(missing_rate, height(history), 1);
                                history.seed = repmat(seed, height(history), 1);
                                history.rg = repmat(rg_value, height(history), 1);
                                history.K0 = repmat(K0_value, height(history), 1);
                                history.maxiter = repmat(maxiter_value, height(history), 1);
                                history.rho = repmat(rho_value, height(history), 1);
                                history.gamma = repmat(gamma_value, height(history), 1);
                                history.parameter_settings = repmat({parameter_settings}, height(history), 1);
                                writetable(history, history_file);

                                fprintf('PSNR: %.6f dB\n', psnr_value);
                                fprintf('MSE: %.6f\n', mse_value);
                                fprintf('RMSE: %.6f\n', rmse_value);
                                fprintf('SSIM: %.6f\n', ssim_value);
                                fprintf('Best iteration PSNR: %.6f dB at iter %d\n', best_iter_psnr, best_iter);
                                fprintf('Best global PSNR: %.6f dB\n', best_global_psnr);
                                fprintf('Elapsed time: %.2f seconds\n', elapsed_time);

                                if psnr_value > best.psnr
                                    best.psnr = psnr_value;
                                    best.ssim = ssim_value;
                                    best.mse = mse_value;
                                    best.rmse = rmse_value;
                                    best.case_id = case_id;
                                    best.rg = rg_value;
                                    best.K0 = K0_value;
                                    best.maxiter = maxiter_value;
                                    best.rho = rho_value;
                                    best.gamma = gamma_value;
                                    best.time = elapsed_time;
                                    best.Xori = Xori;
                                    best.Rtensor = Rtensor;
                                    best.Mtensor = Mtensor;
                                    best.history = history;

                                    save_best_outputs(result_dir, img_name, I, observed, best);
                                    fprintf('*** New best PSNR for %s: %.6f dB ***\n', img_name, best.psnr);
                                end

                                if psnr_value > best_all.psnr
                                    best_all.psnr = psnr_value;
                                    best_all.ssim = ssim_value;
                                    best_all.mse = mse_value;
                                    best_all.rmse = rmse_value;
                                    best_all.image = img_name;
                                    best_all.missing_rate = missing_rate;
                                    best_all.case_id = case_id;
                                    best_all.rg = rg_value;
                                    best_all.K0 = K0_value;
                                    best_all.maxiter = maxiter_value;
                                    best_all.rho = rho_value;
                                    best_all.gamma = gamma_value;
                                    best_all.time = elapsed_time;
                                end
                            catch ME
                                error_message = string(ME.message);
                                fprintf('Run failed: %s\n', ME.message);
                            end

                            row = {case_id, img_name, 'GLSKF-tSVD', ...
                                'observed_global_update', missing_rate, seed, rg_value, K0_value, ...
                                maxiter_value, rho_value, gamma_value, psnr_value, mse_value, rmse_value, ...
                                ssim_value, best_iter_psnr, best_iter, best_global_psnr, ...
                                elapsed_time, char(convergence_status), parameter_settings, ...
                                history_file, char(error_message)};
                            summary_data(end + 1, :) = row; %#ok<SAGROW>
                            summary_table = cell2table(summary_data, 'VariableNames', summary_cols);
                            writetable(summary_table, summary_path);
                            write_metrics(result_dir, best);
                        end
                    end
                end
            end
        end

        fprintf('\n========================================\n');
        fprintf('Image %s finished.\n', img_name);
        fprintf('Missing rate: %.2f\n', missing_rate);
        fprintf('Best case: %d\n', best.case_id);
        fprintf('Best PSNR: %.6f dB\n', best.psnr);
        fprintf('Best SSIM: %.6f\n', best.ssim);
        fprintf('Best params: rg=%g, K0=%g, maxiter=%g, rho=%g, gamma=%g\n', ...
            best.rg, best.K0, best.maxiter, best.rho, best.gamma);
        fprintf('Result directory: %s\n', result_dir);
        fprintf('========================================\n');

        diary('off');
    end
end

fprintf('\n========================================\n');
fprintf('All images finished.\n');
fprintf('Best image: %s\n', best_all.image);
fprintf('Best missing rate: %.2f\n', best_all.missing_rate);
fprintf('Best PSNR: %.6f dB\n', best_all.psnr);
fprintf('Best case: %d\n', best_all.case_id);
fprintf('Best params: rg=%g, K0=%g, maxiter=%g, rho=%g, gamma=%g\n', ...
    best_all.rg, best_all.K0, best_all.maxiter, best_all.rho, best_all.gamma);
fprintf('========================================\n');

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

function save_best_outputs(result_dir, img_name, I, observed, best)
    imwrite(uint8(best.Xori), fullfile(result_dir, [img_name, '_best_recovered.png']));
    imwrite(uint8(abs(best.Mtensor)), fullfile(result_dir, [img_name, '_best_global_M.png']));
    imwrite(uint8(abs(best.Rtensor)), fullfile(result_dir, [img_name, '_best_local_R.png']));

    fig = figure('Visible', 'off', 'Position', [100, 100, 1500, 500]);
    subplot(1, 3, 1); imshow(I); title('Original', 'FontSize', 12);
    subplot(1, 3, 2); imshow(observed); title('Observed', 'FontSize', 12);
    subplot(1, 3, 3); imshow(uint8(best.Xori));
    title(sprintf('Recovered PSNR %.2f dB', best.psnr), 'FontSize', 12);
    saveas(fig, fullfile(result_dir, [img_name, '_best_compare.png']));
    close(fig);

    best_result = best; %#ok<NASGU>
    save(fullfile(result_dir, 'best_result.mat'), 'best_result', '-v7.3');
    writetable(best.history, fullfile(result_dir, 'best_iteration_history.csv'));
end

function write_metrics(result_dir, best)
    fid = fopen(fullfile(result_dir, 'metrics.txt'), 'w');
    fprintf(fid, 'best_case=%d\n', best.case_id);
    fprintf(fid, 'best_psnr=%.6f\n', best.psnr);
    fprintf(fid, 'best_ssim=%.6f\n', best.ssim);
    fprintf(fid, 'best_mse=%.6f\n', best.mse);
    fprintf(fid, 'best_rmse=%.6f\n', best.rmse);
    fprintf(fid, 'best_rg=%.6f\n', best.rg);
    fprintf(fid, 'best_K0=%.6f\n', best.K0);
    fprintf(fid, 'best_maxiter=%.6f\n', best.maxiter);
    fprintf(fid, 'best_rho=%.6f\n', best.rho);
    fprintf(fid, 'best_gamma=%.6f\n', best.gamma);
    fprintf(fid, 'best_time=%.6f\n', best.time);
    fclose(fid);
end
