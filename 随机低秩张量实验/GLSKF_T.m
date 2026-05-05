%% GLSKF-T 随机低秩张量实验脚本
clear; clc; close all;

base_dir = fileparts(mfilename('fullpath'));
if isempty(base_dir)
    base_dir = pwd;
end

addpath(fullfile(base_dir, 'GLSKF-T'));

data_dir = fullfile(base_dir, 'data');
result_root = fullfile(base_dir, 'results', 'GLSKF-T');
if ~exist(result_root, 'dir')
    mkdir(result_root);
end

seed_list = [920, 921, 922];
miss_list = [0.8, 0.9, 0.95];

lengthscaleU        = ones(1, 2) * 30;
varianceU           = ones(1, 2);
lengthscaleR        = ones(1, 2) * 5;
varianceR           = ones(1, 2);
tapering_range      = 7;
d_MaternU           = 3;
d_MaternR           = 3;
maxiter             = 20;
K0                  = 10;
epsilon             = 1e-4;

rg_list             = [7,10];
rho_list            = [1,5,10,15,20];
gamma_list          = [1,5,10,15,20];

all_summary = {};

for si = 1:length(seed_list)
    seed = seed_list(si);
    for mi = 1:length(miss_list)
        missing_rate = miss_list(mi);
        miss_tag = round(missing_rate * 100);
        data_name = sprintf('S%d_miss%d', seed, miss_tag);
        data_path = fullfile(data_dir, [data_name, '.mat']);

        data = load(data_path);
        Xtrue = double(data.X);
        Omega = logical(data.Omega);
        Y = double(data.Y);
        tensor_size = size(Xtrue);

        result_dir = fullfile(result_root, data_name);
        history_dir = fullfile(result_dir, 'iteration_logs');
        if ~exist(result_dir, 'dir')
            mkdir(result_dir);
        end
        if ~exist(history_dir, 'dir')
            mkdir(history_dir);
        end

        diary(fullfile(result_dir, 'console_output.txt'));
        diary_cleanup = onCleanup(@() diary('off')); %#ok<NASGU>

        fprintf('\n========================================\n');
        fprintf('Method: GLSKF-T\n');
        fprintf('Data: %s\n', data_name);
        fprintf('Size: %d x %d x %d\n', tensor_size(1), tensor_size(2), tensor_size(3));
        fprintf('Missing rate: %.1f%%\n', missing_rate * 100);
        fprintf('Seed: %d\n', seed);
        fprintf('Observed entries: %d / %d\n', sum(Omega(:)), numel(Omega));
        fprintf('========================================\n\n');

        case_num = length(rg_list) * length(rho_list) * length(gamma_list);
        cols = {'Case', 'Data', 'Method', 'Variant', 'MissingRate', 'Seed', ...
            'rg', 'rho', 'gamma', 'MSE', 'RMSE', 'RSE', 'MAE', ...
            'FinalRelativeChange', 'Time', 'HistoryFile'};
        summary_data = cell(case_num, numel(cols));

        best_mse = inf;
        best_rmse = inf;
        best_rse = inf;
        best_mae = inf;
        best_rg = 0;
        best_rho = 0;
        best_gamma = 0;
        best_time = 0;
        best_X = [];
        best_R = [];
        best_M = [];
        best_history = table();
        best_relative_change = NaN;

        case_idx = 0;
        for rg_idx = 1:length(rg_list)
            for rho_idx = 1:length(rho_list)
                for gamma_idx = 1:length(gamma_list)
                    rg = rg_list(rg_idx);
                    rho = rho_list(rho_idx);
                    gamma = gamma_list(gamma_idx);
                    case_idx = case_idx + 1;

                    fprintf('\n[%d/%d] Params: rg=%g, rho=%g, gamma=%g\n', ...
                        case_idx, case_num, rg, rho, gamma);

                    tic;
                    [Xhat, Rtensor, Mtensor, ~, iter_history] = GLSKF_tSVD( ...
                        Xtrue, Omega, lengthscaleU, lengthscaleR, ...
                        varianceU, varianceR, tapering_range, ...
                        d_MaternU, d_MaternR, rg, rho, gamma, ...
                        maxiter, K0, epsilon);
                    run_time = toc;

                    Xhat = min(max(double(Xhat), 0), 1);
                    diff_value = Xtrue - Xhat;
                    mse_val = mean(diff_value(:) .^ 2);
                    rmse_val = sqrt(mse_val);
                    rse_val = norm(diff_value(:)) / max(norm(Xtrue(:)), eps);
                    mae_val = mean(abs(diff_value(:)));

                    iter_history = convert_history(iter_history, data_name, seed, missing_rate, ...
                        rg, rho, gamma);
                    if height(iter_history) > 0
                        best_relative_change_case = iter_history.relative_change(end);
                    else
                        best_relative_change_case = NaN;
                    end

                    history_file = fullfile(history_dir, sprintf('case_%03d_history.csv', case_idx));
                    writetable(iter_history, history_file);

                    fprintf('MSE: %.10f\n', mse_val);
                    fprintf('RMSE: %.10f\n', rmse_val);
                    fprintf('RSE: %.10f\n', rse_val);
                    fprintf('MAE: %.10f\n', mae_val);
                    fprintf('Time: %.2f seconds\n', run_time);

                    summary_data(case_idx, :) = {case_idx, data_name, 'GLSKF-T', ...
                        'observed_global_update', missing_rate, seed, rg, rho, gamma, ...
                        mse_val, rmse_val, rse_val, mae_val, best_relative_change_case, ...
                        run_time, history_file};

                    if mse_val < best_mse
                        best_mse = mse_val;
                        best_rmse = rmse_val;
                        best_rse = rse_val;
                        best_mae = mae_val;
                        best_rg = rg;
                        best_rho = rho;
                        best_gamma = gamma;
                        best_time = run_time;
                        best_X = Xhat;
                        best_R = Rtensor;
                        best_M = Mtensor;
                        best_history = iter_history;
                        best_relative_change = best_relative_change_case;
                        fprintf('*** New best MSE = %.10f ***\n', best_mse);
                    end
                end
            end
        end

        summary_table = cell2table(summary_data, 'VariableNames', cols);
        writetable(summary_table, fullfile(result_dir, 'summary.csv'));
        writetable(summary_table, fullfile(result_dir, '实验总结.xlsx'));
        writetable(best_history, fullfile(result_dir, '最佳迭代.csv'));
        writetable(best_history, fullfile(result_dir, 'best_iteration_history.csv'));

        fid = fopen(fullfile(result_dir, 'metrics.txt'), 'w');
        fprintf(fid, 'data=%s\n', data_name);
        fprintf(fid, 'size=%d x %d x %d\n', tensor_size(1), tensor_size(2), tensor_size(3));
        fprintf(fid, 'missing_rate=%.2f\n', missing_rate);
        fprintf(fid, 'seed=%d\n', seed);
        fprintf(fid, 'method=GLSKF-T\n');
        fprintf(fid, 'variant=observed_global_update\n');
        fprintf(fid, 'best_rg=%.6f\n', best_rg);
        fprintf(fid, 'best_rho=%.6f\n', best_rho);
        fprintf(fid, 'best_gamma=%.6f\n', best_gamma);
        fprintf(fid, 'tapering_range=%.6f\n', tapering_range);
        fprintf(fid, 'MSE=%.10f\n', best_mse);
        fprintf(fid, 'RMSE=%.10f\n', best_rmse);
        fprintf(fid, 'RSE=%.10f\n', best_rse);
        fprintf(fid, 'MAE=%.10f\n', best_mae);
        fprintf(fid, 'relative_change=%.10f\n', best_relative_change);
        fprintf(fid, 'time=%.6f\n', best_time);
        fprintf(fid, 'status=ok\n');
        fclose(fid);

        save(fullfile(result_dir, 'result.mat'), ...
            'data_name', 'Xtrue', 'Omega', 'Y', 'best_X', 'best_R', 'best_M', ...
            'best_history', 'best_mse', 'best_rmse', 'best_rse', 'best_mae', ...
            'best_relative_change', 'best_time', 'best_rg', 'best_rho', ...
            'best_gamma', 'missing_rate', 'seed', '-v7.3');

        all_summary(end + 1, :) = {data_name, 'GLSKF-T', 'observed_global_update', ...
            missing_rate, seed, best_rg, best_rho, best_gamma, ...
            best_mse, best_rmse, best_rse, best_mae, best_relative_change, best_time}; %#ok<SAGROW>

        fprintf('\nResult saved to: %s\n', result_dir);
        diary('off');
    end
end

all_cols = {'Data', 'Method', 'Variant', 'MissingRate', 'Seed', 'BestRg', ...
    'BestRho', 'BestGamma', 'MSE', 'RMSE', 'RSE', 'MAE', ...
    'RelativeChange', 'Time'};
all_table = cell2table(all_summary, 'VariableNames', all_cols);
writetable(all_table, fullfile(result_root, 'all_summary.csv'));
writetable(all_table, fullfile(result_root, '全部实验总结.xlsx'));

fprintf('\nAll GLSKF-T low rank tensor experiments finished.\n');
