%% HaLRTC 随机低秩张量实验批量脚本
% 使用原作者 HaLRTC 算法进行合成低秩张量修复
clear; clc; close all;

%% 路径设置
base_dir = fileparts(mfilename('fullpath'));
if isempty(base_dir)
    base_dir = pwd;
end
addpath(fullfile(base_dir, 'HaLRTC'));

%% 参数设置
seed_list = [920, 921, 922];
missing_rates = [0.8, 0.9, 0.95];
rho = [1e-4, 1e-5];
maxIter = 500;
epsilon = 1e-5;
alpha = [1, 1, 1] / 3;

%% 数据与结果目录
data_dir = fullfile(base_dir, 'data');
result_root = fullfile(base_dir, 'results', 'HaLRTC');
if ~exist(result_root, 'dir')
    mkdir(result_root);
end

all_summary = {};
all_cols = {'数据名称', '方法', '缺失率', 'seed', '最佳rho', 'MSE', 'RMSE', 'RSE', 'MAE', '运行时间', '状态'};

%% 批量处理
for si = 1:numel(seed_list)
    seed = seed_list(si);
    for mi = 1:numel(missing_rates)
        missing_rate = missing_rates(mi);
        miss_tag = round(missing_rate * 100);
        data_name = sprintf('S%d_miss%d', seed, miss_tag);
        data_path = fullfile(data_dir, [data_name, '.mat']);

        fprintf('\n========================================\n');
        fprintf('Data: %s\n', data_name);
        fprintf('Missing rate: %.2f\n', missing_rate);
        fprintf('Seed: %d\n', seed);
        fprintf('========================================\n');

        %% 读取数据
        if ~exist(data_path, 'file')
            warning('Data file not found: %s', data_path);
            all_summary(end + 1, :) = {data_name, 'HaLRTC', missing_rate, seed, NaN, NaN, NaN, NaN, NaN, NaN, 'file_not_found'}; %#ok<SAGROW>
            continue;
        end

        data = load(data_path);
        Xtrue = double(data.X);
        Omega = logical(data.Omega);

        if isfield(data, 'Y')
            Y = double(data.Y);
        else
            Y = Xtrue .* Omega;
        end

        tensor_size = size(Xtrue);
        fprintf('Size: %d x %d x %d\n', tensor_size(1), tensor_size(2), tensor_size(3));
        fprintf('Observed entries: %d / %d\n', sum(Omega(:)), numel(Omega));

        %% 创建结果目录
        result_dir = fullfile(result_root, data_name);
        if ~exist(result_dir, 'dir')
            mkdir(result_dir);
        end

        %% 对每个 rho 运行 HaLRTC
        best_mse = inf;
        best = struct();
        summary_cols = {'rho', 'MSE', 'RMSE', 'RSE', 'MAE', '运行时间', '迭代次数', '相对变化终值', '状态'};
        combo_data = cell(numel(rho), numel(summary_cols));

        for ri = 1:numel(rho)
            rho_val = rho(ri);
            status = 'ok';
            error_msg = '';

            fprintf('Running rho = %.0e ...\n', rho_val);

            try
                tic;
                X_init = Y;
                X_init(~Omega) = mean(Y(Omega));
                [Xhat, errList, history] = HaLRTC(Y, Omega, alpha, rho_val, maxIter, epsilon, X_init, Xtrue);
                elapsed_time = toc;

                Xhat = max(0, min(1, Xhat));

                diff_v = Xtrue(:) - Xhat(:);
                mse_val = mean(diff_v .^ 2);
                rmse_val = sqrt(mse_val);
                rse_val = norm(diff_v) / max(norm(Xtrue(:)), eps);
                mae_val = mean(abs(diff_v));
                final_rel_change = errList(end);

                fprintf('  MSE: %.10f\n', mse_val);
                fprintf('  RMSE: %.10f\n', rmse_val);
                fprintf('  RSE: %.10f\n', rse_val);
                fprintf('  MAE: %.10f\n', mae_val);
                fprintf('  Time: %.2f s\n', elapsed_time);
                fprintf('  Iterations: %d\n', length(errList));
            catch ME
                status = 'error';
                error_msg = getReport(ME, 'extended', 'hyperlinks', 'off');
                warning('HaLRTC failed for %s rho=%.0e: %s', data_name, rho_val, ME.message);

                Xhat = Y;
                elapsed_time = 0;
                mse_val = NaN;
                rmse_val = NaN;
                rse_val = NaN;
                mae_val = NaN;
                final_rel_change = NaN;
                history = table();
                errList = [];
            end

            combo_data(ri, :) = {rho_val, mse_val, rmse_val, rse_val, mae_val, elapsed_time, length(errList), final_rel_change, status};

            % 按 MSE 最低选择最佳 rho
            if strcmp(status, 'ok') && isfinite(mse_val)
                if mse_val < best_mse
                    best_mse = mse_val;
                    best.Xhat = Xhat;
                    best.errList = errList;
                    best.history = history;
                    best.mse = mse_val;
                    best.rmse = rmse_val;
                    best.rse = rse_val;
                    best.mae = mae_val;
                    best.time = elapsed_time;
                    best.rho = rho_val;
                    best.final_rel_change = final_rel_change;
                    best.status = status;
                    best.error = error_msg;
                end
            end
        end

        if isempty(fieldnames(best))
            best.Xhat = Y;
            best.errList = [];
            best.history = table();
            best.mse = NaN;
            best.rmse = NaN;
            best.rse = NaN;
            best.mae = NaN;
            best.time = 0;
            best.rho = NaN;
            best.final_rel_change = NaN;
            best.status = 'all_failed';
            best.error = 'All rho candidates failed';
            best_mse = NaN;
        end

        fprintf('Best rho = %.0e, MSE = %.10f\n', best.rho, best.mse);

        %% 保存 summary.csv 和 实验总结.xlsx
        summary_table = cell2table(combo_data, 'VariableNames', summary_cols);
        writetable(summary_table, fullfile(result_dir, 'summary.csv'));
        writetable(summary_table, fullfile(result_dir, '实验总结.xlsx'));

        %% 保存 best_iteration_history.csv 和 最佳迭代.csv
        if ~isempty(best.history) && height(best.history) > 0
            writetable(best.history, fullfile(result_dir, 'best_iteration_history.csv'));
            writetable(best.history, fullfile(result_dir, '最佳迭代.csv'));
        else
            empty_history = table();
            writetable(empty_history, fullfile(result_dir, 'best_iteration_history.csv'));
            writetable(empty_history, fullfile(result_dir, '最佳迭代.csv'));
        end

        %% 保存 metrics.txt
        fid = fopen(fullfile(result_dir, 'metrics.txt'), 'w');
        fprintf(fid, '数据名称=%s\n', data_name);
        fprintf(fid, '尺寸=%d x %d x %d\n', tensor_size(1), tensor_size(2), tensor_size(3));
        fprintf(fid, '缺失率=%.2f\n', missing_rate);
        fprintf(fid, 'seed=%d\n', seed);
        fprintf(fid, '方法=HaLRTC\n');
        fprintf(fid, 'alpha=[%s]\n', num2str(alpha, '%.6f '));
        fprintf(fid, 'rho候选=[%s]\n', num2str(rho, '%.0e '));
        fprintf(fid, '最佳rho=%.0e\n', best.rho);
        fprintf(fid, 'maxIter=%d\n', maxIter);
        fprintf(fid, 'epsilon=%.0e\n', epsilon);
        fprintf(fid, 'MSE=%.10f\n', best.mse);
        fprintf(fid, 'RMSE=%.10f\n', best.rmse);
        fprintf(fid, 'RSE=%.10f\n', best.rse);
        fprintf(fid, 'MAE=%.10f\n', best.mae);
        fprintf(fid, '相对变化终值=%.10f\n', best.final_rel_change);
        fprintf(fid, '运行时间=%.6f\n', best.time);
        fprintf(fid, '状态=%s\n', best.status);
        if ~isempty(best.error)
            fprintf(fid, '错误=%s\n', best.error);
        end
        fclose(fid);

        %% 保存 result.mat
        recovered = best.Xhat;
        best_history = best.history;
        best_errList = best.errList;
        best_mse_val = best.mse;
        best_rmse_val = best.rmse;
        best_rse_val = best.rse;
        best_mae_val = best.mae;
        best_time_val = best.time;
        best_rho_val = best.rho;
        best_status = best.status;

        save(fullfile(result_dir, 'result.mat'), '-v7.3', ...
            'data_name', 'Xtrue', 'Omega', 'Y', 'recovered', ...
            'alpha', 'rho', 'best_rho_val', 'maxIter', 'epsilon', ...
            'missing_rate', 'seed', 'summary_table', ...
            'best_history', 'best_errList', ...
            'best_mse_val', 'best_rmse_val', 'best_rse_val', 'best_mae_val', ...
            'best_time_val', 'best_status');

        %% 全局汇总
        all_summary(end + 1, :) = {data_name, 'HaLRTC', missing_rate, seed, best.rho, ...
            best.mse, best.rmse, best.rse, best.mae, best.time, best.status}; %#ok<SAGROW>

        fprintf('Results saved to: %s\n', result_dir);
    end
end

%% 保存全局汇总
if ~isempty(all_summary)
    all_table = cell2table(all_summary, 'VariableNames', all_cols);
    writetable(all_table, fullfile(result_root, 'all_summary.csv'));
    writetable(all_table, fullfile(result_root, '全部实验总结.xlsx'));
end

fprintf('\n========================================\n');
fprintf('All random low-rank tensor experiments finished.\n');
fprintf('========================================\n');
