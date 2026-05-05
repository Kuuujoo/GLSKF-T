%% SPC random low rank tensor experiment
clear; clc;
set(0, 'DefaultFigureVisible', 'off');

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
addpath(fullfile(script_dir, 'SPC'));

seed_list = [920, 921, 922];
missing_rates = [0.8, 0.9, 0.95];

TVQV_list = {'tv', 'qv'};
rho_tv = [0.5 0.5 0];
rho_qv = [0.5 0.5 0];
K = 10;
SNR = 25;
nu = 0.01;
maxiter = 1000;
tol = 1e-5;
out_im = 0;

data_dir = fullfile(script_dir, 'data');
result_root = fullfile(script_dir, 'results', 'SPC');
if ~exist(result_root, 'dir')
    mkdir(result_root);
end

all_summary = {};
summary_cols = {'Data','Method','Variant','MissingRate','Seed','rho1','rho2','rho3','K','SNR','nu','maxiter','tol','MSE','RMSE','RSE','MAE','RelativeChange','Time','Status','ResultDir'};

for seed_idx = 1:numel(seed_list)
    seed = seed_list(seed_idx);
    for rate_idx = 1:numel(missing_rates)
        missing_rate = missing_rates(rate_idx);
        miss_tag = round(missing_rate * 100);
        data_name = sprintf('S%d_miss%d', seed, miss_tag);
        data_path = fullfile(data_dir, [data_name '.mat']);
        if ~exist(data_path, 'file')
            error('Data file not found: %s', data_path);
        end

        data = load(data_path);
        Xtrue = double(data.X);
        Omega = logical(data.Omega);
        Y = double(data.Y);

        result_dir = fullfile(result_root, data_name);
        if ~exist(result_dir, 'dir')
            mkdir(result_dir);
        end

        fprintf('\n============================================================\n');
        fprintf('SPC random tensor: %s\n', data_name);
        fprintf('size: %d x %d x %d\n', size(Xtrue, 1), size(Xtrue, 2), size(Xtrue, 3));
        fprintf('missing_rate: %.2f, observed: %d/%d\n', missing_rate, nnz(Omega), numel(Omega));
        fprintf('============================================================\n');

        best_mse = inf;
        best = struct();
        combo_summary = {};

        for mode_idx = 1:numel(TVQV_list)
            TVQV = TVQV_list{mode_idx};
            if strcmp(TVQV, 'tv')
                rho = rho_tv;
            else
                rho = rho_qv;
            end

            status = 'ok';
            error_msg = '';
            parameter_settings = sprintf('TVQV=%s, rho=[%.4g %.4g %.4g], K=%d, SNR=%g, nu=%g, maxiter=%d, tol=%g', ...
                upper(TVQV), rho(1), rho(2), rho(3), K, SNR, nu, maxiter, tol);

            fprintf('Run SPC variant=%s\n', upper(TVQV));
            try
                rng(seed);
                tic;
                [X, Z, G, U, histo, histo_R, metric_histo] = SPC_func(Y, Omega, TVQV, rho, K, SNR, nu, maxiter, tol, out_im, Xtrue);
                elapsed_time = toc;
                X = min(1, max(0, double(X)));
                [mse_value, rmse_value, rse_value, mae_value] = tensor_metrics(Xtrue, X);
            catch ME
                status = 'error';
                error_msg = getReport(ME, 'extended', 'hyperlinks', 'off');
                warning('SPC variant %s failed: %s', upper(TVQV), ME.message);

                elapsed_time = 0;
                X = Y;
                Z = [];
                G = [];
                U = {};
                histo = [];
                histo_R = [];
                metric_histo = [];
                mse_value = NaN;
                rmse_value = NaN;
                rse_value = NaN;
                mae_value = NaN;
            end

            history = make_history_table(metric_histo, histo, elapsed_time, Xtrue, data_name, upper(TVQV), missing_rate, seed, parameter_settings, status);
            rel_value = NaN;
            if ~isempty(history)
                rel_value = history.relative_change(end);
            end

            combo_summary(end + 1, :) = {data_name, 'SPC', upper(TVQV), missing_rate, seed, rho(1), rho(2), rho(3), K, SNR, nu, maxiter, tol, mse_value, rmse_value, rse_value, mae_value, rel_value, elapsed_time, status, result_dir}; %#ok<SAGROW>

            if strcmp(status, 'ok') && isfinite(mse_value) && mse_value < best_mse
                best_mse = mse_value;
                best.variant = TVQV;
                best.rho = rho;
                best.X = X;
                best.Z = Z;
                best.G = G;
                best.U = U;
                best.histo = histo;
                best.histo_R = histo_R;
                best.metric_histo = metric_histo;
                best.history = history;
                best.mse = mse_value;
                best.rmse = rmse_value;
                best.rse = rse_value;
                best.mae = mae_value;
                best.relative_change = rel_value;
                best.time = elapsed_time;
                best.status = status;
                best.error = error_msg;
            end
        end

        if isempty(fieldnames(best))
            best.variant = 'failed';
            best.rho = [NaN NaN NaN];
            best.X = Y;
            best.Z = [];
            best.G = [];
            best.U = {};
            best.histo = [];
            best.histo_R = [];
            best.metric_histo = [];
            best.history = make_history_table([], [], 0, Xtrue, data_name, 'failed', missing_rate, seed, 'all_failed', 'all_failed');
            best.mse = NaN;
            best.rmse = NaN;
            best.rse = NaN;
            best.mae = NaN;
            best.relative_change = NaN;
            best.time = 0;
            best.status = 'all_failed';
            best.error = 'All SPC variants failed';
        end

        writetable(best.history, fullfile(result_dir, '最佳迭代.csv'));
        writetable(best.history, fullfile(result_dir, 'best_iteration_history.csv'));

        summary_table = cell2table(combo_summary, 'VariableNames', summary_cols);
        writetable(summary_table, fullfile(result_dir, 'summary.csv'));
        writetable(summary_table, fullfile(result_dir, '实验总结.xlsx'));

        recovered = best.X; %#ok<NASGU>
        save(fullfile(result_dir, 'result.mat'), ...
            'data_name', 'Xtrue', 'Omega', 'Y', 'recovered', 'missing_rate', 'seed', ...
            'best', 'TVQV_list', 'K', 'SNR', 'nu', 'maxiter', 'tol', '-v7.3');

        fid = fopen(fullfile(result_dir, 'metrics.txt'), 'w');
        fprintf(fid, 'data=%s\n', data_name);
        fprintf(fid, 'method=SPC\n');
        fprintf(fid, 'variant=%s\n', upper(best.variant));
        fprintf(fid, 'seed=%d\n', seed);
        fprintf(fid, 'missing_rate=%.2f\n', missing_rate);
        fprintf(fid, 'rho=[%.6g %.6g %.6g]\n', best.rho(1), best.rho(2), best.rho(3));
        fprintf(fid, 'K=%d\nSNR=%g\nnu=%g\nmaxiter=%d\ntol=%g\n', K, SNR, nu, maxiter, tol);
        fprintf(fid, 'MSE=%.10g\nRMSE=%.10g\nRSE=%.10g\nMAE=%.10g\nrelative_change=%.10g\n', best.mse, best.rmse, best.rse, best.mae, best.relative_change);
        fprintf(fid, 'time=%.10g\nstatus=%s\nerror=%s\n', best.time, best.status, best.error);
        fclose(fid);

        all_summary(end + 1, :) = {data_name, 'SPC', upper(best.variant), missing_rate, seed, best.rho(1), best.rho(2), best.rho(3), K, SNR, nu, maxiter, tol, best.mse, best.rmse, best.rse, best.mae, best.relative_change, best.time, best.status, result_dir}; %#ok<SAGROW>
        all_table = cell2table(all_summary, 'VariableNames', summary_cols);
        writetable(all_table, fullfile(result_root, 'all_summary.csv'));
        writetable(all_table, fullfile(result_root, '全部实验总结.xlsx'));

        fprintf('Saved SPC result: %s, MSE=%.6g, RMSE=%.6g\n', result_dir, best.mse, best.rmse);
    end
end

fprintf('\nSPC random low rank tensor script is ready.\n');

function [mse_value, rmse_value, rse_value, mae_value] = tensor_metrics(Xtrue, Xhat)
    diff_value = double(Xtrue(:)) - double(Xhat(:));
    mse_value = mean(diff_value .^ 2);
    rmse_value = sqrt(mse_value);
    rse_value = norm(diff_value) / max(norm(double(Xtrue(:))), eps);
    mae_value = mean(abs(diff_value));
end

function history = make_history_table(metric_histo, histo, elapsed_time, Xtrue, dataset, variant, missing_rate, seed, parameter_settings, status)
    if ~isempty(metric_histo)
        valid_idx = find(any(metric_histo ~= 0, 2));
        metric_histo = metric_histo(valid_idx, :);
        iteration = valid_idx(:);
        mse_curve = metric_histo(:, 1);
        rmse_curve = metric_histo(:, 2);
        rse_curve = metric_histo(:, 3);
        mae_curve = metric_histo(:, 4);
    elseif ~isempty(histo)
        histo = double(histo(:));
        iteration = (1:numel(histo))';
        mse_curve = histo ./ max(numel(Xtrue), 1);
        rmse_curve = sqrt(max(mse_curve, 0));
        rse_curve = NaN(numel(histo), 1);
        mae_curve = NaN(numel(histo), 1);
    else
        history = table();
        history.dataset = string.empty(0, 1);
        history.method = string.empty(0, 1);
        history.variant = string.empty(0, 1);
        history.seed = zeros(0, 1);
        history.missing_rate = zeros(0, 1);
        history.iteration = zeros(0, 1);
        history.elapsed_time_seconds = zeros(0, 1);
        history.MSE = zeros(0, 1);
        history.RMSE = zeros(0, 1);
        history.RSE = zeros(0, 1);
        history.MAE = zeros(0, 1);
        history.relative_change = zeros(0, 1);
        history.parameter_settings = string.empty(0, 1);
        history.convergence_status = string.empty(0, 1);
        return;
    end

    n = numel(iteration);
    elapsed_curve = linspace(0, elapsed_time, n)';
    relative_change = [NaN; abs(diff(mse_curve)) ./ max(abs(mse_curve(1:end-1)), eps)];

    history = table();
    history.dataset = repmat(string(dataset), n, 1);
    history.method = repmat("SPC", n, 1);
    history.variant = repmat(string(variant), n, 1);
    history.seed = repmat(seed, n, 1);
    history.missing_rate = repmat(missing_rate, n, 1);
    history.iteration = iteration;
    history.elapsed_time_seconds = elapsed_curve;
    history.MSE = mse_curve;
    history.RMSE = rmse_curve;
    history.RSE = rse_curve;
    history.MAE = mae_curve;
    history.relative_change = relative_change;
    history.parameter_settings = repmat(string(parameter_settings), n, 1);
    history.convergence_status = repmat(string(status), n, 1);
end
