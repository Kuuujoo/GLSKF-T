%% SPC video experiment script
clear; clc;
set(0, 'DefaultFigureVisible', 'off');

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
addpath(fullfile(script_dir, 'SPC'));

seed = 920;
height = 144;
width = 176;
fps = 30;
show_idx = [40, 200];
missing_rates = [ 0.8 ];

video_files = struct( ...
    'video_name', { 'akiyo_qcif'}, ...
    'filename', {'akiyo_qcif_gray.yuv'} ...
);

TVQV_list = {'tv', 'qv'};
rho_tv = [0.5 0.5 0];
rho_qv = [0.5 0.5 0];
K = 10;
SNR = 25;
nu = 0.01;
maxiter = 1000;
tol = 1e-5;
out_im = 0;

video_data_dir = fullfile(script_dir, '处理后视频');
result_root = fullfile(script_dir, 'results', 'SPC');
if ~exist(result_root, 'dir')
    mkdir(result_root);
end

all_summary = {};
summary_cols = {'dataset','method','variant','missing_rate','seed','MSE','RMSE','PSNR','SSIM','elapsed_time_seconds','status','result_dir'};

for video_idx = 1:numel(video_files)
    video_name = video_files(video_idx).video_name;
    video_path = fullfile(video_data_dir, video_files(video_idx).filename);
    I_uint8 = read_yuv420_gray(video_path, height, width);
    frame_num = size(I_uint8, 3);
    I = double(I_uint8);

    fprintf('\n============================================================\n');
    fprintf('SPC video: %s\n', video_path);
    fprintf('size: %d x %d x %d\n', height, width, frame_num);
    fprintf('============================================================\n');

    for rate_idx = 1:numel(missing_rates)
        missing_rate = missing_rates(rate_idx);
        rng(seed);
        Omega = rand(height, width, frame_num) > missing_rate;
        T = I .* Omega;
        Q = logical(Omega);
        observed_uint8 = uint8(T);

        result_dir = fullfile(result_root, sprintf('%s_%d', video_name, round(missing_rate * 100)));
        if ~exist(result_dir, 'dir')
            mkdir(result_dir);
        end

        fprintf('\nSPC dataset=%s missing_rate=%.2f observed=%d/%d\n', video_name, missing_rate, nnz(Q), numel(Q));

        best_psnr = -inf;
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
            fprintf('Run SPC variant=%s\n', upper(TVQV));

            try
                tic;
                [X, Z, G, U, histo, histo_R] = SPC(T, Q, TVQV, rho, K, SNR, nu, maxiter, tol, out_im);
                elapsed_time = toc;

                X_uint8 = uint8(max(0, min(255, round(X))));
                [mse_value, rmse_value, psnr_value] = video_metrics(I_uint8, X_uint8);
                ssim_value = video_ssim(I_uint8, X_uint8);
            catch ME
                status = 'error';
                error_msg = getReport(ME, 'extended', 'hyperlinks', 'off');
                warning('SPC variant %s failed: %s', upper(TVQV), ME.message);

                elapsed_time = 0;
                X = T;
                Z = [];
                G = [];
                U = {};
                histo = [];
                histo_R = [];
                X_uint8 = observed_uint8;
                mse_value = NaN;
                rmse_value = NaN;
                psnr_value = NaN;
                ssim_value = NaN;
            end

            history = make_history_table(histo, elapsed_time, numel(I), video_name, TVQV, missing_rate, seed, ...
                sprintf('rho=[%.4g %.4g %.4g], K=%d, SNR=%g, nu=%g, maxiter=%d, tol=%g', rho(1), rho(2), rho(3), K, SNR, nu, maxiter, tol), ...
                status);

            combo_summary(end + 1, :) = {video_name, 'SPC', upper(TVQV), missing_rate, seed, mse_value, rmse_value, psnr_value, ssim_value, elapsed_time, status, result_dir}; %#ok<SAGROW>

            if strcmp(status, 'ok') && isfinite(psnr_value) && psnr_value > best_psnr
                best_psnr = psnr_value;
                best.variant = TVQV;
                best.rho = rho;
                best.X = X;
                best.X_uint8 = X_uint8;
                best.Z = Z;
                best.G = G;
                best.U = U;
                best.histo = histo;
                best.histo_R = histo_R;
                best.history = history;
                best.mse = mse_value;
                best.rmse = rmse_value;
                best.psnr = psnr_value;
                best.ssim = ssim_value;
                best.time = elapsed_time;
                best.status = status;
                best.error = error_msg;
            end
        end

        if isempty(fieldnames(best))
            best.variant = 'failed';
            best.rho = [NaN NaN NaN];
            best.X = T;
            best.X_uint8 = observed_uint8;
            best.Z = [];
            best.G = [];
            best.U = {};
            best.histo = [];
            best.histo_R = [];
            best.history = make_history_table([], 0, numel(I), video_name, 'failed', missing_rate, seed, 'all_failed', 'all_failed');
            best.mse = NaN;
            best.rmse = NaN;
            best.psnr = NaN;
            best.ssim = NaN;
            best.time = 0;
            best.status = 'all_failed';
            best.error = 'All SPC variants failed';
        end

        history_path = fullfile(result_dir, 'best_iteration_history.csv');
        writetable(best.history, history_path);

        summary_table = cell2table(combo_summary, 'VariableNames', summary_cols);
        writetable(summary_table, fullfile(result_dir, 'summary.csv'));

        save(fullfile(result_dir, 'Omega.mat'), 'Omega', 'seed', 'missing_rate', 'video_name', '-v7.3');
        save(fullfile(result_dir, 'result.mat'), ...
            'video_name', 'video_path', 'height', 'width', 'frame_num', 'fps', ...
            'missing_rate', 'seed', 'Omega', 'I_uint8', 'observed_uint8', ...
            'best', 'TVQV_list', 'K', 'SNR', 'nu', 'maxiter', 'tol', '-v7.3');

        write_yuv420_gray(best.X_uint8, fullfile(result_dir, sprintf('%s_recovered.yuv', video_name)));
        save_frame_comparisons(I_uint8, observed_uint8, best.X_uint8, show_idx, result_dir, missing_rate, best);
        plot_time_curve(best.history.elapsed_time_seconds, best.history.MSE, 'MSE', fullfile(result_dir, 'mse_time_curve.png'));
        plot_time_curve(best.history.elapsed_time_seconds, best.history.RMSE, 'RMSE', fullfile(result_dir, 'rmse_time_curve.png'));

        metrics_path = fullfile(result_dir, 'metrics.txt');
        fid = fopen(metrics_path, 'w');
        fprintf(fid, 'dataset=%s\n', video_name);
        fprintf(fid, 'method=SPC\n');
        fprintf(fid, 'variant=%s\n', upper(best.variant));
        fprintf(fid, 'missing_rate=%.2f\n', missing_rate);
        fprintf(fid, 'seed=%d\n', seed);
        fprintf(fid, 'rho=[%.6g %.6g %.6g]\n', best.rho(1), best.rho(2), best.rho(3));
        fprintf(fid, 'K=%d\nSNR=%g\nnu=%g\nmaxiter=%d\ntol=%g\n', K, SNR, nu, maxiter, tol);
        fprintf(fid, 'mse=%.10g\nrmse=%.10g\npsnr=%.10g\nssim=%.10g\n', best.mse, best.rmse, best.psnr, best.ssim);
        fprintf(fid, 'time=%.10g\nstatus=%s\n', best.time, best.status);
        fprintf(fid, 'history_file=%s\n', history_path);
        fprintf(fid, 'error=%s\n', best.error);
        fclose(fid);

        all_summary(end + 1, :) = {video_name, 'SPC', upper(best.variant), missing_rate, seed, best.mse, best.rmse, best.psnr, best.ssim, best.time, best.status, result_dir}; %#ok<SAGROW>
        all_table = cell2table(all_summary, 'VariableNames', summary_cols);
        writetable(all_table, fullfile(result_root, 'summary.csv'));

        fprintf('Saved SPC result: %s, PSNR=%.4f, RMSE=%.4f\n', result_dir, best.psnr, best.rmse);
    end
end

if ~isempty(all_summary)
    all_table = cell2table(all_summary, 'VariableNames', summary_cols);
    writetable(all_table, fullfile(result_root, 'summary.csv'));
end

fprintf('\nSPC video experiments are ready to run.\n');

function video = read_yuv420_gray(video_path, height, width)
    if ~exist(video_path, 'file')
        error('Video file not found: %s', video_path);
    end
    fid = fopen(video_path, 'rb');
    cleaner = onCleanup(@() fclose(fid));
    y_size = height * width;
    uv_size = y_size / 4;
    yuv420_frame_size = y_size + 2 * uv_size;
    info = dir(video_path);

    if mod(info.bytes, yuv420_frame_size) == 0
        frame_num = info.bytes / yuv420_frame_size;
        video = zeros(height, width, frame_num, 'uint8');
        for k = 1:frame_num
            y = fread(fid, y_size, 'uint8=>uint8');
            if numel(y) ~= y_size
                error('Failed to read frame %d from %s', k, video_path);
            end
            video(:, :, k) = reshape(y, [width, height])';
            fseek(fid, 2 * uv_size, 'cof');
        end
        return;
    end

    if mod(info.bytes, y_size) == 0
        frame_num = info.bytes / y_size;
        raw = fread(fid, y_size * frame_num, 'uint8=>uint8');
        video = reshape(raw, [width, height, frame_num]);
        video = permute(video, [2 1 3]);
        return;
    end

    error('Video size does not match QCIF gray or YUV420 format: %s', video_path);
end

function write_yuv420_gray(video, save_path)
    video = uint8(max(0, min(255, round(video))));
    [height, width, frame_num] = size(video);
    uv_size = height * width / 4;
    fid = fopen(save_path, 'wb');
    cleaner = onCleanup(@() fclose(fid));
    for k = 1:frame_num
        fwrite(fid, video(:, :, k)', 'uint8');
        fwrite(fid, uint8(128 * ones(uv_size, 1)), 'uint8');
        fwrite(fid, uint8(128 * ones(uv_size, 1)), 'uint8');
    end
end

function [mse_value, rmse_value, psnr_value] = video_metrics(original, recovered)
    diff_value = double(original(:)) - double(recovered(:));
    mse_value = mean(diff_value .^ 2);
    rmse_value = sqrt(mse_value);
    if mse_value <= 0
        psnr_value = Inf;
    else
        psnr_value = 10 * log10(255^2 / mse_value);
    end
end

function value = video_ssim(original, recovered)
    if exist('ssim', 'file') ~= 2
        value = NaN;
        return;
    end
    frame_num = size(original, 3);
    values = NaN(frame_num, 1);
    for k = 1:frame_num
        values(k) = ssim(recovered(:, :, k), original(:, :, k));
    end
    values = values(isfinite(values));
    if isempty(values)
        value = NaN;
    else
        value = mean(values);
    end
end

function history = make_history_table(histo, elapsed_time, data_count, dataset, variant, missing_rate, seed, parameter_settings, status)
    if isempty(histo)
        history = table();
        history.dataset = string.empty(0, 1);
        history.method = string.empty(0, 1);
        history.variant = string.empty(0, 1);
        history.missing_rate = zeros(0, 1);
        history.seed = zeros(0, 1);
        history.iteration = zeros(0, 1);
        history.elapsed_time_seconds = zeros(0, 1);
        history.MSE = zeros(0, 1);
        history.RMSE = zeros(0, 1);
        history.PSNR = zeros(0, 1);
        history.SSIM = zeros(0, 1);
        history.parameter_settings = string.empty(0, 1);
        history.convergence_status = string.empty(0, 1);
        return;
    end

    histo = double(histo(:));
    iteration = (1:numel(histo))';
    mse_curve = histo ./ max(data_count, 1);
    rmse_curve = sqrt(max(mse_curve, 0));
    psnr_curve = 10 * log10(255^2 ./ max(mse_curve, eps));
    elapsed_curve = linspace(0, elapsed_time, numel(histo))';

    history = table();
    history.dataset = repmat(string(dataset), numel(histo), 1);
    history.method = repmat("SPC", numel(histo), 1);
    history.variant = repmat(string(upper(variant)), numel(histo), 1);
    history.missing_rate = repmat(missing_rate, numel(histo), 1);
    history.seed = repmat(seed, numel(histo), 1);
    history.iteration = iteration;
    history.elapsed_time_seconds = elapsed_curve;
    history.MSE = mse_curve;
    history.RMSE = rmse_curve;
    history.PSNR = psnr_curve;
    history.SSIM = NaN(numel(histo), 1);
    history.parameter_settings = repmat(string(parameter_settings), numel(histo), 1);
    history.convergence_status = repmat(string(status), numel(histo), 1);
end

function plot_time_curve(time_values, metric_values, y_label, save_path)
    figure('Position', [100, 100, 700, 500]);
    if isempty(time_values) || isempty(metric_values)
        text(0.5, 0.5, 'No history available', 'HorizontalAlignment', 'center');
    else
        plot(time_values, metric_values, 'LineWidth', 2);
    end
    xlabel('Time (s)', 'FontName', 'Times New Roman', 'FontSize', 14);
    ylabel(y_label, 'FontName', 'Times New Roman', 'FontSize', 14);
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
    grid on;
    box on;
    saveas(gcf, save_path);
    close(gcf);
end

function save_frame_comparisons(original, observed, recovered, show_idx, result_dir, missing_rate, best)
    frame_num = size(original, 3);
    for idx = show_idx
        if idx > frame_num
            continue;
        end
        figure('Position', [100, 100, 1200, 400]);
        subplot(1, 3, 1);
        imshow(original(:, :, idx));
        title(sprintf('Original frame %d', idx), 'FontName', 'Times New Roman');
        axis off;

        subplot(1, 3, 2);
        imshow(observed(:, :, idx));
        title(sprintf('Observed %.0f%% missing', missing_rate * 100), 'FontName', 'Times New Roman');
        axis off;

        subplot(1, 3, 3);
        imshow(recovered(:, :, idx));
        title(sprintf('SPC %s PSNR %.2f dB', upper(best.variant), best.psnr), 'FontName', 'Times New Roman');
        axis off;

        saveas(gcf, fullfile(result_dir, sprintf('frame_%03d_compare.png', idx)));
        close(gcf);
    end
end
