%% GLSKF-T 视频修复实验脚本
clear; clc; close all;

%% 路径设置
base_dir = fileparts(mfilename('fullpath'));
if isempty(base_dir)
    base_dir = pwd;
end
video_root = fileparts(base_dir);
addpath(base_dir);

seed = 920;
rng(seed);

%% 视频列表，使用处理后 YUV 文件
video_list = {
    'akiyo_qcif_gray.yuv', 144, 176;
    'news_qcif_gray.yuv',  144, 176;
};

%% 缺失率列表
missing_rate_list = [0.8, 0.9, 0.95];

%% 固定算法参数
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
show_frame_list     = [40, 100];

%% 参数搜索网格
rg_list             = [7, 10];
rho_list            = [7, 10, 15];
gamma_list          = [7, 10, 15];

%% 结果根目录
result_root = fullfile(video_root, 'results', 'GLSKF-T');
if ~exist(result_root, 'dir')
    mkdir(result_root);
end

all_summary = {};

%% 外层循环：视频 x 缺失率
for vi = 1:size(video_list, 1)
    video_name   = video_list{vi, 1};
    h            = video_list{vi, 2};
    w            = video_list{vi, 3};
    [~, vid_stem, ~] = fileparts(video_name);

    video_dir  = fullfile(video_root, '处理后视频');
    video_path = fullfile(video_dir, video_name);
    I_all = read_yuv420_gray(video_path, h, w);
    frame_num = size(I_all, 3);

    for mi = 1:length(missing_rate_list)
        missing_rate = missing_rate_list(mi);

        % 保留脚本内掩码生成逻辑和种子设置
        rng(seed);
        Omega    = rand(h, w, frame_num) > missing_rate;
        observed = uint8(double(I_all) .* double(Omega));

        exp_name   = sprintf('%s_miss%02d', vid_stem, round(missing_rate * 100));
        result_dir = fullfile(result_root, exp_name);
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
        fprintf('Video: %s\n', video_name);
        fprintf('Input: %s\n', video_path);
        fprintf('Size: %d x %d x %d\n', h, w, frame_num);
        fprintf('Missing rate: %.1f%%\n', missing_rate * 100);
        fprintf('Seed: %d\n', seed);
        fprintf('Observed pixels: %d / %d\n', sum(Omega(:)), numel(Omega));
        fprintf('========================================\n\n');

        case_num = length(rg_list) * length(rho_list) * length(gamma_list);
        cols = {'Case', 'Video', 'Method', 'Variant', 'MissingRate', 'Seed', ...
                'rg', 'rho', 'gamma', 'PSNR', 'MSE', 'RMSE', ...
                'BestEpochPSNR', 'BestEpoch', 'Time', 'HistoryFile'};
        summary_data = cell(case_num, numel(cols));

        best_psnr        = -inf;
        best_rg          = 0;
        best_rho         = 0;
        best_gamma       = 0;
        best_epoch       = 0;
        best_epoch_psnr  = 0;
        best_time        = 0;
        best_mse         = 0;
        best_rmse        = 0;
        best_X           = [];
        best_R           = [];
        best_M           = [];
        best_hist        = [];
        best_history     = table();

        case_idx = 0;
        for rg_idx = 1:length(rg_list)
            for rho_idx = 1:length(rho_list)
                for gamma_idx = 1:length(gamma_list)
                    rg    = rg_list(rg_idx);
                    rho   = rho_list(rho_idx);
                    gamma = gamma_list(gamma_idx);
                    case_idx = case_idx + 1;

                    fprintf('\n[%d/%d] Params: rg=%g, rho=%g, gamma=%g\n', ...
                        case_idx, case_num, rg, rho, gamma);
                    fprintf('--------------------------------------------------\n');

                    tic;
                    [Xori, Rtensor, Mtensor, psnr_hist, iter_history] = GLSKF_tSVD( ...
                        I_all, Omega, lengthscaleU, lengthscaleR, ...
                        varianceU, varianceR, tapering_range, ...
                        d_MaternU, d_MaternR, rg, rho, gamma, ...
                        maxiter, K0, epsilon);
                    run_time = toc;

                    diff     = double(I_all) - double(Xori);
                    mse_val  = mean(diff(:) .^ 2);
                    rmse_val = sqrt(mse_val);
                    psnr_val = 10 * log10(255^2 / max(mse_val, eps));
                    [epoch_psnr, epoch_idx] = max(psnr_hist);

                    iter_history.dataset = repmat({vid_stem}, height(iter_history), 1);
                    iter_history.method = repmat({'GLSKF-T'}, height(iter_history), 1);
                    iter_history.variant = repmat({'observed_global_update'}, height(iter_history), 1);
                    iter_history.missing_rate = repmat(missing_rate, height(iter_history), 1);
                    iter_history.seed = repmat(seed, height(iter_history), 1);
                    iter_history.rg = repmat(rg, height(iter_history), 1);
                    iter_history.rho = repmat(rho, height(iter_history), 1);
                    iter_history.gamma = repmat(gamma, height(iter_history), 1);
                    history_file = fullfile(history_dir, sprintf('case_%03d_history.csv', case_idx));
                    writetable(iter_history, history_file);

                    fprintf('Final PSNR: %.6f dB\n', psnr_val);
                    fprintf('MSE: %.6f\n', mse_val);
                    fprintf('RMSE: %.6f\n', rmse_val);
                    fprintf('BestEpochPSNR: %.6f dB\n', epoch_psnr);
                    fprintf('BestEpoch: %d\n', epoch_idx);
                    fprintf('Time: %.2f seconds\n', run_time);

                    summary_data(case_idx, :) = {case_idx, vid_stem, 'GLSKF-T', ...
                        'observed_global_update', missing_rate, seed, rg, rho, gamma, ...
                        psnr_val, mse_val, rmse_val, epoch_psnr, epoch_idx, ...
                        run_time, history_file};

                    if psnr_val > best_psnr
                        best_psnr       = psnr_val;
                        best_rg         = rg;
                        best_rho        = rho;
                        best_gamma      = gamma;
                        best_epoch      = epoch_idx;
                        best_epoch_psnr = epoch_psnr;
                        best_time       = run_time;
                        best_mse        = mse_val;
                        best_rmse       = rmse_val;
                        best_X          = Xori;
                        best_R          = Rtensor;
                        best_M          = Mtensor;
                        best_hist       = psnr_hist;
                        best_history    = iter_history;
                        fprintf('*** New best PSNR = %.6f dB ***\n', best_psnr);
                    end
                end
            end
        end

        fprintf('\n========================================\n');
        fprintf('Search finished\n');
        fprintf('Best rg: %g\n', best_rg);
        fprintf('Best rho: %g\n', best_rho);
        fprintf('Best gamma: %g\n', best_gamma);
        fprintf('Best PSNR: %.6f dB\n', best_psnr);
        fprintf('BestEpochPSNR: %.6f dB\n', best_epoch_psnr);
        fprintf('BestEpoch: %d\n', best_epoch);
        fprintf('Best time: %.2f seconds\n', best_time);
        fprintf('========================================\n');

        summary_table = cell2table(summary_data, 'VariableNames', cols);
        writetable(summary_table, fullfile(result_dir, 'summary.csv'));
        writetable(summary_table, fullfile(result_dir, '实验总结.xlsx'));
        writetable(best_history, fullfile(result_dir, 'best_iteration_history.csv'));

        fid = fopen(fullfile(result_dir, 'metrics.txt'), 'w');
        fprintf(fid, 'video_name=%s\n',          vid_stem);
        fprintf(fid, 'video_path=%s\n',          video_path);
        fprintf(fid, 'size=%d x %d x %d\n',      h, w, frame_num);
        fprintf(fid, 'missing_rate=%.2f\n',       missing_rate);
        fprintf(fid, 'seed=%d\n',                 seed);
        fprintf(fid, 'method=GLSKF-T\n');
        fprintf(fid, 'variant=observed_global_update\n');
        fprintf(fid, 'best_rg=%.6f\n',            best_rg);
        fprintf(fid, 'best_rho=%.6f\n',           best_rho);
        fprintf(fid, 'best_gamma=%.6f\n',         best_gamma);
        fprintf(fid, 'tapering_range=%.6f\n',     tapering_range);
        fprintf(fid, 'psnr=%.6f\n',               best_psnr);
        fprintf(fid, 'mse=%.6f\n',                best_mse);
        fprintf(fid, 'rmse=%.6f\n',               best_rmse);
        fprintf(fid, 'best_epoch_psnr=%.6f\n',    best_epoch_psnr);
        fprintf(fid, 'best_epoch=%d\n',           best_epoch);
        fprintf(fid, 'time=%.6f\n',               best_time);
        fclose(fid);

        save(fullfile(result_dir, 'result.mat'), ...
            'vid_stem', 'video_path', 'I_all', 'Omega', 'observed', ...
            'best_X', 'best_R', 'best_M', 'best_hist', 'best_history', ...
            'best_psnr', 'best_mse', 'best_rmse', ...
            'best_epoch_psnr', 'best_epoch', 'best_time', ...
            'best_rg', 'best_rho', 'best_gamma', ...
            'missing_rate', 'seed', '-v7.3');

        save_frames_and_curves(result_dir, I_all, observed, best_X, best_history, ...
            show_frame_list, frame_num, missing_rate, best_psnr);

        all_summary(end + 1, :) = {vid_stem, 'GLSKF-T', 'observed_global_update', ...
            missing_rate, seed, best_rg, best_rho, best_gamma, best_psnr, ...
            best_mse, best_rmse, best_epoch_psnr, best_epoch, best_time}; %#ok<SAGROW>

        fprintf('\nResult saved to: %s\n', result_dir);
        diary('off');
    end
end

all_cols = {'Video', 'Method', 'Variant', 'MissingRate', 'Seed', 'BestRg', ...
    'BestRho', 'BestGamma', 'PSNR', 'MSE', 'RMSE', 'BestEpochPSNR', ...
    'BestEpoch', 'Time'};
all_table = cell2table(all_summary, 'VariableNames', all_cols);
writetable(all_table, fullfile(result_root, 'all_summary.csv'));
writetable(all_table, fullfile(result_root, '全部实验总结.xlsx'));

fprintf('\nAll GLSKF-T video experiments finished.\n');

function video = read_yuv420_gray(video_path, height, width)
    fid = fopen(video_path, 'rb');
    if fid < 0
        error('Cannot open video file: %s', video_path);
    end
    cleaner = onCleanup(@() fclose(fid));

    info = dir(video_path);
    y_size = height * width;
    uv_size = y_size / 4;
    yuv420_frame_size = y_size + 2 * uv_size;

    if mod(info.bytes, yuv420_frame_size) == 0
        frame_num = info.bytes / yuv420_frame_size;
        video = zeros(height, width, frame_num, 'uint8');
        for k = 1:frame_num
            y = fread(fid, y_size, 'uint8=>uint8');
            if numel(y) ~= y_size
                error('Incomplete Y frame in %s', video_path);
            end
            video(:, :, k) = reshape(y, [width, height])';
            fseek(fid, 2 * uv_size, 'cof');
        end
    elseif mod(info.bytes, y_size) == 0
        frame_num = info.bytes / y_size;
        video = zeros(height, width, frame_num, 'uint8');
        for k = 1:frame_num
            y = fread(fid, y_size, 'uint8=>uint8');
            if numel(y) ~= y_size
                error('Incomplete gray frame in %s', video_path);
            end
            video(:, :, k) = reshape(y, [width, height])';
        end
    else
        error('Unexpected YUV file size: %s', video_path);
    end
end

function save_frames_and_curves(result_dir, I_all, observed, recovered, history, show_frame_list, frame_num, missing_rate, best_psnr)
    show_frames = show_frame_list(show_frame_list <= frame_num);
    for fi = 1:length(show_frames)
        idx = show_frames(fi);
        ori_frame = I_all(:, :, idx);
        obs_frame = observed(:, :, idx);
        rec_frame = uint8(recovered(:, :, idx));

        imwrite(ori_frame, fullfile(result_dir, sprintf('frame_%03d_original.png', idx)));
        imwrite(obs_frame, fullfile(result_dir, sprintf('frame_%03d_observed.png', idx)));
        imwrite(rec_frame, fullfile(result_dir, sprintf('frame_%03d_recovered.png', idx)));

        fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 400]);
        subplot(1, 3, 1);
        imshow(ori_frame);
        title(sprintf('Original frame %d', idx), 'FontSize', 13, 'FontName', 'Times New Roman');
        axis off;

        subplot(1, 3, 2);
        imshow(obs_frame);
        title(sprintf('Observed frame %d, missing %.0f%%', idx, missing_rate * 100), ...
            'FontSize', 13, 'FontName', 'Times New Roman');
        axis off;

        subplot(1, 3, 3);
        imshow(rec_frame);
        title(sprintf('Recovered frame %d, PSNR %.2f dB', idx, best_psnr), ...
            'FontSize', 13, 'FontName', 'Times New Roman');
        axis off;

        saveas(fig, fullfile(result_dir, sprintf('frame_%03d_compare.png', idx)));
        close(fig);
    end

    plot_time_curve(history.elapsed_time_seconds, history.MSE, ...
        'Time (s)', 'MSE', fullfile(result_dir, 'best_mse_time_curve.png'));
    plot_time_curve(history.elapsed_time_seconds, history.RMSE, ...
        'Time (s)', 'RMSE', fullfile(result_dir, 'best_rmse_time_curve.png'));
end

function plot_time_curve(x, y, x_label, y_label, save_path)
    fig = figure('Visible', 'off', 'Position', [100, 100, 700, 500]);
    plot(x, y, 'LineWidth', 2);
    grid on;
    xlabel(x_label, 'FontName', 'Times New Roman', 'FontSize', 14);
    ylabel(y_label, 'FontName', 'Times New Roman', 'FontSize', 14);
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
    saveas(fig, save_path);
    close(fig);
end
