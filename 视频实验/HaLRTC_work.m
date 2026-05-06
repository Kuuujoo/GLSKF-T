%% HaLRTC 视频实验批量脚本
% 使用原作者 HaLRTC 算法进行灰度视频修复
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
alpha = [1, 1, 1] / 3;
key_frames = [40, 100];
maxP = 255;

%% 视频列表
video_list = {
    'news_qcif_gray.yuv',   144, 176, 'news';
    'akiyo_qcif_gray.yuv',  144, 176, 'akiyo';
};

video_data_dir = fullfile(base_dir, '处理后视频');
result_root = fullfile(base_dir, 'results', 'HaLRTC');
if ~exist(result_root, 'dir')
    mkdir(result_root);
end

all_summary = {};
all_cols = {'数据集', '方法', '缺失率', 'seed', '最佳rho', 'MSE', 'RMSE', 'PSNR', 'SSIM', '运行时间', '状态'};

%% 批量处理
for vi = 1:size(video_list, 1)
    video_filename = video_list{vi, 1};
    h = video_list{vi, 2};
    w = video_list{vi, 3};
    dataset = video_list{vi, 4};

    video_path = fullfile(video_data_dir, video_filename);
    I_uint8 = read_yuv420_gray(video_path, h, w);
    I = double(I_uint8);
    frame_num = size(I, 3);

    fprintf('\n============================================================\n');
    fprintf('Video: %s (%s)\n', dataset, video_filename);
    fprintf('Size: %d x %d x %d\n', h, w, frame_num);
    fprintf('============================================================\n');

    for mi = 1:numel(missing_rates)
        missing_rate = missing_rates(mi);
        miss_tag = round(missing_rate * 100);

        exp_name = sprintf('%s_miss%d', dataset, miss_tag);
        result_dir = fullfile(result_root, exp_name);
        if ~exist(result_dir, 'dir')
            mkdir(result_dir);
        end

        %% 生成掩码
        rng(seed);
        Omega = rand(h, w, frame_num) > missing_rate;
        T = I .* Omega;

        fprintf('\nDataset: %s | Missing rate: %.2f | Observed: %d/%d\n', ...
            dataset, missing_rate, sum(Omega(:)), numel(Omega));

        %% 保存 Omega
        save(fullfile(result_dir, 'Omega.mat'), 'Omega', 'seed', 'missing_rate', 'dataset', '-v7.3');

        %% 对每个 rho 运行 HaLRTC
        best_mse = inf;
        best = struct();
        summary_cols = {'rho', 'MSE', 'RMSE', 'PSNR', 'SSIM', '运行时间', '迭代次数', '状态'};
        combo_data = cell(numel(rho_candidates), numel(summary_cols));

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

                diff_v = I(:) - Xhat(:);
                mse_val = mean(diff_v .^ 2);
                rmse_val = sqrt(mse_val);
                psnr_val = 10 * log10(maxP^2 / max(mse_val, eps));
                ssim_val = video_ssim(I_uint8, Xhat_uint8);

                fprintf('  MSE: %.10f\n', mse_val);
                fprintf('  RMSE: %.10f\n', rmse_val);
                fprintf('  PSNR: %.6f dB\n', psnr_val);
                fprintf('  SSIM: %.6f\n', ssim_val);
                fprintf('  Time: %.2f s\n', elapsed_time);
                fprintf('  Iterations: %d\n', length(errList));
            catch ME
                status = 'error';
                error_msg = getReport(ME, 'extended', 'hyperlinks', 'off');
                warning('HaLRTC failed for %s rho=%.0e: %s', dataset, rho, ME.message);

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

            combo_data(ri, :) = {rho, mse_val, rmse_val, psnr_val, ssim_val, elapsed_time, length(errList), status};

            % 按 MSE 最低选择最佳 rho（MSE 相同时选 RMSE 最低）
            if strcmp(status, 'ok') && isfinite(mse_val)
                if mse_val < best_mse || (mse_val == best_mse && rmse_val < best.rmse)
                    best_mse = mse_val;
                    best.Xhat = Xhat;
                    best.Xhat_uint8 = Xhat_uint8;
                    best.errList = errList;
                    best.history = history;
                    best.mse = mse_val;
                    best.rmse = rmse_val;
                    best.psnr = psnr_val;
                    best.ssim = ssim_val;
                    best.time = elapsed_time;
                    best.rho = rho;
                    best.status = status;
                    best.error = error_msg;
                end
            end
        end

        if isempty(fieldnames(best))
            best.Xhat = T;
            best.Xhat_uint8 = uint8(T);
            best.errList = [];
            best.history = table();
            best.mse = NaN;
            best.rmse = NaN;
            best.psnr = NaN;
            best.ssim = NaN;
            best.time = 0;
            best.rho = NaN;
            best.status = 'all_failed';
            best.error = 'All rho candidates failed';
            best_mse = NaN;
        end

        fprintf('Best rho = %.0e, MSE = %.10f\n', best.rho, best.mse);

        %% 保存关键帧恢复图
        for fi = 1:numel(key_frames)
            kf = key_frames(fi);
            if kf <= frame_num
                rec_frame = best.Xhat_uint8(:, :, kf);
                frame_path = fullfile(result_dir, ...
                    sprintf('%s_miss%d_帧%03d_recovered.png', dataset, miss_tag, kf));
                imwrite(rec_frame, frame_path);
            end
        end

        %% 保存 summary.csv 和 实验总结.xlsx
        summary_table = cell2table(combo_data, 'VariableNames', summary_cols);
        writetable(summary_table, fullfile(result_dir, 'summary.csv'));
        writetable(summary_table, fullfile(result_dir, '实验总结.xlsx'));

        %% 保存 best_iteration_history.csv
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
        fprintf(fid, '数据集=%s\n', dataset);
        fprintf(fid, '视频文件=%s\n', video_filename);
        fprintf(fid, '尺寸=%d x %d x %d\n', h, w, frame_num);
        fprintf(fid, '缺失率=%.2f\n', missing_rate);
        fprintf(fid, 'seed=%d\n', seed);
        fprintf(fid, '方法=HaLRTC\n');
        fprintf(fid, 'alpha=[%s]\n', num2str(alpha, '%.6f '));
        fprintf(fid, 'rho候选=[%s]\n', num2str(rho_candidates, '%.0e '));
        fprintf(fid, '最佳rho=%.0e\n', best.rho);
        fprintf(fid, 'maxIter=%d\n', maxIter);
        fprintf(fid, 'epsilon=%.0e\n', epsilon);
        fprintf(fid, 'MSE=%.10f\n', best.mse);
        fprintf(fid, 'RMSE=%.10f\n', best.rmse);
        fprintf(fid, 'PSNR=%.6f\n', best.psnr);
        fprintf(fid, 'SSIM=%.6f\n', best.ssim);
        fprintf(fid, '运行时间=%.6f\n', best.time);
        fprintf(fid, '状态=%s\n', best.status);
        if ~isempty(best.error)
            fprintf(fid, '错误=%s\n', best.error);
        end
        fclose(fid);

        %% 保存 result.mat
        recovered = best.Xhat;
        Xtrue = I;
        Y = T;
        best_history = best.history;
        best_errList = best.errList;
        best_mse_val = best.mse;
        best_rmse_val = best.rmse;
        best_psnr_val = best.psnr;
        best_ssim_val = best.ssim;
        best_time_val = best.time;
        best_rho_val = best.rho;
        best_status = best.status;

        save(fullfile(result_dir, 'result.mat'), '-v7.3', ...
            'dataset', 'Xtrue', 'Omega', 'Y', 'recovered', ...
            'alpha', 'rho_candidates', 'best_rho_val', 'maxIter', 'epsilon', ...
            'missing_rate', 'seed', 'summary_table', ...
            'best_history', 'best_errList', ...
            'best_mse_val', 'best_rmse_val', 'best_psnr_val', 'best_ssim_val', ...
            'best_time_val', 'best_status');

        %% 全局汇总
        all_summary(end + 1, :) = {dataset, 'HaLRTC', missing_rate, seed, best.rho, ...
            best.mse, best.rmse, best.psnr, best.ssim, best.time, best.status}; %#ok<SAGROW>

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
fprintf('All video experiments finished.\n');
fprintf('========================================\n');

%% 辅助函数：读取 YUV420 灰度视频
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

%% 辅助函数：视频 SSIM（逐帧平均）
function ssim_val = video_ssim(original, recovered)
    ssim_val = NaN;
    if exist('ssim', 'file') ~= 2
        return;
    end
    frame_num = size(original, 3);
    vals = NaN(frame_num, 1);
    for k = 1:frame_num
        try
            vals(k) = ssim(recovered(:, :, k), original(:, :, k));
        catch
            vals(k) = NaN;
        end
    end
    vals = vals(isfinite(vals));
    if ~isempty(vals)
        ssim_val = mean(vals);
    end
end
