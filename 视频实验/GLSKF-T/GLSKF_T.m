%% GLSKF-T 视频修复实验脚本
clear; clc;

%% 路径设置
base_dir = fileparts(mfilename('fullpath'));
if isempty(base_dir)
    base_dir = pwd;
end
video_root = fileparts(base_dir);
addpath(base_dir);

rng(920);

%% 视频列表
video_list = {
    %'akiyo_qcif_gray.mp4',  144, 176;
    'news_qcif_gray.mp4',   144, 176;
};

%% 缺失率列表
missing_rate_list = [ 0.8];

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
rg_list             = [7,10];
rho_list            = [7,10,15];
gamma_list          = [7,10,15];

%% 结果根目录
result_root = fullfile(video_root, 'results', 'GLSKF-T');
if ~exist(result_root, 'dir')
    mkdir(result_root);
end

%% 外层循环：视频 x 缺失率
for vi = 1:size(video_list, 1)
    video_name   = video_list{vi, 1};
    h            = video_list{vi, 2};
    w            = video_list{vi, 3};
    [~, vid_stem, ~] = fileparts(video_name);

    % 读取视频
    video_dir  = fullfile(video_root, '处理后视频');
    video_path = fullfile(video_dir, video_name);

    vobj = VideoReader(video_path);
    frame_num = round(vobj.Duration * vobj.FrameRate);
    I_all = zeros(h, w, frame_num, 'uint8');
    k = 0;
    while hasFrame(vobj)
        k = k + 1;
        raw = readFrame(vobj);
        if size(raw, 3) == 3
            raw = rgb2gray(raw);
        end
        I_all(:, :, k) = imresize(raw, [h, w]);
    end
    frame_num = k;
    I_all = I_all(:, :, 1:frame_num);

    for mi = 1:length(missing_rate_list)
        missing_rate = missing_rate_list(mi);

        % 生成掩码（固定种子保证可复现）
        rng(920);
        Omega    = rand(h, w, frame_num) > missing_rate;
        observed = uint8(double(I_all) .* double(Omega));

        % 本次实验保存目录
        exp_name   = sprintf('%s_miss%02d', vid_stem, round(missing_rate * 100));
        result_dir = fullfile(result_root, exp_name);
        if ~exist(result_dir, 'dir')
            mkdir(result_dir);
        end

        diary(fullfile(result_dir, 'console_output.txt'));
        diary_cleanup = onCleanup(@() diary('off'));

        fprintf('\n========================================\n');
        fprintf('视频：%s\n', video_name);
        fprintf('尺寸：%d x %d x %d\n', h, w, frame_num);
        fprintf('缺失率：%.1f%%\n', missing_rate * 100);
        fprintf('观测像素：%d / %d\n', sum(Omega(:)), numel(Omega));
        fprintf('========================================\n\n');

        % 参数搜索准备
        case_num = length(rg_list) * length(rho_list) * length(gamma_list);
        cols = {'Case', 'Video', 'MissingRate', 'rg', 'rho', 'gamma', ...
                'PSNR', 'MSE', 'RMSE', 'BestEpochPSNR', 'BestEpoch', 'Time'};
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

        case_idx = 0;
        for rg_idx = 1:length(rg_list)
            for rho_idx = 1:length(rho_list)
                for gamma_idx = 1:length(gamma_list)
                    rg    = rg_list(rg_idx);
                    rho   = rho_list(rho_idx);
                    gamma = gamma_list(gamma_idx);
                    case_idx = case_idx + 1;

                    fprintf('\n[%d/%d] 测试参数：rg=%g, rho=%g, gamma=%g\n', ...
                        case_idx, case_num, rg, rho, gamma);
                    fprintf('--------------------------------------------------\n');

                    tic;
                    [Xori, Rtensor, Mtensor, psnr_hist] = GLSKF_tSVD( ...
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

                    fprintf('最终 PSNR：%.6f dB\n', psnr_val);
                    fprintf('MSE：%.6f\n', mse_val);
                    fprintf('RMSE：%.6f\n', rmse_val);
                    fprintf('BestEpochPSNR：%.6f dB\n', epoch_psnr);
                    fprintf('BestEpoch：%d\n', epoch_idx);
                    fprintf('耗时：%.2f 秒\n', run_time);

                    summary_data(case_idx, :) = {case_idx, vid_stem, missing_rate, ...
                        rg, rho, gamma, psnr_val, mse_val, rmse_val, ...
                        epoch_psnr, epoch_idx, run_time};

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
                        fprintf('*** 发现更优结果：PSNR = %.6f dB ***\n', best_psnr);
                    end
                end
            end
        end

        fprintf('\n========================================\n');
        fprintf('参数搜索完成\n');
        fprintf('最优 rg：%g\n', best_rg);
        fprintf('最优 rho：%g\n', best_rho);
        fprintf('最优 gamma：%g\n', best_gamma);
        fprintf('最优 PSNR：%.6f dB\n', best_psnr);
        fprintf('BestEpochPSNR：%.6f dB\n', best_epoch_psnr);
        fprintf('BestEpoch：%d\n', best_epoch);
        fprintf('最优耗时：%.2f 秒\n', best_time);
        fprintf('========================================\n');

        %% 保存实验总结 Excel
        summary_table = cell2table(summary_data, 'VariableNames', cols);
        writetable(summary_table, fullfile(result_dir, '实验总结.xlsx'));

        %% 保存 metrics.txt
        fid = fopen(fullfile(result_dir, 'metrics.txt'), 'w');
        fprintf(fid, 'video_name=%s\n',          vid_stem);
        fprintf(fid, 'video_path=%s\n',          video_path);
        fprintf(fid, 'size=%d x %d x %d\n',      h, w, frame_num);
        fprintf(fid, 'missing_rate=%.2f\n',       missing_rate);
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

        %% 保存 .mat
        save(fullfile(result_dir, 'result.mat'), ...
            'vid_stem', 'video_path', 'I_all', 'Omega', 'observed', ...
            'best_X', 'best_R', 'best_M', 'best_hist', ...
            'best_psnr', 'best_mse', 'best_rmse', ...
            'best_epoch_psnr', 'best_epoch', 'best_time', ...
            'best_rg', 'best_rho', 'best_gamma', ...
            'missing_rate', '-v7.3');

        %% 保存图片
        show_frames = show_frame_list(show_frame_list <= frame_num);
        for fi = 1:length(show_frames)
            idx = show_frames(fi);
            ori_frame = I_all(:, :, idx);
            obs_frame = observed(:, :, idx);
            rec_frame = uint8(best_X(:, :, idx));

            % 单独图片（无文字）
            imwrite(ori_frame, fullfile(result_dir, sprintf('frame_%03d_original.png', idx)));
            imwrite(obs_frame, fullfile(result_dir, sprintf('frame_%03d_observed.png', idx)));
            imwrite(rec_frame, fullfile(result_dir, sprintf('frame_%03d_recovered.png', idx)));

            % 对比图（三合一，带标题）
            figure('Visible', 'off', 'Position', [100, 100, 1200, 400]);

            subplot(1, 3, 1);
            imshow(ori_frame);
            title(sprintf('原始帧 %d', idx), 'FontSize', 13, 'FontName', 'SimHei');
            axis off;

            subplot(1, 3, 2);
            imshow(obs_frame);
            title(sprintf('观测帧 %d  (缺失 %.0f%%)', idx, missing_rate * 100), ...
                'FontSize', 13, 'FontName', 'SimHei');
            axis off;

            subplot(1, 3, 3);
            imshow(rec_frame);
            title(sprintf('修复帧 %d  PSNR=%.2f dB', idx, best_psnr), ...
                'FontSize', 13, 'FontName', 'SimHei');
            axis off;

            saveas(gcf, fullfile(result_dir, sprintf('frame_%03d_compare.png', idx)));
            close(gcf);
        end

        fprintf('\n结果已保存到：%s\n', result_dir);
        diary('off');
    end
end

fprintf('\n所有实验完成。\n');
