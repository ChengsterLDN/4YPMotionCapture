function dual_camera_recorder
%% Dual Webcam Live View + Simultaneous Video Recording + Timer
% Displays live video from two webcams and records synchronized
% video streams into separate folders with a visible recording timer.

clc; close all;

% --- List connected webcams ---
camList = webcamlist;
disp('Available webcams:');
disp(camList);

% --- Select cameras (adjust indices as needed) ---
cam1_index = 1;
cam2_index = 3;

% --- Initialize webcams ---
cam1 = webcam(cam1_index);
cam2 = webcam(cam2_index);

% Optional: set resolution
cam1.Resolution = '640x480';
cam2.Resolution = '640x480';

% --- Create output folders ---
folder1 = fullfile(pwd, 'Camera1Video');
folder2 = fullfile(pwd, 'Camera2Video');
if ~exist(folder1, 'dir'); mkdir(folder1); end
if ~exist(folder2, 'dir'); mkdir(folder2); end

% --- Create UI ---
fig = uifigure('Name', 'Dual Camera Recorder', 'Position', [100 100 1050 600]);

ax1 = uiaxes(fig, 'Position', [50 150 400 300]);
title(ax1, 'Camera 1'); axis(ax1, 'off');

ax2 = uiaxes(fig, 'Position', [600 150 400 300]);
title(ax2, 'Camera 2'); axis(ax2, 'off');

statusLabel = uilabel(fig, ...
    'Text', 'Ready', ...
    'Position', [425 120 250 30], ...
    'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', ...
    'FontSize', 12);

timerLabel = uilabel(fig, ...
    'Text', '', ...
    'Position', [450 90 200 25], ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 12, ...
    'FontColor', [1 0 0], ...
    'FontWeight', 'bold');

btnStart = uibutton(fig, 'push', ...
    'Text', '🎬 Start Recording', ...
    'FontSize', 14, ...
    'BackgroundColor', [0.2 0.7 0.2], ...
    'FontWeight', 'bold', ...
    'Position', [300 30 200 60]);

btnStop = uibutton(fig, 'push', ...
    'Text', '🛑 Stop Recording', ...
    'FontSize', 14, ...
    'BackgroundColor', [0.9 0.3 0.3], ...
    'FontWeight', 'bold', ...
    'Position', [550 30 200 60], ...
    'Enable', 'off');

% --- Timer for live preview ---
previewTimer = timer('ExecutionMode', 'fixedRate', ...
                     'Period', 0.05, ...
                     'TimerFcn', @(~,~) updatePreviews());

start(previewTimer);

% --- Shared state variables ---
isRecording = false;
videoWriter1 = [];
videoWriter2 = [];
recordStartTime = [];

% --- Button Callbacks ---
btnStart.ButtonPushedFcn = @(btn, event) startRecording();
btnStop.ButtonPushedFcn  = @(btn, event) stopRecording();

% --- Close handler ---
fig.CloseRequestFcn = @(src, event) closeApp();

disp('🎥 Live preview started. Click "🎬 Start Recording" to begin.');

%% ===== Nested Helper Functions (can access parent vars) =====

    function updatePreviews()
        try
            img1 = snapshot(cam1);
            img2 = snapshot(cam2);
            imshow(img1, 'Parent', ax1);
            imshow(img2, 'Parent', ax2);

            if isRecording
                writeVideo(videoWriter1, img1);
                writeVideo(videoWriter2, img2);
                elapsed = toc(recordStartTime);
                timerLabel.Text = sprintf('Recording: %s', formatTime(elapsed));
            end
        catch
            % Ignore temporary camera errors
        end
    end

    function startRecording()
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        file1 = fullfile(folder1, sprintf('video_cam1_%s.avi', timestamp));
        file2 = fullfile(folder2, sprintf('video_cam2_%s.avi', timestamp));

        fps = 20;
        videoWriter1 = VideoWriter(file1, 'Motion JPEG AVI');
        videoWriter1.FrameRate = fps;
        open(videoWriter1);

        videoWriter2 = VideoWriter(file2, 'Motion JPEG AVI');
        videoWriter2.FrameRate = fps;
        open(videoWriter2);

        isRecording = true;
        recordStartTime = tic;
        btnStart.Enable = 'off';
        btnStop.Enable = 'on';
        statusLabel.Text = sprintf('🔴 Recording... (%s)', timestamp);
        timerLabel.Text = 'Recording: 00:00:00';
        disp('🎬 Recording started.');
    end

    function stopRecording()
        if isRecording
            isRecording = false;
            close(videoWriter1);
            close(videoWriter2);
            btnStart.Enable = 'on';
            btnStop.Enable = 'off';
            statusLabel.Text = '✅ Recording stopped and saved.';
            timerLabel.Text = '';
            disp('🛑 Recording stopped and saved.');
        end
    end

    function closeApp()
        try
            if isvalid(previewTimer)
                stop(previewTimer);
                delete(previewTimer);
            end
        catch
        end
        if isRecording
            stopRecording();
        end
        try
            clear cam1 cam2;
        catch
        end
        if isvalid(fig)
            delete(fig);
        end
        disp('🧹 Clean exit: Cameras released and GUI closed.');
    end

    function tStr = formatTime(seconds)
        hrs = floor(seconds / 3600);
        mins = floor(mod(seconds, 3600) / 60);
        secs = floor(mod(seconds, 60));
        tStr = sprintf('%02d:%02d:%02d', hrs, mins, secs);
    end

end
