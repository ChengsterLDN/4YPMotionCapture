%% Dual Webcam Live Preview and Simultaneous Photo Capture (Separate Folders)
% Displays live video from two webcams and saves images from each
% camera to its own folder when you press the "📸 Capture" button.

clc; clear; close all;

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

% --- Create output folders if not existing ---
folder1 = fullfile(pwd, 'Camera1');
folder2 = fullfile(pwd, 'Camera2');
if ~exist(folder1, 'dir'); mkdir(folder1); end
if ~exist(folder2, 'dir'); mkdir(folder2); end

% --- Create UI ---
fig = uifigure('Name', 'Dual Camera Live View', 'Position', [100 100 1000 550]);

ax1 = uiaxes(fig, 'Position', [50 150 400 300]);
title(ax1, 'Camera 1');
axis(ax1, 'off');

ax2 = uiaxes(fig, 'Position', [550 150 400 300]);
title(ax2, 'Camera 2');
axis(ax2, 'off');

% Status label
lbl = uilabel(fig, ...
    'Text', 'Ready', ...
    'Position', [420 120 200 30], ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 12, ...
    'FontWeight', 'bold');

% Capture button
btn = uibutton(fig, 'push', ...
    'Text', '📸 Capture', ...
    'FontSize', 14, ...
    'BackgroundColor', [0.2 0.7 0.2], ...
    'FontWeight', 'bold', ...
    'Position', [400 50 200 60], ...
    'ButtonPushedFcn', @(btn,event) capturePhotos(cam1, cam2, folder1, folder2, lbl));

% --- Timer for live preview ---
previewTimer = timer('ExecutionMode', 'fixedRate', ...
                     'Period', 0.05, ...   % ~20 FPS
                     'TimerFcn', @(~,~) updatePreviews(cam1, cam2, ax1, ax2));

start(previewTimer);

% --- Clean up when window closes ---
fig.CloseRequestFcn = @(~,~) closeApp(cam1, cam2, previewTimer, fig);

disp('🎥 Live preview started. Press the 📸 button to capture.');

function capturePhotos(cam1, cam2, folder1, folder2, lbl)
    % Capture frames from both cameras
    img1 = snapshot(cam1);
    img2 = snapshot(cam2);

    % Generate timestamped filenames
    timestamp = datestr(now, 'yyyymmdd_HHMMSS_FFF');
    filename1 = fullfile(folder1, sprintf('cam1_%s.jpg', timestamp));
    filename2 = fullfile(folder2, sprintf('cam2_%s.jpg', timestamp));

    % Save images
    imwrite(img1, filename1);
    imwrite(img2, filename2);

    % Update status text
    lbl.Text = sprintf('Saved:\n%s\n%s', ...
        ['Camera1/' extractAfter(filename1, 'Camera1/')], ...
        ['Camera2/' extractAfter(filename2, 'Camera2/')]);
    fprintf('📸 Saved %s and %s\n', filename1, filename2);
end

function updatePreviews(cam1, cam2, ax1, ax2)
    % Continuously fetch frames and display them
    try
        img1 = snapshot(cam1);
        img2 = snapshot(cam2);

        imshow(img1, 'Parent', ax1);
        imshow(img2, 'Parent', ax2);
    catch
        % Ignore temporary frame grab errors
    end
end

function closeApp(cam1, cam2, previewTimer, fig)
    try
        stop(previewTimer);
        delete(previewTimer);
    catch
    end

    clear cam1 cam2;
    delete(fig);
    disp('🛑 Live preview stopped. Cameras released.');
end
