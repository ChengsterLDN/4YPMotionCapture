%% Stereo HSV-Based Live Tracker with Video Recording, Trajectory Saving, Red Boxes, and Kalman Filter Smoothing
clc; close all; clear;

%% --- Load stereo calibration parameters ---
S = load('stereoParams.mat');
stereoParams = S.stereoParams;
calibSize = stereoParams.CameraParameters1.ImageSize;

%% --- Connect to two webcams ---
camList = webcamlist;
if numel(camList) < 2
    error('Two webcams are required.');
end

camL = webcam(1);
camR = webcam(3);

% Try setting resolution to calibration size
try
    resStr = sprintf('%dx%d', calibSize(2), calibSize(1));
    camL.Resolution = resStr;
    camR.Resolution = resStr;
catch
    warning('Could not set resolution. Will resize frames instead.');
end

%% --- Capture first frame ---
leftFrame = snapshot(camL);
rightFrame = snapshot(camR);
leftFrame  = imresize(leftFrame, [calibSize(1) calibSize(2)]);
rightFrame = imresize(rightFrame, [calibSize(1) calibSize(2)]);

%% --- Select multiple pixels for different colors ---
figure('Name','Select Colors to Track','Position',[100 100 1000 400]);
imshow(leftFrame);
title('Click on each object color to track. Press Enter when done.');

[xList, yList] = ginput();   % multiple clicks
xList = round(xList); yList = round(yList);

numColors = length(xList);
if numColors == 0
    error('No colors selected.');
end

% Get HSV and RGB for each selected pixel
HSVList = zeros(numColors,3);
RGBList = zeros(numColors,3);
for i = 1:numColors
    rgb = impixel(leftFrame, xList(i), yList(i));
    HSVList(i,:) = rgb2hsv(double(rgb)/255);
    RGBList(i,:) = double(rgb)/255;  % For plotting
    hold on;
    plot(xList(i), yList(i), '*', 'Color', RGBList(i,:), 'MarkerSize', 15, 'LineWidth',2);
    fprintf('Color %d selected at (%d,%d): RGB=[%.0f %.0f %.0f], HSV=[%.3f %.3f %.3f]\n', ...
        i, xList(i), yList(i), rgb(1), rgb(2), rgb(3), HSVList(i,1), HSVList(i,2), HSVList(i,3));
end
hold off;

% Fixed HSV tolerances
hTol = 0.04; sTol = 0.3; vTol = 0.35;
hMinList = max(0, HSVList(:,1)-hTol);
hMaxList = min(1, HSVList(:,1)+hTol);
sMinList = max(0, HSVList(:,2)-sTol);
sMaxList = min(1, HSVList(:,2)+sTol);
vMinList = max(0, HSVList(:,3)-vTol);
vMaxList = min(1, HSVList(:,3)+vTol);

%% --- Initialize storage and Kalman filters ---
positions3D = cell(numColors,1);
frameIdx = 0;

% Kalman filters for each object (left and right separately)
kalmanL = cell(numColors,1);
kalmanR = cell(numColors,1);
for c = 1:numColors
    % Initialize at the selected pixel
    kalmanL{c} = configureKalmanFilter('ConstantVelocity', [xList(c), yList(c)], [1 1]*1e5, [25 10], 25);
    kalmanR{c} = configureKalmanFilter('ConstantVelocity', [xList(c), yList(c)], [1 1]*1e5, [25 10], 25);
end

prevCentroidsL = nan(numColors,2);
prevCentroidsR = nan(numColors,2);

maxMove = 50; % maximum pixels allowed per frame

%% --- Setup folders and video writers ---
if ~exist('Camera1','dir'), mkdir('Camera1'); end
if ~exist('Camera2','dir'), mkdir('Camera2'); end
if ~exist('trajectories','dir'), mkdir('trajectories'); end

timestamp = datestr(now,'yyyymmdd_HHMMSS');
leftVideoFile  = fullfile('Camera1', ['leftCam_' timestamp '.avi']);
rightVideoFile = fullfile('Camera2', ['rightCam_' timestamp '.avi']);

vwLeft = VideoWriter(leftVideoFile);
vwLeft.FrameRate = 30;  
open(vwLeft);

vwRight = VideoWriter(rightVideoFile);
vwRight.FrameRate = 30;
open(vwRight);

%% --- Live tracking figure ---
hFig = figure('Name','Live Stereo Multi-Color Tracker','Position',[100 100 1400 500]);
disp('Press Ctrl+C or close the figure to stop.');

while true
    frameIdx = frameIdx + 1;

    if ~ishandle(hFig)
        disp('Figure closed by user. Stopping live tracking.');
        break;
    end

    % Capture frames
    leftFrame  = snapshot(camL);
    rightFrame = snapshot(camR);

    leftFrame  = imresize(leftFrame, [calibSize(1), calibSize(2)]);
    rightFrame = imresize(rightFrame, [calibSize(1), calibSize(2)]);

    % Write frames to video
    writeVideo(vwLeft, leftFrame);
    writeVideo(vwRight, rightFrame);

    % Rectify
    [leftRect, rightRect] = rectifyStereoImages(leftFrame, rightFrame, stereoParams);

    % HSV
    hsvLeft  = rgb2hsv(im2double(leftRect));
    hsvRight = rgb2hsv(im2double(rightRect));

    centroidsL = nan(numColors,2);
    centroidsR = nan(numColors,2);

    % --- Process each color ---
    for c = 1:numColors
        maskL = (hsvLeft(:,:,1)>=hMinList(c) & hsvLeft(:,:,1)<=hMaxList(c)) & ...
                (hsvLeft(:,:,2)>=sMinList(c) & hsvLeft(:,:,2)<=sMaxList(c)) & ...
                (hsvLeft(:,:,3)>=vMinList(c) & hsvLeft(:,:,3)<=vMaxList(c));
        maskR = (hsvRight(:,:,1)>=hMinList(c) & hsvRight(:,:,1)<=hMaxList(c)) & ...
                (hsvRight(:,:,2)>=sMinList(c) & hsvRight(:,:,2)<=sMaxList(c)) & ...
                (hsvRight(:,:,3)>=vMinList(c) & hsvRight(:,:,3)<=vMaxList(c));

        se = strel('disk',5);
        maskL = imfill(imclose(imopen(maskL,se),se),'holes');
        maskR = imfill(imclose(imopen(maskR,se),se),'holes');

        statsL = regionprops(maskL,'Centroid','Area');
        statsR = regionprops(maskR,'Centroid','Area');

        % Select largest area blob
        if ~isempty(statsL)
            [~, idxL] = max([statsL.Area]);
            detectedL = statsL(idxL).Centroid;
        else
            detectedL = [NaN NaN];
        end
        if ~isempty(statsR)
            [~, idxR] = max([statsR.Area]);
            detectedR = statsR(idxR).Centroid;
        else
            detectedR = [NaN NaN];
        end

        % --- Kalman filter + max motion threshold ---
        % Left
        if ~any(isnan(detectedL))
            predictedL = predict(kalmanL{c});
            if ~any(isnan(prevCentroidsL(c,:))) && norm(detectedL - prevCentroidsL(c,:)) > maxMove
                % ignore sudden jump
                centroidsL(c,:) = prevCentroidsL(c,:);
            else
                centroidsL(c,:) = correct(kalmanL{c}, detectedL);
            end
        else
            centroidsL(c,:) = predict(kalmanL{c});
        end

        % Right
        if ~any(isnan(detectedR))
            predictedR = predict(kalmanR{c});
            if ~any(isnan(prevCentroidsR(c,:))) && norm(detectedR - prevCentroidsR(c,:)) > maxMove
                centroidsR(c,:) = prevCentroidsR(c,:);
            else
                centroidsR(c,:) = correct(kalmanR{c}, detectedR);
            end
        else
            centroidsR(c,:) = predict(kalmanR{c});
        end

        prevCentroidsL(c,:) = centroidsL(c,:);
        prevCentroidsR(c,:) = centroidsR(c,:);

        % Triangulate 3D
        if all(~isnan([centroidsL(c,:) centroidsR(c,:)]))
            point3D = triangulate(centroidsL(c,:), centroidsR(c,:), stereoParams);
            positions3D{c} = [positions3D{c}; point3D];
        else
            positions3D{c} = [positions3D{c}; [NaN NaN NaN]];
        end
    end

    %% --- Visualization with red boxes around markers ---
    if ishandle(hFig)
        rectSize = 10; % size of red rectangle

        % Left camera
        subplot(1,3,1); imshow(leftRect); hold on;
        for c = 1:numColors
            if all(~isnan(centroidsL(c,:)))
                plot(centroidsL(c,1), centroidsL(c,2), '*', 'Color', RGBList(c,:), 'MarkerSize', 10, 'LineWidth',1.5);
                rectangle('Position',[centroidsL(c,1)-rectSize/2, centroidsL(c,2)-rectSize/2, rectSize, rectSize], ...
                          'EdgeColor','r','LineWidth',2);
            end
        end
        title('Left Camera'); hold off;

        % Right camera
        subplot(1,3,2); imshow(rightRect); hold on;
        for c = 1:numColors
            if all(~isnan(centroidsR(c,:)))
                plot(centroidsR(c,1), centroidsR(c,2), '*', 'Color', RGBList(c,:), 'MarkerSize', 10, 'LineWidth',1.5);
                rectangle('Position',[centroidsR(c,1)-rectSize/2, centroidsR(c,2)-rectSize/2, rectSize, rectSize], ...
                          'EdgeColor','r','LineWidth',2);
            end
        end
        title('Right Camera'); hold off;

        % 3D Trajectories
        subplot(1,3,3); hold on;
        for c = 1:numColors
            pts = positions3D{c};
            valid = ~any(isnan(pts),2);
            plot3(pts(valid,1), pts(valid,2), pts(valid,3), '.-', 'Color', RGBList(c,:), 'LineWidth', 1.5);
        end
        xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
        grid on; axis equal;
        title('3D Trajectories'); view(0,-90); hold off;

        drawnow;
    end
end

%% --- Cleanup and save trajectories ---
clear camL camR;

% Close video writers
close(vwLeft);
close(vwRight);
disp(['Saved left video to Camera1: ' leftVideoFile]);
disp(['Saved right video to Camera2: ' rightVideoFile]);

% Save trajectories
for c = 1:numColors
    traj = positions3D{c};
    h = HSVList(c,1);
    s = HSVList(c,2);
    v = HSVList(c,3);
    hStr = sprintf('%04d', round(h*1000));
    sStr = sprintf('%04d', round(s*1000));
    vStr = sprintf('%04d', round(v*1000));
    filename = sprintf('trajectory_H%s_S%s_V%s_%s.mat', hStr, sStr, vStr, timestamp);
    filepath = fullfile('trajectories', filename);
    save(filepath, 'traj');
    fprintf('Saved trajectory for color HSV [%.3f %.3f %.3f] to %s\n', h, s, v, filepath);
end
%%
figure
hold on
for c = 1:numColors
    pts = positions3D{c};
    valid = ~any(isnan(pts),2);
    plot3(pts(valid,1), pts(valid,3),  pts(valid,2), '.-', 'Color', RGBList(c,:), 'LineWidth', 1.5);
end
xlabel('X (mm)'); ylabel('Z (mm)'); zlabel('Y (mm)');
grid on; axis equal; hold off
title('3D Trajectories');  view(20,30);hold off;