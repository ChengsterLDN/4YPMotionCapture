%% Stereo HSV-Based Live Tracker (Median Smoothing, No Kalman)
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

%% --- Select multiple colors ---
figure('Name','Select Colors to Track','Position',[100 100 1000 400]);
imshow(leftFrame);
title('Click on each object color to track. Press Enter when done.');

[xList, yList] = ginput();
xList = round(xList); yList = round(yList);
numColors = length(xList);
if numColors == 0, error('No colors selected.'); end

HSVList = zeros(numColors,3);
RGBList = zeros(numColors,3);
for i = 1:numColors
    rgb = impixel(leftFrame, xList(i), yList(i));
    HSVList(i,:) = rgb2hsv(double(rgb)/255);
    RGBList(i,:) = double(rgb)/255;
    hold on;
    plot(xList(i), yList(i), '*', 'Color', RGBList(i,:), 'MarkerSize', 15, 'LineWidth',2);
    fprintf('✅ Color %d selected at (%d,%d): RGB=[%.0f %.0f %.0f], HSV=[%.3f %.3f %.3f]\n', ...
        i, xList(i), yList(i), rgb(1), rgb(2), rgb(3), HSVList(i,1), HSVList(i,2), HSVList(i,3));
end
hold off;

%% --- HSV Tolerance ---
hTol = 0.04; sTol = 0.3; vTol = 0.5;
hMinList = max(0, HSVList(:,1)-hTol);
hMaxList = min(1, HSVList(:,1)+hTol);
sMinList = max(0, HSVList(:,2)-sTol);
sMaxList = min(1, HSVList(:,2)+sTol);
vMinList = max(0, HSVList(:,3)-vTol);
vMaxList = min(1, HSVList(:,3)+vTol);

%% --- Initialize ---
positions3D = cell(numColors,1);
recentCentroidsL = cell(numColors,1);
recentCentroidsR = cell(numColors,1);
recent3D = cell(numColors,1);
windowSize = 5; % median filter window (frames)

%% --- Setup folders and video writers ---
if ~exist('Camera1Video','dir'), mkdir('Camera1Video'); end
if ~exist('Camera2Video','dir'), mkdir('Camera2Video'); end
if ~exist('trajectories','dir'), mkdir('trajectories'); end

timestamp = datestr(now,'yyyymmdd_HHMMSS');
leftVideoFile  = fullfile('Camera1Video', ['Cam1_' timestamp '.avi']);
rightVideoFile = fullfile('Camera2Video', ['Cam2_' timestamp '.avi']);

vwLeft = VideoWriter(leftVideoFile); vwLeft.FrameRate = 30; open(vwLeft);
vwRight = VideoWriter(rightVideoFile); vwRight.FrameRate = 30; open(vwRight);

%% --- Live Tracking ---
hFig = figure('Name','Stereo Multi-Color Tracker (Median Filter)','Position',[100 100 1400 500]);
disp('Press Ctrl+C or close the figure to stop.');

while true
    if ~ishandle(hFig), disp('Figure closed. Stopping...'); break; end

    % Capture
    leftFrame  = snapshot(camL);
    rightFrame = snapshot(camR);
    leftFrame  = imresize(leftFrame, [calibSize(1), calibSize(2)]);
    rightFrame = imresize(rightFrame, [calibSize(1), calibSize(2)]);
    writeVideo(vwLeft, leftFrame);
    writeVideo(vwRight, rightFrame);

    % Rectify
    [leftRect, rightRect] = rectifyStereoImages(leftFrame, rightFrame, stereoParams);
    hsvLeft  = rgb2hsv(im2double(leftRect));
    hsvRight = rgb2hsv(im2double(rightRect));

    centroidsL = nan(numColors,2);
    centroidsR = nan(numColors,2);

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

        if ~isempty(statsL)
            [~, idxL] = max([statsL.Area]);
            centroidsL(c,:) = statsL(idxL).Centroid;
        end
        if ~isempty(statsR)
            [~, idxR] = max([statsR.Area]);
            centroidsR(c,:) = statsR(idxR).Centroid;
        end

        % Triangulate if both found
        if all(~isnan([centroidsL(c,:) centroidsR(c,:)]))
            point3D = triangulate(centroidsL(c,:), centroidsR(c,:), stereoParams);

            % --- Store history for median filtering ---
            recentCentroidsL{c} = [recentCentroidsL{c}; centroidsL(c,:)];
            recentCentroidsR{c} = [recentCentroidsR{c}; centroidsR(c,:)];
            recent3D{c} = [recent3D{c}; point3D];

            % Limit window size
            if size(recent3D{c},1) > windowSize
                recentCentroidsL{c}(1,:) = [];
                recentCentroidsR{c}(1,:) = [];
                recent3D{c}(1,:) = [];
            end

            % --- Median filtering ---
            smoothL = median(recentCentroidsL{c},1);
            smoothR = median(recentCentroidsR{c},1);
            smooth3D = median(recent3D{c},1);

            positions3D{c} = [positions3D{c}; smooth3D];
            centroidsL(c,:) = smoothL;
            centroidsR(c,:) = smoothR;
        else
            positions3D{c} = [positions3D{c}; [NaN NaN NaN]];
        end
    end

    %% --- Visualization ---
    rectSize = 10;

    subplot(1,3,1); imshow(leftRect); hold on;
    for c = 1:numColors
        if all(~isnan(centroidsL(c,:)))
            plot(centroidsL(c,1), centroidsL(c,2), '*', 'Color', RGBList(c,:), 'MarkerSize', 10, 'LineWidth', 1.5);
            rectangle('Position',[centroidsL(c,1)-rectSize/2, centroidsL(c,2)-rectSize/2, rectSize, rectSize], ...
                      'EdgeColor','r','LineWidth',2);
        end
    end
    title('Left Camera'); hold off;

    subplot(1,3,2); imshow(rightRect); hold on;
    for c = 1:numColors
        if all(~isnan(centroidsR(c,:)))
            plot(centroidsR(c,1), centroidsR(c,2), '*', 'Color', RGBList(c,:), 'MarkerSize', 10, 'LineWidth', 1.5);
            rectangle('Position',[centroidsR(c,1)-rectSize/2, centroidsR(c,2)-rectSize/2, rectSize, rectSize], ...
                      'EdgeColor','r','LineWidth',2);
        end
    end
    title('Right Camera'); hold off;

    subplot(1,3,3); hold on;
    for c = 1:numColors
        pts = positions3D{c};
        valid = ~any(isnan(pts),2);
        plot3(pts(valid,1), pts(valid,3), pts(valid,2), '.-', 'Color', RGBList(c,:), 'LineWidth', 1.5);
    end
    xlabel('X (mm)'); ylabel('Z (mm)'); zlabel('Y (mm)');
    grid on; title('3D Trajectories (Median Filter)'); view(20,30); hold off;

    drawnow limitrate;
end

%% --- Cleanup ---
clear camL camR;
close(vwLeft); close(vwRight);
disp(['Saved videos: ' leftVideoFile ' and ' rightVideoFile]);

for c = 1:numColors
    traj = positions3D{c};
    hStr = sprintf('%04d', round(HSVList(c,1)*1000));
    sStr = sprintf('%04d', round(HSVList(c,2)*1000));
    vStr = sprintf('%04d', round(HSVList(c,3)*1000));
    filename = sprintf('trajectory_H%s_S%s_V%s_%s.mat', hStr, sStr, vStr, timestamp);
    filepath = fullfile('trajectories', filename);
    save(filepath, 'traj');
    fprintf('💾 Saved trajectory HSV=[%.3f %.3f %.3f] → %s\n', HSVList(c,1), HSVList(c,2), HSVList(c,3), filepath);
end

%% Plot final trajectories
figure;
hold on;
for c = 1:numColors
    pts = positions3D{c};
    valid = ~any(isnan(pts),2);
    plot3(pts(valid,1), pts(valid,3), pts(valid,2), '.-', 'Color', RGBList(c,:), 'LineWidth', 1.5);
end
xlabel('X (mm)'); ylabel('Z (mm)'); zlabel('Y (mm)');
grid on; axis equal;
title('3D Trajectories (Smoothed)'); view(20,30);
hold off;
