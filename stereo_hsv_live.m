%% Stereo HSV-Based Live Object Tracker (Single Pixel Selection)
clc; close all; clear;

%% --- Load stereo calibration parameters ---
S = load('stereoParams.mat');
stereoParams = S.stereoParams;
calibSize = stereoParams.CameraParameters1.ImageSize;

%% --- Connect to two webcams ---
camList = webcamlist;
if numel(camList) < 2
    error('Two webcams are required for stereo tracking.');
end

fprintf('Available webcams:\n');
disp(camList);

% Auto-connect to first two webcams
camL = webcam(1);
camR = webcam(3);

% Try setting resolution to calibration size
try
    resStr = sprintf('%dx%d', calibSize(2), calibSize(1));
    camL.Resolution = resStr;
    camR.Resolution = resStr;
catch
    warning('Could not set resolution directly. Will resize instead.');
end

%% --- Capture first frame for color selection ---
leftFrame  = snapshot(camL);
rightFrame = snapshot(camR);

% Resize to calibration size (for consistency)
leftFrame  = imresize(leftFrame,  [calibSize(1), calibSize(2)]);
rightFrame = imresize(rightFrame, [calibSize(1), calibSize(2)]);

%% --- User selects pixel on left image ---
figure('Name','Select Pixel to Track','Position',[100 100 1000 400]);
subplot(1,2,1);
imshow(leftFrame);
title('Click on object color to track');
[x, y] = ginput(1);
x = round(x); y = round(y);
subplot(1,2,2);
imshow(rightFrame);
title('Right Camera Preview');

% Get RGB at clicked pixel
rgbPixel = impixel(leftFrame, x, y);
if isempty(rgbPixel)
    error('No pixel selected.');
end
selectedColor = rgbPixel(1,:);
selectedHSV = rgb2hsv(double(selectedColor)/255);

fprintf('Selected RGB: [%.0f %.0f %.0f]\n', selectedColor);
fprintf('Selected HSV: [%.3f %.3f %.3f]\n', selectedHSV);

% --- Define HSV tolerance range ---
hTol = 0.04; sTol = 0.3; vTol = 0.35;
hMin = max(0, selectedHSV(1)-hTol); hMax = min(1, selectedHSV(1)+hTol);
sMin = max(0, selectedHSV(2)-sTol); sMax = min(1, selectedHSV(2)+sTol);
vMin = max(0, selectedHSV(3)-vTol); vMax = min(1, selectedHSV(3)+vTol);

fprintf('Tracking HSV range:\nH: [%.2f %.2f], S: [%.2f %.2f], V: [%.2f %.2f]\n',...
    hMin,hMax,sMin,sMax,vMin,vMax);

%% --- Initialize plotting ---
figure('Name','Live Stereo HSV Tracker','Position',[100 100 1400 500]);
positions3D = [];
frameIdx = 0;

%% --- Live tracking loop ---
disp('Press Ctrl+C to stop.');
while true
    % Capture frames
    leftFrame  = snapshot(camL);
    rightFrame = snapshot(camR);

    % Resize to calibration image size
    leftFrame  = imresize(leftFrame,  [calibSize(1), calibSize(2)]);
    rightFrame = imresize(rightFrame, [calibSize(1), calibSize(2)]);

    % Rectify stereo images
    [leftRect, rightRect] = rectifyStereoImages(leftFrame, rightFrame, stereoParams);

    % Convert to HSV
    hsvLeft = rgb2hsv(im2double(leftRect));
    hsvRight = rgb2hsv(im2double(rightRect));

    % Color mask thresholds
    maskLeft = (hsvLeft(:,:,1)>=hMin & hsvLeft(:,:,1)<=hMax) & ...
               (hsvLeft(:,:,2)>=sMin & hsvLeft(:,:,2)<=sMax) & ...
               (hsvLeft(:,:,3)>=vMin & hsvLeft(:,:,3)<=vMax);
    maskRight = (hsvRight(:,:,1)>=hMin & hsvRight(:,:,1)<=hMax) & ...
                (hsvRight(:,:,2)>=sMin & hsvRight(:,:,2)<=sMax) & ...
                (hsvRight(:,:,3)>=vMin & hsvRight(:,:,3)<=vMax);

    % Morphological cleanup
    se = strel('disk',5);
    maskLeft = imclose(imopen(maskLeft,se),se);
    maskRight = imclose(imopen(maskRight,se),se);
    maskLeft = imfill(maskLeft,'holes');
    maskRight = imfill(maskRight,'holes');

    % Find centroids of largest blobs
    statsL = regionprops(maskLeft,'Centroid','Area');
    statsR = regionprops(maskRight,'Centroid','Area');

    if ~isempty(statsL)
        [~, idxL] = max([statsL.Area]);
        centroidLeft = statsL(idxL).Centroid;
    else
        centroidLeft = [NaN NaN];
    end

    if ~isempty(statsR)
        [~, idxR] = max([statsR.Area]);
        centroidRight = statsR(idxR).Centroid;
    else
        centroidRight = [NaN NaN];
    end

    % Triangulate 3D point
    if all(~isnan([centroidLeft centroidRight]))
        point3D = triangulate(centroidLeft, centroidRight, stereoParams);
        positions3D = [positions3D; point3D];
    else
        positions3D = [positions3D; [NaN NaN NaN]];
    end

    % --- Visualization ---
    subplot(1,3,1);
    imshow(leftRect); hold on;
    if all(~isnan(centroidLeft))
        plot(centroidLeft(1), centroidLeft(2), 'r*', 'MarkerSize', 10);
    end
    title('Left Camera'); hold off;

    subplot(1,3,2);
    imshow(rightRect); hold on;
    if all(~isnan(centroidRight))
        plot(centroidRight(1), centroidRight(2), 'r*', 'MarkerSize', 10);
    end
    title('Right Camera'); hold off;

    subplot(1,3,3);
    valid = ~any(isnan(positions3D),2);
    plot3(positions3D(valid,1), positions3D(valid,2), positions3D(valid,3), 'm.-', 'LineWidth', 1.5);
    xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
    grid on; axis equal;
    title('3D Trajectory'); view(0,-90);

    drawnow;
    frameIdx = frameIdx + 1;
end

%% --- Cleanup on stop ---
clear camL camR;
disp('Stopped and cleaned up webcams.');
