%% Stereo HSV-Based Live Tracker for Multiple Colors
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

camL = webcam(1); % Webcam number: You can find this by typing webcamlist in the MATLAB command window
camR = webcam(3); % Webcam number: You can find this by typing webcamlist in the MATLAB command window

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

% Get HSV for each selected pixel
HSVList = zeros(numColors,3);
RGBList = zeros(numColors,3);
for i = 1:numColors
    rgb = impixel(leftFrame, xList(i), yList(i));
    HSVList(i,:) = rgb2hsv(double(rgb)/255);
    RGBList(i,:) = double(rgb)/255;  % For plotting
    % --- Visual feedback ---
    hold on;
    plot(xList(i), yList(i), '*', 'Color', RGBList(i,:), 'MarkerSize', 15, 'LineWidth',2);
    fprintf('Color %d selected at (%d,%d): RGB=[%.0f %.0f %.0f], HSV=[%.3f %.3f %.3f]\n', ...
        i, xList(i), yList(i), rgb(1), rgb(2), rgb(3), HSVList(i,1), HSVList(i,2), HSVList(i,3));
end
hold off;

% Fixed tolerances
hTol = 0.04; sTol = 0.3; vTol = 0.35;

% Compute HSV ranges for each color
hMinList = max(0, HSVList(:,1)-hTol);
hMaxList = min(1, HSVList(:,1)+hTol);
sMinList = max(0, HSVList(:,2)-sTol);
sMaxList = min(1, HSVList(:,2)+sTol);
vMinList = max(0, HSVList(:,3)-vTol);
vMaxList = min(1, HSVList(:,3)+vTol);

fprintf('Tracking %d colors with fixed HSV tolerance.\n', numColors);

%% --- Initialize plotting ---
figure('Name','Live Stereo Multi-Color Tracker','Position',[100 100 1400 500]);
positions3D = cell(numColors,1);
frameIdx = 0;

%% --- Live tracking loop ---
disp('Press Ctrl+C to stop.');
while true
    frameIdx = frameIdx + 1;

    % Capture frames
    leftFrame  = snapshot(camL);
    rightFrame = snapshot(camR);

    % Resize
    leftFrame  = imresize(leftFrame, [calibSize(1), calibSize(2)]);
    rightFrame = imresize(rightFrame, [calibSize(1), calibSize(2)]);

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

        if ~isempty(statsL)
            [~, idxL] = max([statsL.Area]);
            centroidsL(c,:) = statsL(idxL).Centroid;
        end
        if ~isempty(statsR)
            [~, idxR] = max([statsR.Area]);
            centroidsR(c,:) = statsR(idxR).Centroid;
        end

        % Store 3D positions
        if all(~isnan([centroidsL(c,:) centroidsR(c,:)]))
            point3D = triangulate(centroidsL(c,:), centroidsR(c,:), stereoParams);
            positions3D{c} = [positions3D{c}; point3D];
        else
            positions3D{c} = [positions3D{c}; [NaN NaN NaN]];
        end
    end

    % --- Visualization ---
    subplot(1,3,1); imshow(leftRect); hold on;
    for c = 1:numColors
        if all(~isnan(centroidsL(c,:)))
            plot(centroidsL(c,1), centroidsL(c,2), '*', 'Color', RGBList(c,:), 'MarkerSize', 10, 'LineWidth',1.5);
        end
    end
    title('Left Camera'); hold off;

    subplot(1,3,2); imshow(rightRect); hold on;
    for c = 1:numColors
        if all(~isnan(centroidsR(c,:)))
            plot(centroidsR(c,1), centroidsR(c,2), '*', 'Color', RGBList(c,:), 'MarkerSize', 10, 'LineWidth',1.5);
        end
    end
    title('Right Camera'); hold off;

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

%% --- Cleanup ---
clear camL camR;
disp('Stopped and released webcams.');
