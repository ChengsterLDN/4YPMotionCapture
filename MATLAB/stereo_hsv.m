%% Stereo Object Tracker (ROI-based HSV Color Tracking)
clc; close all;

%% --- Load stereo calibration parameters ---
S = load('stereoParams.mat');          % Load .mat file
stereoParams = S.stereoParams;         % Extract stereoParameters object

%% --- Select stereo videos ---
[leftFile, leftPath] = uigetfile({'*.mp4;*.avi'}, 'Select LEFT camera video');
[rightFile, rightPath] = uigetfile({'*.mp4;*.avi'}, 'Select RIGHT camera video');
if isequal(leftFile,0) || isequal(rightFile,0)
    error('No video selected. Operation cancelled.');
end

leftVid  = VideoReader(fullfile(leftPath, leftFile));
rightVid = VideoReader(fullfile(rightPath, rightFile));

%% --- Read first frame to pick color region ---
leftFrame  = readFrame(leftVid);
rightFrame = readFrame(rightVid);

figure('Name', 'Select ROI for Object Color', 'Position', [200 200 1000 400]);
subplot(1,2,1);
imshow(leftFrame);
title('LEFT Frame - Draw ROI around object');

subplot(1,2,2);
imshow(rightFrame);
title('RIGHT Frame');

disp('👉 Draw a rectangle around the object in the LEFT frame, then double-click or press Enter.');

subplot(1,2,1);
roi = drawrectangle('Color', 'r', 'LineWidth', 1.5);
wait(roi);
roiPos = round(roi.Position);  % [x y w h]
roiCrop = imcrop(leftFrame, roiPos);

if isempty(roiCrop)
    error('No ROI selected. Try again.');
end

%% --- Compute HSV mean and std from ROI ---
hsvROI = rgb2hsv(im2double(roiCrop));
H = hsvROI(:,:,1); S = hsvROI(:,:,2); V = hsvROI(:,:,3);

hMean = mean(H(:)); sMean = mean(S(:)); vMean = mean(V(:));
hStd  = std(H(:));  sStd  = std(S(:));  vStd  = std(V(:));

scale = 1.5; % tolerance multiplier

hMin = max(0, hMean - scale*hStd); hMax = min(1, hMean + scale*hStd);
sMin = max(0, sMean - scale*sStd); sMax = min(1, sMean + scale*sStd);
vMin = max(0, vMean - scale*vStd); vMax = min(1, vMean + scale*vStd);

fprintf('Selected HSV mean ± %.1fxSTD:\n', scale);
fprintf('H: %.2f ± %.2f  → [%.2f %.2f]\n', hMean, scale*hStd, hMin, hMax);
fprintf('S: %.2f ± %.2f  → [%.2f %.2f]\n', sMean, scale*sStd, sMin, sMax);
fprintf('V: %.2f ± %.2f  → [%.2f %.2f]\n', vMean, scale*vStd, vMin, vMax);

%% --- Reset videos to beginning ---
leftVid.CurrentTime = 0;
rightVid.CurrentTime = 0;

%% --- Initialize storage ---
positions3D = [];
frameIdx = 0;

%% --- Process frames ---
figure('Name', 'Stereo HSV Color Tracker', 'Position', [100 100 1400 600]);

while hasFrame(leftVid) && hasFrame(rightVid)
    frameIdx = frameIdx + 1;
    leftFrame  = readFrame(leftVid);
    rightFrame = readFrame(rightVid);

    % --- Rectify stereo images ---
    [leftRect, rightRect] = rectifyStereoImages(leftFrame, rightFrame, stereoParams);
    
    % --- Detect object in left image ---
    hsvLeft = rgb2hsv(im2double(leftRect));
    H = hsvLeft(:,:,1); S = hsvLeft(:,:,2); V = hsvLeft(:,:,3);
    maskLeft = (H>=hMin & H<=hMax) & (S>=sMin & S<=sMax) & (V>=vMin & V<=vMax);
    maskLeft = imopen(maskLeft, strel('disk',5));
    maskLeft = imclose(maskLeft, strel('disk',10));
    maskLeft = imfill(maskLeft, 'holes');
    statsLeft = regionprops(maskLeft, 'Centroid', 'Area');
    if ~isempty(statsLeft)
        [~, idxMax] = max([statsLeft.Area]);
        centroidLeft = statsLeft(idxMax).Centroid;
    else
        centroidLeft = [NaN NaN];
    end

    % --- Detect object in right image ---
    hsvRight = rgb2hsv(im2double(rightRect));
    H = hsvRight(:,:,1); S = hsvRight(:,:,2); V = hsvRight(:,:,3);
    maskRight = (H>=hMin & H<=hMax) & (S>=sMin & S<=sMax) & (V>=vMin & V<=vMax);
    maskRight = imopen(maskRight, strel('disk',5));
    maskRight = imclose(maskRight, strel('disk',10));
    maskRight = imfill(maskRight, 'holes');
    statsRight = regionprops(maskRight, 'Centroid', 'Area');
    if ~isempty(statsRight)
        [~, idxMax] = max([statsRight.Area]);
        centroidRight = statsRight(idxMax).Centroid;
    else
        centroidRight = [NaN NaN];
    end

    % --- Triangulate 3D position ---
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
        plot(centroidLeft(1), centroidLeft(2),'r*','MarkerSize',10);
    end
    title(sprintf('Left Frame %d', frameIdx));
    hold off;

    subplot(1,3,2);
    imshow(rightRect); hold on;
    if all(~isnan(centroidRight))
        plot(centroidRight(1), centroidRight(2),'r*','MarkerSize',10);
    end
    title(sprintf('Right Frame %d', frameIdx));
    hold off;

    % --- Plot 3D Trajectory ---
    subplot(1,3,3);
    validPoints = ~any(isnan(positions3D),2);
    plot3(positions3D(validPoints,1), positions3D(validPoints,2), positions3D(validPoints,3), 'm.-','LineWidth',1.5);
    axis equal
    xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
    grid on; title('3D Trajectory'); axis equal;
    view(0,-90)

    drawnow;
end

disp('✅ Tracking complete! 3D positions stored in variable "positions3D".');
