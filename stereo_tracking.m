%% Stereo Object Tracker with Interactive Color Selection
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

%% --- Read first frame to pick color ---
leftFrame  = readFrame(leftVid);
rightFrame = readFrame(rightVid);

figure('Name', 'Select Object Color', 'Position', [200 200 1000 400]);
subplot(1,2,1);
imshow(leftFrame);
title('Left Frame - Click on object color to track');

subplot(1,2,2);
imshow(rightFrame);
title('Right Frame');

disp('👉 Click on the object color you want to track in the LEFT frame...');
[x, y] = ginput(1);

% Get RGB at clicked point
rgbPixel = impixel(leftFrame, round(x), round(y));
if isempty(rgbPixel)
    error('No pixel selected. Try clicking on the object.');
end
selectedColor = rgbPixel(1,:);  % Ensure 1x3 vector
selectedHSV = rgb2hsv(double(selectedColor)/255);

fprintf('Selected RGB: [%.0f %.0f %.0f]\n', selectedColor);
fprintf('Selected HSV: [%.2f %.2f %.2f]\n', selectedHSV);

% --- Define HSV tolerance range ---
hTol = 0.05; sTol = 0.3; vTol = 0.3;
hMin = max(0, selectedHSV(1)-hTol); hMax = min(1, selectedHSV(1)+hTol);
sMin = max(0, selectedHSV(2)-sTol); sMax = min(1, selectedHSV(2)+sTol);
vMin = max(0, selectedHSV(3)-vTol); vMax = min(1, selectedHSV(3)+vTol);

fprintf('Tracking HSV range:\nH: [%.2f %.2f], S: [%.2f %.2f], V: [%.2f %.2f]\n',...
    hMin,hMax,sMin,sMax,vMin,vMax);

%% --- Reset videos to beginning ---
leftVid.CurrentTime = 0;
rightVid.CurrentTime = 0;

%% --- Initialize storage ---
positions3D = [];
frameIdx = 0;

%% --- Process frames ---
while hasFrame(leftVid) && hasFrame(rightVid)
    frameIdx = frameIdx + 1;
    leftFrame  = readFrame(leftVid);
    rightFrame = readFrame(rightVid);

    % --- Rectify stereo images ---
    [leftRect, rightRect] = rectifyStereoImages(leftFrame, rightFrame, stereoParams);
    
    % --- Display the rectified frames ---
    %figure('Name', 'Rectified Frames', 'Position', [100 100 1200 600]);
    %imshow(stereoAnaglyph(leftRect, rightRect));
    %title("Rectified Video Frames");
    %drawnow
    

    % --- Detect object in left image ---
    hsvLeft = rgb2hsv(leftRect);
    H = hsvLeft(:,:,1); S = hsvLeft(:,:,2); V = hsvLeft(:,:,3);
    maskLeft = (H>=hMin & H<=hMax) & (S>=sMin & S<=sMax) & (V>=vMin & V<=vMax);
    maskLeft = imopen(maskLeft, strel('disk',5));
    maskLeft = imclose(maskLeft, strel('disk',10));
    maskLeft = imfill(maskLeft, 'holes');
    statsLeft = regionprops(maskLeft, 'Centroid');
    if ~isempty(statsLeft)
        centroidLeft = statsLeft(1).Centroid;
    else
        centroidLeft = [NaN NaN];
    end

    % --- Detect object in right image ---
    hsvRight = rgb2hsv(rightRect);
    H = hsvRight(:,:,1); S = hsvRight(:,:,2); V = hsvRight(:,:,3);
    maskRight = (H>=hMin & H<=hMax) & (S>=sMin & S<=sMax) & (V>=vMin & V<=vMax);
    maskRight = imopen(maskRight, strel('disk',5));
    maskRight = imclose(maskRight, strel('disk',10));
    maskRight = imfill(maskRight, 'holes');
    statsRight = regionprops(maskRight, 'Centroid');
    if ~isempty(statsRight)
        centroidRight = statsRight(1).Centroid;
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
    title(sprintf('Left Rectified Frame %d', frameIdx));
    hold off;

    subplot(1,3,2);
    imshow(rightRect); hold on;
    if all(~isnan(centroidRight))
        plot(centroidRight(1), centroidRight(2),'r*','MarkerSize',10);
    end
    title(sprintf('Right Rectified Frame %d', frameIdx));
    hold off;
    % --- Plot 3D Trajectory ---
    subplot(1,3,3);
    validPoints = ~any(isnan(positions3D),2);
    plot3(positions3D(validPoints,1), positions3D(validPoints,2), positions3D(validPoints,3),'m.-','LineWidth',1.5);
    axis equal
    xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
    grid on; title('3D Trajectory of Tracked Object'); axis equal;
    view(0,-90)
    
    drawnow;
end
