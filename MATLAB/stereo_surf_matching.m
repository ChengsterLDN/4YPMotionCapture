%% Stereo Feature-Based Tracker (KLT + SURF)
clc; close all;

%% --- Load stereo calibration parameters ---
S = load('stereoParams.mat');
stereoParams = S.stereoParams;

%% --- Select stereo videos ---
[leftFile, leftPath] = uigetfile({'*.mp4;*.avi'}, 'Select LEFT camera video');
[rightFile, rightPath] = uigetfile({'*.mp4;*.avi'}, 'Select RIGHT camera video');
if isequal(leftFile,0) || isequal(rightFile,0)
    error('No video selected.');
end

leftVid  = VideoReader(fullfile(leftPath, leftFile));
rightVid = VideoReader(fullfile(rightPath, rightFile));

%% --- Read first frames and rectify ---
leftFrame  = readFrame(leftVid);
rightFrame = readFrame(rightVid);
[leftRect0, rightRect0] = rectifyStereoImages(leftFrame, rightFrame, stereoParams);

figure('Name', 'Select ROI for Tracking', 'Position', [200 200 1000 450]);
subplot(1,2,1);
imshow(leftRect0);
title('Left Rectified Frame - Draw ROI around object to track');
axis on;

subplot(1,2,2);
imshow(rightRect0);
title('Right Rectified Frame - For reference');
axis on;

disp('👉 Draw a rectangle around the object to track in the LEFT rectified image.');
subplot(1,2,1);
hRect = drawrectangle('Color','r','LineWidth',1.5);
wait(hRect);
roi = round(hRect.Position);
close(gcf);

%% --- Detect features within ROI (using SURF) ---
grayL = rgb2gray(leftRect0);
pointsL = detectSURFFeatures(grayL, 'ROI', roi);
pointsL = selectStrongest(pointsL, 100);

if pointsL.Count < 5
    error('Not enough features detected in selected ROI. Try a different area.');
end

% --- Initialize KLT trackers for left and right views ---
trackerL = vision.PointTracker('MaxBidirectionalError', 1);
initialize(trackerL, pointsL.Location, leftRect0);

% For right camera, find corresponding points in the first rectified pair
grayR = rgb2gray(rightRect0);
[featuresL, validPointsL] = extractFeatures(grayL, pointsL);
pointsR = detectSURFFeatures(grayR);
[featuresR, validPointsR] = extractFeatures(grayR, pointsR);
indexPairs = matchFeatures(featuresL, featuresR, 'Unique', true);

matchedPointsL = validPointsL(indexPairs(:,1));
matchedPointsR = validPointsR(indexPairs(:,2));

if matchedPointsL.Count < 5
    error('Could not find stereo correspondences in the first frame.');
end

% Initialize tracker for right view
trackerR = vision.PointTracker('MaxBidirectionalError', 1);
initialize(trackerR, matchedPointsR.Location, rightRect0);

%% --- Initialize storage ---
positions3D = [];
frameIdx = 0;

%% --- Process frames ---
figure('Name','Stereo Feature-Based Tracking','Position',[100 100 1400 500]);
while hasFrame(leftVid) && hasFrame(rightVid)
    frameIdx = frameIdx + 1;
    leftFrame  = readFrame(leftVid);
    rightFrame = readFrame(rightVid);

    [leftRect, rightRect] = rectifyStereoImages(leftFrame, rightFrame, stereoParams);

    % --- Track feature points ---
    [pointsL, validityL] = trackerL(leftRect);
    [pointsR, validityR] = trackerR(rightRect);
    valid = validityL & validityR;

    % --- Triangulate valid points ---
    if nnz(valid) >= 5
        worldPoints = triangulate(pointsL(valid,:), pointsR(valid,:), stereoParams);
        meanPos = mean(worldPoints,1,'omitnan');
        positions3D = [positions3D; meanPos];
    else
        positions3D = [positions3D; [NaN NaN NaN]];
    end

    % --- Visualization ---
    subplot(1,3,1);
    imshow(leftRect); hold on;
    plot(pointsL(valid,1), pointsL(valid,2), 'go', 'MarkerSize', 4);
    title(sprintf('Left Frame %d', frameIdx));
    hold off;

    subplot(1,3,2);
    imshow(rightRect); hold on;
    plot(pointsR(valid,1), pointsR(valid,2), 'go', 'MarkerSize', 4);
    title(sprintf('Right Frame %d', frameIdx));
    hold off;

    subplot(1,3,3);
    validPts = ~any(isnan(positions3D),2);
    plot3(positions3D(validPts,1), positions3D(validPts,2), positions3D(validPts,3),'m.-','LineWidth',1.5);
    xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
    title('3D Trajectory'); grid on; axis equal;
    view(0,-90);
    drawnow;

    % Optional: re-detect if too many points are lost
    if nnz(valid) < 10 && hasFrame(leftVid)
        release(trackerL);
        release(trackerR);
        grayL = rgb2gray(leftRect);
        grayR = rgb2gray(rightRect);
        newPtsL = detectSURFFeatures(grayL);
        newPtsR = detectSURFFeatures(grayR);
        [fL,vL] = extractFeatures(grayL,newPtsL);
        [fR,vR] = extractFeatures(grayR,newPtsR);
        idx = matchFeatures(fL,fR,'Unique',true);
        if size(idx,1) >= 5
            initialize(trackerL, vL(idx(:,1)).Location, leftRect);
            initialize(trackerR, vR(idx(:,2)).Location, rightRect);
        end
    end
end

disp('✅ Tracking finished.');
