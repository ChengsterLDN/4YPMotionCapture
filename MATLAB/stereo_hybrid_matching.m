%% Hybrid Stereo Tracker (Color + Features + KLT)
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

%% --- Read first frame and rectify ---
leftFrame  = readFrame(leftVid);
rightFrame = readFrame(rightVid);
[leftRect0, rightRect0] = rectifyStereoImages(leftFrame, rightFrame, stereoParams);

%% --- Select ROI for initialization ---
figure('Name','Hybrid Tracker Initialization','Position',[200 200 1000 450]);
subplot(1,2,1);
imshow(leftRect0);
title('Left Rectified Frame - Draw ROI around object');
hRect = drawrectangle('Color','r','LineWidth',1.5);
wait(hRect);
roi = round(hRect.Position);
close(gcf);

%% --- Compute color model (HSV range) ---
roiCrop = imcrop(leftRect0, roi);
hsvCrop = rgb2hsv(im2double(roiCrop));
h = hsvCrop(:,:,1); s = hsvCrop(:,:,2); v = hsvCrop(:,:,3);
hMin = max(0, mean(h(:)) - 0.05); hMax = min(1, mean(h(:)) + 0.05);
sMin = max(0, mean(s(:)) - 0.3); sMax = min(1, mean(s(:)) + 0.3);
vMin = max(0, mean(v(:)) - 0.3); vMax = min(1, mean(v(:)) + 0.3);

%% --- Detect initial features within ROI (or color mask) ---
grayL = rgb2gray(leftRect0);
hsvL  = rgb2hsv(im2double(leftRect0));
maskColor = (hsvL(:,:,1)>=hMin & hsvL(:,:,1)<=hMax & ...
             hsvL(:,:,2)>=sMin & hsvL(:,:,2)<=sMax & ...
             hsvL(:,:,3)>=vMin & hsvL(:,:,3)<=vMax);
maskColor = imopen(maskColor, strel('disk',3));
maskColor = imclose(maskColor, strel('disk',5));

pointsL = detectFASTFeatures(grayL, 'ROI', roi);
if pointsL.Count < 5
    % fallback: detect features within color mask
    [y,x] = find(maskColor);
    if isempty(x)
        error('Could not find object region.');
    end
    maskROI = [min(x) min(y) max(x)-min(x) max(y)-min(y)];
    pointsL = detectHarrisFeatures(grayL, 'ROI', maskROI);
end
pointsL = selectStrongest(pointsL, 100);

if pointsL.Count < 5
    error('Still not enough features detected — try better lighting or a different area.');
end

%% --- Find corresponding points in right rectified frame ---
grayR = rgb2gray(rightRect0);
[featuresL, validPointsL] = extractFeatures(grayL, pointsL);
pointsR = detectFASTFeatures(grayR);
[featuresR, validPointsR] = extractFeatures(grayR, pointsR);
idxPairs = matchFeatures(featuresL, featuresR, 'Unique', true, 'MatchThreshold', 40);
matchedPointsL = validPointsL(idxPairs(:,1));
matchedPointsR = validPointsR(idxPairs(:,2));

if matchedPointsL.Count < 5
    warning('Few stereo correspondences found; triangulation may be inaccurate.');
end

%% --- Initialize KLT trackers ---
trackerL = vision.PointTracker('MaxBidirectionalError', 1);
initialize(trackerL, matchedPointsL.Location, leftRect0);

trackerR = vision.PointTracker('MaxBidirectionalError', 1);
initialize(trackerR, matchedPointsR.Location, rightRect0);

%% --- Tracking loop ---
positions3D = [];
frameIdx = 0;
figure('Name','Hybrid Stereo Tracker','Position',[100 100 1400 500]);

while hasFrame(leftVid) && hasFrame(rightVid)
    frameIdx = frameIdx + 1;
    leftFrame  = readFrame(leftVid);
    rightFrame = readFrame(rightVid);
    [leftRect, rightRect] = rectifyStereoImages(leftFrame, rightFrame, stereoParams);

    [pointsL, validL] = trackerL(leftRect);
    [pointsR, validR] = trackerR(rightRect);
    valid = validL & validR;

    % --- If too few valid points, reinitialize using color mask ---
    if nnz(valid) < 8
        hsvL = rgb2hsv(im2double(leftRect));
        maskColor = (hsvL(:,:,1)>=hMin & hsvL(:,:,1)<=hMax & ...
                     hsvL(:,:,2)>=sMin & hsvL(:,:,2)<=sMax & ...
                     hsvL(:,:,3)>=vMin & hsvL(:,:,3)<=vMax);
        maskColor = imopen(maskColor, strel('disk',3));
        maskColor = imclose(maskColor, strel('disk',5));
        stats = regionprops(maskColor, 'BoundingBox');
        if ~isempty(stats)
            roiColor = round(stats(1).BoundingBox);
            grayL = rgb2gray(leftRect);
            grayR = rgb2gray(rightRect);
            newPtsL = detectFASTFeatures(grayL, 'ROI', roiColor);
            newPtsL = selectStrongest(newPtsL, 50);
            [fL,vL] = extractFeatures(grayL, newPtsL);
            newPtsR = detectFASTFeatures(grayR);
            [fR,vR] = extractFeatures(grayR, newPtsR);
            idx = matchFeatures(fL,fR,'Unique',true,'MatchThreshold',50);
            if size(idx,1) >= 5
                release(trackerL);
                release(trackerR);
                initialize(trackerL, vL(idx(:,1)).Location, leftRect);
                initialize(trackerR, vR(idx(:,2)).Location, rightRect);
                [pointsL,validL] = trackerL(leftRect);
                [pointsR,validR] = trackerR(rightRect);
                valid = validL & validR;
            end
        end
    end

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
    plot(pointsL(valid,1), pointsL(valid,2),'go','MarkerSize',4);
    title(sprintf('Left Frame %d', frameIdx));
    hold off;

    subplot(1,3,2);
    imshow(rightRect); hold on;
    plot(pointsR(valid,1), pointsR(valid,2),'go','MarkerSize',4);
    title(sprintf('Right Frame %d', frameIdx));
    hold off;

    subplot(1,3,3);
    validPts = ~any(isnan(positions3D),2);
    plot3(positions3D(validPts,1), positions3D(validPts,2), positions3D(validPts,3),'m.-','LineWidth',1.5);
    xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
    title('3D Trajectory'); grid on; axis equal;
    view(0,-90);
    drawnow;
end

disp('✅ Hybrid tracking finished.');
