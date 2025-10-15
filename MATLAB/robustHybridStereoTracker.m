function robustHybridStereoTracker()
% robustHybridStereoTracker
% Robust hybrid stereo tracker (color + features + KLT + periodic re-detect)
% Saves 3D trajectory in variable positions3D and displays live plots.
%
% Requirements: Computer Vision Toolbox recommended (vision.PointTracker,
% extractFeatures, matchFeatures). Uses VideoReader and stereoParameters.

clc; close all;

%% -------------------- User settings (tune these) -----------------------
minFeaturesToKeep = 10;    % minimum valid point pairs before re-init
targetNumFeatures  = 80;   % number of features to track ideally
reDetectEveryN     = 15;   % periodic re-detection interval (frames)
kltMaxBidirectErr  = 2;    % bidirectional error tolerance (pixels)
kltBlockSize       = [31 31]; % larger block size helps scale/rotation
colorHTol = 0.06; sTol = 0.35; vTol = 0.35; % color model tolerances
templateCorrThreshold = 0.5; % fallback template correlation threshold
maxReinitAttempts = 3;    % how many times to try reinit before giving up
resizeForMatch = 1.0;     % set <1 to speed up matching (e.g. 0.7)
% -----------------------------------------------------------------------

%% --- Load stereo calibration parameters ---
if ~exist('stereoParams.mat','file')
    error('stereoParams.mat not found in current folder. Run stereo calibration first.');
end
S = load('stereoParams.mat');
stereoParams = S.stereoParams;

%% --- Select stereo videos ---
[leftFile, leftPath] = uigetfile({'*.mp4;*.avi;*.mov'}, 'Select LEFT camera video');
[rightFile, rightPath] = uigetfile({'*.mp4;*.avi;*.mov'}, 'Select RIGHT camera video');
if isequal(leftFile,0) || isequal(rightFile,0)
    error('No video selected. Operation cancelled.');
end
leftVid  = VideoReader(fullfile(leftPath, leftFile));
rightVid = VideoReader(fullfile(rightPath, rightFile));

%% --- Read first frames and rectify ---
leftFrame  = readFrame(leftVid);
rightFrame = readFrame(rightVid);
[leftRect0, rightRect0] = rectifyStereoImages(leftFrame, rightFrame, stereoParams);

%% --- User selects ROI on left frame (object region) ---
hFig = figure('Name','Initialization: Draw ROI on LEFT rectified frame','Position',[200 200 1000 450]);
subplot(1,2,1); imshow(leftRect0); title('Left (draw ROI)'); axis on;
subplot(1,2,2); imshow(rightRect0); title('Right (for reference)'); axis on;
subplot(1,2,1); disp('Draw ROI on left frame and press Enter/double-click to confirm.');
hRect = drawrectangle('Color','r','LineWidth',1.5);
wait(hRect);
roi = round(hRect.Position); % [x y w h]
close(hFig);

% Crop and build color model (HSV mean +/- tol)
roiCrop = imcrop(leftRect0, roi);
if isempty(roiCrop)
    error('ROI cropping failed. Try again.');
end
hsvCrop = rgb2hsv(im2double(roiCrop));
hMean = mean(hsvCrop(:,:,1),'all'); sMean = mean(hsvCrop(:,:,2),'all'); vMean = mean(hsvCrop(:,:,3),'all');
hMin = max(0, hMean - colorHTol); hMax = min(1, hMean + colorHTol);
sMin = max(0, sMean - sTol); sMax = min(1, sMean + sTol);
vMin = max(0, vMean - vTol); vMax = min(1, vMean + vTol);

fprintf('Color model H:[%.3f %.3f] S:[%.3f %.3f] V:[%.3f %.3f]\n', hMin,hMax,sMin,sMax,vMin,vMax);

%% --- Initial feature detection (inside ROI or masked color region) ---
grayL0 = rgb2gray(leftRect0);
hsvL0 = rgb2hsv(im2double(leftRect0));
maskColor0 = (hsvL0(:,:,1)>=hMin & hsvL0(:,:,1)<=hMax) & ...
             (hsvL0(:,:,2)>=sMin & hsvL0(:,:,2)<=sMax) & ...
             (hsvL0(:,:,3)>=vMin & hsvL0(:,:,3)<=vMax);
maskColor0 = imopen(maskColor0, strel('disk',3));
maskColor0 = imclose(maskColor0, strel('disk',5));

% Prefer features inside ROI; fallback to color mask bounding box
pointsL = detectMinEigenFeatures(grayL0, 'ROI', roi);
if pointsL.Count < 8
    bw = maskColor0;
    props = regionprops(bw,'BoundingBox','Area');
    if ~isempty(props)
        % pick largest connected region
        areas = [props.Area]; [~, idxMax] = max(areas);
        bb = round(props(idxMax).BoundingBox);
        pointsL = detectMinEigenFeatures(grayL0, 'ROI', bb);
    end
end
pointsL = selectStrongest(pointsL, targetNumFeatures);
if pointsL.Count < 6
    error('Not enough initial features. Try selecting a different ROI or improving lighting.');
end

% Extract descriptors and find correspondences to right frame
grayR0 = rgb2gray(rightRect0);
[featuresL, validPointsL] = extractFeatures(grayL0, pointsL);
pointsR_all = detectMinEigenFeatures(grayR0);
[featuresR_all, validPointsR_all] = extractFeatures(grayR0, pointsR_all);
indexPairs = matchFeatures(featuresL, featuresR_all, 'Unique', true, 'MaxRatio', 0.8);
if isempty(indexPairs)
    error('Could not find stereo correspondences. Try a different ROI or different video pair.');
end
matchedPointsL = validPointsL(indexPairs(:,1));
matchedPointsR = validPointsR_all(indexPairs(:,2));
% Keep up to targetNumFeatures matched pairs
numKeep = min([matchedPointsL.Count, targetNumFeatures]);
matchedPointsL = matchedPointsL(1:numKeep);
matchedPointsR = matchedPointsR(1:numKeep);

fprintf('Initialized with %d matched point pairs.\n', matchedPointsL.Count);

%% --- Initialize KLT trackers ---
trackerL = vision.PointTracker('MaxBidirectionalError', kltMaxBidirectErr, 'BlockSize', kltBlockSize);
initialize(trackerL, matchedPointsL.Location, leftRect0);

trackerR = vision.PointTracker('MaxBidirectionalError', kltMaxBidirectErr, 'BlockSize', kltBlockSize);
initialize(trackerR, matchedPointsR.Location, rightRect0);

% Save initial template for fallback (left view)
templateLeft = imcrop(leftRect0, roi);
templateLeftGray = im2double(rgb2gray(templateLeft));

%% --- Tracking loop variables ---
leftVid.CurrentTime = 0; rightVid.CurrentTime = 0;
positions3D = []; frameIdx = 0; reinitAttempts = 0;
prevCentroid = []; prevBBox = roi; % for template fallback
figure('Name','Robust Hybrid Stereo Tracking','Position',[80 80 1400 520]);

%% --- Main processing loop ---
while hasFrame(leftVid) && hasFrame(rightVid)
    frameIdx = frameIdx + 1;
    leftFrame = readFrame(leftVid); rightFrame = readFrame(rightVid);
    [leftRect, rightRect] = rectifyStereoImages(leftFrame, rightFrame, stereoParams);
    leftGray = rgb2gray(leftRect); rightGray = rgb2gray(rightRect);

    % Track points
    [pointsL_tracked, validL] = trackerL(leftRect);
    [pointsR_tracked, validR] = trackerR(rightRect);
    valid = validL & validR;
    numValid = nnz(valid);

    % If we have enough valid points, triangulate and compute centroid
    if numValid >= minFeaturesToKeep
        worldPts = triangulate(pointsL_tracked(valid,:), pointsR_tracked(valid,:), stereoParams);
        mean3D = mean(worldPts,1,'omitnan');
        positions3D = [positions3D; mean3D];
        % compute 2D centroid in left image for color correction
        centroid2D = mean(pointsL_tracked(valid,:),1,'omitnan');
        prevCentroid = centroid2D;
        prevBBox = round([centroid2D - size(templateLeft)/2, fliplr(size(templateLeft))]); % [x y w h]
    else
        positions3D = [positions3D; [NaN NaN NaN]];
    end

    % ------------------- Periodic re-detection ---------------------
    if mod(frameIdx, reDetectEveryN) == 0 || numValid < minFeaturesToKeep
        % Build color mask in leftRect
        hsvL = rgb2hsv(im2double(leftRect));
        maskColor = (hsvL(:,:,1)>=hMin & hsvL(:,:,1)<=hMax) & ...
                    (hsvL(:,:,2)>=sMin & hsvL(:,:,2)<=sMax) & ...
                    (hsvL(:,:,3)>=vMin & hsvL(:,:,3)<=vMax);
        maskColor = imopen(maskColor, strel('disk',3));
        maskColor = imclose(maskColor, strel('disk',6));
        props = regionprops(maskColor, 'Area', 'BoundingBox', 'Centroid');
        if ~isempty(props)
            [~, idxMax] = max([props.Area]);
            bb = round(props(idxMax).BoundingBox); centColor = props(idxMax).Centroid;
            % detect features inside bb
            newPointsL = detectMinEigenFeatures(leftGray, 'ROI', bb);
            newPointsL = selectStrongest(newPointsL, targetNumFeatures);
            if newPointsL.Count >= 6
                % match to right frame
                [fL, vL] = extractFeatures(leftGray, newPointsL);
                newPointsR = detectMinEigenFeatures(rightGray);
                [fR, vR] = extractFeatures(rightGray, newPointsR);
                idx = matchFeatures(fL, fR, 'Unique', true, 'MaxRatio', 0.8);
                if ~isempty(idx) && size(idx,1) >= 6
                    % Reinitialize trackers with matched pairs
                    matchedL = vL(idx(:,1)); matchedR = vR(idx(:,2));
                    release(trackerL); release(trackerR);
                    initialize(trackerL, matchedL.Location, leftRect);
                    initialize(trackerR, matchedR.Location, rightRect);
                    reinitAttempts = 0;
                    % update template bbox and centroid
                    prevCentroid = centColor;
                    prevBBox = bb;
                end
            end
        end
    end

    % ------------------- Color centroid correction (soft) ----------------
    try
        hsvL = rgb2hsv(im2double(leftRect));
        maskColor = (hsvL(:,:,1)>=hMin & hsvL(:,:,1)<=hMax) & ...
                    (hsvL(:,:,2)>=sMin & hsvL(:,:,2)<=sMax) & ...
                    (hsvL(:,:,3)>=vMin & hsvL(:,:,3)<=vMax);
        maskColor = imopen(maskColor, strel('disk',3));
        props = regionprops(maskColor,'Centroid');
        if ~isempty(props) && ~isempty(prevCentroid)
            colorCent = props(1).Centroid;
            % shift tracked left points partly toward color centroid
            [ptsL_cur, validLnow] = trackerL(leftRect);
            if any(validLnow)
                meanTracked = mean(ptsL_cur(validLnow,:),1,'omitnan');
                shiftVec = (colorCent - meanTracked);
                corrFactor = 0.25; % how strongly to correct (0..1)
                % nudge the internal position estimate in the tracker:
                % reconstruct new point set as old + correction for valid points
                newPts = ptsL_cur;
                newPts(validLnow,:) = newPts(validLnow,:) + corrFactor*shiftVec;
                % reinitialize tracker with corrected points to bias future tracking
                release(trackerL);
                initialize(trackerL, newPts, leftRect);
            end
        end
    catch
        % ignore color correction errors
    end

    % ------------------- Template fallback if still too few points ------------
    if nnz(valid) < minFeaturesToKeep
        % try template matching using last templateLeft (search around prevBBox if available)
        try
            leftSearch = leftGray;
            if exist('prevBBox','var') && ~isempty(prevBBox)
                % define limited search window to speed up and reduce false positives
                x1 = max(1, prevBBox(1)-round(prevBBox(3)));
                y1 = max(1, prevBBox(2)-round(prevBBox(4)));
                x2 = min(size(leftGray,2), prevBBox(1)+2*round(prevBBox(3)));
                y2 = min(size(leftGray,1), prevBBox(2)+2*round(prevBBox(4)));
                leftSearch = leftGray(y1:y2, x1:x2);
                offset = [x1-1 y1-1];
            else
                offset = [0 0];
            end
            c = normxcorr2(im2double(rgb2gray(templateLeft)), im2double(leftSearch));
            [maxC, idxmax] = max(c(:));
            [ypeak,xpeak] = ind2sub(size(c), idxmax);
            topLeft = [xpeak - size(templateLeft,2), ypeak - size(templateLeft,1)] + 1;
            matchCenter = topLeft + [size(templateLeft,2) size(templateLeft,1)]/2 + offset;
            if maxC >= templateCorrThreshold
                % reinit features around matchCenter: detect features in bbox
                newBB = round([matchCenter - fliplr(size(templateLeft))/2, fliplr(size(templateLeft))]);
                newBB(1) = max(newBB(1),1); newBB(2)=max(newBB(2),1);
                newBB(3)=min(newBB(3), size(leftGray,2)-newBB(1)); newBB(4)=min(newBB(4), size(leftGray,1)-newBB(2));
                newPointsL = detectMinEigenFeatures(leftGray, 'ROI', newBB);
                newPointsL = selectStrongest(newPointsL, targetNumFeatures);
                if newPointsL.Count >= 6
                    [fL, vL] = extractFeatures(leftGray, newPointsL);
                    newPointsR = detectMinEigenFeatures(rightGray);
                    [fR, vR] = extractFeatures(rightGray, newPointsR);
                    idx = matchFeatures(fL, fR, 'Unique', true);
                    if size(idx,1) >= 6
                        matchedL = vL(idx(:,1)); matchedR = vR(idx(:,2));
                        release(trackerL); release(trackerR);
                        initialize(trackerL, matchedL.Location, leftRect);
                        initialize(trackerR, matchedR.Location, rightRect);
                        prevBBox = newBB;
                        reinitAttempts = 0;
                    end
                end
            else
                reinitAttempts = reinitAttempts + 1;
            end
        catch
            reinitAttempts = reinitAttempts + 1;
        end
    end

    % If reinit attempts exceed limit, warn and continue
    if reinitAttempts > maxReinitAttempts
        warning('Multiple reinit attempts failed at frame %d. Tracking may be lost.', frameIdx);
        reinitAttempts = 0;
    end

    % ------------------- Visualization -------------------
    subplot(1,3,1); imshow(leftRect); hold on;
    if exist('pointsL_tracked','var')
        plot(pointsL_tracked(valid,1), pointsL_tracked(valid,2), 'go','MarkerSize',4);
    end
    if exist('prevBBox','var') && ~isempty(prevBBox)
        rectangle('Position', prevBBox, 'EdgeColor','r', 'LineWidth',1);
    end
    title(sprintf('Left frame %d (valid=%d)', frameIdx, numValid));
    hold off;

    subplot(1,3,2); imshow(rightRect); hold on;
    if exist('pointsR_tracked','var')
        plot(pointsR_tracked(valid,1), pointsR_tracked(valid,2), 'go','MarkerSize',4);
    end
    title('Right');
    hold off;

    subplot(1,3,3);
    validPts3 = ~any(isnan(positions3D),2);
    if any(validPts3)
        plot3(positions3D(validPts3,1), positions3D(validPts3,2), positions3D(validPts3,3), 'm.-','LineWidth',1.2);
        xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
        grid on; axis equal; view(0,-90); title('3D Trajectory');
    else
        title('3D Trajectory (no valid points yet)');
    end

    drawnow;
end

disp('Tracking finished. positions3D stored in workspace (variable positions3D).');

assignin('base','positions3D',positions3D); % export to base workspace
end
