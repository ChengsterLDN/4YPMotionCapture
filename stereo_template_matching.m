%% Stereo Object Tracker using Template Matching
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

%% --- Read first frame to pick template ---
leftFrame  = readFrame(leftVid);
rightFrame = readFrame(rightVid);

% Rectify the first frames for template selection
[leftRect0, rightRect0] = rectifyStereoImages(leftFrame, rightFrame, stereoParams);

figure('Name', 'Select Template (Left then Right)', 'Position', [200 200 1000 450]);

subplot(1,2,1);
imshow(leftRect0);
title('Left Rectified Frame - Draw rectangle for LEFT template');
axis on;

subplot(1,2,2);
imshow(rightRect0);
title('Right Rectified Frame - Draw rectangle for RIGHT template');
axis on;

% --- LEFT template selection ---
disp('👉 Draw a rectangle on the LEFT rectified frame, then press Enter or double-click to confirm.');
subplot(1,2,1);
hRectL = drawrectangle('Color','r','LineWidth',1.5);
wait(hRectL);                   % Waits for user confirmation
posL = round(hRectL.Position);  % Get rectangle position after confirmation

if isempty(posL)
    error('No rectangle selected for LEFT template.');
end
templateLeft = imcrop(leftRect0, posL);
if isempty(templateLeft)
    error('Failed to crop left template.');
end

% --- RIGHT template selection ---
disp('👉 Draw the corresponding rectangle on the RIGHT rectified frame, then press Enter or double-click to confirm.');
subplot(1,2,2);
hRectR = drawrectangle('Color','r','LineWidth',1.5);
wait(hRectR);
posR = round(hRectR.Position);

if isempty(posR)
    error('No rectangle selected for RIGHT template.');
end
templateRight = imcrop(rightRect0, posR);
if isempty(templateRight)
    error('Failed to crop right template.');
end

close(gcf);


% Convert templates to grayscale double (normxcorr2 works on single channel)
tmplLGray = im2double(rgb2gray(templateLeft));
tmplRGray = im2double(rgb2gray(templateRight));
tmplLSize = size(tmplLGray);  % [h w]
tmplRSize = size(tmplRGray);

% Correlation threshold (tune if needed)
corrThreshold = 0.4;

fprintf('Templates captured. Left template size: %d x %d. Right template size: %d x %d.\n', tmplLSize(1), tmplLSize(2), tmplRSize(1), tmplRSize(2));
fprintf('Correlation threshold = %.2f\n', corrThreshold);

%% --- Reset videos to beginning ---
leftVid.CurrentTime = 0;
rightVid.CurrentTime = 0;

%% --- Initialize storage ---
positions3D = [];
frameIdx = 0;

%% --- Process frames ---
figure('Name','Stereo Template Tracking','Position',[100 100 1400 500]);
while hasFrame(leftVid) && hasFrame(rightVid)
    frameIdx = frameIdx + 1;
    leftFrame  = readFrame(leftVid);
    rightFrame = readFrame(rightVid);

    % --- Rectify stereo images ---
    [leftRect, rightRect] = rectifyStereoImages(leftFrame, rightFrame, stereoParams);

    % --- Convert to grayscale doubles for matching ---
    leftGray  = im2double(rgb2gray(leftRect));
    rightGray = im2double(rgb2gray(rightRect));

    % --- Template matching LEFT using normalized cross-correlation ---
    try
        cLeft = normxcorr2(tmplLGray, leftGray);
        % Find peak
        [maxC_L, maxIdxL] = max(cLeft(:));
        [ypeakL, xpeakL] = ind2sub(size(cLeft), maxIdxL);
        % Compute top-left of matched region in leftGray
        topLeftL = [xpeakL - tmplLSize(2) + 1, ypeakL - tmplLSize(1) + 1];
        centroidLeft = topLeftL + [tmplLSize(2)/2, tmplLSize(1)/2];
        if maxC_L < corrThreshold
            centroidLeft = [NaN NaN];
            maxC_L = NaN;
        end
    catch
        centroidLeft = [NaN NaN];
        maxC_L = NaN;
    end

    % --- Template matching RIGHT ---
    try
        cRight = normxcorr2(tmplRGray, rightGray);
        [maxC_R, maxIdxR] = max(cRight(:));
        [ypeakR, xpeakR] = ind2sub(size(cRight), maxIdxR);
        topLeftR = [xpeakR - tmplRSize(2) + 1, ypeakR - tmplRSize(1) + 1];
        centroidRight = topLeftR + [tmplRSize(2)/2, tmplRSize(1)/2];
        if maxC_R < corrThreshold
            centroidRight = [NaN NaN];
            maxC_R = NaN;
        end
    catch
        centroidRight = [NaN NaN];
        maxC_R = NaN;
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
        plot(centroidLeft(1), centroidLeft(2),'r+','MarkerSize',12,'LineWidth',1.5);
        rectangle('Position',[topLeftL(1) topLeftL(2) tmplLSize(2) tmplLSize(1)],'EdgeColor','r','LineWidth',1);
        text(10,20,sprintf('Corr=%.2f', maxC_L),'Color','y','FontSize',10,'FontWeight','bold','BackgroundColor','black','Margin',1);
    else
        text(10,20,'Left: no match','Color','y','FontSize',10,'FontWeight','bold','BackgroundColor','black','Margin',1);
    end
    title(sprintf('Left Rectified Frame %d', frameIdx));
    hold off;

    subplot(1,3,2);
    imshow(rightRect); hold on;
    if all(~isnan(centroidRight))
        plot(centroidRight(1), centroidRight(2),'r+','MarkerSize',12,'LineWidth',1.5);
        rectangle('Position',[topLeftR(1) topLeftR(2) tmplRSize(2) tmplRSize(1)],'EdgeColor','r','LineWidth',1);
        text(10,20,sprintf('Corr=%.2f', maxC_R),'Color','y','FontSize',10,'FontWeight','bold','BackgroundColor','black','Margin',1);
    else
        text(10,20,'Right: no match','Color','y','FontSize',10,'FontWeight','bold','BackgroundColor','black','Margin',1);
    end
    title(sprintf('Right Rectified Frame %d', frameIdx));
    hold off;

    % --- Plot 3D Trajectory ---
    subplot(1,3,3);
    validPoints = ~any(isnan(positions3D),2);
    plot3(positions3D(validPoints,1), positions3D(validPoints,2), positions3D(validPoints,3),'m.-','LineWidth',1.5);
    axis equal
    xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
    grid on; title('3D Trajectory of Tracked Object'); view(0,-90)
    
    drawnow;
end

disp('Processing finished.');
