%% Single Camera Calibration
imageFolder = '.\calibration\';

%filePattern = fullfile(imageFolder, "chessboard_*.jpg");
%imageFileNames = dir(filePattern);

% Define the folder where the images are stored
%imageFolder = 'path_to_your_image_folder';  % Replace with your folder path
% Create an ImageDatastore object for the images in your folder
imds = imageDatastore(imageFolder, 'FileExtensions', '.jpg', 'LabelSource', 'foldernames');

imageFileNames = imds.Files;
[imagePoints, boardSize] = detectCheckerboardPoints(imageFileNames);

squareSizeInMM = 24;
worldPoints = patternWorldPoints("checkerboard",boardSize,squareSizeInMM);

I = preview(imds); 
imageSize = [size(I, 1),size(I, 2)];
params = estimateCameraParameters(imagePoints,worldPoints, ...
                                  'ImageSize',imageSize);

showReprojectionErrors(params);
figure;
showExtrinsics(params,"PatternCentric");
drawnow;
figure; 
imshow(imageFileNames{1}); 
hold on;
plot(imagePoints(:,1,1), imagePoints(:,2,1),'go');
plot(params.ReprojectedPoints(:,1,1),params.ReprojectedPoints(:,2,1),'r+');
legend('Detected Points','ReprojectedPoints');
hold off;

%%
%% Stereo Pink Object Tracker
% Requirements:
% - params.mat (from stereo calibration)
% - leftVideo.mp4 and rightVideo.mp4


%% Load stereo calibration parameters

%% Read stereo video files
leftVid  = VideoReader('camera1_20251014_222401.mp4');
rightVid = VideoReader('camera2_20251014_222401.mp4');

%% Prepare storage
positions3D = [];
frameIdx = 0;

%% Process each frame
while hasFrame(leftVid) && hasFrame(rightVid)
    frameIdx = frameIdx + 1;
    leftFrame  = readFrame(leftVid);
    rightFrame = readFrame(rightVid);

    % --- Rectify stereo images ---
    [leftRect, rightRect] = rectifyStereoImages(leftFrame, rightFrame, params);

    % --- Detect pink object in left rectified image ---
    hsvLeft = rgb2hsv(leftRect);
    H = hsvLeft(:,:,1); S = hsvLeft(:,:,2); V = hsvLeft(:,:,3);
    pinkMaskLeft = (H > 0.95 | H < 0.05) & S > 0.4 & V > 0.4;
    pinkMaskLeft = imopen(pinkMaskLeft, strel('disk', 5));
    pinkMaskLeft = imclose(pinkMaskLeft, strel('disk', 10));
    pinkMaskLeft = imfill(pinkMaskLeft, 'holes');
    statsLeft = regionprops(pinkMaskLeft, 'Centroid');
    if ~isempty(statsLeft)
        centroidLeft = statsLeft(1).Centroid;
    else
        centroidLeft = [NaN, NaN];
    end

    % --- Detect pink object in right rectified image ---
    hsvRight = rgb2hsv(rightRect);
    H = hsvRight(:,:,1); S = hsvRight(:,:,2); V = hsvRight(:,:,3);
    pinkMaskRight = (H > 0.95 | H < 0.05) & S > 0.4 & V > 0.4;
    pinkMaskRight = imopen(pinkMaskRight, strel('disk', 5));
    pinkMaskRight = imclose(pinkMaskRight, strel('disk', 10));
    pinkMaskRight = imfill(pinkMaskRight, 'holes');
    statsRight = regionprops(pinkMaskRight, 'Centroid');
    if ~isempty(statsRight)
        centroidRight = statsRight(1).Centroid;
    else
        centroidRight = [NaN, NaN];
    end

    % --- Triangulate 3D position ---
    if all(~isnan([centroidLeft centroidRight]))
        point3D = triangulate(centroidLeft, centroidRight, params);
        positions3D = [positions3D; point3D];
    else
        positions3D = [positions3D; [NaN NaN NaN]];
    end

    % --- Visualization (optional) ---
    subplot(1,2,1);
    imshow(leftRect);
    hold on;
    if all(~isnan(centroidLeft))
        plot(centroidLeft(1), centroidLeft(2), 'r*', 'MarkerSize', 10);
    end
    title(sprintf('Left Rectified Frame %d', frameIdx));

    subplot(1,2,2);
    imshow(rightRect);
    hold on;
    if all(~isnan(centroidRight))
        plot(centroidRight(1), centroidRight(2), 'r*', 'MarkerSize', 10);
    end
    title(sprintf('Right Rectified Frame %d', frameIdx));

    drawnow;
end

%% --- Plot 3D Trajectory ---
validPoints = ~any(isnan(positions3D), 2);
figure;
plot3(positions3D(validPoints,1), positions3D(validPoints,2), positions3D(validPoints,3), 'm.-', 'LineWidth', 1.5);
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
grid on;
title('3D Trajectory of Pink Object');
axis equal;
