clc
sub = 200;
% pwd
refDir = '/media/banikr2/DATA/Diesel_block/4_refocus';
segDir = '/media/banikr2/DATA/Diesel_block/5_segmented';
addpath(refDir, segDir);
refFilePaths = dir(refDir);
refFilePaths = refFilePaths(3:end);
segFilePaths = dir(segDir);
segFilePaths = segFilePaths(3:end);
img = imread([refDir, '/', refFilePaths(sub).name]);
seg = imread([segDir,'/',segFilePaths(sub).name]);

% sg_chunk = sg(250:399, 200:349,:);
% imshow(sg_chunk)
% imshow(img21)

%% convert to gray 
segGray = rgb2gray(seg);
% the white background value becomes 255. 
%% create binary mask
mask = segGray~=255;