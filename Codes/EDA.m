clc
sub = 190;
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
figure;imshow(segGray)
% the white background value becomes 255. 
%% create binary mask
mask = segGray~=255;
figure;imshow(mask(400:750, 450:800))
imwrite(mask(400:750, 450:800), 'binarymask_190.png')
%% remove unwanted pixels outside brain
stat = regionprops(mask, 'Area', 'PixelIdxList');
for nn=1:length(stat)
    if stat(nn).Area<1000 % non-brain
        mask(stat(nn).PixelIdxList)= 0;
    end % remove small area
end
figure;imshow(mask(400:750, 450:800))
imwrite(mask(400:750, 450:800), 'binarybrainmask_190.png')
%% run for all segmentation files
mskDir = '/media/banikr2/DATA/Diesel_block/6_binarymask';
for ii=1:length(segFilePaths)
    segPath = fullfile(segDir, segFilePaths(ii).name);
    seg = imread(segPath);
%     size(seg)
    segGray = rgb2gray(seg);
    mask = segGray~=255;
    stat = regionprops(mask, 'Area', 'PixelIdxList');
    for nn=1:length(stat)
        if stat(nn).Area<1000 % non-brain
            mask(stat(nn).PixelIdxList)= 0;
        end % remove small area
    end
    [folder, baseFileNameNoExt, extension] = fileparts(segPath);
    mskPath = fullfile(mskDir, strcat(baseFileNameNoExt,'.png'));
    imwrite(mask, mskPath);
end 
%%
clear all; 
clc
refDir = '/media/banikr2/DATA/Diesel_block/4_refocus';
refFilePaths = dir(refDir);
refFilePaths = refFilePaths(3:end);
mskDir = '/media/banikr2/DATA/Diesel_block/6_binarymask';
mskFilePaths = dir(mskDir);
mskFilePaths = mskFilePaths(3:end);
patchDir = '/media/banikr2/DATA/Diesel_block/patches';
ImageDir = fullfile(patchDir,'Image');
MaskDir = fullfile(patchDir,'Mask');
if ~isfolder(ImageDir)
    cmd = ['mkdir ' ImageDir];
    system(cmd);
end
if ~isfolder(MaskDir)
    cmd = ['mkdir' MaskDir];
    system(cmd); 
end
patchsize = [224, 224];
stepsize = [100, 100];
% overlapMat = zeros(size(img));
for s = 1:length(refFilePaths)
    sub = str2double(cell2mat(regexp(refFilePaths(s).name,'\d*','Match')));
    sprintf('saving image %03d ...', sub)
    img = imread([refDir, '/', refFilePaths(sub).name]);
    msk = imread([mskDir, '/', mskFilePaths(sub).name]);
    count=0;
    for r = 1:stepsize(1):size(img,1)-patchsize(1)
        for c = 1:stepsize(2):size(img,2)-patchsize(2)
            count=count+1;
    %         sprintf('%d %d %d', count, r, c)
    %         overlapMat(r:r + patchsize(1)-1, c:c + patchsize(2)-1) = overlapMat(r:r + patchsize(1)-1, c:c + patchsize(2)-1) + 1;
            imP = img(r:r + patchsize(1)-1, c:c + patchsize(2)-1, :);
            imwrite(imP, strcat(ImageDir,'/', strcat(num2str(sub, '%03d'),'_',num2str(count,'%03d'),'_',num2str(r,'%03d'),'_',num2str(c,'%03d'),'.tif')));
            mkP = msk(r:r + patchsize(1)-1, c:c + patchsize(2)-1);
            imwrite(mkP, strcat(MaskDir,'/', strcat(num2str(sub, '%03d'),'_',num2str(count,'%03d'),'_',num2str(r,'%03d'),'_',num2str(c,'%03d'),'.png')));
            if c+stepsize(2)+patchsize(2)>size(img, 2)
                count=count+1;
    %             sprintf('%d %d %d', count, r, c)
    %             overlapMat(r:r + patchsize(1)-1, size(img,2) - patchsize(2):size(img,2)-1) = overlapMat(r:r + patchsize(1)-1, size(img,2) - patchsize(2):size(img,2)-1) + 1;
                imP = img(r:r + patchsize(1)-1, size(img,2) - patchsize(2):size(img,2)-1, :);
                imwrite(imP, strcat(ImageDir,'/', strcat(num2str(sub, '%03d'),'_',num2str(count,'%03d'),'_',num2str(r,'%03d'),'_',num2str(c,'%03d'),'.tif')));
                mkP = msk(r:r + patchsize(1)-1, size(msk,2) - patchsize(2):size(msk,2)-1);
                imwrite(mkP, strcat(MaskDir,'/', strcat(num2str(sub, '%03d'),'_',num2str(count,'%03d'),'_',num2str(r,'%03d'),'_',num2str(c,'%03d'),'.png')));
            end
            if r+stepsize(1)+patchsize(1)>size(img, 1)
                count=count+1;
    %             sprintf('%d %d %d', count, r, c)
    %             overlapMat(size(img,1) - patchsize(1):size(img,1)-1, c:c + patchsize(2)-1) = overlapMat(size(img,1) - patchsize(1):size(img,1)-1, c:c + patchsize(2)-1)+1;
                imP = img(size(img,1) - patchsize(1):size(img,1)-1, c:c + patchsize(2)-1,:);
                imwrite(imP, strcat(ImageDir,'/', strcat(num2str(sub, '%03d'),'_',num2str(count,'%03d'),'_',num2str(r,'%03d'),'_',num2str(c,'%03d'),'.tif')));
                mkP = msk(size(msk,1) - patchsize(1):size(msk,1)-1, c:c + patchsize(2)-1);
                imwrite(mkP, strcat(MaskDir,'/', strcat(num2str(sub, '%03d'),'_',num2str(count,'%03d'),'_',num2str(r,'%03d'),'_',num2str(c,'%03d'),'.png')));
            end
            if r+stepsize(1)+patchsize(1)>size(img, 1) && c+stepsize(2)+patchsize(2)>size(img, 2)
               count=count+1;
    %            sprintf('%d %d %d', count, r, c)
    %            overlapMat(size(img,1) - patchsize(1):size(img,1)-1, size(img,2) - patchsize(2):size(img,2)-1) = overlapMat(size(img,1) - patchsize(1):size(img,1)-1, size(img,2) - patchsize(2):size(img,2)-1)+1;
               imP = img(size(img,1) - patchsize(1):size(img,1)-1, size(img,2) - patchsize(2):size(img,2)-1, :);
               imwrite(imP, strcat(ImageDir,'/', strcat(num2str(sub, '%03d'),'_',num2str(count,'%03d'),'_',num2str(r,'%03d'),'_',num2str(c,'%03d'),'.tif')));
               mkP = msk(size(msk,1) - patchsize(1):size(msk,1)-1, size(msk,2) - patchsize(2):size(msk,2)-1);
               imwrite(mkP, strcat(MaskDir,'/', strcat(num2str(sub, '%03d'),'_',num2str(count,'%03d'),'_',num2str(r,'%03d'),'_',num2str(c,'%03d'),'.png')));
            end
        end
    end
    if s == 5
        break;
    end
end
% imshow(overlapMat)

%%
clc
slices = dir(refDir);
slices = slices(3:end);
for s = 1:length(slices)
    sub = str2double(cell2mat(regexp(slices(s).name,'\d*','Match')));
    num2str(sub, '%03d')
end