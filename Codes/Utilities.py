from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

def patchify(img, msk, patchsize, stepsize, retovermat=False, plotovermat=False):
    """
    img: image to patchify
    msk: binary mask to patchify
    patchsize: e.g. [100,100,3]
    step: for windown sliding e.g. [100,100]
    retovermat: return the overlap matrix, default = False
    plotovermat: visualize the patching, default = False
    """
    count=0
    imgPatch = []
    mskPatch = []
    overlapMat = np.zeros_like(img[..., 0])
    for r in range(0, img.shape[0]-patchsize[0], stepsize[0]):
        for c in range(0, img.shape[1]-patchsize[1], stepsize[1]):
            count += 1
            overlapMat[r:r+patchsize[0], c:c+patchsize[1]] += 1
            imgPatch.append(img[r:r+patchsize[0], c:c+patchsize[1], :])
            mskPatch.append(msk[r:r+patchsize[0], c:c+patchsize[1]])
            if c+stepsize[1]+patchsize[1] > img.shape[1]:
                count += 1
                overlapMat[r:r+patchsize[0], img.shape[1]-patchsize[1]:img.shape[1]] += 1
                imgPatch.append(img[r:r+patchsize[0], img.shape[1]-patchsize[1]:img.shape[1], :])
                mskPatch.append(msk[r:r+patchsize[0], img.shape[1]-patchsize[1]:img.shape[1]])
            if r+stepsize[0]+patchsize[0] > img.shape[0]:
                count += 1
                overlapMat[img.shape[0]-patchsize[0]:img.shape[0], c:c+patchsize[1]] += 1
                imgPatch.append(img[img.shape[0]-patchsize[0]:img.shape[0], c:c+patchsize[1], :])
                mskPatch.append(msk[img.shape[0]-patchsize[0]:img.shape[0], c:c+patchsize[1]])
            if r+stepsize[0]+patchsize[0] > img.shape[0] and c+stepsize[1]+patchsize[1] > img.shape[1]:
                count += 1
                overlapMat[img.shape[0]-patchsize[0]:img.shape[0], img.shape[1]-patchsize[1]:img.shape[1]] += 1
                imgPatch.append(img[img.shape[0]-patchsize[0]:img.shape[0], img.shape[1]-patchsize[1]:img.shape[1], :])
                mskPatch.append(msk[img.shape[0]-patchsize[0]:img.shape[0], img.shape[1]-patchsize[1]:img.shape[1]])
    imgPatch = np.array(imgPatch).astype(img.dtype)
    mskPatch = np.array(mskPatch).astype(msk.dtype)
    if plotovermat:
        c = plt.imshow(overlapMat)
        plt.colorbar(c)
        plt.show()
    if retovermat:
        return imgPatch, mskPatch, overlapMat
    else:
        return imgPatch, mskPatch

class BlockSet(Dataset):
    def __init__(self, imList, mkList): #, transform=None):
        self.imList = imList
        self.mkList = mkList
        # self.transform = transform

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, index):
        im = Image.open(self.imList[index]).convert('RGB')
        mk = Image.open(self.mkList[index]).convert('L')
        im = np.asarray(im)
        mk = np.asarray(mk)
        im, mk = patchify(im, mk, [224, 224], [100, 100])
        im = np.transpose(im, (0, 3, 1, 2))
        mkB = np.ones((2, 64, 224, 224))
        mkB[1, ...] = mk == 255
        mkB[0, ...] = mkB[0, ...] - mkB[1, ...]
        mkB = np.transpose(mkB, (1, 0, 2, 3))
        return im, mkB




# +---------+
# | scratch |
# +---------+
if __name__ == '__main__':
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])
    # t = transforms.Compose([transforms.RandomResizedCrop(224),
    #                         transforms.RandomHorizontalFlip(),
    # #                       transforms.ToTensor(),
    #                         normalize])
    # mkout = np.zeros((2, 64, 224, 224))
    # print(mkout.shape)
    refDir = r'/media/banikr2/DATA/Diesel_block/4_refocus'
    segDir = r'/media/banikr2/DATA/Diesel_block/5_segmented'
    mskDir = r'/media/banikr2/DATA/Diesel_block/6_binarymask'
    refFiles = sorted(glob(os.path.join(refDir, '*.tif')))  # .sort()
    segFiles = sorted(glob(os.path.join(segDir, '*.tif')))  # .sort()
    mskFiles = sorted(glob(os.path.join(mskDir, '*.png')))
    unitSet = BlockSet([refFiles[20]], [mskFiles[20]])
    x, y = next(iter(unitSet))
    print(x.shape, y.shape, np.unique(y))
    plt.imshow(y[20, 0, ...])
    plt.show()
    plt.imshow(y[20, 1, ...])
    plt.show()