from torch.utils.data import Dataset
from torchvision import transforms
import skimage.io as skio
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, imList, mkList, transform=None):
        self.imList = imList
        self.mkList = mkList
        self.transform = transform

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, index):
        im = skio.imread(self.imList[index], plugin="tifffile")
        mk = skio.imread(self.mkList[index])
        # Normalize here?
        imP, mkP = patchify(im, mk, [224, 224], [100, 100])
        return imP, mkP

# +---------+
# | scratch |
# +---------+
if __name__ != '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    t = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
                normalize])