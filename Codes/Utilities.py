import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
from glob import glob
from Codes.txfromfunc import Normalize

norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def patchify(img, msk, patchsize, stepsize, retovermat=False, plotovermat=False):
    """
    img: image array to patchify
    msk: binary mask array to patchify
    patchsize: e.g. [100,100,3]
    step: for windown sliding e.g. [100,100]
    retovermat: return the overlap matrix, default = False
    plotovermat: visualize the patching, default = False
    """
    count=0
    imgPatch = []
    mskPatch = []
    overlapMat = np.zeros_like(img[..., 0])
    for r in range(0, img.shape[0] - patchsize[0], stepsize[0]):
        for c in range(0, img.shape[1] - patchsize[1], stepsize[1]):
            # patches inside
            count += 1
            # print(count, r, c)
            overlapMat[r:r + patchsize[0], c:c + patchsize[1]] += 1
            imgPatch.append(img[r:r + patchsize[0], c:c + patchsize[1], :])
            mskPatch.append(msk[r:r + patchsize[0], c:c + patchsize[1]])
            if c + stepsize[1] + patchsize[1] > img.shape[1]:
                count += 1
                # print(count, r, c, "column ends")
                overlapMat[r:r + patchsize[0], img.shape[1] - patchsize[1]:img.shape[1]] += 1
                imgPatch.append(img[r:r + patchsize[0], img.shape[1] - patchsize[1]:img.shape[1], :])
                mskPatch.append(msk[r:r + patchsize[0], msk.shape[1] - patchsize[1]:img.shape[1]])
            if r + stepsize[0] + patchsize[0] > img.shape[0]:
                count += 1
                # print(count, r, c, "row ends")
                overlapMat[img.shape[0] - patchsize[0]:img.shape[0], c:c + patchsize[1]] += 1
                imgPatch.append(img[img.shape[0] - patchsize[0]:img.shape[0], c:c + patchsize[1], :])
                mskPatch.append(msk[img.shape[0] - patchsize[0]:msk.shape[0], c:c + patchsize[1]])
            if r + stepsize[0] + patchsize[0] > img.shape[0] and c + stepsize[1] + patchsize[1] > img.shape[1]:
                count += 1
                # print(count, r, c, "row and column ends")
                overlapMat[img.shape[0] - patchsize[0]:img.shape[0], img.shape[1] - patchsize[1]:img.shape[1]] += 1
                imgPatch.append(
                    img[img.shape[0] - patchsize[0]:img.shape[0], img.shape[1] - patchsize[1]:img.shape[1], :])
                mskPatch.append(msk[msk.shape[0] - patchsize[0]:msk.shape[0], msk.shape[1] - patchsize[1]:msk.shape[1]])

    # print("Total patch: ", count)
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


normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

t = T.Compose([
               T.RandomResizedCrop(224),
               T.RandomHorizontalFlip(),
               T.ToTensor(),
               normalize
              ])

class BlockSet(Dataset):
    def __init__(self, imList, mkList, tform=None):
        self.imList = imList
        self.mkList = mkList
        self.tform = tform

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, index):
        im = Image.open(self.imList[index]).convert('RGB')
        mk = Image.open(self.mkList[index]).convert('L')
        im = np.asarray(im)
        mk = np.asarray(mk)
        im, mk = patchify(im, mk, [224, 224], [100, 100]) # dtype('uint8') # (64, 224, 224, 3)
        i_t_s = torch.empty([64, 3, 224, 224])
        m_t_s = torch.empty([64, 224, 224])
        for l in range(0, len(im)): # PIL must follow [H, W, C] format
            PIL_i = Image.fromarray(im[l, ...])#.convert('RGB')
            PIL_m = Image.fromarray(mk[l, ...])#.convert('L')
            i_t = self.tform(PIL_i) # [3, 224, 224]
            # m_t = self.tform(PIL_m)
            m_t = T.ToTensor()(PIL_m)
            i_t_s[l, ...] = i_t 
            m_t_s[l, ...] = m_t
        m_t_o = torch.ones(2, 64, 224, 224)
        print(">>", torch.unique(m_t_s), m_t_s.shape)
        m_t_o[1, ...] = m_t_s == 1.
        m_t_o[0, ...] = m_t_o[0, ...] - m_t_o[1, ...]
        m_t_o = torch.transpose(m_t_o, 1, 0) #, 2, 3)
        return i_t_s, m_t_o # [64, 3, 224, 224], [64, 2, 224, 224]

class BlockSet2(Dataset):
    def __init__(self, imList, mkList, img_transform=None, msk_transform=None):
        self.imList = imList
        self.mkList = mkList
        self.img_transform = img_transform
        self.mask_transform = msk_transform

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, item):
        imgPIL = Image.open(self.imList[item]).convert('RGB')
        mskPIL = Image.open(self.mkList[item]).convert('L')
        imar = np.asarray(imgPIL)
        mask = np.asarray(mskPIL)
        mask = mask[:, :, None].repeat(3, axis=2) #(224, 224, 3)
        # print(mask.shape)
        seed = random.randrange(sys.maxsize)  # get a random seed so that we can reproducibly do the transofrmations
        if self.img_transform is not None:
            random.seed(seed)  # apply this seed to img transforms
            imtr = self.img_transform(imar)
        if self.mask_transform is not None:
            random.seed(seed)
            mask_new = self.mask_transform(mask)
            mask_new = np.asarray(mask_new)[:, :, 0].squeeze()
            mktr = F.to_tensor(mask_new)  # converts to tensor with [0,255] ~ [0,1]
        # fixme: do we need to one-hot? Depends on the loss function.
            # torch.Size([3, 224, 224])
        #
        # imar = np.transpose(imtr, (2, 0, 1))
        # imar = imar[None, ...]
        ###
        # mkar = np.asarray(mskPIL)
        # mkoh = np.ones(np.append([2], np.array(mkar.shape)), dtype=mkar.dtype)
        # mkoh[1, ...] = mkar == 255
        # mkoh[0, ...] = mkoh[0, ...] - mkoh[1, ...]
        # mkoh = mkoh[None, ...]
        return imtr, mktr #mkoh

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
    # dum = torch.randn([2, 64, 224, 224])
    # # dum = torch.transpose(dum, (1, 0, 2, 3))
    # dum = torch.transpose(dum, 1, 0) #, 2, 3)
    # print(dum.shape, "TT")
    refDir = r'/media/banikr2/DATA/Diesel_block/4_refocus'
    segDir = r'/media/banikr2/DATA/Diesel_block/5_segmented'
    mskDir = r'/media/banikr2/DATA/Diesel_block/6_binarymask'
    refFiles = sorted(glob(os.path.join(refDir, '*.tif')))  # .sort()
    segFiles = sorted(glob(os.path.join(segDir, '*.tif')))  # .sort()
    mskFiles = sorted(glob(os.path.join(mskDir, '*.png')))
    im = Image.open(refFiles[20]).convert('RGB')
    mk = Image.open(mskFiles[20]).convert('L')
    im = np.asarray(im)
    mk = np.asarray(mk)
    patchsize = [224, 224]
    stepsize = [100, 100]
    # im, mk = patchify(im, mk, [224, 224], [100, 100])
    # i, m = patchify(im, mk, patchsize, stepsize, plotovermat=True)
    unitSet = BlockSet([refFiles[20]], [mskFiles[20]], tform=t)
    x, y = next(iter(unitSet))
    print(x[20, ...].numpy().shape, type(x))
    unloader = T.ToPILImage()
    def tensor_to_PIL(tensor):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        return image
    #
    # print(x.shape, y.shape, torch.unique(y), torch.max(x), torch.min(x), torch.unique(y[:, 0, ...]))
    Pimage = tensor_to_PIL(x[32, ...])
    plt.imshow(Pimage)
    plt.show()
    plt.imshow(tensor_to_PIL(y[32, 0, ...]))
    plt.show()
    plt.imshow(tensor_to_PIL(y[32, 1, ...]))
    plt.show()
