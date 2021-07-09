import torch
from torch.utils.data import Dataset as BaseDataset
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

torch.manual_seed(1001)

def weights_init(m):
    xavier = torch.nn.init.xavier_uniform_
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
        #print m.weight.data
        #print m.bias.data
        xavier(m.weight.data)
#         print 'come xavier'
        #xavier(m.bias.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear')!=-1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5: return [2, 3, 4]
    # Two dimensional
    elif len(shape) == 4: return [2, 3]
    # Exception - Unknown
    else: raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def patchify(img, msk, patchsize, stepsize, retovermat=False, plotovermat=False, resprnt = False):
    """
    img: image array to patchify
    msk: binary mask array to patchify
    patchsize: e.g. [100,100,3]
    step: for windown sliding e.g. [100,100]
    retovermat: return the overlap matrix, default = False
    plotovermat: visualize the patching, default = False
    resprnt: to print
    """
    count=0
    imgPatch = []
    mskPatch = []
    overlapMat = np.zeros_like(img[..., 0])
    coor = []
    for r in range(0, img.shape[0] - patchsize[0], stepsize[0]):
        for c in range(0, img.shape[1] - patchsize[1], stepsize[1]):
            # patches inside
            count += 1
            if resprnt:
                print(count, r, c)
            overlapMat[r:r + patchsize[0], c:c + patchsize[1]] += 1
            imgPatch.append(img[r:r + patchsize[0], c:c + patchsize[1], :])
            mskPatch.append(msk[r:r + patchsize[0], c:c + patchsize[1]])
            coor.append([r, c])
            if c + stepsize[1] + patchsize[1] > img.shape[1]:
                count += 1
                if resprnt:
                    print(count, r, img.shape[1] - patchsize[1], "column ends")
                overlapMat[r:r + patchsize[0], img.shape[1] - patchsize[1]:img.shape[1]] += 1
                imgPatch.append(img[r:r + patchsize[0], img.shape[1] - patchsize[1]:img.shape[1], :])
                mskPatch.append(msk[r:r + patchsize[0], msk.shape[1] - patchsize[1]:img.shape[1]])
                coor.append([r, img.shape[1] - patchsize[1]])
            if r + stepsize[0] + patchsize[0] > img.shape[0]:
                count += 1
                if resprnt:
                    print(count, img.shape[0] - patchsize[0], c, "row ends")
                overlapMat[img.shape[0] - patchsize[0]:img.shape[0], c:c + patchsize[1]] += 1
                imgPatch.append(img[img.shape[0] - patchsize[0]:img.shape[0], c:c + patchsize[1], :])
                mskPatch.append(msk[img.shape[0] - patchsize[0]:msk.shape[0], c:c + patchsize[1]])
                coor.append([img.shape[0] - patchsize[0], c])
            if r + stepsize[0] + patchsize[0] > img.shape[0] and c + stepsize[1] + patchsize[1] > img.shape[1]:
                count += 1
                if resprnt:
                    print(count, img.shape[0] - patchsize[0], img.shape[1] - patchsize[1], "both row and column ends")
                overlapMat[img.shape[0] - patchsize[0]:img.shape[0], img.shape[1] - patchsize[1]:img.shape[1]] += 1
                imgPatch.append(
                    img[img.shape[0] - patchsize[0]:img.shape[0], img.shape[1] - patchsize[1]:img.shape[1], :])
                mskPatch.append(msk[msk.shape[0] - patchsize[0]:msk.shape[0], msk.shape[1] - patchsize[1]:msk.shape[1]])
                coor.append([img.shape[0] - patchsize[0], img.shape[1] - patchsize[1]])

    if resprnt: print("Total patch: ", count)
    imgPatch = np.array(imgPatch).astype(img.dtype)
    mskPatch = np.array(mskPatch).astype(msk.dtype)
    if plotovermat:
        clmp = plt.imshow(overlapMat)
        plt.colorbar(clmp)
        plt.show()
    if retovermat:
        return imgPatch, mskPatch, coor, overlapMat
    else:
        return imgPatch, mskPatch, coor


normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))
        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))
        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

t = T.Compose([
               T.RandomResizedCrop(224),
               T.RandomHorizontalFlip(),
               T.ToTensor(),
               normalize
              ])

class BlockSet(BaseDataset):
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

class BlockSet2(BaseDataset):
    def __init__(self, imList, mkList, img_transform=None, msk_transform=None):
        self.imList = imList
        self.mkList = mkList
        self.img_transform = img_transform
        self.msk_transform = msk_transform

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, item):
        imgPIL = Image.open(self.imList[item]).convert('RGB')
        mskPIL = Image.open(self.mkList[item]).convert('L')
        imar = np.asarray(imgPIL)
        mask = np.asarray(mskPIL)
        mask = mask[:, :, None].repeat(3, axis=2)
        # (224, 224, 3)
        seed = random.randrange(sys.maxsize)  # get a random seed so that we can reproducibly do the transofrmations
        if self.img_transform is not None:
            random.seed(seed)  # apply this seed to img transforms
            imtr = self.img_transform(imar)
            # imtr = normalize(torch.tensor(imtr))
            # imtr = F.to_tensor(np.array(imtr))
            # imtr = T.ToTensor(imtr)
        if self.msk_transform is not None:
            random.seed(seed)
            mask_new = self.msk_transform(mask) # mask--> ndarray
            # print(mask_new.dtype)
            # [10, 3, 224, 224]
            mask_new = np.asarray(mask_new)[:, :, 0].squeeze()
            mktr = F.to_tensor(mask_new)  # converts to tensor with [0,255] ~ [0,1]
        # fixme: do we need to one-hot? Depends on the loss function.
            # torch.Size([3, 224, 224])

        # imar = np.transpose(imtr, (2, 0, 1))
        # imar = imar[None, ...]

        # mkar = np.asarray(mskPIL)
        # mkoh = torch.ones(np.append([2], np.array(mktr.shape)), dtype=mkar.dtype)
        mkoh = torch.ones([2, 224, 224])
        mkoh[1, ...] = mktr == 1.
        mkoh[0, ...] = mkoh[0, ...] - mkoh[1, ...]
        # mkoh = mkoh[None, ...]
        return imtr, mktr, mkoh # updated: mktr -> mask_new

def dice_loss(input, target, epsilon = 1e-6): # 3 -> 6
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    uniques = np.unique(target.cpu().numpy())
    assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"
    probs = input #F.softmax(input, dim=1) #channel dimension added.
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=0)  #
    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)  # b,c,1,1
    den1 = torch.sum(den1, dim=0)
    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)  # b,c,1,1
    den2 = torch.sum(den2, dim=0)
    dice = 2 * ((num + epsilon) / (den1 + den2 + epsilon))
    # print(dice)
    dice_fg = dice[1:]  # we ignore bg dice val, and take the fg
    # print("dice fg", dice_fg)
    dice_loss = -1 * torch.sum(dice)  # /dice_fg.size(0)
    return dice_loss, dice_fg, torch.sum(dice)

def focal_loss(y_pred, y_true, clweight, gamma=2.0):
    # y_true = y_true.type(torch.cuda.FloatTensor)
    clweight = torch.tensor(clweight, dtype=y_pred.dtype, device=y_pred.device)
    y_pred = torch.clamp(y_pred,  1e-6, 1-1e-6)
    cross_entropy = -y_true * torch.log(y_pred)
    axis = identify_axis(y_pred.shape)
    floss = torch.mean(
                  torch.mean(
                    cross_entropy*torch.pow(1-y_pred, gamma),
                  axis),
            0)*clweight #.cuda()
    return torch.sum(floss), floss
# class BCEDiceLoss(nn.Module):
#     """Linear combination of BCE and Dice losses3D"""
#
#     def __init__(self, alpha=1, beta=1, classes=4):
#         super(BCEDiceLoss, self).__init__()
#         self.alpha = alpha
#         self.bce = nn.BCEWithLogitsLoss()
#         self.beta = beta
#         self.dice = DiceLoss(classes=classes)
#         self.classes=classes
#
#     def forward(self, input, target):
#         target_expanded = expand_as_one_hot(target.long(), self.classes)
#         assert input.size() == target_expanded.size(), "'input' and 'target' must have the same shape"
#         loss_1 = self.alpha * self.bce(input, target_expanded)
#         loss_2, channel_score = self.beta * self.dice(input, target_expanded)
#         return  (loss_1+loss_2) , channel_score
# +---------+
# | scratch |
# +---------+
if __name__ == '__main__':
    img_transform = T.Compose([
        T.ToPILImage(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True), #these need to be in a reproducible order, first affine transforms and then color
        # transforms.RandomResizedCrop(size=patch_size),
        # transforms.RandomRotation(180),
        # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=.5),
        # transforms.RandomGrayscale(),
        T.ToTensor()
        # ques: Normalize?
        # The pathological example didn't normalize.
    ])

    msk_transform = T.Compose([
        T.ToPILImage(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True), #these need to be in a reproducible order, first affine transforms and then color
        # transforms.RandomResizedCrop(size=patch_size, interpolation=Image.NEAREST),
        # transforms.RandomRotation(180),
        # transforms.ToTensor()
    ])

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

    unitSet = BlockSet2([refFiles[290]], [mskFiles[290]], img_transform=img_transform, msk_transform=msk_transform)
    x, y, z = next(iter(unitSet))
    print(torch.unique(y), y.shape)
    print(">>")
    print(torch.unique(z), z.shape)
    # print(x[20, ...].numpy().shape, type(x))
    # unloader = T.ToPILImage()
    # def tensor_to_PIL(tensor):
    #     image = tensor.cpu().clone()
    #     image = image.squeeze(0)
    #     image = unloader(image)
    #     return image
    # #
    # # print(x.shape, y.shape, torch.unique(y), torch.max(x), torch.min(x), torch.unique(y[:, 0, ...]))
    # Pimage = tensor_to_PIL(x[32, ...])
    # plt.imshow(Pimage)
    # plt.show()
    # plt.imshow(tensor_to_PIL(y[32, 0, ...]))
    # plt.show()
    # plt.imshow(tensor_to_PIL(y[32, 1, ...]))
    # plt.show()
