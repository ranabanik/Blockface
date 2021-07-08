# testing UNet_BN

import os
from glob import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from Codes.Utilities import BlockSet2, patchify
from Codes.Network import Unet_BN
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

patch_size = 224
# print(random.randrange(sys.maxsize))

img_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True), #these need to be in a reproducible order, first affine transforms and then color
        # transforms.RandomResizedCrop(size=patch_size),
        # transforms.RandomRotation(180),
        # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=.5),
        # transforms.RandomGrayscale(),
        transforms.ToTensor()
    # ques: Normalize?
    # The pathological example didn't normalize.
    ])

msk_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True), #these need to be in a reproducible order, first affine transforms and then color
        # transforms.RandomResizedCrop(size=patch_size, interpolation=Image.NEAREST),
        # transforms.RandomRotation(180),
        # transforms.ToTensor()
])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
modelDir = r'/media/banikr2/DATA/Diesel_block/model'
# imgDir = r'/media/banikr2/DATA/Diesel_block/OneImage/Image'
# mskDir = r'/media/banikr2/DATA/Diesel_block/OneImage/Mask'
imgDir = r'/media/banikr2/DATA/Diesel_block/patches/Image'
mskDir = r'/media/banikr2/DATA/Diesel_block/patches/Mask'
valImg = np.random.randn(900, 900, 3)
# print(valImg.shape)
valMsk = np.random.randn(900, 900)
_, _, co, overlapmat = patchify(valImg, valMsk, [224, 224], [100, 100], retovermat=True, plotovermat=True, resprnt=False)
# print(overlapmat.shape)
# print(co[10])

imgFiles = sorted(glob(os.path.join(imgDir, '*.tif')))
print(imgFiles[320-64:])
mskFiles = sorted(glob(os.path.join(mskDir, '*.png')))
validSet = BlockSet2(imgFiles[320-64:], mskFiles[320-64:], img_transform=img_transform, msk_transform=msk_transform)
validLoader = DataLoader(validSet, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

FILEPATH_MODEL_LOAD = os.path.join(modelDir, 'UNet_BN_200_1_sub_BCE_noT.pth')
train_states = torch.load(FILEPATH_MODEL_LOAD)
model = Unet_BN(n_classes=2, in_channels=3).to(device)
model.load_state_dict(train_states['train_states_best']['model_state_dict'])
model.eval()
# mask =
count = 0
mask = np.zeros([900, 900])
# print(mask.shape)
for bIdx, sample in enumerate(validLoader):
    img, seg = sample[0].to(device), sample[2].to(device)
    pred = model(img)
    # print(pred.shape, np.max(pred.data[1].cpu().numpy()))
    lbl_pred = pred.data.max(1)[1].cpu().numpy()[:, :].astype('uint8')
    # print(lbl_pred.shape)
    mask[co[count][0]:co[count][0]+224, co[count][1]:co[count][1]+224] += lbl_pred[0, ...]
    # plt.imshow(lbl_pred[0, ...], cmap='gray')
    # plt.show()
    seg = seg.cpu().numpy()
    # plt.imshow(seg[0, 1, ...])
    # plt.show()
    count += 1

# print(count)
plt.imshow(mask/overlapmat)
plt.show()