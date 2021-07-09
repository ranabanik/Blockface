# BCE loss with one image overfit, with no transformation, reaches dice 1.0 in less than 30 epochs.
# fixme: trying with one subject
import numpy as np
import os
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from Codes.Network import Unet_BN
from Codes.Utilities import BlockSet2, weights_init, dice_loss, focal_loss
import segmentation_models_pytorch as smp
from PIL import Image
import time
import pickle

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
modelDir = r'/media/banikr2/DATA/Diesel_block/model'
imgDir = r'/media/banikr2/DATA/Diesel_block/patches/Image'
mskDir = r'/media/banikr2/DATA/Diesel_block/patches/Mask'
imgFiles = sorted(glob(os.path.join(imgDir, '*.tif')))
mskFiles = sorted(glob(os.path.join(mskDir, '*.png')))

patch_size = 224
# print(random.randrange(sys.maxsize))

img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True), #these need to be in a reproducible order, first affine transforms and then color
        transforms.RandomResizedCrop(size=patch_size),
        transforms.RandomRotation(180),
        # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=.5),
        # transforms.RandomGrayscale(),
        transforms.ToTensor()
    # ques: Normalize?
    # The pathological example didn't normalize.
    ])

msk_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True), #these need to be in a reproducible order, first affine transforms and then color
        transforms.RandomResizedCrop(size=patch_size, interpolation=Image.NEAREST),
        transforms.RandomRotation(180),
        # transforms.ToTensor()
])

trainSet = BlockSet2(imgFiles[0:300], mskFiles[0:300], img_transform=img_transform, msk_transform=msk_transform)
# torch.Size([3, 224, 224]) torch.Size([2, 224, 224])
trainLoader = DataLoader(trainSet, batch_size=10, shuffle=True, num_workers=1, pin_memory=True)
validSet = BlockSet2(imgFiles[300:], mskFiles[300:], img_transform=img_transform, msk_transform=msk_transform)
validLoader = DataLoader(validSet, batch_size=10, shuffle=False, num_workers=1, pin_memory=True)
# x, y, z = next(iter(trainLoader))
# print(x.shape, y.shape, z.shape, x.dtype, y.dtype, z.dtype)
# print(torch.max(x), torch.min(x), torch.unique(y), torch.unique(z))
# model = Unet_BN(n_classes=2, in_channels=3).to(device)
model = smp.Unet(
    encoder_name="resnet101",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2                       # model output channels (number of classes in your dataset)
).to(device)
learning_rate = 1e-4
clwt = [0.2, 0.8]
lmbd = 0.3

# bce = nn.BCEWithLogitsLoss()
bce = nn.BCELoss()
#
# out = model(x.to(device))
# # print(out.shape)
#
# print("loss: ", bce(out, z.to(device)))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
FILEPATH_MODEL_LOAD = None
"""%%%%%%% Saving Criteria of Model %%%%%%%"""
if FILEPATH_MODEL_LOAD is not None:  # Only needed for loading models or transfer learning
    train_states = torch.load(FILEPATH_MODEL_LOAD)
    model.load_state_dict(train_states['train_states_latest']['model_state_dict'])
    optimizer.load_state_dict(train_states['train_states_latest']['optimizer_state_dict'])
    train_states_best = train_states['train_states_best']
    loss_valid_min = train_states_best['loss_valid_min']
    model_save_criteria = train_states_best['model_save_criteria']
else:
    train_states = {}
    model_save_criteria = 0

model_ = 'UNet_smp_dice_w_tform'
FILEPATH_MODEL_SAVE = os.path.join(modelDir, '{}.pth'.format(model_))
# this_project_dir = os.path.join(modelDir, TIME_STAMP)
FILEPATH_LOG = os.path.join(modelDir, '{}.bin'.format(model_))
max_epochs = 50
loss_epoch_train = []
loss_epoch_valid = []
dice_score_train = []
dice_score_valid = []
for epoch in range(max_epochs):
    print('training ...')
    running_loss = 0
    running_dice = 0
    running_time_batch = 0
    time_batch_start = time.time()
    model.train()
    for bIdx, sample in enumerate(trainLoader):
        time_batch_load = time.time() - time_batch_start
        time_compute_start = time.time()
        optimizer.zero_grad()
        img, seg = sample[0].to(device), sample[2].to(device)
        out = model(img)
        # print(out.shape, seg.shape)
        dloss, dice, _ = dice_loss(out, seg)
        # bloss = bce(out, seg)
        floss, _ = focal_loss(out, seg, clwt, gamma=2.0)
        loss = lmbd * dloss + (1 - lmbd) * floss
        loss.backward()
        optimizer.step()
        running_loss += dloss.item()
        mean_loss = running_loss / (bIdx + 1)
        running_dice += dice.item()  # dice_per_channel
        mean_dice = running_dice / (bIdx + 1)
        # print time stats
        time_compute = time.time() - time_compute_start
        time_batch = time_batch_load + time_compute
        running_time_batch += time_batch
        time_batch_avg = running_time_batch / (bIdx + 1)
        print(
            'epoch: {}/{}, batch: {}/{}, loss-train: {:.4f}, batch time taken: {:.2f}s, eta_epoch: {:.2f} hours'.format(
                epoch + 1, max_epochs,
                bIdx + 1, len(trainLoader),
                mean_loss,
                time_batch,
                time_batch_avg * (len(trainLoader) - (bIdx + 1)) / 3600))
    # scheduler.step()
    loss_epoch_train.append(mean_loss)
    dice_score_train.append(mean_dice)
    print('validating...')
    running_loss = 0
    running_dice = 0
    model.eval()
    with torch.no_grad():
        for vIdx, sample in enumerate(validLoader):
            img, seg = sample[0].to(device), sample[2].to(device)
            # segC = sample[2].to(device)
            out = model(img)
            dloss, dice, _ = dice_loss(out, seg)
            # bloss = bce(out, seg)
            floss, _ = focal_loss(out, seg, clwt, gamma=2.0)
            loss = lmbd * dloss + (1 - lmbd) * floss
            running_loss += loss.item()
            mean_loss = running_loss / (vIdx + 1)
            running_dice += dice.item()  # dice_per_channel
            mean_dice = running_dice / (vIdx + 1)
            print('epoch: {}/{}, batch: {}/{}, mean-loss: {:.4f}'.format(
                epoch + 1, max_epochs,
                vIdx + 1, len(validLoader),
                mean_loss))
    loss_epoch_valid.append(mean_loss)
    dice_score_valid.append(mean_dice)
    # print('loss_epoch_valid: {}'.format(loss_epoch_valid))

    chosen_criteria = mean_dice  #
    print('Criteria at the end of epoch {} is {:.4f}'.format(epoch + 1, chosen_criteria))
    # fv.write('Criteria at the end of epoch {} is {:.4f}'.format(epoch + 1, chosen_criteria))
    if chosen_criteria > model_save_criteria:
        print('criteria increased from {:.4f} to {:.4f}, saving model ...'
              .format(model_save_criteria, chosen_criteria))
        train_states_best = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_save_criteria': chosen_criteria
        }
        train_states['train_states_best'] = train_states_best
        torch.save(train_states, FILEPATH_MODEL_SAVE)
        model_save_criteria = chosen_criteria
    if (epoch + 1) % 10 == 0:  # save model every 10 epochs
        train_states_latest = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_save_criteria': chosen_criteria}
        train_states['train_states_latest'] = train_states_latest
        torch.save(train_states, FILEPATH_MODEL_SAVE)
    print("Updating log...")
    log = {
        'loss_train': loss_epoch_train,
        'loss_valid': loss_epoch_valid,
        'dice_train': dice_score_train,
        'dice_valid': dice_score_valid}
    with open(FILEPATH_LOG, 'wb') as pfile:
            pickle.dump(log, pfile)