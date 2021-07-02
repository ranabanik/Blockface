import numpy as np
import os
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as F
from torch.utils.tensorboard import SummaryWriter
from Codes.Network import UNet3D, UNet
from Codes.Utilities import BlockSet2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from time import time
import math
import random
import sys
phases = ["train", "valid"]
validation_phases = ["valid"]
in_channels = 3
n_classes = 2
wf = 2
depth = 5
padding = True
up_mode = 'upconv'
batch_norm = True
ignore_index = -100
edge_weight = 1.1 # ques: we didn't use?
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
modelDir = r'/media/banikr2/DATA/Diesel_block/model'
imgDir = r'/media/banikr2/DATA/Diesel_block/patches/Image'
mskDir = r'/media/banikr2/DATA/Diesel_block/patches/Mask'
imgFiles = sorted(glob(os.path.join(imgDir, '*.tif')))
mskFiles = sorted(glob(os.path.join(mskDir, '*.png')))
#helper function for pretty printing of current time and remaining time
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time()
    s = now - since
    es = s / (percent+.00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
patch_size = 224
# print(random.randrange(sys.maxsize))

img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True), #these need to be in a reproducible order, first affine transforms and then color
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
        transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True), #these need to be in a reproducible order, first affine transforms and then color
        transforms.RandomResizedCrop(size=patch_size, interpolation=Image.NEAREST),
        transforms.RandomRotation(180),
        # transforms.ToTensor()
    ])

dataSet = {}
dataLoader = {}
unitSet = BlockSet2(imgFiles, mskFiles, img_transform=img_transform, msk_transform=msk_transform)
# for phase in phases:
dataSet[phases[0]] = BlockSet2(imgFiles[0:300], mskFiles[0:300], img_transform=img_transform, msk_transform=msk_transform)
dataLoader[phases[0]] = DataLoader(dataSet[phases[0]], batch_size=10, shuffle=True, num_workers=1, pin_memory=True)
dataSet[phases[1]] = BlockSet2(imgFiles[300:], mskFiles[300:], img_transform=img_transform, msk_transform=msk_transform)
dataLoader[phases[1]] = DataLoader(dataSet[phases[1]], batch_size=10, shuffle=False, num_workers=1, pin_memory=True)
# x, y = next(iter(unitSet)) # y : [0, 255], shape: (224, 224)
# unitLoader = DataLoader(unitSet, batch_size=10, shuffle=True, num_workers=1,pin_memory=True)
# x, y = next(iter(unitLoader))
# print(torch.max(x), torch.min(x), type(y), torch.max(y), torch.min(y), x.shape, y.shape)
# tensor(1.) tensor(0.) <class 'torch.Tensor'> tensor(1.) tensor(0.) torch.Size([10, 3, 224, 224]) torch.Size([10, 1, 224, 224])
model = UNet(n_classes=n_classes, in_channels=in_channels, padding=padding, depth=depth, wf=wf, up_mode=up_mode, batch_norm=batch_norm).to(device)
# fixme: weight initialization?
# same spatial size output: torch.Size([10, 2, 224, 224])
optim = torch.optim.Adam(model.parameters())
# ques: or typically best way to start is SGD
# optim = torch.optim.SGD(model.parameters(),
#                           lr=.1,
#                           momentum=0.9,
#                           weight_decay=0.0005)
class_weight = torch.from_numpy(np.array([0, 1])).type('torch.FloatTensor').to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight, reduce=False)
writer = SummaryWriter()
num_epochs = 100
best_loss_on_test = np.Infinity
start_time = time()
for epoch in range(num_epochs):
    all_acc = {key: 0 for key in phases}
    all_loss = {key: torch.zeros(1).to(device) for key in phases}
    # print("<<<", all_loss)
    cmatrix = {key: np.zeros((2, 2)) for key in phases}
    for phase in phases:  # iterate through both training and validation states
        if phase == 'train':
            model.train()  # Set model to training mode
        else:  # when in eval mode, we don't want parameters to be up-dated
            model.eval()  # Set model to evaluate mode
        for ii, (X, y) in enumerate(dataLoader[phase]):  # for each of the batches
            X = X.to(device)  # [Nbatch, 3, H, W]
            # y_weight = y_weight.type('torch.FloatTensor').to(device)
            y = y.type('torch.LongTensor').to(device)  # [Nbatch, H, W] with class indices (0, 1)
            # print(y.shape)
            y = y.squeeze(1)
            # print("target:", y.shape, type(y), y.dtype)
            with torch.set_grad_enabled(phase == 'train'):
                prediction = model(X)  # [N, Nclass, H, W]
                # print(prediction.shape)
                # print("prediction:", prediction.shape, type(prediction), prediction.dtype)
                loss_matrix = criterion(prediction, y)
                loss = (loss_matrix).mean()  #
                if phase == "train":  # in case we're in train mode, need to do back propogation
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    train_loss = loss
                # print(loss)
                # print(">>>>>", loss.detach().view(1, -1), type(loss.detach().view(1, -1)), loss.detach().view(1, -1).dtype)
                # all_loss[phase] = torch.cat((all_loss[phase], loss.detach().view(1, -1)))
                # print(all_loss)
                if phase in validation_phases:  # if this phase is part of validation, compute confusion matrix
                    p = prediction[:, :, :, :].detach().cpu().numpy()
                    cpredflat = np.argmax(p, axis=1).flatten()
                    yflat = y.cpu().numpy().flatten()

                    cmatrix[phase] = cmatrix[phase] + confusion_matrix(yflat, cpredflat, labels=range(n_classes))

            all_acc[phase] = (cmatrix[phase] / cmatrix[phase].sum()).trace()
            # all_loss[phase] = all_loss[phase].cpu().numpy().mean()

            # save metrics to tensorboard
            writer.add_scalar(f'{phase}/loss', all_loss[phase], epoch)
            if phase in validation_phases:
                writer.add_scalar(f'{phase}/acc', all_acc[phase], epoch)
                writer.add_scalar(f'{phase}/TN', cmatrix[phase][0, 0], epoch)
                writer.add_scalar(f'{phase}/TP', cmatrix[phase][1, 1], epoch)
                writer.add_scalar(f'{phase}/FP', cmatrix[phase][0, 1], epoch)
                writer.add_scalar(f'{phase}/FN', cmatrix[phase][1, 0], epoch)
                writer.add_scalar(f'{phase}/TNR', cmatrix[phase][0, 0] / (cmatrix[phase][0, 0] + cmatrix[phase][0, 1]),
                                  epoch)
                writer.add_scalar(f'{phase}/TPR', cmatrix[phase][1, 1] / (cmatrix[phase][1, 1] + cmatrix[phase][1, 0]),
                                  epoch)

        print('%s ([%d/%d] %d%%), train loss: ' % (timeSince(start_time, (epoch + 1) / num_epochs), #%.4f test loss: %.4f
                                                                       epoch + 1, num_epochs,
                                                                       (epoch + 1) / num_epochs * 100))
                                                                       # all_loss["train"], all_loss["val"]), end="")

        # if current loss is the best we've seen, save model state with all variables
        # necessary for recreation
        # if all_loss["valid"] < best_loss_on_test:
        #     best_loss_on_test = all_loss["valid"]
        #     print("  **")
        #     state = {'epoch': epoch + 1,
        #              'model_dict': model.state_dict(),
        #              'optim_dict': optim.state_dict(),
        #              'best_loss_on_test': all_loss,
        #              'n_classes': n_classes,
        #              'in_channels': in_channels,
        #              'padding': padding,
        #              'depth': depth,
        #              'wf': wf,
        #              'up_mode': up_mode, 'batch_norm': batch_norm}
        #
        #     torch.save(state, os.path.join(modelDir, "{}_unet_best_model.pth".format(epoch)))
        # else:
        #     print("")
if __name__ != '__main__':
    imgPIL = Image.open(imgFiles[10]).convert('RGB')
    imar = np.asarray(imgPIL)
    imar = np.transpose(imar, (2, 0, 1))
    imar = imar[None, ...]
    print(imar.shape)

    out = model(x)['out']
    # print(model)
    out = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    print(out.shape)
    # (21, 224, 224)
    plt.imshow(out)
    plt.show()


    def decode_segmap(image, nc=21):
        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb


    rgb = decode_segmap(om)
    plt.imshow(rgb);
    plt.axis('off');
    plt.show()


    class DeepLabV3Block(nn.Module):
        def __init__(self, use_pretrained=True):
            super(DeepLabV3Block, self).__init__()
            deeplabv3 = models.segmentation.deeplabv3_resnet101(pretrained=use_pretrained)
            self.b4fl = nn.Sequential(*list(deeplabv3.children())[:-1])
            self.fl = nn.Conv2d(21, 2, kernel_size=(1, 1), stride=(1, 1))

        def forward(self, x):
            x = self.b4fl(x)
            x = self.fl(x)
            return x


    # models.segmentation.deeplabv3_resnet101()
    # print(models.segmentation.deeplabv3_resnet101())
    # b4fl = nn.Sequential(*list(models.segmentation.deeplabv3_resnet101().children())[:-1])
    # print(b4fl) #*list(b4fl.children())[-1])
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    # print(b4fl)
    # fl = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    # model = DeepLabV3Block(use_pretrained=True)
    # out = model(x)
    # print(model)
    out = model(x)['out']
    # # print(model)
    # om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    # print(om.shape)
    # print(np.unique(om))
    # (21, 224, 224)
    # plt.imshow(out)
    # plt.show()

    # +-----------------------------------------+
    # | Dataset output and transformation works |
    # +-----------------------------------------+
    I = Image.open(imgFiles[idx]).convert('RGB')
    plt.imshow(I)
    plt.show()
    M = Image.open(mskFiles[idx]).convert('L')
    plt.imshow(M)
    plt.show()
    unitLoader = DataLoader(unitSet, batch_size=1, shuffle=True, num_workers=1)
    # x, y = next(iter(unitLoader))
    print(x.shape, y.shape, type(x), type(y))
    unloader = transforms.ToPILImage()
    def tensor_to_PIL(tensor):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        return image


    #     #
    #     # print(x.shape, y.shape, torch.unique(y), torch.max(x), torch.min(x), torch.unique(y[:, 0, ...]))
    Pimage = tensor_to_PIL(x)
    plt.imshow(Pimage)
    plt.show()
    plt.imshow(y)
    plt.show()

