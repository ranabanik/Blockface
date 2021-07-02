import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        bn = True
        bs = True
        super(UNet3D, self).__init__()
        self.ec0_1_32 = self.encoder(self.in_channel, 32, bias=bs, batchnorm=bn) #True
        self.ec1_32_64 = self.encoder(32, 64, bias=bs, batchnorm=bn)
        self.ec2_64_64 = self.encoder(64, 64, bias=bn, batchnorm=bn)
        self.ec3_64_128 = self.encoder(64, 128, bias=bs, batchnorm=bn)
        self.ec4_128_128 = self.encoder(128, 128, bias=bs, batchnorm=bn)
        self.ec5_128_256 = self.encoder(128, 256, bias=bs, batchnorm=bn)
        self.ec6_256_256 = self.encoder(256, 256, bias=bs, batchnorm=bn)
        self.ec7_256_512 = self.encoder(256, 512, bias=bs, batchnorm=bn)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9_512_512 = self.decoder(512, 512, kernel_size=2, stride=2, bias=True)
        self.dc8_768_256 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc7_256_256 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc6_256_256 = self.decoder(256, 256, kernel_size=2, stride=2, bias=True)
        self.dc5_384_128 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc4_128_128 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc3_128_128 = self.decoder(128, 128, kernel_size=2, stride=2, bias=True)
        self.dc2_192_64 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc1_64_64 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc0_64_nClasses = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=True)


    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def center_crop(self, layer, target_size): # layer size = target size
        """
        :param layer: from encoder path(syn)
        :param target_size: recent output of decoder(dn) [5x1] sized vector
        :return: center_croped layer that matches target_size
        """
        # print('Layer size: ', layer.shape)
        # print('target size: ', target_size)
        batch_size, n_channels, layer_width, layer_height, layer_depth = layer.size()
        xy1 = (layer_width - target_size[2]) // 2
        xy2 = (layer_height - target_size[3]) // 2
        xy3 = (layer_depth - target_size[4]) //2
        return layer[:, :, xy1:(xy1 + target_size[2]), xy2:(xy2 + target_size[3]), xy3:(xy3 + target_size[4])]

    def forward(self, x):
        e0 = self.ec0_1_32(x)
        # print("e0: %s" % (str(e0.size())))
        syn0 = self.ec1_32_64(e0)
        # print("syn0: %s" % (str(syn0.size())))
        e1 = self.pool0(syn0)
        # print("e1: %s" % (str(e1.size())))
        e2 = self.ec2_64_64(e1)
        # print("e2: %s" % (str(e2.size())))
        syn1 = self.ec3_64_128(e2)
        # print("syn1: %s" % (str(syn1.size())))
        del e0, e1, e2

        e3 = self.pool1(syn1)
        # print("e3: %s" % (str(e3.size())))
        e4 = self.ec4_128_128(e3)
        # print("e4: %s" % (str(e4.size())))
        syn2 = self.ec5_128_256(e4)
        # print("syn2: %s" % (str(syn2.size())))
        del e3, e4

        e5 = self.pool2(syn2)
        # print("e5: %s" % (str(e5.size())))
        e6 = self.ec6_256_256(e5)
        # print("e6: %s" % (str(e6.size())))
        e7 = self.ec7_256_512(e6)
        # print("e7: %s" % (str(e7.size())))
        del e5, e6
        # print("block e7 size = %s" % (str(e7.size())))
        # print("block dc9 size = %s" % (str(self.dc9(e7).size())))
        # print("block syn2 size = %s" % (str(syn2.size())))
        d9_demo = self.dc9_512_512(e7)
        # print("d9_demo: ", d9_demo.size())
        d9 = torch.cat((d9_demo, self.center_crop(syn2, d9_demo.size())), 1) #[16, 512, 10, 10, 10] , syn2
        # print("d9: %s" % (str(d9.size())))
        del e7, syn2, d9_demo
        d8 = self.dc8_768_256(d9)
        # print("d8: %s" % (str(d8.size())))
        d7 = self.dc7_256_256(d8)
        # print("d7: %s" % (str(d7.size())))
        del d9, d8
        # print("block d7 size = %s" % (str(d7.size())))
        d6_demo = self.dc6_256_256(d7)
        d6 = torch.cat((d6_demo, self.center_crop(syn1, d6_demo.size())), 1)
        # print("d6: %s" % (str(d6.size())))
        del d7, syn1, d6_demo

        d5 = self.dc5_384_128(d6)
        # print("d5: %s" % (str(d5.size())))
        d4 = self.dc4_128_128(d5)
        # print("d4: %s" % (str(d4.size())))
        del d6, d5
        # print("d3_1: %s" % (str(self.dc3(d4).size())))
        d3_demo = self.dc3_128_128(d4)
        d3 = torch.cat((d3_demo, self.center_crop(syn0, d3_demo.size())), 1)
        # print("d3: %s" % (str(d3.size())))
        del d4, syn0, d3_demo
        # print("block d3 size = %s" % (str(d3.size())))

        d2 = self.dc2_192_64(d3)
        # print("d2: %s" % (str(d2.size())))
        d1 = self.dc1_64_64(d2)
        # print("block d2 size = %s" % (str(d2.size())))
        # del d3, d2
        # print("d1: %s" % (str(d1.size())))
        d0 = self.dc0_64_nClasses(d1)
        # print("Here", d0.size())
        out = F.softmax(d0, dim=1)
        # out = torch.sigmoid(d0)
        # print("No sigmoid")
        return out #out #out

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
