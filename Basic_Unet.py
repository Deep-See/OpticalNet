import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import init

class UNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(UNet, self).__init__()

        self.encoder1 = self.conv_block(img_ch, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.encoder2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.encoder3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.encoder4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = self.conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        # self.decoder4 = self.conv_block(512, 512)  #---- remove skip connection
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # self.decoder3 = self.conv_block(512, 256)
        self.decoder3 = self.conv_block(256, 256) #---- remove skip connection
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.decoder2 = self.conv_block(256, 128)
        self.decoder2 = self.conv_block(128, 128)    #---- remove skip connection
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.decoder1 = self.conv_block(128, 64)
        self.decoder1 = self.conv_block(64, 64)    #---- remove skip connection
        self.final_conv = nn.Conv2d(64, output_ch, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1 if in_channels == 1 else 0.2 if in_channels == 256 else 0.3),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        # dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        # dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        # dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.final_conv(dec1))

# ========        for debug and see each layers output ====== #
    # import matplotlib.pyplot as plt
    # import numpy as np

    # arr = dec4.cpu().detach().numpy()
    # img = arr[0, 0]

    # plt.figure(figsize=(10, 8))
    # im = plt.imshow(img, cmap='viridis')
    # plt.colorbar(im)
    # plt.title("Visualization of dec4 (first channel)")

    # plt.show()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class BinaryDNN(nn.Module):
    def __init__(self):
        super(BinaryDNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 64, 4096)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4096, 64 * 64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        # x = self.sigmoid(x)   --- in solver.py
        x = x.view(-1, 1, 64, 64)
        return x


class CustomResNet34(nn.Module):
    def __init__(self):
        super(CustomResNet34, self).__init__()
        resnet = models.resnet34(pretrained=False)   #put False for training from scratch

        # 初始层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1.weight.data = torch.mean(resnet.conv1.weight.data, dim=1, keepdim=True)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 残差块
        self.layer1 = resnet.layer1  # 3 个残差块
        self.layer2 = resnet.layer2  # 4 个残差块
        self.layer3 = resnet.layer3  # 6 个残差块
        self.layer4 = resnet.layer4  # 3 个残差块

        # 上采样层
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(64 * 64, 90 * 90)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.upsample(x)

        x = x.view(-1, 1, 64, 64)


        return x

class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        resnet = models.resnet18(pretrained=False)   #put False for training from scratch

        # 初始层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1.weight.data = torch.mean(resnet.conv1.weight.data, dim=1, keepdim=True)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 残差块
        self.layer1 = resnet.layer1  # 3 个残差块
        self.layer2 = resnet.layer2  # 4 个残差块
        self.layer3 = resnet.layer3  # 6 个残差块
        self.layer4 = resnet.layer4  # 3 个残差块

        # 上采样层
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(64 * 64, 90 * 90)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.upsample(x)

        x = x.view(-1, 1, 64, 64)

        return x