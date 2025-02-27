import torch.nn as nn
import torch
import torchvision
import pytorch_lightning as pl
import torchvision.models as models
import torch.nn.functional as F


class SimpleConvNet(pl.LightningModule):
    def __init__(self, num_classes=1):
        super().__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

        self.residual = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=2, padding=0)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
           
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # Binary classification
        )
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x_res = self.residual(x1)
        x_res = F.interpolate(x_res, size=(x3.shape[2], x3.shape[3]), mode='bilinear', align_corners=False)
        x3 += x_res

        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.global_avg_pool(x5)

        x6 = self.classifier(x6)
        return x6


class TransConvNet(pl.LightningModule):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        # Load ResNet-50 Pretrained Model
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Modify First Layer to Handle 3-Channel Images
        self.resnet.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove Fully Connected Layer
        self.resnet.fc = nn.Identity()

        # Freeze Early Layers for Transfer Learning
        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze entire model initially
        
        # Unfreeze the last few layers (fine-tuning)
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True  # Unfreeze last residual block

        # Custom Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        x = self.resnet(x)  # Feature extraction using ResNet
        x = self.classifier(x)  # Classification head
        return x
        

class UNet(pl.LightningModule):
  def __init__(self, n_classes=1, in_ch=3):
      super().__init__()
      #######################
      # Start YOUR CODE    #
      #######################
      # number of filter's list for each expanding and respecting contracting layer
      c = [16, 32, 64, 128]

      # first convolution layer receiving the image
      # encoder layers

      # decoder layers

      # last layer returning the output
      #######################
      # END OF YOUR CODE    #
      #######################
  def forward(self,x):
      #######################
      # Start YOUR CODE    #
      #######################
      # encoder

      # decoder

      #######################
      # END OF YOUR CODE    #
      #######################
      return x


def conv3x3_bn(ci, co):
    #######################
    # Start YOUR CODE    #
    #######################
    pass
    #######################
    # end YOUR CODE    #
    #######################

def encoder_conv(ci, co):
    #######################
    # Start YOUR CODE    #
    #######################
    pass
    #######################
    # end YOUR CODE    #
    #######################

class deconv(nn.Module):
  def __init__(self, ci, co):
    super(deconv, self).__init__()
    #######################
    # Start YOUR CODE    #
    #######################
    pass
    #######################
    # end YOUR CODE    #
    #######################

  def forward(self, x1, x2):
      #######################
      # Start YOUR CODE    #
      #######################
      x=x1
      #######################
      # end YOUR CODE    #
      #######################
      return x
