import torch.nn as nn
import torch
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F


class SimpleConvNet(pl.LightningModule):
    def __init__(self):
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

        self.residual = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=4)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
           
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        

        self.classifier = nn.Sequential(
            # linear layers
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=128, out_features=1)
        )
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x_res = self.residual(x1)
        x3 += x_res

        x4 = self.conv4(x3)
        x5 = self.global_avg_pool(x4)
        x5 = self.classifier(x5)
        return x5


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
