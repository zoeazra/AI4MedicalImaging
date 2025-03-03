import torch.nn as nn
import torch
import torchvision
import pytorch_lightning as pl
import torchvision.models as models
import torch.nn.functional as F

class SimpleConvNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.layers = nn.Sequential(
            # conv block 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # conv block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.classifier = nn.Sequential(
            # linear layers
            nn.AdaptiveAvgPool2d(output_size=(4, 4)),
            nn.Flatten(),
            nn.Linear(in_features=4 * 4 * 32, out_features=60),
            nn.ReLU(),
            nn.Linear(in_features=60, out_features=1)
        )
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        x = self.layers(x)
        x = self.classifier(x)
        return x

class CustomConvNet(pl.LightningModule):
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
      super().__init__()
      c = [16, 32, 64, 128, 256]  # Number of filters

    # Encoder (Contracting path)
      self.enc1 = encoder_conv(in_ch, c[0])
      self.enc2 = encoder_conv(c[0], c[1])
      self.enc3 = encoder_conv(c[1], c[2])
      self.enc4 = encoder_conv(c[2], c[3])

    # Bottleneck
      self.bottleneck = nn.Sequential(
        conv3x3_bn(c[3], c[4]),
        conv3x3_bn(c[4], c[4])
      )

    # Decoder (Expanding path)
      self.dec4 = deconv(c[4], c[3])
      self.dec3 = deconv(c[3], c[2])
      self.dec2 = deconv(c[2], c[1])
      self.dec1 = deconv(c[1], c[0])

    # Output layer
      self.final = nn.Conv2d(c[0], n_classes, kernel_size=1)

      #######################
      # END OF YOUR CODE    #
      #######################
  def forward(self,x):
      #######################
      # Start YOUR CODE    #
      #######################
      # encoder
      x1 = self.enc1(x)
      x2 = self.enc2(x1)
      x3 = self.enc3(x2)
      x4 = self.enc4(x3)

        # Bottleneck
      x_b = self.bottleneck(x4)

    # Decoding path with skip connections
      x = self.dec4(x_b, x4)
      x = self.dec3(x, x3)
      x = self.dec2(x, x2)
      x = self.dec1(x, x1)

      #######################
      # END OF YOUR CODE    #
      #######################
      return torch.sigmoid(self.final(x))


def conv3x3_bn(ci, co):
    #######################
    # Start YOUR CODE    #
    #######################
    return nn.Sequential(
        nn.Conv2d(ci, co, kernel_size=3, padding=1),
        nn.BatchNorm2d(co),
        nn.LeakyReLU(inplace=True)
    )
    #######################
    # end YOUR CODE    #
    #######################

def encoder_conv(ci, co):
    #######################
    # Start YOUR CODE    #
    #######################
    return nn.Sequential(
        conv3x3_bn(ci, co),
        conv3x3_bn(co, co),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    #######################
    # end YOUR CODE    #
    #######################

class deconv(nn.Module):
  def __init__(self, ci, co):
    super(deconv, self).__init__()
    #######################
    # Start YOUR CODE    #
    #######################
    super().__init__()
    self.upconv = nn.ConvTranspose2d(ci, co, kernel_size=2, stride=2)
    self.conv = nn.Sequential(
        conv3x3_bn(ci, co),
        conv3x3_bn(co, co)
    )
    #######################
    # end YOUR CODE    #
    #######################

  def forward(self, x1, x2):
      #######################
      # Start YOUR CODE    #
      #######################
      
    x1 = self.upconv(x1)
    x = torch.cat([x1, x2], dim=1)  # Skip connection
    return self.conv(x)
      #######################
      # end YOUR CODE    #
      #######################
