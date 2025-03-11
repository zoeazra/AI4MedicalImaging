import os
import glob
import nibabel as nib
import torch
import numpy as np
from scipy.ndimage import rotate
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision import transforms
import cv2


# Data loader
class Scan_DataModule(pl.LightningDataModule):
  def __init__(self, config,transform=True):
    super().__init__()
    self.train_data_dir   = config['train_data_dir']
    self.val_data_dir     = config['val_data_dir']
    self.test_data_dir    = config['test_data_dir']
    self.batch_size       = config['batch_size']
    if transform:
      self.train_transforms = transforms.Compose([
                Random_Rotate(0.1),
                Random_Flip(),
                Random_GaussianBlur(),
                transforms.ToTensor()
            ])
    else:
      self.train_transforms = transforms.Compose([transforms.ToTensor()])
    self.val_transforms  = transforms.Compose([transforms.ToTensor()])

  def setup(self, stage=None):
    self.train_dataset = Scan_Dataset(self.train_data_dir, transform = self.train_transforms)
    self.val_dataset   = Scan_Dataset(self.val_data_dir  , transform = self.val_transforms)
    self.test_dataset = Scan_Dataset(self.test_data_dir  , transform = self.val_transforms)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False)

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = False)


class Scan_DataModule_Segm(pl.LightningDataModule):
  def __init__(self, config, transform=False):
    super().__init__()
    self.train_data_dir   = config['train_data_dir']
    self.val_data_dir     = config['val_data_dir']
    self.test_data_dir    = config['test_data_dir']
    self.batch_size       = config['batch_size']
    if transform:
      self.train_transforms = transforms.Compose([Random_Rotate_Seg(0.1),
                                                  Random_Flip_Seg(),
                                                  Random_GaussianBlur_Seg(),
                                                  ToTensor_Seg()])
    else:
      self.train_transforms = transforms.Compose([ToTensor_Seg()])
    self.val_transforms   = transforms.Compose([ToTensor_Seg()])

  def setup(self, stage=None):
    self.train_dataset = Scan_Dataset_Segm(self.train_data_dir, transform = self.train_transforms)
    self.val_dataset   = Scan_Dataset_Segm(self.val_data_dir  , transform = self.val_transforms)
    self.test_dataset = Scan_Dataset_Segm(self.test_data_dir  , transform = self.val_transforms)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle=False)


# Data module
class Scan_Dataset(Dataset):
    def __init__(self, data_dir, transform=False):
        self.transform = transform
        self.data_list = sorted(glob.glob(os.path.join(data_dir, 'img*.nii.gz')))


    def __len__(self):
        """defines the size of the dataset (equal to the length of the data_list)"""
        return len(self.data_list)

    def __getitem__(self, idx):
        """ensures each item in data_list is randomly and uniquely assigned an index (idx) so it can be loaded"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # loading image
        image_name = self.data_list[idx]
        image = nib.load(image_name).get_fdata()
        # image = np.transpose(image, (2, 0, 1))

        # setting label from image name
        label = int(image_name.split('.')[0][-1])
        label = torch.tensor(label)

        # apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


class Scan_Dataset_Segm(Dataset):
  def __init__(self, data_dir, transform=False):
    self.transform = transform
    self.img_list = sorted(glob.glob(os.path.join(data_dir,'img*.nii.gz')))
    self.msk_list = sorted(glob.glob(os.path.join(data_dir,'msk*.nii.gz')))

  def __len__(self):
    """defines the size of the dataset (equal to the length of the data_list)"""
    return len(self.img_list)

  def __getitem__(self, idx):
    """ensures each item in data_list is randomly and uniquely assigned an index (idx) so it can be loaded"""

    if torch.is_tensor(idx):
      idx = idx.tolist()

    # loading image
    image_name = self.img_list[idx]
    image = nib.load(image_name).get_fdata()

    # loading mask
    mask_name = self.msk_list[idx]
    mask = nib.load(mask_name).get_fdata()
    mask = np.expand_dims(mask, axis=2)

    # make sample
    sample = {'image': image, 'mask': mask}

    # apply transforms
    if self.transform:
      sample = self.transform(sample)

    return sample


# data augmentation. You can edit this to add additional augmentation options
class Random_Rotate(object):
  """Rotate ndarrays in sample."""
  def __init__(self, probability):
    assert isinstance(probability, float) and 0 < probability <= 1, 'Probability must be a float number between 0 and 1'
    self.probability = probability

  def __call__(self, sample):
    if float(torch.rand(1, dtype=torch.float64)) < self.probability:
      angle = float(torch.randint(low=-10, high=11, size=(1,)))
      sample = rotate(sample, angle, axes=(0, 1), reshape=False, order=3, mode='nearest')
    return sample.copy()


class Random_Rotate_Seg(object):
  """Rotate ndarrays in sample."""
  def __init__(self, probability):
    assert isinstance(probability, float) and 0 < probability <= 1, 'Probability must be a float number between 0 and 1'
    self.probability = probability

  def __call__(self, sample):
    image, mask = sample['image'], sample['mask']
    if float(torch.rand(1, dtype=torch.float64)) < self.probability:
      angle = float(torch.randint(low=-10, high=11, size=(1,)))
      image = rotate(image, angle, axes=(0, 1), reshape=False, order=3, mode='nearest')
      mask = rotate(mask, angle, axes=(0, 1), reshape=False, order=0, mode='nearest')
    return {'image': image.copy(), 'mask': mask.copy()}


class ToTensor_Seg(object):
  """applies ToTensor for dict input"""
  def __call__(self, sample):
    image, mask = sample['image'], sample['mask']
    image = transforms.ToTensor()(image)
    mask = transforms.ToTensor()(mask)
    return {'image': image.clone(), 'mask': mask.clone()}


class Random_Flip(object):
    """Randomly apply horizontal and/or vertical flip to a NumPy image array with a given probability."""
    def __init__(self, probability=0.5, horizontal=True, vertical=True):
        """
        Args:
        - probability (float): Probability of applying a flip.
        - horizontal (bool): Whether to allow horizontal flipping.
        - vertical (bool): Whether to allow vertical flipping.
        """
        assert isinstance(probability, float) and 0 < probability <= 1, 'Probability must be a float between 0 and 1'
        self.probability = probability
        self.horizontal = horizontal
        self.vertical = vertical

    def __call__(self, sample):
        # Apply flipping based on probability
        if np.random.rand() < self.probability:
            if self.horizontal and np.random.rand() > 0.5:
                sample = np.fliplr(sample)  # Horizontal flip

            if self.vertical and np.random.rand() > 0.5:
                sample = np.flipud(sample)  # Vertical flip

        return sample.copy()

class Random_GaussianBlur(object):
    """Randomly apply Gaussian blur to a NumPy image array with a given probability."""
    def __init__(self, probability=0.4, kernel_size=7, sigma=(2.0, 5.0)):
        """
        Args:
        - probability (float): Probability of applying Gaussian blur.
        - kernel_size (int): Size of the Gaussian kernel (must be odd).
        - sigma (tuple): Range of standard deviation for Gaussian kernel.
        """
        assert isinstance(probability, float) and 0 < probability <= 1, 'Probability must be a float between 0 and 1'
        assert kernel_size % 2 == 1, "Kernel size must be an odd number."
        self.probability = probability
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, sample):
        if np.random.rand() < self.probability:
            # Randomly select a sigma value within the specified range
            sigma_value = np.random.uniform(self.sigma[0], self.sigma[1])

            # Apply Gaussian blur using OpenCV
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma_value)

        return sample.copy()
  


class Random_Flip_Seg(object):
    """Randomly apply horizontal and/or vertical flip to an image-mask pair with a given probability."""
    def __init__(self, probability=0.5, horizontal=True, vertical=True):
        """
        Args:
        - probability (float): Probability of applying a flip.
        - horizontal (bool): Whether to allow horizontal flipping.
        - vertical (bool): Whether to allow vertical flipping.
        """
        assert isinstance(probability, float) and 0 < probability <= 1, 'Probability must be a float between 0 and 1'
        self.probability = probability
        self.horizontal = horizontal
        self.vertical = vertical

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        if np.random.rand() < self.probability:
            if self.horizontal and np.random.rand() > 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)

            if self.vertical and np.random.rand() > 0.5:
                image = np.flipud(image)
                mask = np.flipud(mask)

        return {'image': image.copy(), 'mask': mask.copy()}


class Random_GaussianBlur_Seg(object):
    """Randomly apply Gaussian blur to an image-mask pair with a given probability (only applied to the image)."""
    def __init__(self, probability=0.4, kernel_size=7, sigma=(2.0, 5.0)):
        """
        Args:
        - probability (float): Probability of applying Gaussian blur.
        - kernel_size (int): Size of the Gaussian kernel (must be odd).
        - sigma (tuple): Range of standard deviation for Gaussian kernel.
        """
        assert isinstance(probability, float) and 0 < probability <= 1, 'Probability must be a float between 0 and 1'
        assert kernel_size % 2 == 1, "Kernel size must be an odd number."
        self.probability = probability
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if np.random.rand() < self.probability:
            sigma_value = np.random.uniform(self.sigma[0], self.sigma[1])
            image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), sigma_value)

        return {'image': image.copy(), 'mask': mask.copy()}