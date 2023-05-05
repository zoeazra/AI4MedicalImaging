"""
Author: adopted from Ricardo Eugenio González Valenzuela by Oliver Gurney-Champion | Spring 2023
Date modified: March 2023

Skin cancer is the most common cancer globally, with melanoma being the most deadly form. Dermoscopy is a skin imaging
modality that has demonstrated improvement for diagnosis of skin cancer compared to unaided visual inspection. However,
clinicians should receive adequate training for those improvements to be realized. In order to make expertise more
widely available, the International Skin Imaging Collaboration (ISIC) has developed the ISIC Archive, an international
repository of dermoscopic images, for both the purposes of clinical training, and for supporting technical research
toward automated algorithmic analysis by hosting the ISIC Challenges.

The dataset is already pre-processed for you in batches of 128x256x256x3 (128 images, of size 256x256 rgb)
"""

#loading libraries

import os
import numpy as np
import argparse
import glob
import torchio
import torch
import torchmetrics
import torch.nn.functional as F
import nibabel as nib

from sys import platform
from Data_loader import Scan_Dataset_Segm, Scan_DataModule_Segm, ToTensor_Seg
from visualization import show_data_Unet
from CNNs import UNet
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

#seeding for reproducible results
SEED=42
#torch.backends.cudnn.deterministic = True
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
np.random.RandomState(SEED)

if platform == "linux" or platform == "linux2":
    data_dir = '/project/gpuuva019/practical_2_data/segmentation'
else:
    #set data location on your local computer. Data can be downloaded from:
    # https://surfdrive.surf.nl/files/index.php/s/epjCz4fip1pkWN7
    # PW: deeplearningformedicalimaging
    data_dir = 'C:\scratch\Surf\Documents\Onderwijs\DeepLearning_MedicalImaging\opgaven\opgave 2\AI-Course_StudentChallenge\data\segmentation'

print('data is loaded from ' + data_dir)
# view data
nn_set = 'train' # ['train', 'val', 'test']
index = 0
dataset = Scan_Dataset_Segm(os.path.join(data_dir, nn_set))
show_data_Unet(dataset,index,n_images_display=5)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device)

models = {'unet': UNet}

optimizers = {'adam': torch.optim.Adam,
              'sgd': torch.optim.SGD}

metrics = {'acc': torchmetrics.Accuracy(task='binary').to(device),
           'f1': torchmetrics.F1Score(task='binary').to(device),
           'precision': torchmetrics.Precision(task='binary').to(device),
           'recall': torchmetrics.Recall(task='binary').to(device)}


class Segmenter(pl.LightningModule):
    def __init__(self, *args):
        super().__init__()

        # defining model
        self.model_name = config_segm['model_name']
        assert self.model_name in models, f'Model name "{self.model_name}" is not available. List of available names: {list(models.keys())}'
        self.model = models[self.model_name]().to(device)

        # assigning optimizer values
        self.optimizer_name = config_segm['optimizer_name']
        self.lr = config_segm['optimizer_lr']

    def step(self, batch, nn_set):
        X, y = batch['image'], batch['mask']
        X, y = X.float().to(device), y.to(device).float()
        y_hat = self.model(X)
        y_prob = torch.sigmoid(y_hat)
        del X, y_hat, batch

        pos_weight = torch.tensor([config_segm['loss_pos_weight']]).float().to(device)
        loss = F.binary_cross_entropy_with_logits(y, y_prob, pos_weight=pos_weight)
        self.log(f"{nn_set}_loss", loss, on_step=False, on_epoch=True)

        for i, (metric_name, metric_fn) in enumerate(metrics.items()):
            score = metric_fn(y_prob, y.int())
            self.log(f'{nn_set}_{metric_name}', score, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self.step(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.step(batch, 'test')

    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        assert self.optimizer_name in optimizers, f'Optimizer name "{self.optimizer_name}" is not available. List of available names: {list(models.keys())}'
        return optimizers[self.optimizer_name](self.parameters(), lr=self.lr)


class Scan_DataModule_Segm_Test(pl.LightningDataModule):
    def __init__(self, test_data_dir, batch_size=32):
        super().__init__()
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.test_transforms = transforms.Compose([ToTensor_Seg()])

    def setup(self, stage=None):
        self.test_dataset = Scan_Dataset_Segm(self.test_data_dir, transform=self.test_transforms)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)


def run(config_segm):
    logger = TensorBoardLogger(config_segm['bin'], name=config_segm['experiment_name'])
    if not config_segm['checkpoint_folder_path']:
        data = Scan_DataModule_Segm(config_segm)
        segmenter = Segmenter(config_segm)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_f1')
        trainer = pl.Trainer(max_epochs=config_segm['max_epochs'],
                             logger=logger, callbacks=[checkpoint_callback],
                             default_root_dir=config_segm['bin'],
                             log_every_n_steps=1)
        trainer.fit(segmenter, data)
    else:
        # change these paths
        test_data_dir = os.path.join(data_dir, 'test')
        # load best model
        PATH = glob.glob(os.path.join(config_segm['checkpoint_folder_path'], '*'))[0]
        model = Segmenter.load_from_checkpoint(PATH)
        model.eval()

        # make test dataloader
        test_data = Scan_DataModule_Segm_Test(test_data_dir)

        # test model
        trainer = pl.Trainer()
        trainer.test(model, dataloaders=test_data, verbose=True)

        # get and store predictions
        if config_segm['predictions_path']:
            image_list = glob.glob(test_data_dir+'/img*.nii.gz')
            predictions_dir = config_segm['predictions_path']
            os.makedirs(predictions_dir, exist_ok=True)
            for image in image_list:
                X = torch.tensor(nib.load(image).get_fdata())
                X = torch.permute(X, [2, 0, 1])
                X = torch.unsqueeze(X, 0)
                X = X.float()
                y = model(X)
                pred = torch.sigmoid(y)
                pred_image = torchio.Image(tensor=pred.cpu().detach())
                pred_path = predictions_dir + 'pred_' + image.rsplit('/', 1)[1]
                pred_image.save(pred_path)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    # Optimizer hyperparameters
    parser.add_argument('--optimizer_lr', default=0.1, type=float, nargs='+',
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Minibatch size')
    parser.add_argument('--model_name', default='unet', type=str,
                        help='defines model to use')
    parser.add_argument('--optimizer_name', default='sgd', type=str,
                        help='optimizer options: adam and sgd (default)')
    # Other hyperparameters
    parser.add_argument('--max_epochs', default=1, type=int,
                        help='Max number of epochs')
    parser.add_argument('--experiment_name', default='test1', type=str,
                        help='name of experiment')
    parser.add_argument('--checkpoint_folder_path', default=False, type=str,
                        help='name of experiment')
    parser.add_argument('--predictions_path', default=False, type=str,
                        help='path where to store predictions')

    args = parser.parse_args()
    config_segm = vars(args)

    config_segm.update({
        'train_data_dir': os.path.join(data_dir, 'train'),
        'val_data_dir': os.path.join(data_dir, 'val'),
        'test_data_dir': os.path.join(data_dir, 'test'),
        'bin': 'models/',
        'loss_pos_weight': 1})

    run(config_segm)
    # Feel free to add any additional functions, such as plotting of the loss curve here
