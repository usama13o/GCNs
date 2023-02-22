import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)
import torch

import argparse

parser = argparse.ArgumentParser(description='Kmeans file makes !!')
parser.add_argument('-k', '--kmeans',  help='kmeans cluster numbers ', required=False,type=int)
args = parser.parse_args()
print(f"Args k choosen  = {args.kmeans}")

class AE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=512, input_height=32):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(latent_dim=latent_dim,
                                        input_height=input_height,
                                        first_conv=False,
                                        maxpool1=False)

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2, 3))

    def training_step(self, batch, batch_idx):
        # print(batch)
        x, _ = batch
        #if channels are less than 3, repeat channels
        if x.shape[0] < 3:
            print(x.shape)
            x = x.repeat(3, 1, 1)
            print(x.shape)
        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        # print(x.shape,x_encoded.shape)
        # x_encoded = self.fc_mu(x_encoded)

        x_hat = vae.decoder(x_encoded)
        # print(x_hat.shape)

        # reconstruction loss
        recon_loss_ = self.gaussian_likelihood(x_hat, self.log_scale, x) # old recon_loss 
        # print(recon_loss.shape)
        # recon_loss = torch.nn.MSELoss()(x_hat,x,reduction='mean')
        recon_loss = F.mse_loss(x, x_hat, reduction="none")
        recon_loss = recon_loss.sum(dim=[1, 2, 3]).mean(dim=[0])
       
       
        # print(recon_loss.shape)
        # elbo =recon_loss_
        loss = (-recon_loss_.mean() + recon_loss)
        self.log_dict({
            'reconstruction': loss,
        })

        return loss

class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=32):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(latent_dim=latent_dim,
                                        input_height=input_height,
                                        first_conv=False,
                                        maxpool1=False)

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu),
                                       torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        # print(batch)
        x, _ = batch
        #if channels are less than 3, repeat channels
        if x.shape[0] < 3:
            print(x.shape)
            x = x.repeat(3, 1, 1)
            print(x.shape)
        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = vae.decoder(z)

        # reconstruction loss
        recon_loss_ = self.gaussian_likelihood(x_hat, self.log_scale, x) # old recon_loss 
        # print(recon_loss.shape)
        recon_loss = torch.nn.MSELoss()(x_hat,x)
        # print(recon_loss.shape)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss_) # with old recon_loss
        # elbo = (kl + recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss_': recon_loss.mean(),
            'recon_loss': recon_loss_.mean(),
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return elbo

import torch.utils.data as data
import numpy as np
import datetime
from os import listdir
import random
from os.path import join
from os.path import basename
from PIL import Image
import h5py

class wss_dataset_class(data.Dataset):
    def __init__(self,
                 root_dir,
                 split,
                 transform=None,
                 preload_data=False,
                 train_pct=0.8,
                 balance=True):
        super(wss_dataset_class, self).__init__()
        #train dir
        img_dir = root_dir

        self.image_filenames = sorted(
            [join(img_dir, x) for x in listdir(img_dir) if is_image_file(x)])
        self.target_filenames = [
            list(
                map(int, [
                    x.split('-')[-1][:-4][1],
                    x.split('-')[-1][:-4][4],
                    x.split('-')[-1][:-4][7]
                ])) for x in self.image_filenames
        ]
        sp = self.target_filenames.__len__()
        sp = int(train_pct * sp)
        # random.shuffle(self.image_filenames)
        if split == 'train':
            self.image_filenames = self.image_filenames[:sp]
        elif split == 'all':
            self.image_filenames = self.image_filenames
        else:
            self.image_filenames = self.image_filenames[sp:]
            # find the mask for the image
            self.target_filenames = self.target_filenames[sp:]
            print(len(self.target_filenames))
        print('Number of {0} images: {1} patches'.format(
            split, self.__len__()))
        assert len(self.image_filenames) == len(self.target_filenames)

        # report the number of images in the dataset

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [
                open_image_np(ii)[0] for ii in self.image_filenames
            ]
            print('Loading is done\n')

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second +
                       datetime.datetime.now().microsecond)
        target = self.target_filenames[index]
        if sum(target) == 2:
            target = 0#3
        else:
            target = np.array(target).argmax()
        # load the nifti images
        if not self.preload_data:
            input = open_image_np(self.image_filenames[index])
        else:
            input = np.copy(self.raw_images[index])

        # handle exceptions
        if self.transform:
            input = self.transform(input)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


def is_image_file(filename):
    return any(
        filename.endswith(extension)
        for extension in [".nii.gz", 'png', 'tiff', 'jpg', "bmp"])


def open_image(filename):
    """
    Open an image (*.jpg, *.png, etc).
    Args:
    filename: Name of the image file.
    returns:
    A PIL.Image.Image object representing an image.
    """
    image = Image.open(filename)
    return image


def open_image_np(path):
    im = open_image(path)
    array = np.array(im)
    return array
class HDF5Dataset(data.Dataset):
    """
    Load an HDF5 dataset.
    Args:
    path_train: Path to the HDF5 file.
    path_train_y: Path to the HDF5 file labels.
    transform
    """
    def __init__(self, path,path_y, transform=None, limit=False):
        super(HDF5Dataset, self).__init__()
        self.path = path
        self.path_y = path_y
        self.limit = limit
        self.transform = transform
    def __getitem__(self, index):
        with h5py.File(self.path, 'r') as hf:
            input = hf['x'][index]
        with h5py.File(self.path_y, 'r') as hf:
            target = hf['y'][index].squeeze()
        if self.transform:
            input = self.transform(input)
        return input, torch.tensor(target)
    def __len__(self):
        if self.limit is not False:
            return self.limit
        with h5py.File(self.path, 'r') as hf:
            return len(hf['x'])
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.ConvertImageDtype(torch.float),
])
from medmnist.dataset import PathMNIST, BreastMNIST,OCTMNIST,ChestMNIST,PneumoniaMNIST,DermaMNIST,RetinaMNIST,BloodMNIST,TissueMNIST,OrganAMNIST,OrganCMNIST,OrganSMNIST
data = PathMNIST(root='/home/uz1/DATA!/medmnist', download=True,split='train',transform=transform)
data_class_name = data.__class__.__name__
print(data)
loader = DataLoader(data, batch_size=128, drop_last=True, num_workers=16)
vae = VAE(input_height=data[0][0].shape[1], latent_dim=1024)
vae = vae.load_from_checkpoint("/home/uz1/projects/GCN/logging/epoch=128-step=45278.ckpt")


num_points = 20000 if len(data) > 20000 else len(data)
# real test data  -  vis of means of data and class clusters
test = [data[x][0] for x in range(num_points)]
y_test = [data[x][1] for x in range(num_points)]
test = torch.stack(test, 0).to(vae.device)
if vae.__class__.__name__ == 'VAE':
    x_encoded = vae.encoder(test)

    mu, log_var = vae.fc_mu(x_encoded), vae.fc_var(x_encoded)
    std = torch.exp(log_var / 2)
    q = torch.distributions.Normal(mu, std)
    z = q.rsample()
    z=z.detach().cpu().numpy()
    print(z.shape)
elif vae.__class__.__name__ == 'deeplab':
    z = vae.encode(test)[0].detach().cpu().numpy()


from sklearn.cluster import KMeans
import math
x_em = z
k=KMeans(n_clusters=args.kmeans).fit(x_em.reshape(x_em.shape[0],-1))
#count unique values + return counts
unique, counts = np.unique(k.labels_, return_counts=True)
print("unique values:", unique)
print("counts:", counts)
import pickle
with open(f"/home/uz1/projects/GCN/kmeans-model-{args.kmeans}-{data_class_name}.pkl", "wb") as f:
    pickle.dump(k, f)
del test
print("############## DONE ! ################")