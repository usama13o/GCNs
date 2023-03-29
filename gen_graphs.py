# %%
import pickle
from symbol import return_stmt
import time
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)
import torch
from componants import DecoderBlock

import torch.utils.data as data
import numpy as np
import datetime
from os import listdir
import random
from os.path import join
from os.path import basename
from PIL import Image
import h5py

import argparse

args = argparse.ArgumentParser()
args.add_argument("--batch_size", type=int, default=128, help="size of the batches")
args.add_argument("--epochs", type=int, default=1000)
args.add_argument("--lr", type=float, default=0.001)
args.add_argument("--weight_decay", type=float, default=0.0)
args.add_argument("--train", type=bool, default=False)
args.add_argument("--dataset", type=str, default="")
args.add_argument("--k", type=int, default=8)
args.add_argument("--model", type=str, default="GCN")
args.add_argument("--vae_input_h", type=int, default=32, help="input height for the vae")
args.add_argument("--nclusters", type=int, default=8)
args.add_argument("-size", "--image_transform_size", type=int, default=128)
args.add_argument("-use-comb-vae","--use_combined", type=bool, default=False)
args.add_argument("-pz","--patch_size", type=int, default=0)
args = args.parse_args()

# log file name  - date and time
log_file_name = "vae_log_k-" + str(args.k) + "_" + time.strftime("%Y%m%d-%H%M%S")
# init log file txt
# modify print to be log to txt (Doesn't seem to do the trick !)

print("init gen graphs\n with args \n", args)   
class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate."""

    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)
def resize_conv1x1(in_planes, out_planes, scale=1):
    """upsample + 1x1 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv1x1(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv1x1(in_planes, out_planes))
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Resdecoder(nn.Module):
    def __init__(self, x,latent_dim=256, input_height=224, first_conv=False, maxpool1=False):
        # super().__init__(   DecoderBlock, [2, 2, 2, 2],latent_dim=latent_dim, input_height=input_height, first_conv=first_conv, maxpool1=maxpool1)
        super().__init__()
        self.block = DecoderBlock
        self.latent_dim = latent_dim
        self.expansion = self.block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height
        self.layers = [2, 2, 2, 2]
        self.upscale_factor = 8
        

        _, self.c, self.h, self.w = x.shape
        block = self.block
        layers = self.layers
        out_put_shape = self.c # * self.h * self.w
        self.linear = torch.nn.Linear(self.latent_dim, out_put_shape * self.h * self.w)
        self.inplanes = out_put_shape



        self.layer1 = self._make_layer(block, 256, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 128, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 64, layers[2], scale=2)

        if self.maxpool1:
            self.layer4 = self._make_layer(block, 64, layers[3], scale=2)
            self.upscale_factor *= 2
        else:
            self.layer4 = self._make_layer(block, 64, layers[3])

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=self.input_height // self.upscale_factor)

        self.conv1 = nn.Conv2d(64 * block.expansion, 3, kernel_size=3, stride=1, padding=1, bias=False)
    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)

        x = x.view(x.size(0), self.c, self.h, self.w)
        x = self.upscale1(x)
        # print('in to l1',x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upscale(x)

        x = self.conv1(x)

        return x

from torchvision.models import swin_transformer
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

# %% [markdown]
# Let's use CIFAR-10 already split up and transformed.
#
# The Lightning Datamodule has 3 dataloaders, train, val, test
#

# %%
class DivideIntoPatches:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        height, width = img.shape[-2:]
        patches = []
        for i in range(0, height - self.patch_size + 1, self.patch_size):
            for j in range(0, width - self.patch_size + 1, self.patch_size):
                patches.append(img[ :, i:i + self.patch_size, j:j + self.patch_size])
        return torch.stack(patches, dim=0)

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

# %%
from torchvision import transforms
from torch.utils.data import DataLoader

from medmnist.dataset import PathMNIST, BreastMNIST,OCTMNIST,ChestMNIST,PneumoniaMNIST,DermaMNIST,RetinaMNIST,BloodMNIST,TissueMNIST,OrganAMNIST,OrganCMNIST,OrganSMNIST
# %%
# lr logging
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from callbacks import TestReconCallback_vae
import os
import glob

callbacks = []
lr_monitor = LearningRateMonitor(logging_interval="epoch")
callbacks.append(lr_monitor)

# save checkpoint on last epoch only
ckpt = ModelCheckpoint(f"/home/uz1/projects/GCN/logging/",
                       monitor="elbo",
                       save_weights_only=True)
callbacks.append(ckpt)

# add test for mid-train recon viewing

# %%
pl.seed_everything(1234)

vae = VAE(input_height=args.vae_input_h, latent_dim=1024)

# %%

# vae = vae.load_from_checkpoint(r"C:\Users\Usama\data\ckpt\epoch=0-step=8999.ckpt")

name = glob.glob(fr"C:\Users\Usama\data\ckpt\{args.dataset}\{args.patch_size}\{args.image_transform_size}\*")[-1]
print("VAE model path :  ",name)
vae = vae.load_from_checkpoint(name)


from torchvision import transforms
from torch.utils.data import DataLoader
## images as 128x128 to 16 patches (each 32x32)
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.RandomResizedCrop((128,128)),
#     transforms.ConvertImageDtype(torch.float),
#     transforms.Resize((args.image_transform_size, args.image_transform_size)),
# ])

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((args.image_transform_size, args.image_transform_size)),
            transforms.ConvertImageDtype(torch.float),
            DivideIntoPatches(patch_size=args.patch_size), # takes an image tensor and returns a list of patches stacked as (H // patch_size **2 x H x W x C)
        ])
print("Image TRANSFROMS #### ", transform)
if args.dataset == "wss":
    n_classes=4
    data_128 = wss_dataset_class("/home/uz1/data/wsss/train/1.training", 'all',
                         transform)
if args.dataset == "pcam":
    n_classes=3
    data_128 = data
    data_128.transform = transform
    data_128.limit=30000
    valid_dataset= HDF5Dataset("/home/uz1/DATA!/pcam/pcam/validation_split.h5","/home/uz1/DATA!/pcam/Labels/Labels/camelyonpatch_level_2_split_valid_y.h5",limit=10000,transform=transform)
    if args.k == 4:
        with open("/home/uz1/projects/GCN/kmeans-model-4.pkl", "rb") as f:
            k = pickle.load(f)
    elif args.k == 8:
        with open("/home/uz1/projects/GCN/kmeans-model.pkl", "rb") as f:
            k = pickle.load(f)
if args.dataset == 'pathmnist':
    n_classes=9
    from medmnist.dataset import PathMNIST
    data_128 = PathMNIST(root=r"C:\Users\Usama\data", download=True,split='train',transform=transform)
    valid_dataset = PathMNIST(root=r"C:\Users\Usama\data", download=True,split='val',transform=transform)
    
    if args.patch_size != 0:
        with open(fr"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-{args.image_transform_size}-{args.patch_size}-{args.k}-PathMNIST.pkl", "rb") as f:
            k = pickle.load(f)
    args.nclusters = args.k
if args.dataset == 'dermamnist':
    from medmnist.dataset import DermaMNIST
    data_128 = DermaMNIST(root=r"C:\Users\Usama\data",download=True,split='train',transform=transform)
    valid_dataset = DermaMNIST(root=r"C:\Users\Usama\data", download=True,split='val',transform=transform)
    n_classes = 7
    if args.patch_size != 0:
        with open(fr"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-{args.image_transform_size}-{args.patch_size}-{args.k}-DermaMNIST.pkl", "rb") as f:
            k = pickle.load(f)
    else:
        raise ValueError("Patch size must be specified for DermaMNIST")
if args.dataset == 'bloodmnist':
    from medmnist.dataset import BloodMNIST
    data_128 = BloodMNIST(root='/home/uz1/DATA!/medmnist', download=True,split='train',transform=transform)
    valid_dataset = BloodMNIST(root='/home/uz1/DATA!/medmnist', download=True,split='val',transform=transform)
    n_classes = 8
    if args.k == 16:
        with open(r"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-16-BloodMNIST.pkl", "rb") as f:
            k = pickle.load(f)
    if args.k == 32:
        with open(r"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-32-BloodMNIST.pkl", "rb") as f:
            k = pickle.load(f)
    if args.k == 64:
        with open(r"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-64-BloodMNIST.pkl", "rb") as f:
            k = pickle.load(f)
    if args.k == 128:
        with open(r"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-128-BloodMNIST.pkl", "rb") as f:
            k = pickle.load(f)
    if args.k == 8:
        with open(r"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-7-medmnist-blood.pkl", "rb") as f:
            k = pickle.load(f)
if args.dataset == 'combinedmnist':    
    from datasets import  combined_medinst_dataset
    n_classes = 91
    data_128 = combined_medinst_dataset(root='/home/uz1/DATA!/medmnist', split='train',transform=transform)
    if args.k == 16:
        with open(r"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-16-combined_medinst_dataset.pkl", "rb") as f:
            k = pickle.load(f)
    if args.k == 32:
        with open(r"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-32-combined_medinst_dataset.pkl", "rb") as f:
            k = pickle.load(f)
    if args.k == 64:
        with open(r"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-64-combined_medinst_dataset.pkl", "rb") as f:
            k = pickle.load(f)
    if args.k == 128:
        with open(r"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-128-combined_medinst_dataset.pkl", "rb") as f:
            k = pickle.load(f)


        
loader = DataLoader(data_128, batch_size=32, drop_last=True, num_workers=16)


args.nclusters = args.k

# if use_combined change Kmeans and vae to
if args.use_combined:
    if args.k == 16:
        with open(r"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-16-combined_medinst_dataset.pkl", "rb") as f:
            k = pickle.load(f)
    if args.k == 32:
        with open(r"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-32-combined_medinst_dataset.pkl", "rb") as f:
            k = pickle.load(f)
    if args.k == 64:
        with open(r"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-64-combined_medinst_dataset.pkl", "rb") as f:
            k = pickle.load(f)
    if args.k == 128:
        with open(r"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-128-combined_medinst_dataset.pkl", "rb") as f:
            k = pickle.load(f)
    vae = vae.load_from_checkpoint(r"C:\Users\Usama\projects\GCNs\kmeans\logging/combined_medinst_dataset/epoch=13-step=56672.ckpt")

    print("### USING COMBINED VAE AND KMEANS ###") 
# %%
# %%
from torchvision.transforms import ToPILImage,ToTensor
from matplotlib.pyplot import imshow, figure
import numpy as np
from torchvision.utils import make_grid
from skimage import graph, io, segmentation, color
from matplotlib import pyplot as plt
from torch_geometric.utils import to_dense_adj, grid,dense_to_sparse
from monai.data import GridPatchDataset, DataLoader, PatchIter
patch_iter = PatchIter(patch_size=(32, 32), start_pos=(0, 0))
#patching each image
pil = ToPILImage()
to_tensor=ToTensor()
def get_embedding_vae(x,vae):

    x_encoded = vae.encoder(x)
    mu, log_var = vae.fc_mu(x_encoded), vae.fc_var(x_encoded)
    std = torch.exp(log_var / 2)
    q = torch.distributions.Normal(mu, std)
    z = q.rsample()
    return z
from math import sqrt
# %%
# %%
def calculate_cluster_dist_stat(label):
    s = torch.zeros(16,8)
    label=label+1
    l=label.copy()
    for i in range(16):
        print(label.reshape(4,4))
        l[i]=0
        unique_n = torch.unique(a[i].reshape(4,4) * l.reshape(4,4),return_counts=True)
        no_n = sum(unique_n[1][1:])
        print(torch.unique(a[i].reshape(4,4) * l.reshape(4,4),return_counts=True) ,no_n,l[i],)
        perc_n = unique_n[1][1:]/no_n

        w = torch.zeros(8,)
        w[unique_n[0][1:].long()-1] = perc_n
        s[i] = w
    return s

import numpy as np

from monai.data import GridPatchDataset, DataLoader, PatchIter

# image-level patch generator, "grid sampling"
# %%
# Install required packages.
import os
import torch
# Helper function for visualization.
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

from torch_geometric.data import Data,Dataset



# %% [markdown]
# ## Build a GCN
# #### Cluster assignment given using the KMEANS + edge features (statistics !?)
# #### pool features using the assignmentinto cluster node features
# #### build and classify images  - somehow

# %%
from tkinter import Label
from monai.data.iterable_dataset import IterableDataset
from monai.transforms import apply_transform
import numpy as np

from monai.data import GridPatchDataset, DataLoader, PatchIter

#patch yeilding



# image-level dataset
images = data_128
# image-level patch generator, "grid sampling"
patch_iter = PatchIter(patch_size=(32, 32), start_pos=(0, 0))
# patch-level intensity shifts
# patch_intensity = RandShiftIntensity(offsets=1.0, prob=1.0)


class GridPatchDatasetWithLabels(IterableDataset):

    """
    Takes a list or Dataset of iamges and their labels 
    returns:
    Patchs , coords, image label
    """
    def __init__(
        self,
        data ,
        patch_iter ,
        transform=None,
        with_coordinates=True,
        kmeans=None,
    ) -> None:
        super().__init__(data=data, transform=None)
        self.patch_iter = patch_iter
        self.patch_transform = transform
        self.with_coordinates = with_coordinates

    def __iter__(self):
        for  image,label in super().__iter__():
            for patch, *others in self.patch_iter(image):
                out_patch = patch
                if self.patch_transform is not None:
                    out_patch = apply_transform(self.patch_transform, patch, map_items=False)
                if self.with_coordinates and len(others) > 0:  # patch_iter to yield at least 2 items: patch, coords
                    yield torch.tensor(out_patch), others[0],label
                else:
                    yield torch.tensor(out_patch),label

    def __getitem__(self, index):
        patches=[]
        coords=[]
        for patch,coord in self.patch_iter(self.data[index][0]):
            out_patch = patch
            if self.patch_transform is not None:
                out_patch = apply_transform(self.patch_transform, patch, map_items=False)
            patches.append(out_patch)
            coords.append(coord)
        return torch.stack([torch.tensor(patches)],0),coords,self.data[index][1]

# construct the dataset
ds = GridPatchDatasetWithLabels(data=images,
                      patch_iter=patch_iter,)

# %%
def get_embedding_vae(x,vae):

    x_encoded = vae.encoder(x)
    mu, log_var = vae.fc_mu(x_encoded), vae.fc_var(x_encoded)
    std = torch.exp(log_var / 2)
    q = torch.distributions.Normal(mu, std)
    z = q.rsample()
    return z


from torch_geometric.data import Data,Dataset
from torch_geometric.utils import to_dense_adj, grid,dense_to_sparse

def filter_a(data):

    if data.y==3:
        return False
    else:
        True
def populateSBatched(labels, n_clusters=8, s=None, n_graphs=10, n_patches=16):
    """
    Calculates the S cluster assignment transform of input patch features 
    for multiple graphs and returns the S and the aggregated adjacency matrix for each graph.
    Shape : (number of patches, number of clusters)

    Args:
        labels (ndarray): Labels for each node in each graph. Shape is (n_graphs*n_patches,)
        n_clusters (int): Number of clusters
        s (ndarray): Existing S assignment transform to update
        n_graphs (int): Number of graphs
        n_patches (int): Number of patches in each image

    Returns:
        tuple: S assignment transform and aggregated adjacency matrix for each graph
    """
    # n_patches = n_graphs * n_patches
    div = int(np.sqrt(n_patches))

    S_list = []
    adj_list = []

    for i in range(n_graphs):
        start = i * n_patches
        end = start + n_patches
        graph_labels = labels[start:end]

        s_graph = np.zeros((n_patches, n_clusters))
        s_graph[np.arange(n_patches), graph_labels] = 1

        # calc adj matrix
        adj_graph = to_dense_adj(grid(div, div)[0]).reshape(n_patches, n_patches)

        S_list.append(s_graph)
        adj_list.append(np.matmul(np.matmul(s_graph.transpose(1, 0), adj_graph), s_graph))

    S = np.concatenate(S_list)
    adj = np.stack(adj_list)

    return S, adj

def populateS(labels,n_clusters=8,s=None):
    """"
    Calculates the S cluster assigment transform of input patch features 
    and returns the (S) and the aggregated (out_adj) as well.
    shape : ( number of patches , number of clusters)
    """
    # print("S is " ,s==None)
    n_patches=len(labels)
    div = int(sqrt(n_patches))
 
    s = np.zeros((n_patches,n_clusters))
    s[np.arange(n_patches), labels] = 1
         # TODO optimise this!


    #calc adj matrix
    adj = to_dense_adj(grid(n_patches//div,n_patches//div)[0]).reshape(n_patches,n_patches)
    return s , np.matmul(np.matmul(s.transpose(1, 0),adj ), s)

from torch_geometric.data import Data,Dataset
from torch_geometric.utils import to_dense_adj, grid,dense_to_sparse

from monai.data import GridPatchDataset, DataLoader, PatchIter


class ImageTOGraphDatasetBatched(Dataset):
    """ 
    Dataset takes holds the kmaens classifier and vae encoder. On each input image we encode then get k mean label then formulate graph as Data object
    """
    def __init__(self,data,vae,kmeans,norm_adj=True,return_x_only=None):
        self.data=data
        self.vae=vae
        self.vae.cuda()
        self.return_x_only=return_x_only

        self.kmeans = kmeans
        self.norm_adj = norm_adj
        self.n_patches = self.data[1][0].shape[0]

        # if self.data[1][0].dim() >3:
        #     self.__getitem__ = self.__getitem__patches   
            # add attr

    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        #given that index is a list of indeices , load the patches and get the embedding
        # p = self.data[index][0] wont work 
    
        b=len(index) 
        patches = torch.concat([self.data[i][0] for i in index],0).cuda()
        n_graphs = patches.shape[0] // self.n_patches
        n_clusters = self.kmeans.cluster_centers_.shape[0]
        ys = np.concatenate([self.data[i][1] for i in index],0)
        z=get_embedding_vae(patches,self.vae.cuda()).clone().detach().cpu().numpy()
        label=self.kmeans.predict(z)
        s,out_adj = populateSBatched(label,n_clusters,n_graphs=n_graphs,n_patches=self.n_patches)

        z = z.reshape((b, self.n_patches, -1))
        s = s.reshape((b,self.n_patches, n_clusters))

        s = np.transpose(s,(0,2,1))
        x = np.matmul(s, z).reshape(b,n_clusters,-1)
        del z,s,patches
        if self.return_x_only:
            return torch.tensor(x),torch.tensor(label),torch.tensor(ys)
        #if normlaise adj 
        if self.norm_adj:
            out_adj = out_adj.div(out_adj.sum(1))
            #nan to 0 in tensor 
            out_adj = out_adj.nan_to_num(0)
            #assert if there is nan in tensor
            assert out_adj.isnan().any() == False , "Found nan in out_adj"
        return Data(x=torch.tensor(x).float(),edge_index=dense_to_sparse(out_adj)[0],y=torch.tensor(self.data[index][1]),edge_attr=dense_to_sparse(out_adj)[1])
class ImageTOGraphDataset(Dataset):
    """ 
    Dataset takes holds the kmaens classifier and vae encoder. On each input image we encode then get k mean label then formulate graph as Data object
    """
    def __init__(self,data,vae,kmeans,norm_adj=True,return_x_only=None):
        self.data=data
        self.vae=vae
        self.return_x_only=return_x_only

        self.kmeans = kmeans
        self.norm_adj = norm_adj

        # if self.data[1][0].dim() >3:
        #     self.__getitem__ = self.__getitem__patches   
            # add attr

    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):

        patches = self.data[index][0]
        z=get_embedding_vae(patches.cuda(),self.vae.cuda()).clone().detach().cpu().numpy()
        label=self.kmeans.predict(z)
        s,out_adj = populateS(label,self.kmeans.cluster_centers_.shape[0])
        x = np.matmul(s.transpose(1,0) , z)
        del z
        del s,patches
        if self.return_x_only:
            return torch.tensor(x),torch.tensor(label),torch.tensor(self.data[index][1])
        #if normlaise adj 
        if self.norm_adj:
            out_adj = out_adj.div(out_adj.sum(1))
            #nan to 0 in tensor 
            out_adj = out_adj.nan_to_num(0)
            #assert if there is nan in tensor
            assert out_adj.isnan().any() == False , "Found nan in out_adj"
        return Data(x=torch.tensor(x).float(),edge_index=dense_to_sparse(out_adj)[0],y=torch.tensor(self.data[index][1]),edge_attr=dense_to_sparse(out_adj)[1])


from torch_geometric.loader import DataLoader as GraphDataLoader
# toech data loader 
from torch.utils.data import DataLoader
#import samplesr
import torch


ImData = ImageTOGraphDatasetBatched(data=data_128,vae=vae,kmeans=k,return_x_only=True)
ImData2 = ImageTOGraphDataset(data=data_128,vae=vae,kmeans=k,return_x_only=True)
sampler = torch.utils.data.sampler.BatchSampler(
    torch.utils.data.sampler.RandomSampler(ImData),
    batch_size=args.batch_size,
    drop_last=True)
train_loader = GraphDataLoader(ImData, batch_size=1, shuffle=False, num_workers=0,sampler = sampler)

# #time how long it takes to load data 
import time
start = time.time()
next(iter(train_loader))
end = time.time()
print("Time to load data: {}s".format(end-start))
#how much for all data
print("Time to load all data: {} min".format(((end-start)*len(ImData)/128)// 60))
#test using tqdm load trhough all data
from tqdm import tqdm
# loop through train loader
for i,_ in tqdm( enumerate(train_loader), total=len(train_loader)):
    if i == 5:
        break


train_loader2 = GraphDataLoader(ImData2, batch_size=args.batch_size, shuffle=True, num_workers=0,)
#time how long it takes to load data
start = time.time()
next(iter(train_loader2))
end = time.time()
print("Time to load data before : {}s".format(end-start))
print("Time to load all data: {} min".format(((end-start)*len(ImData)/128) // 60))
for i,_ in tqdm(enumerate(train_loader2), total=len(train_loader2)):
    if i == 5:
        break

#print dataset stats
print("Dataset stats:")
print("Number of graphs: {}".format(len(ImData)))
# print("Number of features: {}".format(ImData.num_features))
print("batch size: {}".format(train_loader.batch_size))

# ImData_valid = ImageTOGraphDataset(data=valid_dataset,vae=vae,kmeans=k,return_x_only=True)
# valid_loader = GraphDataLoader(ImData_valid,batch_size=16,shuffle=True,num_workers=0)
#import dataloader from torch
from torch.utils.data import DataLoader
# train_loader = DataLoader(ImData, batch_size=args.batch_size,)
train_loader = DataLoader(ImData, batch_size=1,sampler = sampler)
# for all data in the dataset: create a h5y file to store the data and save to disk
import h5py
import os
import numpy as np
from torch_geometric.data import Data,Dataset
from tqdm import tqdm
#create h5py file
# if os.path.exists(f'C:\Users\Usama\projects\GCNs\kmeans\graph_data/graph-data---{args.dataset}-{args.patch_size}-{args.nclusters}-{args.image_transform_size}-UC_{args.use_combined}.h5'):
    # print("File already exists at ", f'C:\Users\Usama\projects\GCNs\kmeans\graph_data/graph-data---{args.dataset}-{args.patch_size}-{args.nclusters}-{args.image_transform_size}-UC_{args.use_combined}.h5')

    #exit
    # exit("File already exists")

h5f = h5py.File(fr'C:\Users\Usama\data\graph-data---{args.dataset}-{args.patch_size}-{args.nclusters}-{args.image_transform_size}-UC_{args.use_combined}.h5', 'w')

#create dataset in file

h5f.create_dataset('x', (len(ImData),args.nclusters,vae.fc_mu.out_features))

# add the data to h5py file
n_clusters = args.nclusters
embed_size = vae.fc_mu.out_features
n_patches = data_128[0][0].shape[0]
print(f"n_clusters: {n_clusters}, embed_size: {embed_size}, n_patches: {n_patches}")
#try numpy array
edge_i = []
edge_at = []
ys=[]
tmp=0
print("Started saving data to h5py file")
for i,d in tqdm(enumerate(train_loader),total=len(train_loader),desc="Saving data to h5py file"):

    h5f['x'][tmp:tmp+d[0].squeeze().shape[0]] = d[0].reshape(-1,n_clusters,embed_size)
    tmp=tmp+d[0].squeeze().shape[0]

    ys.append(d[2].squeeze())
    edge_i.append(d[1].squeeze())

edge_i = np.stack(edge_i[:-1])
ys = np.stack(ys[:-1])
ys = np.concatenate([ys.reshape(-1),d[2].reshape(-1)]) 
edge_i = np.concatenate([edge_i.reshape(-1,n_patches),d[1].reshape(-1,n_patches)])

#create datasets for edge index and edge attr and ys
h5f.create_dataset('edge_index', data=edge_i)
h5f.create_dataset('ys', data=ys)

#close the file
h5f.close()
print("Finished saving data to h5py file")
# %%
