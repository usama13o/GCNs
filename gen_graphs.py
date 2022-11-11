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


import argparse

args = argparse.ArgumentParser()
args.add_argument("--batch_size", type=int, default=128, help="size of the batches")
args.add_argument("--epochs", type=int, default=1000)
args.add_argument("--lr", type=float, default=0.001)
args.add_argument("--weight_decay", type=float, default=0.0)
args.add_argument("--train",type=bool,default=False)
args.add_argument("--dataset",type=str,default="")
args.add_argument("--k", type=int, default=8)
args.add_argument("--model",type=str,default="GCN")
args.add_argument("--vae_input_h", type=int, default=32, help="input height for the vae")
args.add_argument("--nclusters", type=int, default=8)
args.add_argument("-size","--image_transform_size",type=int,default=128)
args = args.parse_args()

#log file name  - date and time
log_file_name = "vae_log_k-" + str(args.k) + "_" + time.strftime("%Y%m%d-%H%M%S")
#init log file txt
f = open(f"/home/uz1/projects/GCN/{log_file_name}.txt","w")
f.close()
#modify print to be log to txt (Doesn't seem to do the trick !)
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

# %%
from torchvision import transforms
from torch.utils.data import DataLoader

from medmnist.dataset import PathMNIST, BreastMNIST,OCTMNIST,ChestMNIST,PneumoniaMNIST,DermaMNIST,RetinaMNIST,BloodMNIST,TissueMNIST,OrganAMNIST,OrganCMNIST,OrganSMNIST
# %%
# lr logging
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from callbacks import TestReconCallback_vae
import os

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

vae = vae.load_from_checkpoint("/home/uz1/projects/GCN/logging/epoch=20-step=172031.ckpt")
if args.dataset == "medmnist-path":
    vae = vae.load_from_checkpoint("/home/uz1/projects/GCN/logging/epoch=128-step=45278.ckpt")
if args.dataset == "medmnist-derma":
    vae = vae.load_from_checkpoint("/home/uz1/projects/GCN/logging/DermaMNIST/epoch=378-step=10232.ckpt")
if args.dataset == "medmnist-blood":
    vae = vae.load_from_checkpoint("/home/uz1/projects/GCN/logging/BloodMNIST/epoch=486-step=22401.ckpt")
#

#load images to be patched
from torchvision import transforms
from torch.utils.data import DataLoader
## images as 128x128 to 16 patches (each 32x32)
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomResizedCrop((128,128)),
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize((args.image_transform_size,args.image_transform_size)),
])
print("Image TRANSFROMS #### ",transform)
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
if args.dataset == 'medmnist-path':
    n_classes=9
    from medmnist.dataset import PathMNIST, BreastMNIST,OCTMNIST,ChestMNIST,PneumoniaMNIST,DermaMNIST,RetinaMNIST,BloodMNIST,TissueMNIST,OrganAMNIST,OrganCMNIST,OrganSMNIST
    data_128 = PathMNIST(root='/home/uz1/DATA!/medmnist', download=True,split='train',transform=transform)
    valid_dataset = PathMNIST(root='/home/uz1/DATA!/medmnist', download=True,split='val',transform=transform)
    with open("/home/uz1/projects/GCN/kmeans-model-8-medmnist-path.pkl", "rb") as f:
        k = pickle.load(f)
    if args.nclusters==16:
        with open("/home/uz1/projects/GCN/kmeans-model-16-pathmnist.pkl", "rb") as f:
            k = pickle.load(f)
if args.dataset == 'medmnist-derma':
    from medmnist.dataset import DermaMNIST
    data_128 = DermaMNIST(root='/home/uz1/DATA!/medmnist', download=True,split='train',transform=transform)
    valid_dataset = DermaMNIST(root='/home/uz1/DATA!/medmnist', download=True,split='val',transform=transform)
    n_classes = 7
    with open("/home/uz1/projects/GCN/kmeans-model-8-medmnist-derma.pkl", "rb") as f:
        k = pickle.load(f)
if args.dataset == 'medmnist-blood':
    from medmnist.dataset import BloodMNIST
    data_128 = BloodMNIST(root='/home/uz1/DATA!/medmnist', download=True,split='train',transform=transform)
    valid_dataset = BloodMNIST(root='/home/uz1/DATA!/medmnist', download=True,split='val',transform=transform)
    n_classes = 8
    with open("/home/uz1/projects/GCN/kmeans-model-7-medmnist-blood.pkl", "rb") as f:
        k = pickle.load(f)
loader = DataLoader(data_128, batch_size=32, drop_last=True, num_workers=16)






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
def populateS(labels,n_clusters=8,s=None):
    """"
    Calculates the S cluster assigment transform of input patch features 
    and returns the (S) and the aggregated (out_adj) as well.
    shape : ( number of patches , number of clusters)
    """
    # print("S is " ,s==None)
    n_patches=len(labels)
    div = int(sqrt(n_patches))
    if s == None:
        s = np.zeros((n_patches,n_clusters))
        for i in range(s.shape[0]):
            s[i][labels[i]] = 1
    else:
        s=s

    #calc adj matrix
    adj = to_dense_adj(grid(n_patches//div,n_patches//div)[0]).reshape(n_patches,n_patches)
    return s , np.matmul(np.matmul(s.transpose(1, 0),adj ), s)
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

class ImageTOGraphDataset(Dataset):
    """ 
    Dataset takes holds the kmaens classifier and vae encoder. On each input image we encode then get k mean label then formulate graph as Data object
    """
    def __init__(self,data,vae,kmeans,norm_adj=True,return_x_only=None,nclusters=8):
        self.data=data
        self.vae=vae
        self.return_x_only=return_x_only
        self.kmeans=kmeans
        self.norm_adj = norm_adj
        self.nclusters=nclusters
        self.patch_iter = PatchIter(patch_size=(32, 32), start_pos=(0, 0))
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):

        patches = []
        for x in self.patch_iter(self.data[index][0]):
            patches.append(x[0])

        patches = torch.stack([torch.tensor(np.array(patches))],0).squeeze()

        z=get_embedding_vae(patches,self.vae).clone().detach().double().cpu().numpy()
        label=self.kmeans.predict(z)
        s,out_adj = populateS(label,self.nclusters)
        x = np.matmul(s.transpose(1,0) , z)
        if self.return_x_only:
            return x,label,self.data[index][1]
        #if normlaise adj
        if self.norm_adj:
            out_adj = out_adj.div(out_adj.sum(1))
            #nan to 0 in tensor
            out_adj = out_adj.nan_to_num(0)
            #assert if there is nan in tensor
            assert out_adj.isnan().any() == False , "Found nan in out_adj"
        return Data(x=torch.tensor(x).float(),edge_index=dense_to_sparse(out_adj)[0],y=torch.tensor(self.data[index][1]),edge_attr=dense_to_sparse(out_adj)[1])


from torch_geometric.loader import DataLoader as GraphDataLoader

ImData = ImageTOGraphDataset(data=data_128,vae=vae,kmeans=k,return_x_only=True,nclusters=args.nclusters)
train_loader =GraphDataLoader(ImData, batch_size=args.batch_size, shuffle=True, num_workers=0,)
#print dataset stats
print("Dataset stats:")
print("Number of graphs: {}".format(len(ImData)))
# print("Number of features: {}".format(ImData.num_features))
print("batch size: {}".format(train_loader.batch_size))

# ImData_valid = ImageTOGraphDataset(data=valid_dataset,vae=vae,kmeans=k,return_x_only=True)
# valid_loader = GraphDataLoader(ImData_valid,batch_size=16,shuffle=True,num_workers=0)
#import dataloader from torch
from torch.utils.data import DataLoader
train_loader = DataLoader(ImData, batch_size=args.batch_size,)
# for all data in the dataset: create a h5y file to store the data and save to disk
import h5py
import os
import numpy as np
import torch
from torch_geometric.data import Data,Dataset
from tqdm import tqdm
#create h5py file
h5f = h5py.File(f'graph-data---{args.dataset}-{args.nclusters}.h5', 'w')
#create dataset in file
# h5f.create_dataset('x',(len(ImData),ImData[0].x.shape[0],ImData[0].x.shape[1]))
# h5f.create_dataset('x', (len(ImData),ImData[1].x.shape[0],ImData[1].x.shape[1]))
h5f.create_dataset('x', (len(ImData),ImData[1][0].shape[0],ImData[1][0].shape[1]))
# h5f.create_dataset('edge_index', (len(ImData),ImData[1].edge_index.shape[0],ImData[1].edge_index.shape[1]),maxshape=(None,2,None))
# h5f.create_dataset('edge_attr', (len(ImData),ImData[1].edge_index.shape[0]))
# add the data to h5py file
n_clusters = ImData[1][0].shape[0]
embed_size = ImData[1][0].shape[1]
#try numpy array
edge_i = []
edge_at = []
ys=[]
tmp=0
for i,d in tqdm(enumerate(train_loader),total=len(train_loader)):
    # print(ImData[i].x)
    # print(ImData[i].x.shape)
    # h5f["x"].resize((h5f["x"].shape[0],ImData[i].x.shape[0]), axis = 0)
    # h5f["x"][-ImData[i].x.shape[0]:,:,:] = ImData[i].x
    # h5f['x'][i:i+d[0].shape[0]] = d[0].reshape(-1,8,1024)
    # move index by size of d[0].shape[0]
    h5f['x'][tmp:tmp+d[0].shape[0]] = d[0].reshape(-1,n_clusters,embed_size)
    tmp=tmp+d[0].shape[0]
    print("from {} to {}".format(tmp,d[0].shape[0]+tmp))


    #resize for edge index
    # h5f["edge_index"].resize((h5f["edge_index"].shape[0],ImData[i].edge_index.shape[0],ImData[i].edge_index.shape[1]), axis = 0)
    # h5f["edge_index"][:,:,-ImData[i].edge_index.shape[1]:] = ImData[i].edge_index.numpy()
    # h5f['edge_index'][i,:,] = ImData[i].edge_index.numpy()
    # h5f['edge_attr'][i] = ImData[i].edge_attr.numpy()
    # h5f['y'][i] = ImData[i].y.numpy()
    # dst[i] = temsp[i].x.numpy()
    # edge_i.append(ImData[i].edge_index.numpy())
    # edge_at.append(ImData[i].edge_attr.numpy())
    ys.append(d[2])
    edge_i.append(d[1])

edge_i = np.stack(edge_i[:-1])
ys = np.stack(ys[:-1])
ys = np.concatenate([ys.reshape(-1),d[2].reshape(-1)]) 
edge_1 = np.concatenate([edge_i.reshape(-1,16),d[1].reshape(-1,16)])
# edge_at = np.array(edge_at)
# ys = np.array(ys)
#create datasets for edge index and edge attr and ys
h5f.create_dataset('edge_index', data=edge_i)
h5f.create_dataset('ys', data=ys)
# h5f.create_dataset('edge_attr', data=edge_at)
# h5f.create_dataset('y', data=ys)
#close the file
h5f.close()
# np.savez(fn,x=dst)
# read the h5py file and print the data
# h5f = h5py.File(f'graph-data--{args.dataset}.h5', 'r')
# del dst
# # dst = np.load(fn)
# print(h5f['x'][0])
# print(h5f['x'][0].shape)
# # for i in range(5):
# #     print(f"Data is equal at {i} : ",(h5f['x'][i] == ImData[i][0])
#     # print(f"Data is equal at {i} : ",np.array_equal(dst['x'][i],temsp[i].x.numpy()))
# print(h5f['x'].shape)


# #compare shape 0 are all equal
# print(f"Shape 0 is equal: ",(h5f['x'].shape[0] == h5f['edge_index'].shape[0]))
# print(f"Shape 0 is equal: ",(h5f['x'].shape[0] == h5f['edge_attr'].shape[0]))
# print(f"Shape 0 is equal: ",(h5f['x'].shape[0] == h5f['y'].shape[0]))

# h5f.close()



# %%
