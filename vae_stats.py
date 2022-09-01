
# %%
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
args.add_argument("--batch_size", type=int, default=32)
args.add_argument("--epochs", type=int, default=100)
args.add_argument("--lr", type=float, default=0.001)
args.add_argument("--weight_decay", type=float, default=0.0)
args.add_argument("--train",type=bool,default=False)
args.add_argument("--dataset",type=str,default="")

args = args.parse_args()
print(args._get_args())

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
                ])) for x in self.image_filenames ]
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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop((32, 32)),
    transforms.ConvertImageDtype(torch.float),
])
if args.dataset == "wss":
    data = wss_dataset_class("/home/uz1/data/wsss/train/1.training", 'all',
                         transform)
elif args.dataset == "pcam":
    data = HDF5Dataset("/home/uz1/DATA!/pcam/pcam/training_split.h5","/home/uz1/DATA!/pcam/Labels/Labels/camelyonpatch_level_2_split_train_y.h5",transform=transform)
else:
    raise ValueError("Dataset not supported")
loader = DataLoader(data, batch_size=32, drop_last=True, num_workers=16)


# %%
# %%
if args.dataset == "wss":
    val_data = wss_dataset_class("/home/uz1/data/wsss/train/1.training",
                                        'valid', transform)
    val_loader = DataLoader(data, batch_size=16, drop_last=True, num_workers=16)

# %%
# lr logging
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from callbacks import TestReconCallback
import os

callbacks = []
lr_monitor = LearningRateMonitor(logging_interval="epoch")
callbacks.append(lr_monitor)

# save checkpoint on last epoch only
ckpt = ModelCheckpoint("/home/uz1/projects/GCN/logging/",
                       monitor="elbo",
                       save_weights_only=True)
callbacks.append(ckpt)

# add test for mid-train recon viewing
test = [data[x][0] for x in range(16)]
test = torch.stack(test, 0)
testRecon = TestReconCallback(test)
callbacks.append(testRecon)

# %%
pl.seed_everything(1234)

vae = VAE(input_height=data[0][0].shape[1], latent_dim=1024)
trainer = pl.Trainer(gpus=1,
                     max_epochs=500,
                     progress_bar_refresh_rate=10,
                     callbacks=callbacks)
if args.train:
# vae = vae.load_from_checkpoint("/home/uz1/projects/GCN/logging/final.cpkt")
    vae = vae.load_from_checkpoint("/home/uz1/projects/GCN/logging/epoch=461-step=145529.ckpt")
    trainer.fit(vae, loader)


# %%

vae = vae.load_from_checkpoint("/home/uz1/projects/GCN/logging/epoch=20-step=172031.ckpt")

# 

#load images to be patched 
from torchvision import transforms
from torch.utils.data import DataLoader
## images as 128x128 to 16 patches (each 32x32)
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomResizedCrop((128,128)),
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize((128,128)),
])
if args.dataset == "wss":
    data_128 = wss_dataset_class("/home/uz1/data/wsss/train/1.training", 'all',
                         transform)
if args.dataset == "pcam":
    data_128 = data
    data_128.transform = transform
    data_128.limit=3000
loader = DataLoader(data_128, batch_size=32, drop_last=True, num_workers=16)






# %%
# %%
from torchvision.transforms import ToPILImage,ToTensor
from matplotlib.pyplot import imshow, figure
import numpy as np
from torchvision.utils import make_grid
from skimage import graph, io, segmentation, color
from matplotlib import pyplot as plt
import pickle
from torch_geometric.utils import to_dense_adj, grid,dense_to_sparse
from monai.data import GridPatchDataset, DataLoader, PatchIter
patch_iter = PatchIter(patch_size=(32, 32), start_pos=(0, 0))
#patching each image
with open("/home/uz1/projects/GCN/kmeans-model-4.pkl", "rb") as f:
    k = pickle.load(f)
pil = ToPILImage()
to_tensor=ToTensor()
def get_embedding_vae(x,vae):

	x_encoded = vae.encoder(x)
	mu, log_var = vae.fc_mu(x_encoded), vae.fc_var(x_encoded)
	std = torch.exp(log_var / 2)
	q = torch.distributions.Normal(mu, std)
	z = q.rsample()
	return z 
def populateS(labels,shape=(16,8),s=None):
    """"
    Calculates the S cluster assigment transform of input patch features 
    and returns the (S) and the aggregated (out_adj) as well.
    """
    # print("S is " ,s==None)
    if s == None: 
        s= np.zeros(shape)
        for i in range(s.shape[0]):
            s[i][labels[i]] = 1
    else:
        s=s
    
    #calc adj matrix
    adj = to_dense_adj(grid(shape[0]//4,shape[0]//4)[0]).reshape(shape[0],shape[0])
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
def populateS(labels,shape=(16,8)):
    """"
    Calculates the S cluster assigment transform of input patch features 
    and returns the (S) and the aggregated (out_adj) as well.
    """
    s = np.zeros(shape) 

    for i in range(s.shape[0]):
        s[i][labels[i]] = 1
    
    #calc adj matrix
    adj = to_dense_adj(grid(shape[0]//4,shape[0]//4)[0]).reshape(shape[0],shape[0])
    return s , np.matmul(np.matmul(s.transpose(1, 0),adj ), s)


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
    def __init__(self,data,vae,kmeans,return_labels=False):
        self.data=data
        self.vae=vae
        self.kmeans=kmeans
        self.return_labels=return_labels
        self.patch_iter = PatchIter(patch_size=(32, 32), start_pos=(0, 0))
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        
        patches = []
        for x in self.patch_iter(self.data[index][0]):
            patches.append(x[0])
            
        patches = torch.stack([torch.tensor(patches)],0).squeeze()

        z=get_embedding_vae(patches,self.vae).clone().detach().cpu().numpy()
        label=self.kmeans.predict(z)
        s,out_adj = populateS(label)
        x = np.matmul(s.transpose(1,0) , z)
        # if return labels as well
        if self.return_labels:
            return Data(x=torch.tensor(x).float(),edge_index=dense_to_sparse(out_adj)[0],y=[self.data[index][1],list(label)],edge_attr=dense_to_sparse(out_adj)[1])
        return Data(x=torch.tensor(x).float(),edge_index=dense_to_sparse(out_adj)[0],y=self.data[index][1],edge_attr=dense_to_sparse(out_adj)[1])

class KDataset(Dataset):
    
    """
    Dataset to store the cluster repr of images as embedding feature maps 
    and edge as a grid formation ? - not really we use cluster feature relations as a 

    """

    def __init__(self,zs,data_128,labels,root=None,transform=None,pre_transform=None,pre_filter=None):
        super(KDataset,self).__init__(root,transform,pre_transform,pre_filter)
        self.zs=zs
        self.data = data_128
        self.labels = labels
    

    def __getitem__(self,index):
        
        s,out_adj = populateS(self.labels[index])
        x = np.matmul(s.transpose(1,0) , self.zs[index])

        
        return Data(x=x,edge_index=dense_to_sparse(out_adj)[0],y=torch.tensor(self.data[index][1]),edge_attr=dense_to_sparse(out_adj)[1])
    def __len__(self):
        return len(self.zs)


from torch_geometric.loader import DataLoader as GraphDataLoader
ImData = ImageTOGraphDataset(data=data_128,vae=vae,kmeans=k,return_labels=True)
train_loader =GraphDataLoader(ImData, batch_size=32, shuffle=True, num_workers=0)

valid_dataset= HDF5Dataset("/home/uz1/DATA!/pcam/pcam/validation_split.h5","/home/uz1/DATA!/pcam/Labels/Labels/camelyonpatch_level_2_split_valid_y.h5",limit=10000,transform=transform)
ImData_valid = ImageTOGraphDataset(data=valid_dataset,vae=vae,kmeans=k)
valid_loader = GraphDataLoader(ImData_valid,batch_size=16,shuffle=True,num_workers=0)


# %%
# if zs is not None:
#     k_data = KDataset(zs,data_128,labels,pre_filter=filter_a)
#     from torch_geometric.loader import DataLoader
#     print("No final data: ",len(k_data))

#     train_loader = DataLoader(k_data, batch_size=8, shuffle=True)
#     for data in train_loader:
#         print(data)
#         break

# %%

# %%
import torch
from torch.nn import Linear,Sequential
from torch_geometric.nn import GCNConv,GATConv,EdgeConv,DynamicEdgeConv
from torch_geometric.nn import global_mean_pool


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class GCN(torch.nn.Module):
    '''
    Can add edge_atttr using the second out of dense_to_sparse
    '''
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        
        gcon = EdgeConv
        self.conv1 = gcon(Sequential(Linear(2*1024, 512)))

        self.conv2 =gcon(Sequential(Linear(2*512, 256)))#gcon(512, 256)
        self.conv3 =gcon(Sequential(Linear(2*256, 256)))#gcon(512, 256)
        self.conv4 =gcon(Sequential(Linear(2*256, 128)))#gcon(512, 256)
        self.conv5 =gcon(Sequential(Linear(2*128, 64)))#gcon(512, 256)
        self.conv6 =gcon(Sequential(Linear(2*64, 32)))#gcon(512, 256)
        self.classifier = Linear(32, 3)

    def forward(self, x, edge_index,batch):
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)
        h = h.relu()
        h = self.conv4(h, edge_index)
        h = h.relu()
        h = self.conv5(h, edge_index)
        h = h.relu()
        h = self.conv6(h, edge_index)
        # h = h.tanh()  # Final GNN embedding space.
        h = global_mean_pool(h,batch)  # [batch_size, hidden_channels]

        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out#, h


# %%
# print(f'Embedding shape: {list(out.shape)}')

import time
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GraphUNet
from torch.functional import F
# model = GraphUNet(9,128,2,4,act=F.tanh)




# calculate the occurence of each cluster assignment and the taragets for each cluster
m=4
cluisters = np.zeros(m)
#init one for each class 1 , 0 
clusters_0 = np.zeros(m)
clusters_1 = np.zeros(m)
#coutn each class
count_0 = 0
count_1 = 0
for data in tqdm(train_loader):
    for k,i in enumerate(data.y[1]):
        cluisters[i]+=1
        if data.y[0][k].item() ==0:
            clusters_0[i]+=1
            count_0+=1
        else:
            clusters_1[i]+=1
            count_1+=1

# create bar graph of the occurence of each cluster with the target for each cluster

import matplotlib.pyplot as plt

plt.bar(range(m),cluisters,color='b')
plt.xticks(range(m),range(m))
plt.title('Occurence of each cluster')
# save
plt.savefig("./cluster_occurence.png")

plt.close()
# plot the occurence of each cluster for each class
plt.bar(range(m),clusters_0,color='b')
plt.xticks(range(m),range(m))
plt.title('Occurence of each cluster for class 0')
# save
plt.savefig("./cluster_occurence_0.png")
plt.close()

plt.bar(range(m),clusters_1,color='r')
plt.xticks(range(m),range(m))
plt.title('Occurence of each cluster for class 1')
# save
plt.savefig("./cluster_occurence_1.png")
#close
plt.close()

#plot class 0 and class 1
plt.bar(range(2),[count_0,count_1],color='g')
plt.xticks(range(2),['class 0','class 1'])
plt.title('Occurence of each class')
# save
plt.savefig("./class_occurence.png")