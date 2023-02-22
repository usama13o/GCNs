import os.path as osp
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool,DynamicEdgeConv
import numpy as np
import PIL as pl
from torch_geometric.data import Data

import torch_geometric
from torch_geometric.transforms import BaseTransform
from skimage import future
from torch_scatter import scatter_min
from torch_scatter import scatter_mean
from torch_geometric.data import Data
from skimage import graph, data, io, segmentation, color
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from skimage import draw
import numpy as np
# tensor prep
import torch

from bisect import bisect_right
import datetime
import glob
import math
from os import listdir
from random import sample
import numpy as np
from sklearn.feature_selection import SelectFdr
import tifffile
from torch import dtype
import torch.utils.data as data
from pathlib import Path
# Standard libraries
import torch.utils.data as data
import numpy as np
import pickle 
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
from os.path import join
from math import sqrt
from torch_geometric.utils import to_dense_adj, grid,dense_to_sparse
def get_embedding_vae(x,vae):

	x_encoded = vae.encoder(x)
	mu, log_var = vae.fc_mu(x_encoded), vae.fc_var(x_encoded)
	std = torch.exp(log_var / 2)
	q = torch.distributions.Normal(mu, std)
	z = q.rsample()
	return z
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
         # TODO optimise this!
    else:
        s=s

    #calc adj matrix
    adj = to_dense_adj(grid(n_patches//div,n_patches//div)[0]).reshape(n_patches,n_patches)
    return s , np.matmul(np.matmul(s.transpose(1, 0),adj ), s)

class ImageToClusterHD5(data.Dataset):
    """ 
    Dataset takes holds the kmaens classifier and vae encoder. On each input image we encode then get k mean label then formulate graph as Data object
    """
    def __init__(self,data,norm_adj=True,split=None,n_clusters=None):
        #read h5 file into self .data
        self.data = h5py.File(data,'r')
        self.x = self.data['x']
        self.ys = self.data['ys'][:].reshape(-1)
        self.labels = self.data['edge_index'][:].reshape(-1,self.data['edge_index'].shape[2])
        self.nc = n_clusters
        # print("number of clusters", self.nc)
        # print("len of x", self.x.shape)
        # print("len of l", self.labels.shape)
        # print("len of y", self.ys.shape)
        self.x= self.x[:self.labels.shape[0]]
        self.ys = self.ys[:self.labels.shape[0]]
        print("->len of x", self.x.shape)
        print("->len of l", self.labels.shape)
        print("->len of y", self.ys.shape)

        assert len(self.x) == len(self.labels) , "x and labels should be same length"
        self.norm_adj = norm_adj
        # self.num_classes=9 They have a builtin property for this
        if split == 'train':
            self.x = self.x[:int(len(self.x)*.8)]
            self.labels = self.labels[:int(len(self.labels)*.8)]
            self.ys = self.ys[:int(len(self.ys)*.8)]
        elif split == 'val':
            self.x = self.x[int(len(self.x)*.8):]
            self.labels = self.labels[int(len(self.labels)*.8):]
            self.ys = self.ys[int(len(self.ys)*.8):]        
        
    def __len__(self):
        return len(self.x)
    def __getitem__(self,index):

        label = self.labels[index]
        s,out_adj = populateS(label,n_clusters=label.shape[0] if self.nc == None else self.nc)
        x = self.x[index][:]
        if self.norm_adj:
            out_adj = out_adj.div(out_adj.sum(1))
            #nan to 0 in tensor 
            out_adj = out_adj.nan_to_num(0)
            #assert if there is nan in tensor
            assert out_adj.isnan().any() == False , "Found nan in out_adj"
        # assert self.ys[index] != None , "Found None in ys"
        return Data(x=torch.tensor(x).float(),edge_index=dense_to_sparse(out_adj)[0],y=torch.tensor([self.ys[index]]),edge_attr=dense_to_sparse(out_adj)[1])

dataset_raw = ImageToClusterHD5(data="/home/uz1/projects/GCN/GraphGym/run/graph-data---pathmnist-32-256-UC_True.h5")
from typing import Optional
from sklearn.utils import shuffle
from torchvision import datasets
from torch_geometric.data import DataLoader
import torchvision.transforms as T
from torch_geometric.utils import grid, to_dense_adj
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import SubsetRandomSampler
batch_size = 128
max_nodes = dataset_raw[0].x.shape[0] * batch_size
print("max nodes", max_nodes)
train_loader = DataLoader(dataset_raw, batch_size=batch_size , shuffle=True)
val_loader = DataLoader(dataset_raw, batch_size=batch_size , shuffle=True)

dataset = dataset_raw
dataset.num_features = dataset[0].x.shape[1]
dataset.num_nodes = dataset[0].x.shape[0]
dataset.num_classes = 9
class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 normalize=False,
                 lin=True):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))

    def forward(self, x, adj, mask=None):
        # print("in frwd ", x.shape) # batch_size x num_nodes, in_channels
        # print("in frwd adj", adj.shape)
        # batch_size, num_nodes, in_channels = x.size()

        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
            # print("in frwd ", x.shape)
            x = self.bns[step](x.permute(0, 2, 1)).permute(0, 2, 1)
            # print("after bn",x.shape)
            # x = x.permute(0, 2, 1)

        return x

class DGNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 normalize='batch',
                 lin=True):
        super(DGNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(EdgeConv2d(in_channels, hidden_channels, norm=normalize))

        self.convs.append(EdgeConv2d(hidden_channels, hidden_channels,norm=normalize))

        self.convs.append(EdgeConv2d(hidden_channels, out_channels, norm=normalize))

    def forward(self, x, adj, mask=None):
        # batch_size, num_nodes, in_channels = x.size()

        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
            # print("in frwd ", x.shape)
            # x = x.permute(0, 2, 1)

        return x

class DiffPool(torch.nn.Module):
    def __init__(self):
        super(DiffPool, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(dataset.num_features, 64, num_nodes)
        self.gnn1_embed = GNN(dataset.num_features, 64, 64)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(64, 64, num_nodes)
        self.gnn2_embed = GNN(64, 64, 64, lin=False)

        self.gnn3_embed = GNN(64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(64, 64)
        self.lin2 = torch.nn.Linear(64, dataset.num_classes)

        self.stem = Stem(64,3,128)

    def forward(self, x, adj, mask=None,return_clusters=False):
        x_temp = x
        #add stem downsampling
        # b,n,f = x.shape 
        # x_stem=x
        # x=x.reshape(b,-1,128)
        # print(x.shape,adj.shape)
        d = int(x.shape[0] - to_dense_adj(adj).shape[1])
        # print("d" , d)
        adj = torch.nn.functional.pad(to_dense_adj(adj), (0, d,0, d), mode='constant', value=0)
        s1 = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        # print("1st adj ", adj.shape)
        x, adj, l1, e1 = dense_diff_pool(x,adj , s1, mask)
        #x_1 = s_0.t() @ z_0
        #adj_1 = s_0.t() @ adj_0 @ s_0
        # print("2nd adj ", adj.shape)

        s2 = self.gnn2_pool(x,adj)
        x = self.gnn2_embed(x, adj)
        x, adj, l2, e2 = dense_diff_pool(x,  adj, s2)
        # print("3nd adj ", adj.shape)

        x = self.gnn3_embed(x, adj)

        x = x.view(max(mask)+1,-1,x.shape[2]).mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        if return_clusters : return x_stem, x_temp, s1,s2
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2

class DynDiffPool(torch.nn.Module):
    def __init__(self):
        super(DynDiffPool, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = DGNN(dataset.num_features, 64, num_nodes)
        self.gnn1_embed = DGNN(dataset.num_features, 64, 64)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = DGNN(64, 64, num_nodes)
        self.gnn2_embed = DGNN(64, 64, 64, lin=False)

        self.gnn3_embed = DGNN(64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(64, 64)
        self.lin2 = torch.nn.Linear(64, dataset.num_classes)

        self.stem = Stem(64,3,128)

    def forward(self, x, adj, mask=None,return_clusters=False):
        x_temp = x
        #add stem downsampling
        b,n,f = x.shape 
        x = self.stem(x.view(b,3,int(n**.5),-1))
        x_stem=x
        x=x.reshape(b,128,-1,1)
        ad = dense_knn_matrix(x,16)
        # print(x.shape,adj.shape)
        s1 = self.gnn1_pool(x, ad, mask)
        x = self.gnn1_embed(x, ad, mask)

        x, adj, l1, e1 = dense_diff_pool(x.reshape(b,-1,64), adj, s1.reshape(b,n,-1), mask)
        #x_1 = s_0.t() @ z_0
        #adj_1 = s_0.t() @ adj_0 @ s_0

        x=x.reshape(b,128,-1,1)
        ad = dense_knn_matrix(x,16)

        s2 = self.gnn2_pool(x, ad)
        x = self.gnn2_embed(x, ad)


        x, adj, l2, e2 = dense_diff_pool(x.reshape(b,-1,64), adj, s2.reshape(b,n,-1), mask)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        if return_clusters : return x_stem, x_temp, s1,s2
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2

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
import torch.nn as nn
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()        
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        # print(x.shape)
        return x

from tqdm import tqdm
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = DiffPool().to(device)
# model = torch.load("/home/uz1/projects/GCN/checkpoint")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.to(device)
losses = AverageMeter()
val_accc = AverageMeter()
test_accc = AverageMeter()
'''
- stem downsamples from 64x64 to 16x16 - adj matches that
- num features per node is 128 - output of stem 
'''


def train(epoch):
    model.train()
    loss_all = 0

    for i,data in enumerate(tqdm(train_loader,total=len(train_loader))):
        data = data.to(device)
        optimizer.zero_grad()
        print(data)
        output,_,_ = model(data.x, data.edge_index,data.batch)
        # print("Done with forward pass")
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        losses.update(loss)
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()

        print(f"iteration {i} loss {losses.avg}")
    return loss_all / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0

    for data in tqdm(loader,total=len(loader)):
        data = data.to(device)
        pred,_,_ = model(data.x, data.edge_index,data.batch)
        # print(pred.shape)
        ###calculate accuracy
        
        correct += pred.argmax().eq(data.y.view(-1)).sum().item()
        # print(correct,len(loader.sampler.indices))
    return correct / len(loader.sampler.indices)


best_val_acc = test_acc = 0
for epoch in range(1, 151):
    train_loss = train(epoch)
    val_acc = test(val_loader)
    val_accc.update(val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
    print(
        f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f} ({losses.avg}), '
        f'Val Acc: {val_acc:.4f} ({val_accc.avg})'
    )
