import numpy as np
import matplotlib.pyplot as plt
from utils import *

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import graph, data, io, segmentation, color
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from skimage import draw
from torch_scatter import scatter_mean
from torch_geometric.data import Data
from skimage import graph, data, io, segmentation, color
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from skimage import draw
# tensor prep
import torch
import PIL as pl 
from torch_geometric.transforms import BaseTransform
from skimage import future
from torch_scatter import scatter_min
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.nn import global_mean_pool

from datasets import GeoFolders_2
DATASET_PATH ="/home/uz1/data/geo/slices/64"
import torch_geometric.transforms as T


from datasets import GeoGnn

import numpy as np
import PIL as pl 
from torch_geometric.transforms import BaseTransform
from skimage import future
from torch_scatter import scatter_min
class ImgToGraph(BaseTransform):
    r"""Converts an image to a superpixel representation using the
    :meth:`skimage.segmentation.slic` algorithm, resulting in a
    :obj:`torch_geometric.data.Data` object holding the centroids of
    superpixels in :obj:`pos` and their mean color in :obj:`x`
    (functional name: :obj:`to_slic`).

    This transform can be used with any :obj:`torchvision` dataset.

    Example::

        from torchvision.datasets import MNIST
        import torchvision.transforms as T
        from torch_geometric.transforms import ToSLIC

        transform = T.Compose([T.ToTensor(), ToSLIC(n_segments=75)])
        dataset = MNIST('/tmp/MNIST', download=True, transform=transform)

    Args:
        add_seg (bool, optional): If set to `True`, will add the segmentation
            result to the data object. (default: :obj:`False`)
        add_img (bool, optional): If set to `True`, will add the input image
            to the data object. (default: :obj:`False`)
        **kwargs (optional): Arguments to adjust the output of the SLIC
            algorithm. See the `SLIC documentation
            <https://scikit-image.org/docs/dev/api/skimage.segmentation.html
            #skimage.segmentation.slic>`_ for an overview.
    """
    def __init__(self, add_seg=False, add_img=False, **kwargs):
        self.add_seg = add_seg
        self.add_img = add_img
        self.kwargs = kwargs

    def __call__(self, img,mask,n_seg=50):
      segments_slic = segmentation.slic(img, n_segments=n_seg,compactness=10, sigma=1,
                          start_label=0)
    
      seg = torch.from_numpy(segments_slic)
      rag = rag_mean_band(img[:,:,:], segments_slic, connectivity=2, mode='similarity', sigma=255.0,ch=img.shape[2])

      img=torch.from_numpy(img)
            
      mask[mask!=0] = 1
      mask = torch.from_numpy(mask)[:,:,:1]
      h, w, c = img.shape
      # pinta ll shapes 
    #   print(seg.shape,img.shape,mask.shape)
      x = scatter_mean(img.view(h * w, c), seg.view(h * w), dim=0)

      pos_y = torch.arange(h, dtype=torch.float)
      pos_y = pos_y.view(-1, 1).repeat(1, w).view(h * w)
      pos_x = torch.arange(w, dtype=torch.float)
      pos_x = pos_x.view(1, -1).repeat(h, 1).view(h * w)

      pos = torch.stack([pos_x, pos_y], dim=-1)
      pos = scatter_mean(pos, seg.view(h * w), dim=0)


      edge_index = np.asarray([[n1,n2] for (n1,n2) in rag.edges]).reshape(2,-1)#connectivity coodinates 
      weights = np.asarray([w[2]['weight'] for w in rag.edges.data()])
      x = np.asarray([n[1]['mean color'] for n in rag.nodes.items()])
      node_class = scatter_min(mask.view(h*w),seg.reshape(h*w),dim=0)[0]

    #   lc = future.graph.show_rag(seg, rag, img[:,:,:3])

    #   pos= np.asarray([n[1]['centroid'] for n in rag.nodes.items()])


      data = Data(x=torch.from_numpy(x), pos=pos,edge_index=torch.tensor(edge_index),edge_weight=torch.tensor(weights).unsqueeze(1),y=node_class[:])

      return data

transform =ImgToGraph()
train_dataset_full = GeoGnn(root="/home/uz1/data/geo/full_image/new_data/out",mask_root="/home/uz1/data/geo/full_image/new_data/out/anno/gimp_anno",tif_root="/home/uz1/data/geo/full_image/new_data/composites",transform=transform)


transform =ImgToGraph()
p=1
train_dataset = GeoFolders_2(
root=DATASET_PATH,  transform=transform,raw_dir='/home/uz1/data/geo/slices_raw/64/',split="none",pick=p,k_labels_path=f'./k_labels_{p}.pickle')
valid_dataset = GeoFolders_2(
root=DATASET_PATH,  transform=transform,raw_dir='/home/uz1/data/geo/slices_raw/64/',split="valid",pick="China_Area_2")


import time
from torch.nn import Linear
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GraphUNet
from torch.functional import F
model = GraphUNet(9,128,9,4)#,act=F.tanh)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.
data_laoder = DataLoader(train_dataset,batch_size=128,shuffle=False)
lin =  Linear(9,2)
def train(data):
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x.float(), data.edge_index)#,data.edge_weight.float())  # Perform a single forward pass.
    out = lin (out)
    loss = criterion(out.softmax(-1), data.y)  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss,out

for epoch in range(1000):
    # track using tqdm
    tqdm_bar = tqdm(data_laoder)
    for i,data in enumerate(tqdm_bar):
        loss,out = train(data)
        if i % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')
            #calculate acc
            pred = out.softmax(-1).argmax(-1)
            acc = (pred == data.y).float().mean()
            print(f'Accuracy: {acc.item():.4f}')
            print("Unique predictions ",np.unique(pred,return_counts=True))
            print("Unique labels ",np.unique(data.y,return_counts=True))
            # visualize_embedding(h, color=data.y, epoch=epoch, loss=loss)
            # o = Data(x=h,pos=data.pos,edge_index=data.edge_index,edge_weight=data.edge_weight,y=out.softmax(dim=-1).argmax(dim=-1))
            # G=to_networkx(o, to_undirected=True)
            # pos_dict = {e:[k,v] for e,(k,v) in enumerate(o.pos[:])}
            # visualize_graph(G, pos=pos_dict,color=o.y)
            # time.sleep(0.3)