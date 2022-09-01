from typing import Iterable
from matplotlib import pyplot as plt
import nibabel as nib
import numpy as np
import os
import torchvision.transforms as tt
import torch 
import torchvision
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz",'png','tiff','jpg',"bmp"])

def open_target_get_class(target):
        target = np.array(Image.open(target))
        target[target < 254] = 1
        target[target > 253] = 0
        # target= not target
        target = target.max()
        return target

def open_target_get_class_with_perc(target,perc):
        target = np.array(Image.open(target))
        target[target < 254] = 1
        target[target > 253] = 0
        # target= not target
        amount_of_1 = np.unique(target,return_counts=True)[1]
        if len(amount_of_1)<2:
            return 0
        else:
            if (amount_of_1[1] / (target.shape[0] * target.shape[1]* target.shape[2]) ) > perc:
                return 1
            else:
                return 0
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
def open_target_np_slides(path):
    im = open_image(path)
    mask= np.array(im)
    li = (np.unique(mask))
    if 29 in li:
        mask[mask==29]=0
    # normal case
    # if len(li)>5:
    if 'fixed' in path:
       # print('found normal slide' + path)
        mask[mask!=255]=0
        mask[mask==255]=2
    #tumour
    else:
       # print('found tumour slide' + path)
        mask = ~mask
        mask[mask!=255]=0

        mask[mask==255]=1
    li = (np.unique(mask,return_counts=True))
    # print(li)
    return mask[:,:,0,np.newaxis]
def open_target_np(path):
    im = open_image(path)
    mask= np.array(im)
    li = (np.unique(mask))
    if 29 in li:
        mask[mask==29]=0
    # normal case
    if len(li)>5:
        mask[mask!=255]=0
        mask[mask==255]=2
    #tumour
    else:
        mask[mask!=255]=0
        mask[mask==255]=1
    return mask[:,:,0,np.newaxis]
def open_target_np_peso(path):
    im = open_image(path)
    mask= np.array(im)
    mask[mask==1]=0
    mask[mask==2]=1
    return mask[:,:,0,np.newaxis]

def open_target_np_glas(path):
    im = open_image(path)
    mask= np.array(im)
    mask[mask!=0]=1
    return mask[:,:,np.newaxis]
def open_image_np(path):
    im = open_image(path)
    array = np.array(im)
    return array

def open_image_np_bw(path):
    im = open_image(path)
    array = np.array(im)
    return array[:,:,np.newaxis].repeat(3,axis=2)
def load_nifti_img(filepath, dtype):
    '''
    NIFTI Image Loader
    :param filepath: path to the input NIFTI image
    :param dtype: dataio type of the nifti numpy array
    :return: return numpy array
    '''
    nim = nib.load(filepath)
    out_nii_array = np.array(nim.get_data(),dtype=dtype)
    out_nii_array = np.squeeze(out_nii_array) # drop singleton dim in case temporal dim exists
    meta = {'affine': nim.get_affine(),
            'dim': nim.header['dim'],
            'pixdim': nim.header['pixdim'],
            'name': os.path.basename(filepath)
            }

    return out_nii_array, meta



def check_exceptions(image, label=None):
    if label is not None:
        if image.shape[:-1] != label.shape[:-1]:
            print('Error: mismatched size, image.shape = {0}, '
                  'label.shape = {1}'.format(image.shape, label.shape))
            #print('Skip {0}, {1}'.format(image_name, label_name))
            raise(Exception('image and label sizes do not match'))

    if image.max() < 1e-6:
        print('Error: blank image, image.max = {0}'.format(image.max()))
        #print('Skip {0} {1}'.format(image_name, label_name))
        raise (Exception('blank image exception'))

    if label.max() < 1e-6:
        print('Error:  label blank, image.max = {0}'.format(label.max()))
        #print('Skip {0} {1}'.format(image_name, label_name))
        raise (Exception('blank label exception'))




class Resize:
    def __init__(self, size,pil=False):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size
        
        self.toPil = tt.ToPILImage() if pil else None
        self.resize = tt.Resize(self._size)

    def __call__(self,x,y=None):
        
        _input=self.toPil(x) if self.toPil is not None else x
        if x.ndim < 3:
           _input=  _input.convert("L")
        _input = self.resize(_input)
        # _input = skimage.util.img_as_ubyte(_input)
        if y is not None:
            _input_y=self.toPil(y).convert("L")
            _input_y = self.resize(_input_y)
            if x.ndim < 3:

                return np.array(_input)[:,:,np.newaxis].repeat(3,axis=2), np.array(_input_y)[:,:,np.newaxis]
            return np.array(_input), np.array(_input_y)[:,:,np.newaxis]
        else:
            return np.array(_input)
def show_closest_images(train_img_embeds,k=50):
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    neigh = NearestNeighbors(n_neighbors=k)
    nn = neigh.fit(train_img_embeds[1])
    ind_list=[]
    for i in range(1700,1800):
        print(i)
        _,ind = nn.kneighbors(train_img_embeds[1][i].reshape(1,-1))
        ind_list.extend(*ind)
    topk = np.unique(ind_list,return_counts=True)[0][np.argpartition(np.unique(ind_list,return_counts=True)[1], -4)[-4:]]
    print("topk --> ",topk)
    topk_imgs=torch.cat([train_img_embeds[0][topk]],dim=0)
    img_g = torchvision.utils.make_grid(topk_imgs,nrow=4,normalize=True,range=(-1,1)).permute(1,2,0)
    plt.figure(figsize=(12, 3))
    plt.imshow(img_g)
    plt.axis('off')
def find_similar_images(query_img, query_z, key_embeds, K=8,knn=False,dist_metric='cos'):
        # Find closest K images. We use the euclidean distance here but other like cosine distance can also be used.
        if dist_metric== 'cos':
            dist = torch.cosine_similarity(query_z[None, :], key_embeds[1])
        else:
            dist = torch.cdist(query_z[None, :], key_embeds[1],p=2)
        dist = dist.squeeze(dim=0)
        if knn:
            from sklearn.neighbors import NearestNeighbors 
            neigh = NearestNeighbors(n_neighbors=8)
            nn = neigh.fit(key_embeds[1])
            dist, indices = nn.kneighbors(query_z.reshape(1,-1))
            indices = indices.reshape(-1)
        else:
            dist, indices = torch.sort(dist)
        # Plot K closest images
        imgs_to_display = torch.cat(
            [query_img[None], key_embeds[0][indices[:K]]], dim=0)
        grid = torchvision.utils.make_grid(
            imgs_to_display, nrow=K+1, normalize=True, range=(-1, 1))
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(12, 3))
        plt.imshow(grid)
        plt.axis('off')
        plt.savefig(f'grid_{indices[1]}_{dist_metric}_knn__{str(knn)}.png')
        # plt.show()