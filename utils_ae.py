from typing import Iterable
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
