
import glob
import torch
import numpy as np
import PIL.Image

from datasets import DivideIntoPatches
import argparse
from torchvision.models import swin_transformer
from componants import DecoderBlock
from torch import nn
import torch.nn.functional as F

def main(args):
    im_s  = args.img_size
    pz = args.patch_size
    bs = args.batch_size

    
    import pytorch_lightning as pl
    from torch import nn
    from torch.nn import functional as F
    from pl_bolts.models.autoencoders.components import (
        resnet18_decoder,
        resnet18_encoder,
    )
    import torch
    import warnings
    warnings.filterwarnings("ignore")

    class VAE(pl.LightningModule):
        def __init__(self, enc_out_dim=512, latent_dim=256, input_height=32):
            super().__init__()

            self.save_hyperparameters()

            # encoder, decoder
            self.input_height = input_height
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
            # print(x.shape)
            if x.shape[2] < 3:
                x = x.repeat(1, 1, 3, 1,1)
            if x.dim() >3: # b,16,3,h,w -> b*16,3,h,w
                x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
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
        def validation_step(self, batch, batch_idx):
            x, _ = batch
            #if channels are less than 3, repeat channels
            # print(x.shape)
            if x.shape[2] < 3:
                x = x.repeat(1, 1, 3, 1,1)
            # print(x.shape)
            if x.dim() >3: # b,16,3,h,w -> b*16,3,h,w
                x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
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
            recon_loss_ = self.gaussian_likelihood(x_hat, self.log_scale, x)
            # print(recon_loss.shape)
            recon_loss = torch.nn.MSELoss()(x_hat,x)


            # kl
            kl = self.kl_divergence(z, mu, std)

            # elbo
            elbo = (kl - recon_loss_) # with old recon_loss
            # elbo = (kl + recon_loss)
            elbo = elbo.mean()

            self.log_dict({
                'val_elbo': elbo,
                'val_kl': kl.mean(),
                'val_recon_loss_': recon_loss.mean(),
                'val_recon_loss': recon_loss_.mean(),
                'val_reconstruction': recon_loss.mean(),
                'val_kl': kl.mean(),
            })

            return elbo
        # using LightningDataModule
    class LitDataModule(pl.LightningDataModule):
        def __init__(self, batch_size,data,val_data=None):
            super().__init__()
            self.save_hyperparameters()
        # or
            self.batch_size = batch_size
            self.data = data
            self.val_data = val_data

        def train_dataloader(self):
            return DataLoader(self.data, batch_size=self.batch_size, drop_last=True, num_workers=24)
        def val_dataloader(self):
            return DataLoader(self.val_data, batch_size=self.batch_size, drop_last=True, num_workers=24)
    from torchvision import transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((im_s, im_s)),
        transforms.ConvertImageDtype(torch.float),
        DivideIntoPatches(patch_size=pz), # takes an image tensor and returns a list of patches stacked as (H // patch_size **2 x H x W x C)
    ])
    # data = wss_dataset_class("/home/uz1/data/wsss/train/1.training", 'all',
                            #  transform)
    # data = HDF5Dataset("/home/uz1/DATA!/pcam/pcam/training_split.h5","/home/uz1/DATA!/pcam/Labels/Labels/camelyonpatch_level_2_split_train_y.h5",transform=transform)
    from medmnist.dataset import PathMNIST, BreastMNIST,OCTMNIST,ChestMNIST,PneumoniaMNIST,DermaMNIST,RetinaMNIST,BloodMNIST,TissueMNIST,OrganAMNIST,OrganCMNIST,OrganSMNIST
    # using a unified ataset of medmnist
    # # batch size should depend oon num_patches by 

    import medmnist
    import difflib
    d = difflib.get_close_matches(args.dataset,medmnist.dataset.INFO.keys())[0]
    d = medmnist.dataset.INFO[d]['python_class']
    if args.dataset == "pathmnist":
        data = PathMNIST(root=r"C:\Users\Usama\data", split='test',transform=transform)
    elif args.dataset == "dermamnist":
        data = DermaMNIST(root=r"C:\Users\Usama\data", split='test',transform=transform,download=True)
    elif args.dataset == "pnemonmnist":
        data = PneumoniaMNIST(root=r"C:\Users\Usama\data", split='train',transform=transform,download=True)
    elif args.dataset == "organsmnist":
        data = OrganSMNIST(root=r"C:\Users\Usama\data", split='train',transform=transform,download=True)
    elif args.dataset == "octmnist":
        data = OCTMNIST(root=r"C:\Users\Usama\data", split='train',transform=transform,download=True)
        val_data = OCTMNIST(root=r"C:\Users\Usama\data", split='val',transform=transform,download=True)
    elif args.dataset == "chestmnist":
        data = ChestMNIST(root=r"C:\Users\Usama\data", split='train',transform=transform,download=True)
        val_data = ChestMNIST(root=r"C:\Users\Usama\data", split='val',transform=transform,download=True)
    elif args.dataset == "breastmnist":
        data = BreastMNIST(root=r"C:\Users\Usama\data", split='train',transform=transform,download=True)
        val_data = BreastMNIST(root=r"C:\Users\Usama\data", split='val',transform=transform,download=True)
    loader = DataLoader(data, batch_size=args.batch_size, drop_last=True, num_workers=24)


    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from callbacks import  TestReconCallback_vae,TestReconCallback_ae
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    import os
    import random
    import datetime
    from math import sqrt

    callbacks = []
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    if args.dataset == "pathmnist":
        val_data= PathMNIST(root=r"C:\Users\Usama\data",download=True,split='val',transform=transform)
    if args.dataset == "dermamnist":
        val_data= DermaMNIST(root=r"C:\Users\Usama\data",download=True,split='val',transform=transform)
    if args.dataset == "pnemonmnist":
        val_data= PneumoniaMNIST(root=r"C:\Users\Usama\data",download=True,split='val',transform=transform)
    if args.dataset == 'organsmnist':
        val_data= OrganSMNIST(root=r"C:\Users\Usama\data",download=True,split='val',transform=transform)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True, num_workers=24)
    # loader = LitDataModule(args.batch_size, data,val_data)

    # save checkpoint on last epoch only
    ckpt_dir = fr"C:\Users\Usama\data\ckpt\{data.__class__.__name__}\{pz}\{im_s}"
    ckpt = ModelCheckpoint(ckpt_dir,
                        monitor="elbo",
                        save_weights_only=True)
    callbacks.append(ckpt)

    # add test for mid-train recon viewing
    test = [data[x][0] for x in random.sample(range(len(data)), 1)]

    test = torch.stack(test, 0).squeeze()
    testRecon = TestReconCallback_vae(test)
    callbacks.append(testRecon)


    early_stop_callback = EarlyStopping(monitor="val_elbo", min_delta=0.00, patience=3, verbose=False, mode="min",stopping_threshold=-2000.00)
    callbacks.append(early_stop_callback)



    pl.seed_everything(1234)

    vae2 = VAE(input_height=data[0][0].shape[2], latent_dim=256)
    print("Using input shape: ", data[0][0].shape, " latent dim: ", 256)
    # model = deeplab(args={'n_channel': 3, 'n_classes': 2})
    trainer = pl.Trainer(gpus=1,
                        max_epochs=10, #auto_scale_batch_size=True,
                        #  progress_bar_refresh_rate=10,
                         
                        callbacks=callbacks)
    
    if args.use_pretrain == True and not os.path.exists(ckpt_dir):
        vae = vae2
        vae.decoder.upscale1.size =  vae2.decoder.upscale1.size
        vae.batch_size = args.batch_size
        trainer = pl.Trainer(gpus=1,
                        max_epochs=10,#auto_scale_batch_size=True,
                        #  progress_bar_refresh_rate=10,
                        
                        callbacks=callbacks,num_sanity_val_steps=0)
        print("Using pretrained model")
        trainer.fit(vae, loader,val_loader)
        print("Training finished")
        
    else:
        print("ckpt found at: ", ckpt_dir)
        ckpt_dir = glob.glob(f"{ckpt_dir}/*.ckpt")[0]
        vae = vae2.load_from_checkpoint(ckpt_dir)

    num_points = 700 if len(data) > 20000 else len(data)
    
    h=int(int(data[0][0].shape[-1]) * sqrt(data[0][0].shape[0]))
    p_z = int(sqrt((h*h) // int(data[0][0].shape[0])))
    # num_points = num_points if (((h // p_z) * (h // p_z) ) * num_points ) % 1600 == 0 else 1600 // ((h // p_z) * (h // p_z))
    num_patches= (h // p_z) * (h // p_z)
    num_batches = 1600 // (num_patches * args.batch_size)
    num_points = (num_patches * args.batch_size) 
    print("given num_patches: ", num_patches, " num_points: ", num_points)

    vae = vae.to('cuda')



    #using mini batch kemeans 
    from sklearn.cluster import MiniBatchKMeans
    import math
    import pickle
    from math import sqrt,ceil
    from tqdm import tqdm
    h = im_s
    p_z = pz
    print("Using a VAE (for kmeans) with h=",h,"and p(z)=",p_z)
    if args.batch_size < args.num_nodes:
        num_sub_batches = ceil(args.num_nodes / num_points)
    else:
        num_sub_batches = 1
    z_list = []
    if not os.path.exists(fr"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-{h}-{p_z}-{args.num_nodes}-{data.__class__.__name__}.pkl"):
        kmeans = MiniBatchKMeans(n_clusters=args.num_nodes,batch_size=args.batch_size,verbose=1)
        for i in tqdm(range(num_batches),total=num_batches,desc="Kmeans"):
            while num_sub_batches > 0:
                test,_ = next(iter(loader))
                test = test.to(vae.device)
                if test.shape[2] < 3:
                    test = test.repeat(1, 1, 3, 1,1)
                if test.dim() >4:
                    test = test.reshape(-1, test.shape[2], test.shape[3], test.shape[4]) # batch, num patches ,channel, height, width to batch, channel, height, width
                x_encoded = vae.encoder(test)
                mu, log_var = vae.fc_mu(x_encoded), vae.fc_var(x_encoded)
                std = torch.exp(log_var / 2)
                q = torch.distributions.Normal(mu, std)
                z = q.rsample()
                z=z.detach().cpu().numpy()
                z_list.append(z)
                num_sub_batches -= 1
            z = np.concatenate(z_list, axis=0)
            kmeans.partial_fit(z)
        print("Done - out shape ",z.shape)
        with open(fr"C:\Users\Usama\projects\GCNs\kmeans\kmeans-model-{h}-{p_z}-{args.num_nodes}-{data.__class__.__name__}.pkl", 'wb') as f:
            pickle.dump(kmeans, f)

    print("Kmeans Done !")


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('-p','--patch_size', type=int, default=32)
    args.add_argument('-i','--img_size', type=int, default=128)
    args.add_argument('-bs','--batch_size', type=int, default=8)
    args.add_argument('-k','--num_nodes', type=int, default=8)
    args.add_argument('-use_pretrain','--use_pretrain', type=bool, default=True)
    args.add_argument('--dataset', type=str, default='pathmnist', )
    args = args.parse_args()

    patch_size = args.patch_size
    img_size = args.img_size


    num_patches = (img_size // patch_size) ** 2

    print("Running with config: \n patch_size: {} \n img_size: {} \n num_patches: {}\n bs: {}\n num nodes:{} ".format(patch_size, img_size, num_patches,args.batch_size,args.num_nodes))

    main(args)
