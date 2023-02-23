import torch
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt


import argparse

def main(pz,im_s,bs):
    try:

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
        import pytorch_lightning as pl
        from torch import nn
        from torch.nn import functional as F
        from pl_bolts.models.autoencoders.components import (
            resnet18_decoder,
            resnet18_encoder,
        )
        import torch

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
                # print(x.shape)
                if x.shape[1] < 3:
                    x = x.repeat(1, 3, 1, 1)
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
                if x.shape[1] < 3:
                    x = x.repeat(1, 3, 1, 1)
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
        from torchvision import transforms
        from torch.utils.data import DataLoader
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.ConvertImageDtype(torch.float),
            DivideIntoPatches(patch_size=patch_size), # takes an image tensor and returns a list of patches stacked as (H // patch_size **2 x H x W x C)
        ])
        # data = wss_dataset_class("/home/uz1/data/wsss/train/1.training", 'all',
                                #  transform)
        # data = HDF5Dataset("/home/uz1/DATA!/pcam/pcam/training_split.h5","/home/uz1/DATA!/pcam/Labels/Labels/camelyonpatch_level_2_split_train_y.h5",transform=transform)
        from medmnist.dataset import PathMNIST, BreastMNIST,OCTMNIST,ChestMNIST,PneumoniaMNIST,DermaMNIST,RetinaMNIST,BloodMNIST,TissueMNIST,OrganAMNIST,OrganCMNIST,OrganSMNIST
        # using a unified ataset of medmnist
        from datasets import combined_medinst_dataset


        data = PathMNIST(root='/home/uz1/DATA!/medmnist', split='test',transform=transform)
        loader = DataLoader(data, batch_size=args.batch_size, drop_last=True, num_workers=16)



        from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
        from callbacks import  TestReconCallback_vae,TestReconCallback_ae
        import os
        import random
        import datetime
        from math import sqrt

        callbacks = []
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        val_data= PathMNIST(root='/home/uz1/DATA!/medmnist', download=True,split='val',transform=transform)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True, num_workers=16)
        len(val_data)

        # save checkpoint on last epoch only
        ckpt = ModelCheckpoint(f"/home/uz1/projects/GCN/logging/{data.__class__.__name__}/{args.patch_size}/",
                            monitor="elbo",
                            save_weights_only=True)
        callbacks.append(ckpt)

        # add test for mid-train recon viewing
        test = [data[x][0] for x in random.sample(range(len(data)), 1)]

        test = torch.stack(test, 0).squeeze()
        testRecon = TestReconCallback_vae(test)
        callbacks.append(testRecon)


        pl.seed_everything(1234)

        vae = VAE(input_height=data[0][0].shape[2], latent_dim=256)
        print("Using input shape: ", data[0][0].shape, " latent dim: ", 256)
        # model = deeplab(args={'n_channel': 3, 'n_classes': 2})
        trainer = pl.Trainer(gpus=1,
                            max_epochs=10,
                            #  progress_bar_refresh_rate=10,
                            
                            callbacks=callbacks,strategy="dp")

        if args.use_pretrain == True:
            vae = vae.load_from_checkpoint("/home/uz1/projects/GCN/logging/PathMNIST/2023_02_18/epoch=9-step=74990.ckpt")
            trainer = pl.Trainer(gpus=1,
                            max_epochs=1,
                            #  progress_bar_refresh_rate=10,
                            
                            callbacks=callbacks,strategy="dp")
            try:
                print("Using pretrained model")
                trainer.fit(vae, loader,val_loader)
                print("Training finished")
            except:
                print("Error in training, trying again with new model")
                vae = VAE(input_height=data[0][0].shape[2], latent_dim=256)
                trainer = pl.Trainer(gpus=1,
                            max_epochs=8,
                            #  progress_bar_refresh_rate=10,
                            
                            callbacks=callbacks,strategy="dp")
                trainer.fit(vae, loader,val_loader)
        # vae = vae.load_from_checkpoint("/home/uz1/projects/GCN/logging/epoch=461-step=145529.ckpt")
        # vae = vae.load_from_checkpoint("/home/uz1/projects/GCN/logging/epoch=20-step=172031.ckpt")

        num_points = 700 if len(data) > 20000 else len(data)
        
        h=int(int(data[0][0].shape[-1]) * sqrt(data[0][0].shape[0]))
        p_z = int(sqrt((h*h) // int(data[0][0].shape[0])))
        num_points = num_points if (((h // p_z) * (h // p_z) ) * num_points ) % 16000 == 0 else 16000 // ((h // p_z) * (h // p_z))

        # real test data  -  vis of means of data and class clusters
        # test = [data[x][0] for x in range(num_points)]
        # y_test = [data[x][1] for x in range(num_points)]
        # print([x for x in random.sample(range(len(data)), num_points)])
        # get test amd y_test wiht random num_points (use a random sample of data) - same for both 
        vae = vae.to('cpu')
        if data[0][0].dim() > 1:
            test = [data[x][0] for x in random.sample(range(len(data)), num_points)]
        else:
            test = [[data[x][0],data[x][1]] for x in random.sample(range(len(data)), num_points)]
        # y_test= [x[1] for x in test]

        test = torch.stack(test, 0).to(vae.device)
        if test.dim() >4:
            test = test.reshape(-1, test.shape[2], test.shape[3], test.shape[4])
        print(test.shape)
        if vae.__class__.__name__ == 'VAE':
            x_encoded = vae.encoder(test)

            mu, log_var = vae.fc_mu(x_encoded), vae.fc_var(x_encoded)
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()
            z=z.detach().cpu().numpy()
            print("Done - out shape ",z.shape)
        elif vae.__class__.__name__ == 'deeplab':
            z = vae.encode(test)[0].detach().cpu().numpy()


        from sklearn.cluster import KMeans
        import math
        import pickle
        from math import sqrt
        x_em = z
        h=int(int(data[0][0].shape[-1]) * sqrt(data[0][0].shape[0]))
        p_z = int(sqrt((h*h) // int(data[0][0].shape[0])))
        print("Using a VAE with h=",h,"and p(z)=",p_z)
        for n in [8,16,32,64,128]:
            k=KMeans(n_clusters=n).fit(x_em.reshape(x_em.shape[0],-1))
            k.labels_.shape 
            #count unique values + return counts
            unique, counts = np.unique(k.labels_, return_counts=True)
            print("unique values:", unique)
            print("counts:", counts)
            with open(f"kmeans-model-{h}-{p_z}-{n}-{data.__class__.__name__}.pkl", "wb") as f:
                pickle.dump(k, f)

        # catch if error is gpu memory error then run again with reduced batch size
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory - reducing batch size")
            args.batch_size = args.batch_size - 2
            print("New batch size: ", args.batch_size)
            #clear gpu memory
            torch.cuda.empty_cache()
            main(args.patch_size, args.img_size,args.batch_size)
        else:
            raise e


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('-p','--patch_size', type=int, default=32)
    args.add_argument('-i','--img_size', type=int, default=128)
    args.add_argument('-bs','--batch_size', type=int, default=8)
    args.add_argument('-use_pretrain','--use_pretrain', type=bool, default=False)

    args = args.parse_args()

    patch_size = args.patch_size
    img_size = args.img_size


    num_patches = (img_size // patch_size) ** 2

    print("Running with config: \n patch_size: {} \n img_size: {} \n num_patches: {}\n bs: {}".format(patch_size, img_size, num_patches,args.batch_size))

    main(args.patch_size, args.img_size,args.batch_size)
