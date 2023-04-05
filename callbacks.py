
import pytorch_lightning as pl
import torch
from matplotlib.pyplot import imshow, figure
import numpy as np
from torchvision.utils import make_grid
import torchvision
class TestReconCallback_vae(pl.Callback):
    '''
    callback for variational autoencoder
    Given a stack of images:
        #generate mean and std of some test data
    
        test # [B,C,H,W]
    Input to model encoder and calculate the mean and std 

    Decode z to recons and plot using plt 
    
    '''
    
    def __init__(self, input_imgs, every_n_epochs=1,logs=None):
        self.logs=logs if logs !=None else ''
        super().__init__()
        self.input_imgs = input_imgs
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)
        
    def on_train_epoch_end(self, trainer, pl_module):
        '''
        Here we ...
        '''
        # print("at callback ", pl_module )
        vae=pl_module
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                # print("Encoding !")
                if input_imgs.dim() == 3:
                    input_imgs = input_imgs.unsqueeze(1)
                    input_imgs = input_imgs.repeat(1, 3, 1, 1)

                x_encoded = vae.encoder(input_imgs)
                # print("encoded, ", x_encoded.shape)
                

                mu, log_var = vae.fc_mu(x_encoded), vae.fc_var(x_encoded)
                std = torch.exp(log_var / 2)
                # Z COMES FROM NORMAL(0, 1)
                num_preds = 16
                p = torch.distributions.Normal(mu, std)
                z = p.rsample()
                # print("smaples, ", z.shape)


                # SAMPLE IMAGES
                pred = vae.decoder(z.to(vae.device)).cpu()
                pred = torch.concat([pred,input_imgs.cpu()],0).cpu()
                # print(pred.shape)
                # print("pred ",pred.shape )
                # grid= make_grid(pred).permute(1, 2, 0).numpy() * 1 # std + mean
                grid = torchvision.utils.make_grid(pred, nrow=16, normalize=True, range=(-1,1))

                pl_module.train()

            
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)
            trainer.logger.experiment.add_embedding(z,label_img=input_imgs,global_step=trainer.global_step)

class TestReconCallback_ae(pl.Callback):
    '''
    callback for Autoencoder
    Given a stack of images:
        #generate mean and std of some test data
    
        test # [B,C,H,W]
    Input to model encoder and calculate the mean and std 

    Decode z to recons and plot using plt 
    
    '''
    
    def __init__(self, input_imgs, every_n_epochs=1,logs=None):
        self.logs=logs if logs !=None else ''
        super().__init__()
        self.input_imgs = input_imgs
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)
        
    def on_epoch_end(self, trainer, pl_module):
        '''
        Here we ...
        '''
        # print("at callback ", pl_module )
        vae=pl_module
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                # print("Encoding !")
                pred= vae.forward(input_imgs)
                
                pred = torch.concat([pred.cpu(),input_imgs.cpu()],0).cpu()
                # print(pred.shape)
                # print("pred ",pred.shape )
                # grid= make_grid(pred).permute(1, 2, 0).numpy() * 1 # std + mean
                grid = torchvision.utils.make_grid(pred.cpu(), nrow=16, normalize=True, range=(-1,1))

                pl_module.train()

            
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)