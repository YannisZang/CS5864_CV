import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3, use_spectral_norm=False):
        super(Discriminator, self).__init__()
    
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        def conv_d(layer):
            if use_spectral_norm:
                return nn.utils.spectral_norm(layer)
            else:
                return layer
            
        self.net = nn.Sequential(
            # in 3, out 128, ksize=4, stride=2, padding=1
            conv_d(nn.Conv2d(input_channels, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # in 128, out 256, ksize=4, stride=2, p=1
            conv_d(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # in 256, out 512
            conv_d(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # in 512, out 1024
            conv_d(nn.Conv2d(512, 1024, 4, 2, 1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            # in 1024, out 1 stride=1
            conv_d(nn.Conv2d(1024, 1, 4, 1, 0))
        )
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.net(x)
        x = x.view(x.size(0)) # flatten to shape (N,)
        
        ##########       END      ##########
        
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.net = nn.Sequential(
            # input: (N, noise_dim, 1, 1) → 4x4x1024
            nn.ConvTranspose2d(noise_dim, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # out 512
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # out 256
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # out 128
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # out 3
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1),
            nn.Tanh()
            
        )
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        x = self.net(x)
        ##########       END      ##########
        
        return x
    

