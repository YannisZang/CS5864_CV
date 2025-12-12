import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss for the original GAN.
    
    Use the torch.nn.functional.binary_cross_entropy_with_logits rather than softmax followed by BCELoss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    # loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    real_labels = torch.ones_like(logits_real)
    loss_real = bce_loss(logits_real, real_labels)
    
    fake_labels = torch.zeros_like(logits_fake)
    loss_fake = bce_loss(logits_fake, fake_labels)
    
    loss = loss_real + loss_fake
     
    ##########       END      ##########
    
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss for the original GAN.
    
    Use the torch.nn.functional.binary_cross_entropy_with_logits rather than softmax followed by BCELoss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    labels = torch.ones_like(logits_fake)
    loss = bce_loss(logits_fake, labels)
    
    ##########       END      ##########
    
    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the LSGAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    # loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    loss_real = 0.5 * torch.mean((scores_real - 1) ** 2)
    loss_fake = 0.5 * torch.mean((scores_fake) ** 2)
    
    loss = loss_real + loss_fake
    
    ##########       END      ##########
    
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the LSGAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    # loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    
    loss = 0.5 * torch.mean((scores_fake - 1) ** 2)
    ##########       END      ##########
    
    return loss

def wgan_discriminator_loss(scores_real, scores_fake):
    """
    Wasserstein GAN discriminator (critic) loss.

    Inputs:
    - scores_real: Tensor of shape (N,) with critic scores for real data.
    - scores_fake: Tensor of shape (N,) with critic scores for fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    # loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    
    loss = torch.mean(scores_fake) - torch.mean(scores_real)
    ##########       END      ##########
    
    return loss


def wgan_generator_loss(scores_fake):
    """
    Wasserstein GAN generator loss.

    Inputs:
    - scores_fake: Tensor of shape (N,) with critic scores for fake data.

    Returns:
    - loss: scalar Tensor
    """
    # loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    loss = -torch.mean(scores_fake)
    
    ##########       END      ##########
    
    return loss

