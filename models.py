import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import math

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, 
                 # n_gene, 
                 activation):
        super(MLP, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            # nn.BatchNorm1d(n_gene),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            # nn.BatchNorm1d(n_gene),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return self.seq(x)

def matrix_poly(matrix, d):
    x = torch.eye(d, device = matrix.device).double()+ torch.div(matrix, d)
    return torch.matrix_power(x, d)

def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

class GRNVAE(nn.Module):
    def __init__(
        self, n_gene, hidden_dim=128, z_dim=1, activation=nn.Tanh,
        train_on_non_zero=False, dropout_augmentation=0.05,
        pretrained_A=None, 
    ):
        super(GRNVAE, self).__init__()
        self.n_gene = n_gene
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.train_on_non_zero = train_on_non_zero
        
        if pretrained_A is None:
            adj_A = torch.ones(n_gene, n_gene) / (n_gene - 1) + 0.0001
            adj_A += torch.randn_like(adj_A) * 0.0001
        else:
            adj_A = pretrained_A
        self.adj_A = nn.Parameter(adj_A, requires_grad=True)
        
        self.inference_zposterior = MLP(1, hidden_dim, 2*z_dim, activation)
        self.generative_pxz = MLP(z_dim, hidden_dim, 1, activation)
        self.dropout_on_input = dropout_augmentation
        # self.dropout_on_adj = nn.Dropout(p = dropout_augmentation)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def get_adj_(self):
        eye_tensor = torch.eye(self.n_gene, device = self.adj_A.device)
        mask = torch.ones_like(self.adj_A) - eye_tensor
        return (self.adj_A * mask)
    
    def get_adj(self):
        return self.get_adj_().cpu().detach().numpy()
    
    def get_adj_h(self):
        # Compute constraint h(A) value following DAG-GNN
        # https://github.com/fishmoon1234/DAG-GNN/blob/master/src/train.py
        return _h_A(self.adj_A, self.n_gene)
    
    def I_minus_A(self):
        eye_tensor = torch.eye(self.n_gene, device = self.adj_A.device)
        # clean up A along diagnal line
        mask = torch.ones_like(self.adj_A) - eye_tensor
        clean_A = self.adj_A * mask
        # clean_A = self.dropout_on_adj(self.adj_A) * mask
        return eye_tensor - clean_A
    
    def reparameterization(self, z_posterior):
        mu = z_posterior[:, :, :self.z_dim]
        std = z_posterior[:, :, self.z_dim:]
        return mu + std * torch.randn_like(std)
        
    def forward(self, x, global_mean, global_std, use_dropout_augmentation=True):                 
        if self.train_on_non_zero:
            non_zero_mask = (x != 0)
        else:
            non_zero_mask = torch.ones_like(x)
                
        if use_dropout_augmentation:
            x = x * (torch.rand_like(x) > self.dropout_on_input)
            
        x = (x - global_mean) / global_std

        # Encoder --------------------------------------------------------------
        I_minus_A = self.I_minus_A()
                
        z_posterior = self.inference_zposterior(x.unsqueeze(-1))
        z_posterior = torch.einsum('ogd,gh->ohd', z_posterior, I_minus_A)
        z_posterior[:, :, self.z_dim:] = torch.sqrt(torch.exp(
            z_posterior[:, :, self.z_dim:]))
        z_mu = z_posterior[:, :, :self.z_dim]
        z_sigma = z_posterior[:, :, self.z_dim:]
        z = self.reparameterization(z_posterior)
        
        # Decoder --------------------------------------------------------------
        z_inv = torch.einsum('ogd,gh->ohd', z, torch.inverse(I_minus_A))
        x_rec = self.generative_pxz(z_inv).squeeze(2)
        
        # Losses ---------------------------------------------------------------
        if self.train_on_non_zero:
            loss_rec_all = (x - x_rec).pow(2)
            loss_rec = torch.sum(loss_rec_all * non_zero_mask)
            # loss_rec = loss_rec + torch.sum(loss_rec_all * zero_mask) * 0.1
            loss_rec = loss_rec / torch.sum(non_zero_mask)
        else:
            loss_rec = torch.mean((x - x_rec).pow(2))
        
        # if direct_loss:
        loss_kl = -0.5 * torch.mean(
            1 + torch.log(z_sigma.pow(2)) - z_mu.pow(2) - z_sigma.pow(2))
        # else:
        #     z_posterior_normal = Normal(z_posterior[:, :, :self.z_dim], 
        #                                 z_posterior[:, :, self.z_dim:])        
        #     z_prior_normal = Normal(
        #         torch.zeros_like(z_posterior[:, :, :self.z_dim]), 
        #         torch.ones_like(z_posterior[:, :, :self.z_dim]))
        #     loss_kl = torch.abs(z_posterior_normal.log_prob(z).mean() - \
        #                         z_prior_normal.log_prob(z).mean())
                
        out = {
            'loss_rec': loss_rec, 'loss_kl': loss_kl, 
            'z_posterior': z_posterior, 'z': z
        }
        return out

class GRNVAE_3dA(nn.Module):
    def __init__(
        self, n_gene, hidden_dim=128, z_dim=1, activation=nn.Tanh,
        train_on_non_zero=False, dropout_augmentation=0.05,
        pretrained_A=None, A_dim=1
    ):
        super(GRNVAE_3dA, self).__init__()
        self.n_gene = n_gene
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.train_on_non_zero = train_on_non_zero
        self.A_dim = A_dim
        
        if pretrained_A is None:
            adj_A = torch.ones(self.A_dim, n_gene, n_gene) / (n_gene - 1) + 0.0001
        else:
            adj_A = pretrained_A
        self.adj_A = nn.Parameter(adj_A, requires_grad=True)
        
        self.inference_zposterior = MLP(1, hidden_dim, 2*z_dim, activation)
        self.generative_pxz = MLP(z_dim, hidden_dim, 1, activation)
        self.dropout_on_visible = nn.Dropout(p = dropout_augmentation)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def get_adj(self):
        eye_tensor = torch.eye(self.n_gene, device = self.adj_A.device)
        mask = torch.ones_like(eye_tensor) - eye_tensor
        return (self.adj_A * mask.repeat(self.A_dim, 1, 1)).cpu().detach().sum(dim=0).numpy()
    
    def get_adj_h(self):
        # Compute constraint h(A) value following DAG-GNN
        # https://github.com/fishmoon1234/DAG-GNN/blob/master/src/train.py
        return _h_A(self.adj_A, self.n_gene)
    
    def I_minus_A(self):
        eye_tensor = torch.eye(self.n_gene, device = self.adj_A.device)
        # clean up A along diagnal line
        mask = torch.ones_like(eye_tensor) - eye_tensor
        clean_A = self.adj_A * mask.repeat(self.A_dim, 1, 1)
        return eye_tensor.repeat(self.A_dim, 1, 1) - clean_A
    
    def reparameterization(self, z_posterior):
        mu = z_posterior[:, :, :, :self.z_dim]
        std = z_posterior[:, :, :, self.z_dim:]
        return mu + std * torch.randn_like(std)
        
    def forward(self, x, global_mean, global_std, use_dropout_augmentation=True): 
        if self.train_on_non_zero:
            non_zero_mask = (x != 0)
        else:
            non_zero_mask = torch.ones_like(x)
        
        if use_dropout_augmentation:
            x = self.dropout_on_visible(x)
        
        cell_min = x.min(1, keepdims=True)[0]
        cell_max = x.max(1, keepdims=True)[0]
        x_cell_norm = (x-cell_min)/(cell_max - cell_min)
            
        x = (x_cell_norm - global_mean) / global_std

        # Encoder --------------------------------------------------------------
        I_minus_A = self.I_minus_A()
                
        z_posterior = self.inference_zposterior(x.unsqueeze(-1))
        z_posterior = torch.einsum('ogd,tgh->tohd', z_posterior, I_minus_A)
        z_posterior[:, :, :, self.z_dim:] = torch.sqrt(torch.exp(
            z_posterior[:, :, :, self.z_dim:]))
        z_mu = z_posterior[:, :, :, :self.z_dim]
        z_sigma = z_posterior[:, :, :, self.z_dim:]
        z = self.reparameterization(z_posterior)
        
        # Decoder --------------------------------------------------------------
        z_inv = torch.einsum('togd,tgh->tohd', z, torch.inverse(I_minus_A))
        x_rec = self.generative_pxz(z_inv).squeeze(3)
        
        # Losses ---------------------------------------------------------------
        if self.train_on_non_zero:
            loss_rec = torch.sum((x.repeat(self.A_dim, 1, 1) - x_rec).pow(2) * non_zero_mask.repeat(self.A_dim, 1, 1))
            loss_rec /= torch.sum(non_zero_mask) * self.A_dim
        else:
            loss_rec = torch.mean((x.repeat(self.A_dim, 1, 1) - x_rec).pow(2))
        
        # if direct_loss:
        loss_kl = -0.5 * torch.mean(
            1 + torch.log(z_sigma.pow(2)) - z_mu.pow(2) - z_sigma.pow(2))
        # else:
        #     z_posterior_normal = Normal(z_posterior[:, :, :self.z_dim], 
        #                                 z_posterior[:, :, self.z_dim:])        
        #     z_prior_normal = Normal(
        #         torch.zeros_like(z_posterior[:, :, :self.z_dim]), 
        #         torch.ones_like(z_posterior[:, :, :self.z_dim]))
        #     loss_kl = torch.abs(z_posterior_normal.log_prob(z).mean() - \
        #                         z_prior_normal.log_prob(z).mean())

        loss_sparse = torch.mean(torch.abs(self.adj_A))
                
        out = {
            'loss_rec': loss_rec, 'loss_kl': loss_kl, 
            'loss_sparse': loss_sparse,
            'z_posterior': z_posterior, 'z': z
        }
        return out
