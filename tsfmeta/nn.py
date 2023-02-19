# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:12:32 2022

@author: MSH
"""


import torch


from .layers import (
    FExtractor_cnn,
    FExtractor_cnn_SE,
    )

class MetaComb_se(torch.nn.Module):


    def __init__(
            self,
            n_features,
            n_forecasters,
            window_x, 
            horizon, 
            Use_z = True, 
            n_known_features = None,
            x_filters = 64,
            z_filters = 64,
            p=0.5 ,
            device = 'cuda',
            ):
        
        super().__init__()
        self.type = 'comb'
        self.n_features = n_features
        self.n_forecasters = n_forecasters  
        self.window_x = window_x     
        self.horizon = horizon          
        self.dim_extract_features = x_filters + z_filters  if Use_z else x_filters       
   
        self.device = device
        self.n_known_features = n_features if n_known_features is None else n_known_features
        self.Use_z =  Use_z       
        
        self.norm_layer_x = torch.nn.InstanceNorm1d(n_features+1)
        
        self.norm_layer_z = torch.nn.InstanceNorm1d(n_known_features)
        
        self.fcn_y = FExtractor_cnn_SE(n_features+1, x_filters, window_x, p=p)
        
        if self.n_known_features>0:
            self.fcn_z = FExtractor_cnn_SE(self.n_known_features, z_filters, horizon, p=p)
        
        self.weight = torch.nn.Sequential(
            torch.nn.Linear(self.dim_extract_features,n_forecasters),                         
            torch.nn.Softmax(dim = 1),
            )
        
        self.LossPred = torch.nn.Sequential(
                torch.nn.Linear(self.dim_extract_features,n_forecasters),
                torch.nn.ReLU(),
                )
        
        
    def forward(self, meta):
       
        x = meta['X'].to(self.device)
        b = meta['B'].to(self.device)
        z = meta['Z'].to(self.device)       
        n_samples, _ ,_  = x.shape
                
        x = self.norm_layer_x(x)
        x = self.fcn_y(x)
        
        if self.Use_z:        
            z = self.norm_layer_z(z)
            z = self.fcn_z(z)        
            xz = torch.concat([x,z],dim = -1)
            weights = self.weight(xz.view(n_samples,-1))
        else:
            weights = self.weight(x.view(n_samples,-1))
        
        y = weights.unsqueeze(-1) * b   

        return y.sum(-2)


class MetaComb(torch.nn.Module):


    def __init__(
            self,
            n_features,
            n_forecasters,
            window_x, 
            horizon, 
            Use_z = True, 
            n_known_features = None,
            x_filters = 64,
            z_filters = 64,
            p=0.5 ,
            device = 'cuda',
            ):
        
        super().__init__()
        self.type = 'comb'
        self.n_features = n_features
        self.n_forecasters = n_forecasters  
        self.window_x = window_x     
        self.horizon = horizon          
        self.dim_extract_features = x_filters + z_filters  if Use_z else x_filters       
   
        self.device = device
        self.n_known_features = n_features if n_known_features is None else n_known_features
        self.Use_z =  Use_z       
        
        self.norm_layer_x = torch.nn.InstanceNorm1d(n_features+1)
        
        self.norm_layer_z = torch.nn.InstanceNorm1d(n_known_features)
        
        self.fcn_y = FExtractor_cnn(n_features+1, x_filters, window_x, p=p)
        
        if self.n_known_features>0:
            self.fcn_z = FExtractor_cnn(self.n_known_features, z_filters, horizon, p=p)
        
        self.weight = torch.nn.Sequential(
            torch.nn.Linear(self.dim_extract_features,n_forecasters),                         
            torch.nn.Softmax(dim = 1),
            )
        
        self.LossPred = torch.nn.Sequential(
                torch.nn.Linear(self.dim_extract_features,n_forecasters),
                torch.nn.ReLU(),
                )
        
        
    def forward(self, meta):
       
        x = meta['X'].to(self.device)
        b = meta['B'].to(self.device)
        z = meta['Z'].to(self.device)       
        n_samples, _ ,_  = x.shape
                
        x = self.norm_layer_x(x)
        x = self.fcn_y(x)
        
        if self.Use_z:        
            z = self.norm_layer_z(z)
            z = self.fcn_z(z)        
            xz = torch.concat([x,z],dim = -1)
            weights = self.weight(xz.view(n_samples,-1))
        else:
            weights = self.weight(x.view(n_samples,-1))
        
        y = weights.unsqueeze(-1) * b   

        return y.sum(-2)  
    

class MetaSelection(MetaComb):   
    
    def __init__(self, 
                 n_features,
                 n_forecasters, 
                 window_x, 
                 horizon, 
                 Use_z = True, 
                 n_known_features = None, 
                 x_filters = 64,
                 z_filters = 64,
                 p=0.5 ,
                 device = 'cuda'):        
       
        super().__init__(
                n_features = n_features, 
                n_forecasters = n_forecasters,
                window_x = window_x, 
                horizon = horizon, 
                Use_z = Use_z,
                n_known_features = n_known_features,
                x_filters = x_filters,
                z_filters = z_filters,
                p = p ,
                device  = device,
                )
        self.type = 'select'    
        
    def forward(self, meta, Use_z=True):
        x = meta['X'].to(self.device)
        b = meta['B'].to(self.device)
        z = meta['Z'].to(self.device)       
        n_samples, _ ,_  = x.shape
        # Normalize
        
        x = self.norm_layer_x(x)
        x = self.fcn_y(x)
        
        
        if self.Use_z:        
            z = self.norm_layer_z(z)
            z = self.fcn_z(z)        
            xz = torch.concat([x,z],dim = -1)
            weights = self.weight(xz.view(n_samples,-1))
        else:
            weights = self.weight(x.view(n_samples,-1))     

        return weights

class MetaLoss(MetaComb):
    
    def __init__(self, 
                 n_features,
                 n_forecasters, 
                 window_x, 
                 horizon, 
                 Use_z = True, 
                 n_known_features = None, 
                 x_filters = 64,
                 z_filters = 64,
                 p=0.5 ,
                 device = 'cuda'):        
       
        super().__init__(
                n_features = n_features, 
                n_forecasters = n_forecasters,
                window_x = window_x, 
                horizon = horizon, 
                Use_z = Use_z,
                n_known_features = n_known_features,
                x_filters = x_filters,
                z_filters = z_filters,
                p = p ,
                device  = device,
                )
        
        self.type = 'loss'        
        
    def forward(self, meta, Use_z=True):
        x = meta['X'].to(self.device)
        b = meta['B'].to(self.device)
        z = meta['Z'].to(self.device)       
        n_samples, _ ,_  = x.shape
        # Normalize
        
        x = self.norm_layer_x(x)
        x = self.fcn_y(x)
                
        if self.Use_z:        
            z = self.norm_layer_z(z)
            z = self.fcn_z(z)        
            xz = torch.concat([x,z],dim = -1)
            loss = self.LossPred(xz.view(n_samples,-1))
        else:               
            loss = self.LossPred(x.view(n_samples,-1))         

        return loss
    
from deepdow.layers.misc import CovarianceMatrix
from deepdow.layers import SparsemaxAllocator,SoftmaxAllocator,AnalyticalMarkowitz, NumericalRiskBudgeting

class MetaComb_risk(MetaComb):
    
    def __init__(self, 
                 n_features,
                 n_forecasters, 
                 window_x, 
                 horizon, 
                 Use_z = True, 
                 n_known_features = None, 
                 x_filters = 64,
                 z_filters = 64,
                 p=0.5 ,
                 device = 'cuda'):        
       
        super().__init__(
                n_features = n_features, 
                n_forecasters = n_forecasters,
                window_x = window_x, 
                horizon = horizon, 
                Use_z = Use_z,
                n_known_features = n_known_features,
                x_filters = x_filters,
                z_filters = z_filters,
                p = p ,
                device  = device,
                )
        
        self.type = 'comb'
        self.weight = torch.nn.Sequential(
            torch.nn.Linear(self.dim_extract_features,n_forecasters),                         
            torch.nn.Softmax(dim = 1),
            )   
        self.cov_layer = CovarianceMatrix(
            sqrt=True, shrinkage_strategy="scaled_identity"
        )     
        
        self.allocater = NumericalRiskBudgeting(n_assets = n_forecasters)
        
    def forward(self, meta, Use_z=True):
        x = meta['X'].to(self.device)
        b = meta['B'].to(self.device)
        z = meta['Z'].to(self.device)     
        e = meta['E'].to(self.device) 
        n_samples, _ ,_  = x.shape
        covmat_sqrt = self.cov_layer(e.swapaxes(1,-1))        
        x = self.norm_layer_x(x)
        x = self.fcn_y(x)
        
        if self.Use_z:        
            z = self.norm_layer_z(z)
            z = self.fcn_z(z)        
            xz = torch.concat([x,z],dim = -1)
            weights = self.weight(xz.view(n_samples,-1))
        else:
            weights = self.weight(x.view(n_samples,-1))
        
        weights = self.allocater(covmat_sqrt,weights)
        y = weights.unsqueeze(-1) * b   

        return y.sum(-2)