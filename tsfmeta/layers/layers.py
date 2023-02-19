# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 17:12:33 2022

@author: MSH
"""

import torch
import torch.nn as nn

class squeeze_excite(nn.Module):

    def __init__(
            self, 
            L_in, 
            squeeze_ratio = 0.1,
            ):
        
        super().__init__()
        
        self.squeeze_to = int(L_in * squeeze_ratio)
        
        if self.squeeze_to == 0:
            self.squeeze_to = 1
        
        self.squeeze = nn.Sequential(
                nn.AdaptiveAvgPool1d(self.squeeze_to),
                nn.Linear(self.squeeze_to, L_in//2, bias=False),
                nn.ReLU(),
                nn.Linear(L_in//2, L_in, bias=False),
                nn.Sigmoid(),              
                
                )        
        
    def forward(self, x):
        
        return x * self.squeeze(x)
        

class FExtractor_cnn(torch.nn.Module):

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            window_x,
            kernel_size = 2, 
            stride = 1, 
            dilation = 1, 
            padding = 0,
            p = 0.5,
            ):        
        
        super().__init__()
        self.fcn = nn.Sequential(
            
            nn.Conv1d(
                    in_channels =  in_channels,
                    out_channels = out_channels, 
                    kernel_size= kernel_size if window_x > 16 else 1, 
                    stride= stride, 
                    padding = padding, 
                    dilation = dilation, 
                    groups = 1, 
                    bias = True,
                    ),                    
            nn.Conv1d(
                    in_channels =  out_channels,
                    out_channels = out_channels*2, 
                    kernel_size= kernel_size *2 if window_x > 16 else 1, 
                    stride= stride, 
                    padding = padding, 
                    dilation = dilation, 
                    groups = 1, 
                    bias = True,
                    ),            
            nn.Conv1d(
                    in_channels = out_channels*2,
                    out_channels = out_channels, 
                    kernel_size = kernel_size * 4 if window_x > 16 else 1, 
                    stride = stride, 
                    padding = padding, 
                    dilation = dilation, 
                    groups = 1, 
                    bias = True,
                    ),    
            
            nn.AdaptiveAvgPool1d(1),            
            nn.Dropout(p=p), 
        )
        
        
    def forward(self, x):
        
        return self.fcn(x)
    
    
class FExtractor_cnn_SE(torch.nn.Module):

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            window_x,
            kernel_size = 2, 
            stride = 1, 
            dilation = 1, 
            padding = 0,
            p = 0.5, 
            ):
        
        super().__init__()

        if window_x <= 16:
            kernel_size = 1
            
        
        Lout1 =    ( window_x + 2 * padding - dilation *(kernel_size -1) -1 )/stride + 1
        
        if window_x <= 16:
            Lout2 =    ( Lout1 + 2 * padding - dilation *(kernel_size -1) -1 )/stride + 1
        else:
            Lout2 =    ( Lout1 + 2 * padding - dilation *(kernel_size *2 -1) -1 )/stride + 1
        
        if Lout2 % 1 != 0 :
            raise ValueError("Lout should be integer!")
        
            
        self.fcn = torch.nn.Sequential(
            
            torch.nn.Conv1d(
                    in_channels =  in_channels,
                    out_channels = out_channels, 
                    kernel_size= kernel_size  if window_x > 16 else 1, 
                    stride= stride, 
                    padding = padding, 
                    dilation = dilation, 
                    groups = 1, 
                    bias = True),
            
            squeeze_excite(int(Lout1),0.1),
                    
            torch.nn.Conv1d(
                    in_channels =  out_channels,
                    out_channels = out_channels*2, 
                    kernel_size= kernel_size *2 if window_x > 16 else 1, 
                    stride= stride, 
                    padding=padding, 
                    dilation = dilation, 
                    groups = 1, 
                    bias = True),
            
            squeeze_excite(int(Lout2),0.1),
            
            torch.nn.Conv1d(
                    in_channels = out_channels*2,
                    out_channels = out_channels, 
                    kernel_size = kernel_size * 4 if window_x > 16 else 1, 
                    stride = stride, 
                    padding = padding, 
                    dilation = dilation, 
                    groups = 1, 
                    bias = True),
            
            
            torch.nn.AdaptiveAvgPool1d(1),
            
            torch.nn.Dropout(p=p), 
        )
        
        
    def forward(self, x):
        
        return self.fcn(x)