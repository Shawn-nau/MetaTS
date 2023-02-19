# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:54:28 2022

@author: MSH
"""
import numpy as np

def metrics (pred, y, scale,mask=None):
    
    err = pred - y
    scale = np.where(scale == 0,(err**2).mean(-1), scale)  
    scale = np.where(scale == 0,1, scale)
    if mask is not None:
        err = err * mask
    
    MAE =  np.abs(err).mean(-1).mean(0)
    MSE =  ( err**2 ).mean(-1).mean(0)
    sMAPE1 =  (np.abs(err).sum(-1)/( pred + y+ 1e-6).sum(-1)  ).mean(0)
    sMAPE2 =  (np.abs(err)/( np.abs(pred) + y + 1e-6) ).mean(-1).mean(0)
      
    RMSSE =  np.sqrt((err**2).mean(-1)/ scale).mean(0)
    
    return {'MAE':MAE, 'MSE':MSE, 'sMAPE1' : sMAPE1, 'sMAPE2' : sMAPE2,'RMSSE' : RMSSE}


