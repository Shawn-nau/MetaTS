# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 16:01:44 2022

@author: MSH
"""


"""Collection of functions related to data."""

import torch
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Tuple, Union
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch_geometric.data import HeteroData
from .hetero_static_graph_temporal_signal import HeteroStaticGraphTemporalSignal as HGT


class MetaDataset(Dataset):
    
    """Dataset for metalearning

    Parameters
    ----------
    X : np.ndarray
        Full features dataset of shape `(n_samples,1,lookback)`, are lags  of Y

  
    B : np.ndarray
        Full base forecasts dataset of shape `(n_samples, n_forecasters, horizon)`.
        
    Y : np.ndarray
        Full targets dataset of shape `(n_samples, horizon)`.
        
    Z : np.ndarray
        Full targets dataset of shape `(n_samples, n_features, lookback + horizon)`.
        
    Feat : None or array-like
        If not None then of shape `(n_samples,n_forecasters)` representing ts features for each sample.

    Yc : np.ndarray
        Full targets dataset of shape `(n_samples, n_forecasters)`representing best model indicator for each sample.
        
    Ye : np.ndarray
        Full targets dataset of shape `(n_samples, n_forecasters)`representing losses of base_forecasters for each sample.  
    
    y : np.ndarray
        Full targets dataset of shape `(n_samples, horizon)`representing the meta forecast for each sample. 
        
    transform : None or callable
        If provided, then a callable that transforms a single sample.
    """

    def __init__(
        self, 
        X : np.ndarray, 
        B: np.ndarray = None,
        Y: np.ndarray = None, 
        Z: np.ndarray = None, 
        MetaFeat: np.ndarray = None, 
        transform: np.ndarray = None,
        Xt_idx: pd.Index = None,
        Yt_tidx: pd.Index = None,
        df_idx: pd.MultiIndex = None,
        features_num: List[str] = None,
        features_cat: List[str] = None,
        forecasters: List[str] = None,
        static_known_features: List[str] = None,
        dynamic_known_features_num : List[str] = None,
        dynamic_known_features_cat : List[str] = None,
        mask: np.ndarray = None,  
        weight: np.ndarray = None, 
        edge_index:List[Union[np.ndarray, None]] = None,
        edge_weight: List[Union[np.ndarray, None]] = None,
    ):
        
        
        """Construct."""
        super().__init__()
        self.X = X
        self.B = B
        self.Y = Y
        self.Z = Z
        
        self.MetaFeat = MetaFeat
        
        self.Xt_idx =  Xt_idx
        self.Yt_tidx =  Yt_tidx
        self.df_idx = df_idx
        self.features_num = features_num
        self.features_cat = features_cat        
        self.mask = mask  if mask is not None else np.ones( (X.shape[0], X.shape[-1] + Y.shape[-1]))
        self.weight = weight if weight is not None else np.ones( (X.shape[0], X.shape[-1] + Y.shape[-1]))
        self.forecasters = forecasters  if forecasters is not None else [] 
        self.static_known_features = static_known_features if static_known_features is not None else [] 
        self.dynamic_known_features_num = dynamic_known_features_num 
        self.dynamic_known_features_cat = dynamic_known_features_cat     
        
        self.feature_names = self.features_num + self.features_cat
        self.known_features = self.dynamic_known_features_num + self.dynamic_known_features_cat + self.static_known_features
        self.unknown_features = [ff for ff in self.feature_names if ff not in self.known_features  ]
        
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        
        self.features_num_idx = [self.feature_names.index(i)-1 for i in self.features_num[1:]]
        
        self.features_cat_idx = [self.feature_names.index(i)-1 for i in self.features_cat]

        self.static_known_features_idx = [self.feature_names.index(i)-1 for i in self.static_known_features]      

        self.dynamic_known_features_num_idx = [self.feature_names.index(i)-1 for i in self.dynamic_known_features_num]  

        self.dynamic_known_features_cat_idx = [self.feature_names.index(i)-1 for i in self.dynamic_known_features_cat] 

        self.known_features_idx =  self.dynamic_known_features_num_idx + self.dynamic_known_features_cat_idx + self.static_known_features_idx
        
        self.unknown_features_idx = [self.feature_names.index(i)-1 for i in self.unknown_features]
        
        self.transform = transform
        # utility
        self.n_features, self.lookback = Z.shape[1], X.shape[2]
        
        if self.Y is not None:
            self.horizon = Y.shape[1]
            if self.B is not None:
                self.BestModel()
            else:
                self.Yc = None
                self.Ye = None
                self.err = None
                
        if self.transform:
            self.Y_trans, self.mask_trans = self.transform(self.Y, self.mask[:,self.lookback:(self.lookback+self.horizon)]  )                
            self.weight_trans = self.transform(self.weight[:,self.lookback:(self.lookback+self.horizon)]  )   
        

    def __len__(self):
        """Compute length."""
        return len(self.X)

    def __getitem__(self, ix: int) -> Dict[str, torch.Tensor]:
        """Get item."""
        if self.Y is not None:
            if self.transform:
               Y_sample = torch.from_numpy(self.Y_trans[ix]).type(torch.float32) 
               mask_sample = torch.from_numpy(self.mask_trans[ix]).type(torch.float32) 
               weight_sample = torch.from_numpy(self.weight_trans[ix]).type(torch.float32) 
            else:
               Y_sample = torch.from_numpy(self.Y[ix]).type(torch.float32)
               mask_sample = torch.from_numpy(self.mask[ix,self.lookback:(self.lookback+self.horizon)]).type(torch.float32)
               weight_sample = torch.from_numpy(self.weight[ix,self.lookback:(self.lookback+self.horizon)]).type(torch.float32)
        else:
            Y_sample = torch.zeros(1, 1)            
            mask_sample = torch.zeros(1, 1)
            weight_sample = torch.zeros(1, 1)
            
        
        X_sample = torch.from_numpy( np.concatenate([self.X[ix],self.Z[ix,:,0:self.lookback]],0) ).type(torch.float32) \
            if self.Z is not None else torch.from_numpy( self.X[ix] ).type(torch.float32)
        B_sample = torch.from_numpy(self.B[ix]).type(torch.float32)        
        E_sample = torch.from_numpy(self.err[ix]).type(torch.float32)  
       
        
        #Y_sample = torch.from_numpy(self.Y[ix]).type(torch.float32) if self.Y is not None else torch.zeros(1, 1)
        Z_sample = torch.from_numpy(self.Z[ix,self.known_features_idx,self.lookback:]).type(torch.float32)  if self.Z is not None else torch.zeros(1, 1)
        
        MetaFeat_sample = torch.from_numpy(self.MetaFeat[ix]).type(torch.float32) if self.MetaFeat is not None else torch.zeros(1, 1)          
        Yc_sample = torch.from_numpy(self.Yc[ix]).type(torch.float32) if self.Y is not None else torch.zeros(1, 1)
        Ye_sample = torch.from_numpy(self.Ye[ix]).type(torch.float32) if self.Y is not None else torch.zeros(1, 1)
        
        #mask_sample = torch.from_numpy(self.mask[ix]).type(torch.float32)         
        

        return dict(
                X = torch.nan_to_num(X_sample,nan = -1),
                B = torch.nan_to_num(B_sample,nan = 0), 
                Y = torch.nan_to_num(Y_sample,nan = 0),
                Z = torch.nan_to_num(Z_sample,nan = -1), 
                mask = torch.nan_to_num(mask_sample,0), 
                metafeat = torch.nan_to_num(MetaFeat_sample,nan =0), 
                Yc = torch.nan_to_num(Yc_sample,nan=0),
                Ye = torch.nan_to_num(Ye_sample,nan=0),
                weight = torch.nan_to_num(weight_sample,nan=0),
                E = torch.nan_to_num(E_sample,nan=0),
                )

    @staticmethod
    def _collate_fn(
        batches: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate function to combine items into mini-batch for dataloader.

        Args:
            batches (List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]): List of samples generated with
                :py:meth:`~__getitem__`.

        Returns:
            Tuple[Dict[str, torch.Tensor], Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]: minibatch
        """
        # collate function for dataloader

        X = torch.stack([batch["X"] for batch in batches])
        B = torch.stack([batch["B"] for batch in batches])
        Y = torch.stack([batch["Y"] for batch in batches])
        Z = torch.stack([batch["Z"] for batch in batches])
        mask = torch.stack([batch["mask"] for batch in batches])
        metafeat = torch.stack([batch["metafeat"] for batch in batches])
        Yc = torch.stack([batch["Yc"] for batch in batches])
        Ye = torch.stack([batch["Ye"] for batch in batches])
        weight = torch.stack([batch["weight"] for batch in batches])       
        E = torch.stack([batch["E"] for batch in batches]) 

        return dict(
                X = X,
                B = B,
                Y = Y,
                Z = Z,
                mask = mask,
                metafeat = metafeat,
                Yc = Yc,
                Ye = Ye,
                weight = weight,
                E = E,
                )
        
    def to_dataloader(
        self, train: bool = True, batch_size: int = 1024, batch_sampler: Union[Sampler, str] = None, **kwargs
    ) -> DataLoader:
        
        """
        Get dataloader from dataset.
        """
        default_kwargs = dict(
            shuffle=train,
            drop_last=train and len(self) > batch_size,
            collate_fn=self._collate_fn,
            batch_size=batch_size,
            batch_sampler=batch_sampler,
        )
        default_kwargs.update(kwargs)
        kwargs = default_kwargs

        return DataLoader(
            self,
            **kwargs,
        )        
    
    def BestModel(self):
        B = np.nan_to_num(self.B , nan  = 0)
      
        Y, mask  = self.Y, self.mask[:,self.lookback:(self.lookback+self.horizon)]
        
        if self.transform:
           Y, mask = self.transform(self.Y,self.mask[:,self.lookback:(self.lookback+self.horizon)])
        
        while len(self.B.shape) > len(Y.shape) :    
            Y = np.expand_dims(np.nan_to_num(Y,nan = 0), axis =1) 
            mask = np.expand_dims(np.nan_to_num(mask,nan = 0), axis =1) 
        err = (Y-B)*mask
        loss = np.nanmean((err)**2,-1)
        if len(loss.shape) ==3:
            loss = loss.mean(-1)
        Yc = loss == loss.min(axis=1,keepdims = True)
        self.Yc, self.Ye, self.err = Yc, loss, (Y-B)
        
    def evaluate(self,pred,metrics):
        ## further consider mask
        Y, mask  = self.Y, self.mask[:,self.lookback:(self.lookback+self.horizon)]
        
        if self.transform:
           Y, mask = self.transform(self.Y,self.mask[:,self.lookback:(self.lookback+self.horizon)])
       
        while len(self.B.shape) > len(Y.shape) :
            Y = np.expand_dims(np.nan_to_num(Y,nan = 0), axis =1) 
            mask = np.expand_dims(np.nan_to_num(mask,nan = 0), axis =1)
            
        scale = np.nanmean((self.X[:,:,1:self.lookback] - self.X[:,:,0:(self.lookback-1)])**2, axis = -1)
        if pred is not None:
            while len(pred.shape) < len(self.B.shape) :
                pred = np.expand_dims(pred,1)           
            
            pred = np.concatenate([ np.nan_to_num(self.B,nan = 0),pred],1)
        else:
            pred = self.B
        
        return metrics(pred, Y, scale, mask)
    
    def add_Base_forecasts(self, model_pred,forecaster):     
        
         
       
        if forecaster in self.forecasters:         
            position = self.forecasters.index(forecaster)
            
            self.B[:,position,::] = model_pred

        else:
            self.forecasters += [forecaster]
            model_pred =  np.expand_dims(model_pred, 1)
            if self.B is None:
                self.B = model_pred
            else:
                self.B = np.concatenate([self.B, model_pred], 1)
            
            
    def XZ_globalReg(self, L, rolling_step, mode = 'Tree', rescale = False):       

        if L + self.horizon  > self.lookback:
            raise ValueError("The window_size + forecasting "
                                 "horizon would exceed the length of the "
                                 "given timeseries!")
        Zy_train = []
        Zy_test = []
        mask = self.mask
        weight = self.weight
        X_test = self.X[:,:,(self.lookback - L): ]
        mask_test = mask[:,self.lookback: ]
        weight_test = weight[:,self.lookback: ]
        
        rollings = (self.lookback - L - self.horizon)//rolling_step
        
        X_train = [self.X[:,:,(self.lookback - L - self.horizon - i*rolling_step):(self.lookback - self.horizon - i*rolling_step)] for i in range(rollings) ]       
        Y_train = [self.X[:,:,(self.lookback  - self.horizon - i*rolling_step):(self.lookback - i*rolling_step)] for i in range(rollings) ] 
        mask_train = [mask[:,(self.lookback  - self.horizon - i*rolling_step):(self.lookback - i*rolling_step)] for i in range(rollings) ]  
        weight_train = [weight[:,(self.lookback  - self.horizon - i*rolling_step):(self.lookback - i*rolling_step)] for i in range(rollings) ]  
        
        idx_tr = [self.df_idx.values  for i in range(rollings) ]
        tdx_tr = [self.Xt_idx[(self.lookback  - self.horizon - i*rolling_step):(self.lookback - i*rolling_step)] for i in range(rollings) ] 
        
        idx_ts = [self.df_idx.values]
        tdx_ts = [self.Yt_tidx]

               
           
        if self.Z is not None:
            Z_test  =  self.Z[:,:,(self.lookback - L):]
            Zx_test = Z_test[:,:,0:L]
            Zy_test = Z_test[:,:,L:L+self.horizon]         
                        
            Z_train = [self.Z[:,:,(self.lookback - L - self.horizon - i*rolling_step):(self.lookback - i*rolling_step)] for i in range(rollings) ]

                
        if mode == 'seq2seq':         
            X_test = [X_test]
            if rescale:
                X_train, scale_y_tr = self.rescale_y(X_train)
                X_test, scale_y_ts = self.rescale_y(X_test)
            else:
                scale_y_tr = [None for i in range(rollings)]
                scale_y_ts = [None]                  
            
            
            
            if len(self.unknown_features)>2:
                unknown_features_tr = [ np.concatenate([X_train[i],Z_train[i][:,self.unknown_features_idx[1:],:L]],1) for i in range(rollings)]
                unknown_features_ts = np.concatenate([X_test[0], Z_test[:,self.unknown_features_idx[1:],:L]],1)                
            elif len(self.unknown_features) == 2 :                            
                unknown_features_tr = [ np.concatenate([X_train[i],Z_train[i][:,:,:L]],1) for i in range(rollings)]
                unknown_features_ts = np.concatenate([X_test[0], Z_test[:,:,:L]],1)
            else:
                unknown_features_tr = X_train 
                unknown_features_ts = X_test[0]
                
            
            target_features = self.dynamic_known_features_num_idx + self.dynamic_known_features_cat_idx
            if len(target_features)  >1:
                 
                target_features_tr = [Z_train[i][:,target_features,:] for i in range(rollings)]
                target_features_ts = Z_test[:,target_features,:]
            elif len(target_features) == 1:
                 
                target_features_tr = [Z_train[i][:,target_features,:] for i in range(rollings)]
                target_features_ts = Z_test[:,target_features,:]
            else:
                target_features_tr = [None for i in range(rollings)]
                target_features_ts = None           
            
            
            static_features = self.static_known_features_idx 
            if len(static_features)  >1:
                 
                static_features_tr = [Z_train[i][:,static_features,0] for i in range(rollings)]
                static_features_ts = Z_test[:,static_features,0]
                
            elif len(static_features) == 1:
                 
                static_features_tr = [Z_train[i][:,:,0] for i in range(rollings)]
                static_features_ts = Z_test[:,:,0]
            else:
                static_features_tr = [None for i in range(rollings)]
                static_features_ts = None             
            
            data_tr = HGT(
                  edge_index = [self.edge_index for i in range(rollings)],
                  edge_weight = [self.edge_weight for i in range(rollings)],
                  
                  unknown_features = unknown_features_tr,
                  static_features = static_features_tr,
                  targets = Y_train,
                  targets_mask = [np.expand_dims(mask_train[i],1) for i in range(rollings)],
                  target_features = target_features_tr,
                  scales = scale_y_tr,
                  tdx = tdx_tr,
                  idx = idx_tr,
                  )
            
            data_ts = HGT(
                  edge_index = [self.edge_index],
                  edge_weight = [self.edge_weight],
                  
                  unknown_features = [unknown_features_ts],
                  static_features = [static_features_ts],
                  targets = [np.expand_dims(self.Y,1)],
                  targets_mask = [np.expand_dims(mask_test,1)],
                  target_features = [target_features_ts],
                  scales = scale_y_ts,
                  tdx = tdx_ts,
                  idx = idx_ts
                  )
            return data_tr, data_ts
            
            '''    
            return ({'X_tr':X_train,
                    'Y_tr': Y_train,
                    'Z_tr': Z_train,
                    'mask_tr': mask_train, 
                    'weight_tr': weight_train, 
                    'idx_tr':idx_tr,
                    'tidx_tr':tidx_tr,
                    'scale_y_tr': scale_y_tr,
                    },
                    {'X_ts': X_test,
                    'Z_ts': Z_test, 
                    'mask_ts': mask_test,
                    'weight_ts': weight_test, 
                    'idx_ts':idx_ts,
                    'tidx_ts':tdx_ts,
                    'scale_y_ts': scale_y_ts,
                    },
                    {'features_num_idx': self.features_num_idx,
                    'features_cat_idx': self.features_cat_idx,
                    'static_known_features_idx': self.static_known_features_idx,
                    'dynamic_known_features_num_idx': self.dynamic_known_features_num_idx,
                    'dynamic_known_features_cat_idx': self.dynamic_known_features_cat_idx,
                    },
                    {'edge_index' : self.edge_index,
                    'edge_weight': self.edge_weight,                    
                    },
                    )     
            '''      
        elif mode == 'Tree' :
            X_train = np.concatenate(X_train,0) 
            Y_train = np.concatenate(Y_train,0)        
            mask_train = np.concatenate(mask_train,0) 
            weight_train = np.concatenate(weight_train,0) 
            
            if self.Z is not None:
                Z_train = np.concatenate(Z_train,0)
                Zx_train = Z_train[:,:,0:L]
                Zy_train = Z_train[:,:,L:L+self.horizon]
                X_test = np.concatenate([X_test,Zx_test],1)  
                X_train = np.concatenate([X_train,Zx_train],1)
            
            ntr, _ , _ = X_train.shape
            nts, _ , _ = X_test.shape            
            
            Xtr,Xts,Ztr,Zts = [ ],[ ],[ ],[ ]  ## position 0 is the lags of target
            
            for i, f in zip(range(len(self.feature_names)),self.feature_names):
                if f in self.static_known_features:
                    Xtr += [ X_train[:,i,0:1] ]
                    Xts += [ X_test[:,i,0:1] ]
                else:
                    if f in self.dynamic_known_features_num + self.dynamic_known_features_cat:  
                        #Xtr += [ X_train[:,i,:],Zy_train[:,i-1,:] ]
                       # Xts += [ X_test[:,i,:],Zy_test[:,i-1,:] ]
                        Xtr += [ X_train[:,i,:]]
                        Xts += [ X_test[:,i,:]]
                        Ztr += [Zy_train[:,i-1,:]]
                        Zts +=[Zy_test[:,i-1,:]]
                    else:
                        Xtr += [ X_train[:,i,:]]
                        Xts += [ X_test[:,i,:] ]
                                            

            
            return {
                'X_tr': np.concatenate( Xtr,-1),
                'Z_tr': np.concatenate( Ztr,-1) if len(Ztr)>0 else Ztr,
                'Y_tr': Y_train.reshape(ntr,-1),
                'mask_tr': mask_train.reshape(ntr,-1),
                'weight_train': weight_train.reshape(ntr,-1),
                'X_ts': np.concatenate( Xts,-1),
                'Z_ts': np.concatenate( Zts,-1)  if len(Zts)>0 else Zts,
                'mask_ts': mask_test,
                'weight_ts': weight_test,
            }
                
    def reduce_mem_usage(self):
        self.X = self.X.astype(np.float32)
        self.B = self.B.astype(np.float32) if self.B is not None else self.B 
        self.Y = self.Y.astype(np.float32) if self.Y is not None else self.Y 
        self.Z = self.Z.astype(np.float32) if self.Z is not None else self.Z    
        self.MetaFeat = self.MetaFeat.astype(np.float32)  if self.MetaFeat is not None else self.MetaFeat
        self.mask = self.mask.astype(np.int8) if self.mask is not None else self.mask
        self.weight = self.weight.astype(np.float16) if self.weight is not None else self.weight
            
    @staticmethod
    def rescale_y(y):
        ## only for postive forecasting
        eps = 1e-8
        scales = []
        y_scale = y.copy()
        for i in range(len(y)):
            
            scale_sum = np.nansum(y[i]*(y[i]>0),axis= -1,keepdims = True) ## avoid sales of -1
            scale_count = np.nansum((y[i]>0),axis= -1, keepdims = True)    
            scale = scale_sum /(scale_count + eps) 
            y_scale[i] = y_scale[i]/(scale + eps)

            scales.append(scale)

        return y_scale, scales          
            
        
class RawData():
    def __init__(self, 
                 npdata: pd.DataFrame,
                 df_idx: pd.MultiIndex, 
                 time_idx: pd.Index,
                 features_num: List[str],
                 features_cat: List[str],
                 static_known_features : List[str]= None,
                 dynamic_known_features_num : List[str]= None,
                 dynamic_known_features_cat : List[str]= None,
                 mask: np.ndarray = None,
                 weight: np.ndarray = None,
                 edge_index:List[Union[np.ndarray, None]] = None,
                 edge_weight: List[Union[np.ndarray, None]] = None,
                 ):

        self.df_idx = df_idx
        self.time_idx = time_idx
        self.features_num = features_num  ### all the num features
        self.features_cat = features_cat  ### all the cat features
        self.npdata = npdata
        self.N = len(df_idx)
        self.C = len(features_num) +  len(features_cat)
        self.T = len(time_idx)
        self.mask = mask
        self.weight = weight
        self.static_known_features = static_known_features  ## static features are all known category features, should be a subset of features_cat and known_features_cat
        self.dynamic_known_features_num =  dynamic_known_features_num  ## a subset of features_num       
        self.dynamic_known_features_cat =  dynamic_known_features_cat  ## a subset of features_cat, including static features  
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        
    def data_segment(
            self, 
            W : int, 
            H :int, 
            nseg :int = 10, 
            step: int = None):
        """
        :param data: shape(N,C,T) time data:shape(N,T)
        :return: s个shot样本 shape为(s,N,C,W+H) s个时间样本 shape 为(s,N,W+H)
        """
        if W + H > self.T:
            raise ValueError("The window_size + forecasting "
                                 "horizon would exceed the length of the "
                                 "given timeseries!")
            
        window_end = self.T
        
        if step is None:
            step = H
        data_slots = []
        
        n = 0
        while n < nseg:
            if (window_end - W - H) < 0 :
                break
            data_slots.append(
                RawData(
                npdata = self.npdata[:,:,(window_end - W - H):window_end],        
                df_idx = self.df_idx,
                time_idx = self.time_idx[(window_end - W - H):window_end],
                features_num = self.features_num,
                features_cat = self.features_cat,                
                static_known_features = self.static_known_features,
                dynamic_known_features_num = self.dynamic_known_features_num,
                dynamic_known_features_cat = self.dynamic_known_features_cat,
                mask = self.mask[:,(window_end - W - H):window_end] if self.mask is not None else None,
                weight = self.weight[:,(window_end - W - H):window_end] if self.weight is not None else None,
                edge_index = self.edge_index,
                edge_weight = self.edge_weight,
                )
                )
            window_end -= step
            n += 1
            
        return data_slots
    
    
    def Raw2Meta(self,H):
        """
        H : max forecasting horizon

        """    
        return MetaDataset(        
                X = np.expand_dims(self.npdata[:,0,0:(self.T-H)],1), 
                Y = self.npdata[:,0,(self.T-H):], 
                Z = self.npdata[:,1:,:], 
                Xt_idx = self.time_idx[0:(self.T-H)],
                Yt_tidx = self.time_idx[(self.T-H):], 
                df_idx = self.df_idx,
                features_num = self.features_num,
                features_cat = self.features_cat,
                static_known_features = self.static_known_features,
                dynamic_known_features_num = self.dynamic_known_features_num,
                dynamic_known_features_cat = self.dynamic_known_features_cat,
                mask = self.mask,    
                weight = self.weight,  
                edge_index = self.edge_index,
                edge_weight = self.edge_weight,
                
                )
        
        

       
class Md_utils:

    def Df_to_rawdata(
            df : pd.DataFrame,
            idx : List[str], 
            tdx : str,
            target: List[str], 
            features_num: List[str], 
            features_cat: List[str],
            static_known_features : List[str] = None ,
            dynamic_known_features_num :List[str] =None, 
            dynamic_known_features_cat :List[str] =None,             
            mask_idx : str = None,
            weight : pd.DataFrame = None,
            edge_index:List[Union[np.ndarray, None]] = None,
            edge_weight: List[Union[np.ndarray, None]] = None,
            ):
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.groupby(idx, sort=False).apply(lambda x: x.ffill().bfill())    
        
        x = df.pivot(index = idx,columns=tdx)
        df_idx = x.index
        
        features = target + features_num + features_cat
        time_idx =  x[features[0]].columns
        
        if mask_idx is not None:
            mask = x[mask_idx].applymap(lambda x: 0 if np.isnan(x) else 1 ).values
        else:
            mask = x[features[0]].applymap(lambda x: 0 if np.isnan(x) else 1 ).values
        x =  [np.expand_dims( x[features[i]].values, 1) for i in range(len(features))]
        
        if weight is not None:
            weight = weight.pivot(index = idx,columns=tdx).values
        else:
            weight = mask
        
        return RawData(
                npdata = np.concatenate(x,1), 
                df_idx = df_idx,
                time_idx = time_idx ,
                features_num = features_num, 
                features_cat = features_cat, 
                static_known_features = static_known_features, 
                dynamic_known_features_num = dynamic_known_features_num,
                dynamic_known_features_cat = dynamic_known_features_cat,
                mask = mask, 
                weight = weight,
                edge_index = edge_index,
                edge_weight = edge_weight,
                )
            
            
            
    def md_concat(md_list,fidx = None, transform = None):
        N = len(md_list)
        fidx = [] if fidx is None else fidx    
        return MetaDataset(        
            X = np.concatenate([md_list[i].X for i in range(N)],0), 
            B = np.concatenate([md_list[i].B[:,fidx,:] for i in range(N)],0) if md_list[0].B is not None else None , 
            Y = np.concatenate([md_list[i].Y for i in range(N)],0) if md_list[0].Y is not None else None, 
            Z = np.concatenate([md_list[i].Z for i in range(N)],0) if md_list[0].Z is not None else None, 
            MetaFeat = np.concatenate([md_list[i].MetaFeat for i in range(N)],0) if md_list[0].MetaFeat is not None else None, 
            transform = transform,
            Xt_idx = None,
            Yt_tidx = None,
            df_idx = None,
            features_num = md_list[0].features_num,
            features_cat = md_list[0].features_cat,
            forecasters = [md_list[0].forecasters[f] for f in fidx ],
            static_known_features = md_list[0].static_known_features,
            dynamic_known_features_num = md_list[0].dynamic_known_features_num,
            dynamic_known_features_cat = md_list[0].dynamic_known_features_cat,
            mask = np.concatenate([md_list[i].mask for i in range(N)],0) if md_list[0].mask is not None else None,  
            weight = np.concatenate([md_list[i].weight for i in range(N)],0) if md_list[0].weight is not None else None, 
            )
        
        
   
    def metadataclass_update(df_slots):
            
            for i in range(len(df_slots)):
                   
                    s = MetaDataset(        
                            X = df_slots[i].X,
                            B = df_slots[i].B,
                            Y = df_slots[i].Y,
                            Z = df_slots[i].Z,
                            MetaFeat = df_slots[i].MetaFeat,
                            transform = df_slots[i].transform,
                            Xt_idx = df_slots[i].Xt_idx,
                            Yt_tidx = df_slots[i].Yt_tidx,
                            df_idx = df_slots[i].df_idx,
                            features_num = df_slots[i].features_num,
                            features_cat = df_slots[i].features_cat,
                            forecasters = df_slots[i].forecasters,
                            static_known_features = df_slots[i].static_known_features,
                            dynamic_known_features_num = df_slots[i].dynamic_known_features_num,
                            dynamic_known_features_cat = df_slots[i].dynamic_known_features_cat,                            
                            mask = df_slots[i].mask ,
                            weight = df_slots[i].weight,
                            edge_index = df_slots[i].edge_index,
                            edge_weight = df_slots[i].edge_weight,
                                            
                            )
                    df_slots[i] = s
            return df_slots

