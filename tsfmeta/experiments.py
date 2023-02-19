# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 23:23:42 2022

@author: MSH
"""
import torch
class meta_learning_run():
    
    def __init__(
        self, 
        model, 
        loss_fun,
        optimizer,
        train_dataloader, 
        val_dataloader,
        device,
        epochs,
        scale_loss=False):
        
        self.model = model 
        self.loss_fun = loss_fun 
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader 
        self.val_dataloader = val_dataloader
        self.device = device
        self.scale_loss = scale_loss
        
        
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_loop()  
            if self.val_dataloader is not None:
                self.val_loop() 
        print("Done!")
        



    def train_loop(self):
        
        #size = len(self.train_dataloader.dataset)
        total_loss = 0
        for batch, meta in enumerate(self.train_dataloader):
             
            # Compute prediction and loss
            pred = self.model(meta)
            if self.model.type == 'comb':
                y = meta['Y'].to(self.device) 
                mask = meta['mask'].to(self.device)
                if self.scale_loss:
                    lookback = meta['X'].shape[2]
                    scale = torch.nanmean((meta['X'][:,0,1:lookback] - meta['X'][:,0,0:(lookback-1)])**2, axis = -1)
                    scale = torch.where(scale == 0,scale+1, scale)   
                    scale = torch.sqrt(scale).to(self.device)  
                    loss = self.loss_fun(pred * mask/scale.unsqueeze(1), y*mask/scale.unsqueeze(1))                  
                else:
                    loss = self.loss_fun(pred * mask, y*mask)
                
            if self.model.type == 'select':
                y = meta['Yc'].to(self.device) 
                mask = (torch.min(meta['mask'],-1,keepdim=True)[0]>0).to(self.device)
                loss = self.loss_fun(pred * mask, y*mask)
            if self.model.type == 'loss':
                y = meta['Ye'].to(self.device) 
                mask = (torch.min(meta['mask'],-1,keepdim=True)[0]>0).to(self.device)
                loss = self.loss_fun(pred * mask, y*mask)
                    
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        total_loss /= batch + 1
        print(total_loss)
        
    
    def val_loop(self):
        #size = len(self.test_dataloader.dataset)
        total_loss = 0
        for batch, meta in enumerate(self.val_dataloader):
            
            pred = self.model(meta)   
            
            if self.model.type == 'comb':
                y = meta['Y'].to(self.device) 
                mask = meta['mask'].to(self.device)
                
                if self.scale_loss:
                    lookback = meta['X'].shape[2]
                    scale = torch.nanmean((meta['X'][:,0,1:lookback] - meta['X'][:,0,0:(lookback-1)])**2, axis = -1)
                    scale = torch.where(scale == 0,scale+1, scale) 
                    scale = torch.sqrt(scale).to(self.device)   
                    loss = self.loss_fun(pred * mask/scale.unsqueeze(1), y*mask/scale.unsqueeze(1))                  
                else:
                    loss = self.loss_fun(pred * mask, y*mask)
                
            if self.model.type == 'select':
                y = meta['Yc'].to(self.device) 
                mask = (torch.min(meta['mask'],-1,keepdim=True)[0]>0).to(self.device)
                loss = self.loss_fun(pred * mask, y*mask)
                
            if self.model.type == 'loss':
                y = meta['Ye'].to(self.device) 
                mask = (torch.min(meta['mask'],-1,keepdim=True)[0]>0).to(self.device)
                loss = self.loss_fun(pred * mask, y*mask)              
                      
               
            total_loss += loss.item()
        total_loss /= batch + 1
        print(total_loss)


        
    def forecast(self,test_dataloader):
        #size = len(self.test_dataloader.dataset)
        total_loss = 0
        total_pred = []
        for batch, meta in enumerate(test_dataloader):
            
               
            y = meta['Y'].to(self.device)         
            mask = meta['mask'].to(self.device) 
                
            pred = self.model(meta)          
       
            
            if self.model.type == 'select':
                b = meta['B'].to(self.device)
                best_idx = torch.argmax(pred, 1, keepdim=True)
                one_hot = torch.FloatTensor(pred.shape).to(self.device)
                one_hot.zero_()
                one_hot.scatter_(1, best_idx, 1)
                pred = (one_hot.unsqueeze(2)*b).sum(-2)
                
                #pred = (pred.unsqueeze(2)*b).sum(-2)

            if self.model.type == 'loss':
                b = meta['B'].to(self.device)
                weights = torch.nn.functional.softmax(torch.exp(-pred),dim=-1)
                pred = (weights.unsqueeze(2)*b).sum(-2)
                
            total_pred.append(pred.detach() )
            if len(y)>0:
                loss = self.acc(pred*mask, y*mask)         
                total_loss += loss.item()   
                
        total_loss /= batch + 1
        print(total_loss)               
        
        return total_loss, torch.cat(total_pred,0)
    
    def acc(self,pred, y):
        
        return torch.nn.functional.mse_loss(pred, y)
        
        
        
        
        
        
        
        
