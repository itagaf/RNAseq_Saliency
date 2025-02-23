import random
import numpy as np
import os

from scipy.stats import gmean, zscore
import scipy


import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter 


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    # def save_checkpoint(self, proj_dir, model, epoch, optimizer, better):
    #     fpath = os.path.join(proj_dir, 'best_checkpoint.pth')
    #     torch.save({'epoch': epoch,
    #                 'model_state_dict': model.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict()}, fpath)


class Net(nn.Module):
    def __init__(self, param_list):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(param_list[0], param_list[1])
        self.linear2 = nn.Linear(param_list[1], param_list[2])
        self.linear3 = nn.Linear(param_list[2], param_list[3])
        self.linear4 = nn.Linear(param_list[3], param_list[4])
        # self.linear5 = nn.Linear(param_list[4], param_list[5])
        self.dropout = nn.Dropout(0.50)
        # self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        # x = self.relu(self.linear1(x))
        x = self.linear1(x)
        x = self.dropout(x)
        # x = self.relu(self.linear2(x))
        x = self.linear2(x)
        x = self.dropout(x)
        # x = self.relu(self.linear3(x))
        x = self.linear3(x)
        x = self.dropout(x)
        # pred = self.relu(self.linear4(x))
        pred = self.linear4(x)
        # pred = self.relu(self.linear5(x))
        # pred = torch.sigmoid(self.linear4(x))
        return pred

def save_checkpoint(proj_dir, model, epoch, optimizer):#, better):
    fpath = os.path.join(proj_dir, 'best_checkpoint.pth')
    
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, fpath)

def train(model, train_loader, val_loader, epochs, optimizer, criterion, proj_dir, early_stop_delta, early_stop_patience, verbose):
    
    print("training model...")
    early_stopping = EarlyStopping(delta=early_stop_delta, patience=early_stop_patience, verbose=verbose)

    train_writer = SummaryWriter(os.path.join(proj_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(proj_dir, 'val'))
    
    for epoch in range(epochs):
        train_loss = 0
        for x, y in train_loader:
            y_pred = model(x)                 
            optimizer.zero_grad()
            loss = criterion(y_pred, y)       
            loss.backward()                   
            optimizer.step()                  
            train_loss += loss.item()


        train_loss /= len(train_loader)
        train_writer.add_scalar(tag='train_loss', scalar_value=train_loss, global_step=epoch)
        
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for x, y in val_loader:
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item() 
                val_acc += ((torch.argmax(y_pred,1) == y).sum() / len(y))
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_writer.add_scalar(tag='val_loss', scalar_value=val_loss, global_step=epoch)
        val_writer.add_scalar(tag='val_acc', scalar_value=val_acc, global_step=epoch)

        print(f'epoch: [{epoch + 1}/{epochs:>2}], loss: {val_loss:.4f}, acc: {val_acc:.4f}')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch: {epoch}!")
            break
        elif (early_stopping.counter == 0):
            save_checkpoint(proj_dir, model, epoch, optimizer)

        
    train_writer.close()
    val_writer.close()
    
    print("done!")
