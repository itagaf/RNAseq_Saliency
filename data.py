import argparse
import random
import pandas as pd
import numpy as np
import os
import pathlib

from datetime import datetime
from scipy.stats import gmean, zscore
import scipy
import re

import requests
import io

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter 

label_set_url = "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"

class Data(Dataset):
    def __init__(self, X_train, y_train, dev):
        self.x = torch.from_numpy(X_train)
        self.y = torch.from_numpy(y_train)
        self.len = self.x.shape[0]
        self.dev = dev
    def __getitem__(self, index):
        return self.x[index].float().to(self.dev), self.y[index].type(torch.LongTensor).to(self.dev)
    def __len__(self):
        return self.len

###
 
def get_url_label_set():
    '''
    This function retrieve all label names (tissue types) of samples, which will be use to retrieve the data 
    '''
    response = requests.get(label_set_url)
    if response.status_code != 200:
        print(f"bad url: {label_set_url} ({response})")
    try:
        content = response.content
        df_label = pd.read_csv(io.BytesIO(content), sep="\t", index_col=0)
        label_set = df_label['SMTSD'].unique()
        return label_set
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
        print(f"Unable to construct DataFrame from URL content. check URL:\n{label_set_url}")
    


def get_local_label_set(path):
    if os.path.isfile(path):
        try:
            with open(path) as f:
                label_set = set(f.read().splitlines())
        except OSError:
            print(f'cannot open{path}')
    else:
        print(f'cannot find {path}\nimport label set from URL')
        label_set = get_url_label_set()        
    
    return label_set


def get_input_filter(path):
    if os.path.isfile(path):
        try:
            with open(path) as f:
                input_filter = set(f.read().splitlines())
            return input_filter

        except OSError:
            print(f'cannot open{path}')
            exit()  
    else:
        print(f'cannot find {path}')
        exit()

###

def augment_data(mat, n_samples):
    N, M = mat.shape
    diff = abs(N - n_samples) # diff - number of samples needed to achieve minimal sample in label
    if N > diff:
        rnd_idx = np.random.permutation(N)[:diff] # sampling #diff random smaples
        rnd_samples = mat[rnd_idx,:]
        rnd_mat = np.random.normal(1,0.5,size=(diff,M)) # generating #diff random factors of mean 1 and std of 0.5 (samples are normalized)
        aug_mat = np.abs(rnd_mat) * rnd_samples # generating augmented samples by multiplying random facotrs and random samples
        return aug_mat
    else:
        n_batchs = int(diff/N) # if the additional number of samples is larger then the exsiting sample size: perform process in batches
        rem = diff - (n_batchs * N)
        aug_mat = None
        for i in range(n_batchs):
            rnd_mat = np.random.normal(1,0.5,size=(N,M))
            bat_aug_mat = np.abs(rnd_mat) * mat
            if aug_mat is None:
                aug_mat = bat_aug_mat
            else:
                aug_mat = np.append(aug_mat, bat_aug_mat, axis=0)

        rnd_idx = np.random.permutation(N)[:rem]
        rnd_samples = mat[rnd_idx,:]
        rnd_mat = np.random.normal(1,0.5, size=(rem,M))
        bat_aug_mat = np.abs(rnd_mat) * rnd_samples
        aug_mat = np.append(aug_mat, bat_aug_mat, axis=0)

        return aug_mat

def process_input_mat(df, input_filter, n_samples_min, label, verbose, is_aug=""):
    
    if input_filter: # use a subset of genes for the inupt
        df = df[df['Name'].isin(input_filter)]
    
    label_mat = df.iloc[:,2:].to_numpy().T # transofrom GTEx's data into model input 
    
    if label_mat.shape[0] < n_samples_min:
        is_aug = "*aug"
        aug_mat = augment_data(label_mat, n_samples_min)
        label_mat = np.append(label_mat, aug_mat, axis=0)
        
    label_mat = zscore(label_mat, axis=1) # normalize the matrix, sample-wise
    
    if verbose:
        print(f'{label_mat.shape[0]:>5}  {is_aug:>5}     {label}')

    return label_mat



def format_labels(label_set):
    mod_set = []
    for label in label_set:
        label_str = re.sub("[()]","",label)
        label_str = label_str.replace(" - ","_")
        label_str = label_str.replace(" ","_").lower()
        mod_set.append(label_str)

    return mod_set



def get_data_from_url(label_set,
                      n_samples_min, 
                      verbose,
                      input_filter,
                      core_url = "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/counts-by-tissue/gene_reads_2017-06-05_v8_",
                      rand_test_sample=False):
    '''
    This function retrieve the data from the GTEx database.
    label_set : list of labels as appear on label's data url
    n_samples_min : number of minimum samples needed in each label. will evoke data augmentation method
    input_filter : a list of genes (ENSEMBEL gene ids) to be used in input.
    rand_test_sample : should be True only for Saliency.ipynb  
    returns a unified expression matrix for all available samples of bulk RNA-seq.

    '''


    label_set = format_labels(label_set)
    x_data = None
    y_data = np.array([])
    cat_dict = dict()       # {label index : label str}
    if verbose:
        print(f"min samples in category requested: {n_samples_min}\n")
        print(f"   N  is augmented  Label")
    
    for i, label in enumerate(label_set): # iterating over labels
        
        label_url = core_url + label + ".gct.gz" # creating slug
        response = requests.get(label_url)
        if response.status_code != 200:
            print(f"bad url ({response}) at label: {label}")
            continue
        try:
            content = response.content
            df = pd.read_csv(io.BytesIO(content),
                             sep="\t",
                             compression="gzip",
                             skiprows=2,
                             index_col=0)
            
            label_mat = process_input_mat(df=df,
                                          input_filter=input_filter,
                                          n_samples_min=n_samples_min,
                                          label=label,
                                          verbose=verbose)
            if rand_test_sample:                                                # for Saliency.ipynb
                rand_idx = np.random.randint(label_mat.shape[0], size=20)
                label_mat = label_mat[rand_idx,:]

            if x_data is None:
                x_data = label_mat
            else:
                x_data = np.append(x_data, label_mat, axis=0)

            y_data = np.append(y_data,[i] * label_mat.shape[0])
            cat_dict[i] = label
           
        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
            print("Error: ", label)
    
    print("Done loading data!")
    print(f"No. of categories: {len(set(y_data))}")
    return x_data, y_data, cat_dict



def get_local_data(in_path, n_samples_min, verbose=False, input_filter=None):
    '''
    This function retrieve data from local path.
    Should NOT be preferred
    '''

    x_data = None
    y_data = np.array([])
    cat_dict = dict()
    
    for i, label in enumerate(os.listdir(in_path)):
        label_path = os.path.join(in_path,label)
        label_str = label.replace(".csv.gz","")
        cat_dict[i] = label_str
        
        is_aug = ""
        try:
            df = pd.read_csv(label_path,
                             sep="\t",
                             compression="gzip",
                             skiprows=2,
                             index_col=0)

            label_mat, label_count = process_input_mat(df=df,
                                                       input_filter=input_filter,
                                                       n_samples_min=n_samples_min,
                                                       label=label,
                                                       verbose=verbose)
        
            if x_data is None:
                x_data = label_mat
            else:
                x_data = np.append(x_data, label_mat, axis=0)
                    
            y_data = np.append(y_data,[i] * label_mat.shape[0])

        except:
                print("Error: ", label_str)
    
    print("All done!")
    print(f"No. of categories: {len(set(y_data))}")
    return x_data, y_data, cat_dict



