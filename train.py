import argparse
import random
import pandas as pd
import numpy as np
import os
import pathlib

from datetime import datetime
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter 

from net import Net, train
from data import Data, get_url_label_set, get_local_label_set, get_input_filter, get_data_from_url, get_local_data
from evaluate import evaluate_class_level, plot_confusion_matrix, evaluate_model

curr_date = datetime.now().strftime("%d_%m_%Y")
        

def main():
    parser = argparse.ArgumentParser(prog="train.py",
        formatter_class=argparse.RawTextHelpFormatter,
        # allow_abbrev=True,
        description="""
GTEx RNASeq NN model train
--------------------------
This program trains a network on the entire available GTEx bulk RNASeq data input
to predict the source tissue of an unknown sample. The main objective of this network 
is to apply saliency map in order to determine the distinguish cahrecteristics of
each tissue which the model was trained on""",
        epilog="\n\n")
    # general args
    m_parser = parser.add_argument_group(title="main args",description="")
    m_parser.add_argument("-i","--in-data",        type=str, dest="data_path",   help= "Local path to input data repository. If no argument passed, will fetch data from GTEx public repo (v8 for this version)")
    m_parser.add_argument("-o","--out-path",        type=str, dest="proj_dir",    help="Path to project dir for outputs (model parmaeters, reports and checkpoints).")
    m_parser.add_argument("-l","--label-set",   type=str, dest="label_set",   help="Path to txt file containing list of unique tissue targets. If not provided, will be provided from the GTEx databases")
    m_parser.add_argument("-f","--input-filter",type=str, dest="input_filter",help="Path to txt file of gene ids to be used as input for the network [CAUTION: NO FILTER RESLUTS WITH INFLATED NETWORK].")
    m_parser.add_argument("-u","--device",      type=str, dest="dev",         help="Type of proccessor to be used for model training (passed to torch)")
    m_parser.add_argument("-m","--min-samples", type=int, dest="min_samples", help="Minimum number of samples to in each tissue class. For each category conataining less samples than specified, data augmentation will be used.")
    m_parser.add_argument("-v","--verbose",     action="store_true",          help="Increase program's verbosity")
    m_parser.add_argument("-t","--test",        type=int, dest="test_opt",    help="Apply model test using test set [1 - output metrices to files, 2 - output to stdout ]",default=0, choices=[0,1,2])

    # specific model related args
    model_parser = parser.add_argument_group(title="model args",description="arguments for model training properties:")
    model_parser.add_argument("-e","--epochs",         dest="epochs",     type=int,  default=100, help="Number of epochs for the model to train.")
    model_parser.add_argument("-b","--batch-size",     dest="batch_size", type=int,  default=16,  help="Size of batch size to be passed for DataLoaders.")
    model_parser.add_argument("-lr","--learning-rate", dest="lr",         type=float,default=0.01,help="Learning rate to passed to torch optimizer.")
    model_parser.add_argument("-d","--delta",          dest="delta",      type=float,default=0.0, help="Set delta (between current and optimal loss) for early stopping.")
    model_parser.add_argument("-p","--patience",       dest="patience",   type=int,  default=5,   help="Number of epochs to pass when eligble to early stop before terminating training.")
    model_parser.add_argument("-w","--weight-decay",   dest="decay",      type=float,default=0,   help="Set weight decay (regularization) on optimizer")
     
    # parser.print_help()

    # ------- PARSE ARGS -----------
    args = parser.parse_args()
    
    if args.verbose:
        print(args)

    # label set
    if args.label_set:
        label_set = get_local_label_set(args.label_set)
    else:
        label_set = get_url_label_set()

    # input filter
    if args.input_filter:
        input_filter = get_input_filter(args.input_filter)
    else:
        print(f'No argument passed for "input filter"\nCUATION: size of input vector will be the complete set of genes present in expression data!!!\nproceed anyway? [y/n]: ')
        while(True):
            x = input().lower()
            if x == "y":
                input_filter = None
                break
            elif x == "n":
                exit() 

    # input filter
    if args.min_samples:
        try:        
            min_samples = int(args.min_samples)
        except ValueError:
            print(f'ERROR: invalid argument for min-samples')
            exit()
    else:
        min_samples = 0

    # data path
    if args.data_path:
        if os.path.isdir(args.data_path):
            x_data, y_data, cat_dict = get_loacl_data(args.data_path, 
                                                      min_to_augment=min_samples, 
                                                      verbose=args.verbose, 
                                                      input_filter=input_filter)
        else:
            print(f"directory does not exists: {args.data_path}")
            exit()
    else:
        print(f"fetching input data from GTEx website (v8)")
        print(f'{args.verbose=}')
        x_data, y_data, cat_dict = get_data_from_url(label_set=label_set, 
                                                     verbose=args.verbose,
                                                     n_samples_min=min_samples,
                                                     input_filter=input_filter)
    # project dir to save outputs
    if args.proj_dir:
        proj_dir = args.proj_dir
    else:
        proj_dir = os.path.join(pathlib.Path.home(),"proj_dir")

    if os.path.isdir(proj_dir):
        proj_dir = proj_dir + "_1"

    print(f"creating project directory at: {proj_dir}")
    


    # splittiing the data into train, cross-validation and test
    X_train, X_cv_test, y_train, y_cv_test = train_test_split(x_data, 
                                                              y_data, 
                                                              test_size=0.4, 
                                                              random_state=42)
    
    if args.test_opt:    
        x_cv, x_test, y_cv, y_test = train_test_split(X_cv_test, 
                                                    y_cv_test, 
                                                    test_size=0.5, 
                                                    random_state=42)
    else:
        x_cv = X_cv_test
        y_cv = y_cv_test

    # deleting temporary variables
    del X_cv_test, y_cv_test

    if args.verbose:
        print(f"{X_train.shape = }")
        print(f"y_train: {len(y_train)}")

        print(f"{x_cv.shape = }")
        print(f"y_cv: {len(y_cv)}")
        if args.test_opt:
            print(f"{x_test.shape = }")
            print(f"y_test: {len(y_test)}")

    
    param_list = [19291,7000, 2916, 540, 54]
    # param_list = [19291,5400, 2916, 540, 54]
    # param_list = [19291,10000, 3300, 1000, 54]
    # param_list = [19291,10000, 5000, 2000, 54]             
    # param_list = [19291,10000, 5000, 2500, 370, 54]
    # param_list = [19291,10000, 3500, 1500, 120, 54]

    model = Net(param_list)

    if args.dev.lower() == "cpu":
        dev = "cpu"
    elif args.dev.lower() == "gpu":
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            print("unable to set device as GPU. will run on CPU")
            dev = "cpu"
    elif torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    model.to(torch.device(dev))
    train_data = Data(X_train, y_train, dev)
    val_data = Data(x_cv, y_cv, dev)

    # unapacking args
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    delta = args.delta
    patience = args.patience
    decay = args.decay

    print(f"{lr=}\n{batch_size=}\n{epochs=}\n{dev=}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size) 
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size)
    


    train(model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            optimizer=optimizer,
            criterion=criterion,
            proj_dir=proj_dir,
            early_stop_delta=delta,
            early_stop_patience=patience,
            verbose=args.verbose)
    
    # apply model testing
    if args.test_opt:
        test_data = Data(x_test, y_test, dev)    
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
        evaluate_model(args.test_opt, model, test_loader, criterion, cat_dict.values(), proj_dir)

    # save current model spec 
    lines = str(model)
    out_file = os.path.join(proj_dir,"model_shape.txt")
    with open(out_file,"w") as out_f:
        out_f.writelines(lines)
        out_f.writelines("\n\nrun_args:\n")
        for arg, value in args.__dict__.items():
            out_f.writelines(f"{arg}={value}\n")

   
if __name__ == "__main__":
    main()
    

