import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import torch
import numpy as np
import os
import pandas as pd

def evaluate_class_level(test_opt, y_true, y_pred_labels, class_names, proj_dir=None):
    report = classification_report(y_true, y_pred_labels, target_names=class_names, digits=4, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report['support'] = report['support'].astype('int32')
    report.index.name = "Label"
    
    if test_opt == 1:
        if proj_dir != None:
            out_path = os.path.join(proj_dir, "class_level_report.csv")
            report.to_csv(out_path)
    elif test_opt == 2:
        print(report)

def eval_cm(cm, ratio=0.5):
    cm_max = (cm - np.diag(cm.diagonal())).max(axis=0) # get highest number of hits, excluding true label   
    cm_sum = (cm - np.diag(cm.diagonal())).sum(axis=0) # get sum of all mismatches
    fail_idx = np.where(cm_max > cm.diagonal())[0]
    bad_idx = np.where((cm_sum / cm.diagonal()) > ratio)[0]
    return fail_idx, bad_idx

def plot_confusion_matrix(y_true, y_pred_labels, class_names, proj_dir=None):
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(12, 10))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=False,
                fmt='c', 
                cmap=sns.color_palette("light:b", as_cmap=True),
                xticklabels=class_names,
                linewidth=.5,
                yticklabels=class_names,
               ax=ax)

    fail_idx, bad_idx = eval_cm(cm) # annotate heat map with underperformed labels
    for i in bad_idx:
        ax.add_patch(patches.Rectangle((i, i),1.0,1.0,edgecolor='purple',fill=False,lw=1)) # labels were sum of mismatches exceeded true label
    for i in fail_idx:
        ax.add_patch(patches.Rectangle((i, i),1.0,1.0,edgecolor='red',fill=False,lw=1)) # labels where the true label did not recieve the most hits 
    for j, ticklbl in enumerate(ax.xaxis.get_ticklabels()):
        if j in bad_idx:
            ticklbl.set_color('purple')
        if j in fail_idx:
            ticklbl.set_color('red')
    
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    if proj_dir != None:
        out_path = os.path.join(proj_dir, "confusion_matrix.png")
        fig.savefig(out_path, format="png", bbox_inches = 'tight')
    plt.close(fig)

    
def evaluate_model(test_opt, model, test_loader, criterion, labels, proj_dir=None):
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    y_true = []  # To store true labels
    y_pred_labels = []  # To store predicted class labels
    
    with torch.no_grad():  # Disable gradient calculations for evaluation
        for i, (x, y) in enumerate(test_loader):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            test_loss += loss.item()
            
            # Get predicted class (numerical label)
            y_pred_labels.extend(torch.argmax(y_pred, dim=1).cpu().numpy())
            
            # Collect true labels
            y_true.extend(y.cpu().numpy())
    
    
    evaluate_class_level(test_opt, y_true, y_pred_labels, labels, proj_dir)
    if test_opt == 1:
        plot_confusion_matrix(y_true, y_pred_labels, labels, proj_dir)