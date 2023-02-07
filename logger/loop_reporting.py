"""
Detials
"""
# imports
import torch

# functions
def accuracy_reporting(y1_pred, y2_pred, y1_gt, y2_gt):
    """
    Detials
    """
    # collecting indecies from data
    y1_pred_in, y1_gt_in = data_extractor(y1_pred, y1_gt)
    y2_pred_in, y2_gt_in = data_extractor(y2_pred, y2_gt)

    y1_acc = accuracy_val(y1_pred_in, y1_gt_in)
    y2_acc = accuracy_val(y2_pred_in, y2_gt_in)

    return y1_acc, y2_acc

def data_extractor(pred, gt):
    """
    Detials
    """
    # pred and ground truth indecies
    _, pred_out = torch.max(pred.data, 1)
    _, gt_out = torch.max(gt.data, 1)

    # returning data
    return pred_out, gt_out

def accuracy_val(pred, gt):
    """
    Detial
    """
    true = (pred == gt).sum().item()

    if true == 0:
        acc = 0
    else:
        acc = true/gt.shape[0]
    
    return acc