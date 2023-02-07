"""
Detials
"""
# imports
import torch
from statistics import mean

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
    pred_collector = []
    gt_collector = []
    # pred and ground truth indecies
    if len(pred.data.shape) == 2: 
        _, pred_collector = torch.max(pred, 1)
        _, gt_collector = torch.max(gt, 1)
    else:
        for i, data in enumerate(pred.data):
            for j, data2 in enumerate(data.data):
                _, pred_max = torch.max(data2, -1)
                pred_collector.append(pred_max)
                _, gt_max = torch.max(gt[i][j].data, -1)
                gt_collector.append(gt_max)
    
        pred_collector = torch.stack(pred_collector)
        gt_collector = torch.stack(gt_collector)
        #print(pred_collector, gt_collector)

    # returning data
    return pred_collector, gt_collector

def accuracy_val(pred, gt):
    """
    Detial
    """
    true = (pred == gt).sum().item()

    if true == 0:
        acc = (0)
    else:
        acc = (true/gt.shape[0])
    
    return acc