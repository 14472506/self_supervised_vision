"""
Detials
"""
# imports 
from logger import training_print_out
from matplotlib import pyplot as plt

import torch

# functions 
def classification_training_loop(epoch, count, model, train_loader, device, optimiser, criterion, print_freque):
    """
    Detials
    """
    # model config
    model.train()

    # logging
    acc_loss = 0
    running_loss = 0

    # accume init here
    for i, data in enumerate(train_loader, 0):
        # extact data 
        x, y_gt = data
        x, y_gt = x.to(device), y_gt.to(device)

        # forward + backward + optimizer_step
        y_pred = model(x)
        loss = criterion(y_pred, y_gt)
        loss.backward()
        optimiser.step()

        running_loss, acc_loss = training_print_out(running_loss, acc_loss, loss, print_freque,
                                    device, i, epoch, count)
        
        # reset gradient
        optimiser.zero_grad()

        count += 1

    # calculating and returning trainig loss over epoch
    epoch_loss = acc_loss/len(train_loader)
    return epoch_loss, count

def classification_validation_loop(model, validation_loader, device, criterion):
    """
    Detials
    """
    # model config
    model.eval()

    # logging here
    loss_acc = 0

    for i, data in enumerate(validation_loader, 0):
        # extract data
        x, y_gt = data
        x, y_gt = x.to(device), y_gt.to(device)

        # forwards 
        with torch.no_grad():
            y_pred = model(x)
        
        # get loss
        loss = criterion(y_pred, y_gt)

        # accumulate loss
        loss_acc += loss.item()
    
    # get val loss for epoch
    val_loss = loss_acc/len(validation_loader)
    
    # return val loss
    return val_loss