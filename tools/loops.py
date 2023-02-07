"""
Detials
"""
# imports 
from logger import training_print_out, accuracy_reporting
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
    data = next(iter(train_loader))
    # accume init here
    #for i, data in enumerate(train_loader, 0):
    # extact data 
    x, y_gt = data
    x, y_gt = x.to(device), y_gt.to(device)

    # forward + backward + optimizer_step
    y_pred = model(x)
    print(y_gt)
    print(y_pred)
    loss = criterion(y_pred, y_gt)
    loss.backward()
    optimiser.step()
    print(loss)

    #running_loss, acc_loss = training_print_out(running_loss, acc_loss, loss, print_freque,
    #                            device, i, epoch, count)
        
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
    data = next(iter(train_loader))
    #for i, data in enumerate(validation_loader, 0):
    # extract data
    x, y_gt = data
    x, y_gt = x.to(device), y_gt.to(device)

    # forwards 
    with torch.no_grad():
        y_pred = model(x)
        
    # get loss
    loss = criterion(y_pred, y_gt)
    print(loss)
    # accumulate loss
    loss_acc += loss.item()
    
    # get val loss for epoch
    val_loss = loss_acc/len(validation_loader)
    
    # return val loss
    return val_loss

# functions 
def multiclass_training_loop(epoch, count, model, train_loader, device, optimiser, criterion, print_freque):
    """
    Detials
    """
    # model config
    model.train()

    # logging
    acc_tot_loss = 0
    acc_l1_loss = 0
    acc_l2_loss = 0 
    running_loss = 0

    y1_batch_acc_acume = 0
    y2_batch_acc_acume = 0
    
    # For testing: Uncomment and remove/comment out loop and anything else in pipeline 
    # to make this work
    data = next(iter(train_loader))

    #for i, data in enumerate(train_loader, 0):
    #    # extact data 
    x, y1_gt, y2_gt = data
    x, y1_gt, y2_gt = x.to(device), y1_gt.to(device), y2_gt.to(device)

    # forward + backward + optimizer_step
    y1_pred, y2_pred = model(x)
    
    # calculating loss
    loss1 = criterion(y1_pred, y1_gt)
    loss2 = 0
    for i in range(9):
        tile_p = y2_pred[:, i]
        tile_gt = y2_gt[:, i]
        loss2 += 0.111*criterion(tile_p, tile_gt)
        print(tile_p, tile_gt)
    loss2 = criterion(y2_pred, y2_gt)
    loss =  loss1 + loss2

    print(loss1.item(), loss2.item())

    loss.backward()
    optimiser.step()

    #running_loss, acc_tot_loss = training_print_out(running_loss, acc_tot_loss, loss, print_freque,
    #                            device, i, epoch, count)
    acc_l1_loss += loss1.item()
    acc_l2_loss += loss2.item()
            
    # reset gradient
    optimiser.zero_grad()
    count += 1

    y1_batch_acc, y2_batch_acc = accuracy_reporting(y1_pred, y2_pred, y1_gt, y2_gt)
    y1_batch_acc_acume += y1_batch_acc
    y2_batch_acc_acume += y2_batch_acc

    #print(loss1.item(), loss2.item())
    #print(y1_batch_acc_acume, y2_batch_acc_acume)


    # calculating and returning trainig loss over epoch
    #epoch_tot_loss = acc_tot_loss/len(train_loader)
    #epoch_l1_loss = acc_l1_loss/len(train_loader)
    #epoch_l2_loss = acc_l2_loss/len(train_loader)
#
    #y1_tot_acc = y1_batch_acc_acume/len(train_loader)
    #y2_tot_acc = y2_batch_acc_acume/len(train_loader)
#
    #data = {"total_loss": epoch_tot_loss,
    #        "y1_loss": epoch_l1_loss,
    #        "y2_loss": epoch_l2_loss,
    #        "y1_acc": y1_tot_acc,
    #        "y2_acc": y2_tot_acc}
#
    #return data, count

def multiclass_validation_loop(model, validation_loader, device, criterion):
    """
    Detials
    """
    # model config
    model.eval()

    # logging here
    acc_tot_loss = 0
    acc_l1_loss = 0
    acc_l2_loss = 0 
    running_loss = 0

    y1_batch_acc_acume = 0
    y2_batch_acc_acume = 0

    for i, data in enumerate(validation_loader, 0):
        # extact data 
        x, y1_gt, y2_gt = data
        x, y1_gt, y2_gt = x.to(device), y1_gt.to(device), y2_gt.to(device)

        # forwards 
        with torch.no_grad():
            y1_pred, y2_pred = model(x)
        
        # get loss
        loss1 = criterion(y1_pred, y1_gt)
        loss2 = criterion(y2_pred, y2_gt)
        loss = loss1 + loss2

        # accumulate loss
        acc_tot_loss += loss.item()
        acc_l1_loss += loss1.item()
        acc_l2_loss += loss2.item()

        y1_batch_acc, y2_batch_acc = accuracy_reporting(y1_pred, y2_pred, y1_gt, y2_gt)
        y1_batch_acc_acume += y1_batch_acc
        y2_batch_acc_acume += y2_batch_acc
    
    # get val loss for epoch
    epoch_tot_loss = acc_tot_loss/len(validation_loader)
    epoch_l1_loss = acc_l1_loss/len(validation_loader)
    epoch_l2_loss = acc_l2_loss/len(validation_loader)

    y1_tot_acc = y1_batch_acc_acume/len(validation_loader)
    y2_tot_acc = y2_batch_acc_acume/len(validation_loader)

    data = {"total_loss": epoch_tot_loss,
            "y1_loss": epoch_l1_loss,
            "y2_loss": epoch_l2_loss,
            "y1_acc": y1_tot_acc,
            "y2_acc": y2_tot_acc}
    
    # return val loss
    return data