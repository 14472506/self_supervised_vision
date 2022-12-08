"""
Detials
"""
# imports 
import torch

# functions 
def RotNet_training_loop(epoch, count, model, train_loader, device, optimiser, criterion, print_freque):
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

        # reset gradient
        optimiser.zero_grad

        # forward + backward + optimizer_step
        y_pred = model(x)
        loss = criterion(y_pred, y_gt)
        loss.backward()
        optimiser.step()

        # print stats
        running_loss += loss.item()
        acc_loss += loss.item()
        if i % print_freque == print_freque-1:
            # get GPU memory usage
            mem_all = torch.cuda.memory_allocated(device) / 1024**3 
            mem_res = torch.cuda.memory_reserved(device) / 1024**3 
            mem = mem_res + mem_all
            mem = round(mem, 2)
            print("[epoch: %s][iter: %s][memory use: %sGB] total_loss: %s" %(epoch, count, mem, running_loss/print_freque))
            
            running_loss = 0

        count += 1

    # calculating and returning trainig loss over epoch
    epoch_loss = acc_loss/len(train_loader)
    return epoch_loss, count

def RotNet_validation_loop(model, validation_loader, device, criterion):
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