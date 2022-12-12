"""
Detials
"""
# imports
import torch

# functions
def training_print_out(running_loss, acc_loss, loss, print_freque,
     device, iter_count, epoch, count):
    """
    Detials
    """
    # print stats
    running_loss += loss.item()
    acc_loss += loss.item()
    if iter_count % print_freque == print_freque-1:
        # get GPU memory usage
        mem_all = torch.cuda.memory_allocated(device) / 1024**3 
        mem_res = torch.cuda.memory_reserved(device) / 1024**3 
        mem = mem_res + mem_all
        mem = round(mem, 2)
        print("[epoch: %s][iter: %s][memory use: %sGB] total_loss: %s" %(epoch, count, mem, running_loss/print_freque))
        
        running_loss = 0
    
    # returning values
    return running_loss, acc_loss