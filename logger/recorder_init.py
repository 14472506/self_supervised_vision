"""
Detials
"""
# functions
def recorder_dict_multiclass():
    rec_dict = {
        "epoch": [],
        "y1_train_acc": [],
        "y2_train_acc": [],
        "y1_validation_acc": [],
        "y2_validation_acc": [],
        "training_tot_loss": [],
        "training_l1_loss": [],
        "training_l2_loss": [],
        "validation_tot_loss": [],
        "validation_l1_loss": [],
        "validation_l2_loss": [],
        "best_tot_epoch": [],
        "best_l1_epoch": [],
        "best_l2_epoch": [],
        "best_comb_epoch":[],
        "best_y1_epoch": [],
        "best_y2_epoch": [],
        "best_tot_training_loss": [],
        "best_l1_training_loss": [],
        "best_l2_training_loss": [],
        "best_comb_train_loss": [],
        "best_tot_validation_loss": [],
        "best_l1_validation_loss": [],
        "best_l2_validation_loss": [],
        "best_comb_validation_loss": [],
        "best_y1_acc_epcoh": [],
        "best_y2_acc_epoch": [],
        "best_y1_acc": [],
        "best_y2_acc": [],
        "best_comb_acc": [],
        "best_comb_acc_epoch": []
    }
    return rec_dict