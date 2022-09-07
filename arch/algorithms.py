"""
Detials
"""
# =============================================================================================== #
# Imports
# =============================================================================================== #
import os

# =============================================================================================== #
# Classes
# =============================================================================================== #
class Algorithm():
    """
    Detials
    """
    def __init__(self, opt):
        """
        Details
        """
        self.set_experiment_dir(opt['exp_dir'])
        self.opt = opt
        self.init_all_networks()
        self.init_all_criterians()
        self.current_epoch = 0
        self.optimizers = {}

        self.best_model_metrics = opt['best_metrics'] if ('best_metrics' in opt) else None

    
    def set_experimental_dir(self, directory_path):
        """
        Detials
        """
        self.exp_dir = directory_path
        if (not os.path.isdir(self.exp_dir)):
            os.makedirs(self.exp_dir)


    def init_all_networks(self):
        """
        Detials
        """
        network_defs = self.opt['networks']
        self.networks = {}
        self.optim_params = {}  

        for name_net, def_net, in network_defs.items(): # def_net: {'def_file', 'pretrained', 'opt', 'optim_params'}
            def_file = def_net['def_file']
            net_opt = def_net['opt']
            self.optim_params[name_net] = def_net['optim_params'] if ('optim_params' in def_net) else None
            pretrained_path = def_net['pretrained'] if ('pretrained' in def_net) else None
            self.networks[name_net] = self.init_all_network(def_file, net_opt, pretrained_path, name_net)self.init_all_criterians()
    

    def init_network(self, net_def_file, net_opt, pretrained_path, name_net):
        """
        Details
        """
        if (not os.path.isfile(net_def_file)):
            raise ValueError('Non existing file: {0}'.format(net_def_file))
        
        network = imp.load_source("") ################# pick up from here 


    def init_all_criterians(self):
        """
        Details
        """
    
    def allocate_tensors(self):
    

