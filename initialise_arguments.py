import argparse

class Config(dict):
    def __init__(self, file_=None, config=None, **kwargs):
        super(Config, self).__init__()
        self.__dict__ = self
        
    def __setattr__(self, key, value):
        """Modified to automatically convert `dict` to Config."""
        if type(value) == dict:
            new_config = Config()
            new_config.update(value, deep=False)
            super(Config, self).__setattr__(key, new_config)
        else:
            super(Config, self).__setattr__(key, value)

    def __getitem__(self, key):
        """Allows convenience access to deeper levels using dots to separate
        levels, for example `config["a.b.c"]`.
        """
        if key == "":
            if len(self.keys()) == 1:
                key = list(self.keys())[0]
            else:
                raise KeyError("Empty string only works for single element Configs.")

        if type(key) == str and "." in key:
            superkey = key.split(".")[0]
            subkeys = ".".join(key.split(".")[1:])
            if superkey not in self:
                # this part enables ints in the access chain, e.g. a.1.b
                try:
                    intkey = int(superkey)
                    if intkey in self:
                        superkey = intkey
                except ValueError:
                    # if we can't convert to int, just continue so a KeyError will be raised
                    pass
            if type(self[superkey]) in (list, tuple):
                try:
                    subkeys = int(subkeys)
                except ValueError:
                    pass
            return self[superkey][subkeys]
        else:
            return super(Config, self).__getitem__(key)

    def __setitem__(self, key, value):
        """Allows convenience access to deeper levels using dots to separate
        levels, for example `config["a.b.c"]`.
        """

        if key == "":
            if len(self.keys()) == 1:
                key = list(self.keys())[0]
            else:
                raise KeyError("Empty string only works for single element Configs.")

        if type(key) == str and "." in key:
            superkey = key.split(".")[0]
            subkeys = ".".join(key.split(".")[1:])
            if superkey != "" and superkey not in self:
                self[superkey] = Config()
            if type(self[superkey]) == list:
                try:
                    subkeys = int(subkeys)
                except ValueError:
                    pass
            self[superkey][subkeys] = value
        elif type(value) == dict:
            super(Config, self).__setitem__(key, Config(config=value))
        else:
            super(Config, self).__setitem__(key, value)

    def __delitem__(self, key):
        """Allows convenience access to deeper levels using dots to separate
        levels, for example `config["a.b.c"]`.
        """

        if type(key) == str and "." in key and key not in self:
            superkey = key.split(".")[0]
            subkeys = ".".join(key.split(".")[1:])
            if superkey not in self:
                raise KeyError(superkey + " not found.")
            else:
                self[superkey].__delitem__(subkeys)
        else:
            super().__delitem__(key)

# all default values stated here are the best hyperparameters in the eICU dataset, not MIMIC
def initialise_arguments():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('-disable_cuda', action='store_true')
    parser.add_argument('-intermediate_reporting', action='store_true')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--batch_size_test', default=64, type=int)
    parser.add_argument('-shuffle_train', action='store_true')
    parser.add_argument('-save_results_csv', action='store_true')
    parser.add_argument('--percentage_data', default=100.0, type=float)
    
    parser.add_argument('--task', default='mortality', type=str, help='can be either mortality or LoS')
    
    parser.add_argument('--mode', default='train', type=str, help='can be either train, which reports intermediate '
                                                                  'results on the training and validation sets each '
                                                                  'epoch, or test, which just runs all the epochs and '
                                                                  'only reports on the test set')
    # ablations
    parser.add_argument('--window', default=24, type=int, help='can be either 24 or 48') # 24 - 48
    
    return parser

def gen_config(parser):
    args = parser.parse_args()
    # prepare config dictionary, add all arguments from args
    c = Config()
    for arg in vars(args):
        c[arg] = getattr(args, arg)
    return c

def initialise_mm_gfn_arguments():
    
    parser = initialise_arguments()
    parser.add_argument('--n_epochs', default=50, type=int)
    parser.add_argument('--seed', default=2024, type=int)
    
    parser.add_argument('--learning_rate', default=0.0007, type=float)
    parser.add_argument('--learning_rate_text', default=0.00002, type=float)
    parser.add_argument('--learning_rate_sd_ts', default=0.001, type=float)
        
    parser.add_argument('--sd_dropout', default=0.25, type=float)
    parser.add_argument('--ts_dropout', default=0.25, type=float)
    parser.add_argument('--txt_dropout', default=0.25, type=float)
    
    parser.add_argument('--main_dropout', default=0.45, type=float)
       
    parser.add_argument('--sd_hidden', default=64, type=int)
    parser.add_argument('--ts_hidden', default=256, type=int) 
    parser.add_argument('--txt_hidden', default=256, type=int)
    parser.add_argument('--txt_out_dim', default=256, type=int)
        
    parser.add_argument('--gru_n_layers', default=1, type=int)
    
    parser.add_argument('--log_interval', default=30, type=int)
    
    parser.add_argument('--BCB_FT_Path', default="./MIMIC-III_mortality_FT_BCB", type=str)

    c = gen_config(parser)
    
    c['freeze_embedding'] = True
    c['bidirectional_gru'] = True 
    
    c['intermediate_reporting'] = True 

    return c