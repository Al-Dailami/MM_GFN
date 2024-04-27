import torch
from my_experiment_template import create_folder
from MM_GFN_model import MM_GFN
from my_experiment_template import MyExperimentTemplate
from initialise_arguments import initialise_mm_gfn_arguments

import numpy as np
import random

class MM_GFN_(MyExperimentTemplate):
    def __init__(self, config, n_epochs, exp_name, base_dir):
        super().__init__(config, n_epochs, exp_name, base_dir)
        
        self.setup_template()
        self.model = MM_GFN(config=self.config, 
                                ts_dim=int((self.train_datareader.no_ts_features)//2),
                                flat_dim=self.train_datareader.no_flat_features).to(device=self.device)
        self.elog.print(self.model)
        
        # self.config.learning_rate_text = 0.00002
        print([n for n, p in self.model.named_parameters() if 'BioBert' not in n and 'sd' not in n and 'ts' not in n])
        print([n for n, p in self.model.named_parameters() if 'BioBert' in n])
        print([n for n, p in self.model.named_parameters() if 'sd' in n or 'ts' in n])
        self.optimiser= torch.optim.Adam([
                {'params': [p for n, p in self.model.named_parameters() if 'BioBert' not in n and 'sd' not in n and 'ts' not in n]},
                {'params':[p for n, p in self.model.named_parameters() if 'BioBert' in n], 'lr': self.config.learning_rate_text},
                {'params':[p for n, p in self.model.named_parameters() if 'sd' in n or 'ts' in n], 'lr': self.config.learning_rate_sd_ts}
            ], lr=self.config.learning_rate)
        
        return


if __name__=='__main__':

    c = initialise_mm_gfn_arguments()
    
    c['exp_name'] = 'MM_RL_GFN_GAT'
    
    if c['window'] == 24:
        c['data_path'] = './MIMIC_III_data_24H/'
    else:
        c['data_path'] = './MIMIC_III_data_48H/'
        
    c['base_dir'] = create_folder('./experiments/{}/{}'.format(str(c.window), c.task), c.exp_name)
    
    c['model_ckpt_path'] = create_folder('./experiments/{}/{}/{}'.format(str(c.window), c.task, c.exp_name), 'best_model_ckpt')
    c['pretrained_embedding'] = torch.load(c['data_path'] + 'vectors.pt')
    c['vocab_size'] = None
    
    
    RANDOM_SEED = c.seed # seed initializations
    np.random.seed(RANDOM_SEED) #numpy
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED) # cpu
    torch.cuda.manual_seed(RANDOM_SEED) #gpu
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True # cudnn
    
    mm_gfn = MM_GFN_(config=c,
            n_epochs=c.n_epochs,
            exp_name=c.exp_name,
            base_dir=c['base_dir'])
    mm_gfn.run()     

    
    
