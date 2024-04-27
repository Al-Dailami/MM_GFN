from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel

from torch_geometric.nn import GCNConv, GATConv, PNAConv


import copy
import numpy as np

class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return self.mse(torch.log(pred), torch.log(actual))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class GNNFusion_Late_BCB(nn.Module): # Late fusion of text embedings using Finetuned BioBert
    def __init__(self, config, device=torch.device('cuda')):
        super(GNNFusion_Late_BCB, self).__init__()
        
        self.BioBert = AutoModel.from_pretrained("./MIMIC-III_mortality_FT_BCB").to(device) # The finetuned model weights
        # self.BioBert = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(device)
        for param in self.BioBert.embeddings.parameters():
            param.requires_grad = False
        for param in self.BioBert.encoder.parameters():
            param.requires_grad = False
        
        BioBertConfig = self.BioBert.config
        text_embed_size =  BioBertConfig.hidden_size
        
        self.BioBert_proj_fc1 = nn.Linear(in_features=text_embed_size, out_features=config['txt_out_dim'])
        self.BioBert_proj_fc2 = nn.Linear(in_features=text_embed_size, out_features=config['txt_out_dim'])
        self.BioBert_proj_fc3 = nn.Linear(in_features=text_embed_size, out_features=config['txt_out_dim'])
        self.BioBert_proj_fc4 = nn.Linear(in_features=text_embed_size, out_features=config['txt_out_dim'])
        self.BioBert_proj_fc5 = nn.Linear(in_features=text_embed_size, out_features=config['txt_out_dim'])
        self.BioBert_proj_fc6 = nn.Linear(in_features=text_embed_size, out_features=config['txt_out_dim'])
        self.BioBert_proj_fc7 = nn.Linear(in_features=text_embed_size, out_features=config['txt_out_dim'])
        
        # self.conv1 = GCNConv(config['txt_hidden'], config['txt_hidden'])
        # self.conv2 = GCNConv(config['txt_hidden'], config['txt_hidden'])
        
        self.conv1 = GATConv(config['txt_hidden'], config['txt_hidden']//8, 8) #aggr='mean'
        self.conv2 = GATConv(config['txt_hidden'], config['txt_hidden'], 1, concat=False)
    
    def forward(self, sd_h, ts_h, text, edge_index):
 
        batch_size, input_dim = sd_h.size() # batch_size * input_dim + 1 * hidden_dim(i)
        out_batch1 = []
        out_batch2 = []
        out_batch3 = []
        
        for p in range(batch_size):
            
            ecg_txts = torch.tensor(np.array([item[0] for item in np.array(text[p][0])]), dtype=torch.long).to(ts_h.device)
            ecg_attn = torch.tensor(np.array([item[1] for item in np.array(text[p][0])]), dtype=torch.long).to(ts_h.device)
            ecg_notes_emb = torch.mean(self.BioBert_proj_fc1(self.BioBert(ecg_txts, ecg_attn)[0][:,0,:]), axis=0)
            
            echo_txts = torch.tensor(np.array([item[0] for item in np.array(text[p][1])]), dtype=torch.long).to(ts_h.device)
            echo_attn = torch.tensor(np.array([item[1] for item in np.array(text[p][1])]), dtype=torch.long).to(ts_h.device)
            echo_notes_emb = torch.mean(self.BioBert_proj_fc2(self.BioBert(echo_txts, echo_attn)[0][:,0,:]), axis=0)
            
            nurse_txts = torch.tensor(np.array([item[0] for item in np.array(text[p][2])]), dtype=torch.long).to(ts_h.device)
            nurse_attn = torch.tensor(np.array([item[1] for item in np.array(text[p][2])]), dtype=torch.long).to(ts_h.device)
            nurse_notes_emb = torch.mean(self.BioBert_proj_fc3(self.BioBert(nurse_txts, nurse_attn)[0][:,0,:]), axis=0)
            
            radio_txts = torch.tensor(np.array([item[0] for item in np.array(text[p][3])]), dtype=torch.long).to(ts_h.device)
            radio_attn = torch.tensor(np.array([item[1] for item in np.array(text[p][3])]), dtype=torch.long).to(ts_h.device)
            radio_notes_emb = torch.mean(self.BioBert_proj_fc4(self.BioBert(radio_txts, radio_attn)[0][:,0,:]), axis=0)
            
            phis_txts = torch.tensor(np.array([item[0] for item in np.array(text[p][4])]), dtype=torch.long).to(ts_h.device)
            phis_attn = torch.tensor(np.array([item[1] for item in np.array(text[p][4])]), dtype=torch.long).to(ts_h.device)
            phis_notes_emb = torch.mean(self.BioBert_proj_fc5(self.BioBert(phis_txts, phis_attn)[0][:,0,:]), axis=0)
            
            resp_txts = torch.tensor(np.array([item[0] for item in np.array(text[p][5])]), dtype=torch.long).to(ts_h.device)
            resp_attn = torch.tensor(np.array([item[1] for item in np.array(text[p][5])]), dtype=torch.long).to(ts_h.device)
            resp_notes_emb = torch.mean(self.BioBert_proj_fc6(self.BioBert(resp_txts, resp_attn)[0][:,0,:]), axis=0)
            
            other_txts = torch.tensor(np.array([item[0] for item in np.array(text[p][6])]), dtype=torch.long).to(ts_h.device)
            other_attn = torch.tensor(np.array([item[1] for item in np.array(text[p][6])]), dtype=torch.long).to(ts_h.device)
            other_notes_emb = torch.mean(self.BioBert_proj_fc7(self.BioBert(other_txts, other_attn)[0][:,0,:]), axis=0)
            
            xg = torch.stack([sd_h[p], ts_h[p], ecg_notes_emb, echo_notes_emb, nurse_notes_emb, radio_notes_emb, phis_notes_emb, resp_notes_emb, other_notes_emb],0).to(sd_h.device)

            m1 = F.relu(self.conv1(xg, edge_index[p].to(sd_h.device)))
            m1 = F.dropout(m1, p=0.5, training=self.training)
            m1 = self.conv2(m1, edge_index[p].to(sd_h.device))
#             mean_out = torch.mean(torch.stack([m1[i] for i in torch.unique(edge_index[p][0])[1:]],0).to(inputs[0].device),0)
            mean_out = torch.mean(torch.stack([xg[i] for i in torch.unique(edge_index[p][1])[1:]],0).to(sd_h.device),0)
            
            
            adj_m =  torch.zeros((9, 9), dtype=torch.float)
            adj_m[edge_index[p][0], edge_index[p][1]] = 1.0

            # Flatten the adjacency matrix
            adj_m_flt = adj_m.view(-1).to(sd_h.device)
            
            out_batch1.append(m1[0]) # Center point
            out_batch2.append(mean_out) # mean of notes embeddings
            out_batch3.append(adj_m_flt) 
        
        output1 = torch.stack(out_batch1,0)
        output2 = torch.stack(out_batch2,0)
        output3 = torch.stack(out_batch3,0)
        
        del out_batch1
        del out_batch2
        del out_batch3
        
        return output1, output2, output3
    

class SD_TS_Encoder(nn.Module):
    def __init__(self, config, flat_dim, ts_dim):
        super(SD_TS_Encoder, self).__init__()

        self.bidirectional = True
        self.n_layers = 1
        
        # dimensions are specified in the order of static, timeseries and text
        self.sd_hidden = config['sd_hidden']
        self.ts_hidden = config['ts_hidden']
        self.txt_hidden = config['txt_hidden']
        
        self.sd_prob = config['sd_dropout']
        self.ts_prob = config['ts_dropout']
        self.main_dropout_prob = config['main_dropout']
                
        if self.bidirectional:
            self.ts_output = self.ts_hidden//2
        else:
            self.ts_output = self.ts_hidden
                            
        self.sd_subnet = nn.Sequential(
            nn.Linear(in_features=flat_dim, out_features=self.sd_hidden),
            nn.GELU(),
            # nn.ReLU(),
            nn.Dropout(p=self.sd_prob)
        )

        self.ts_subnet = nn.GRU(input_size=ts_dim, hidden_size=self.ts_output, num_layers=self.n_layers,
                                bidirectional=self.bidirectional, dropout=self.ts_prob)

        self.ts_dropout = nn.Dropout(p=self.ts_prob)
        
        self.remove_none = lambda x: tuple(xi for xi in x if xi is not None)
        
    def forward(self, sd_x, ts_x):
        '''
        Args:
            sd_x: static demo 
            ts_x: time-series
        '''
        
        batch_size, T, Ft  = ts_x.shape
        
        sd_h = self.sd_subnet(sd_x)
        
        X_separated = torch.split(ts_x, (Ft)//2, dim=2)
        ts1 = torch.transpose(X_separated[0], 0, 1).contiguous()
        gru_output1, _ = self.ts_subnet(ts1)
        gru_output = torch.transpose(gru_output1, 0, 1).contiguous()

        if self.bidirectional:
            ts_h = self.ts_dropout(torch.cat(self.remove_none((gru_output[:,-1,:self.ts_hidden], gru_output[:,0,self.ts_hidden:])), dim=1))  # B * 2*hidden_size
        else:
            ts_h = self.ts_dropout(gru_output[:,-1,:])
    
        return sd_h, ts_h

class MM_GFN(nn.Module):
    '''
    Multimodal Graph-based Fusion Network
    '''
    def __init__(self, config, ts_dim=None, flat_dim=None):
        '''
        Args:
            config - configuration parameters
            ts_dim - number of time-series variables
            flat_dim - number of static demographic features 
            
        Output:
            (return value in forward) output predictions for LoS(1) and mortality(1).
        '''
        super(MM_GFN, self).__init__()

        self.task = config['task']
        
        self.txt_hidden = config['txt_hidden']
        
        self.sd_ts_enc = SD_TS_Encoder(config=config, flat_dim=flat_dim,ts_dim=ts_dim)
        
        self.gnn_fusion = GNNFusion_Late_BCB(config=config)

        self.sigmoid = nn.Sigmoid()
        
        self.g_FC = nn.Linear(in_features=config['txt_hidden'], out_features=config['txt_hidden'])
        
        self.sd_proj_fc = nn.Linear(in_features=config['sd_hidden'], out_features=config['txt_hidden'])
        
        self.sd_ts_proj_fc = nn.Linear(in_features=config['sd_hidden']+config['ts_hidden'], out_features=config['txt_hidden'])

        self.fusion_dim = self.txt_hidden*3+(9*9)
        
        self.final_fc = nn.Linear(in_features=self.fusion_dim, out_features=1)
        
        self.regression_act_fn = nn.Hardtanh(min_val=1/48, max_val=100)

        self.margin = nn.Parameter(torch.ones([]) * 0.1)
        
        self.bce_loss = nn.BCELoss()
        self.msle_loss = MSLELoss()
    
    def forward(self, sd_x, ts_x, text, edge_index):
        '''
        Args:
            x: a list contains multimodal input features [static demo., time-series, clinical notes]
        '''
                
        sd_h, ts_h = self.sd_ts_enc(sd_x,ts_x)
        
        cp, txt_rep, adj_m = self.gnn_fusion(self.sd_proj_fc(sd_h),  ts_h, text, edge_index)
        
        g = self.sigmoid(self.g_FC(txt_rep))
        txt_rep_g = txt_rep*g
        proj_sd_ts = self.sd_ts_proj_fc(torch.cat([sd_h,ts_h],dim=1))
        sd_ts_rep = proj_sd_ts*(1-g)
        cp_g = cp*g
        
        fused_features = torch.cat([txt_rep_g,sd_ts_rep,cp_g,adj_m], dim=1)
        
        
        if str.lower(self.task) == 'mortality':
            prediction_output = self.sigmoid(self.final_fc(fused_features))
        elif str.lower(self.task) == 'los':
            prediction_output = self.regression_act_fn(self.final_fc(fused_features))
        else:
            print('Please chose one of the supported tasks: mortaloty or los ...')
        
        return prediction_output, proj_sd_ts, txt_rep

    def loss(self, y_hat, y):
        # classification loss
        if str.lower(self.task) == 'mortality':
            loss = self.bce_loss(y_hat, y) #* self.alpha
        elif str.lower(self.task) == 'los':
            loss = self.msle_loss(y_hat, y)
        return loss
    
    def contrastive_loss(self, ts_rep, txt_rep):
        # Normalize representations (optional)
        with torch.no_grad():
            self.margin.clamp_(0.001,0.5)
            
        ts_rep = F.normalize(ts_rep, p=2, dim=-1)
        txt_rep = F.normalize(txt_rep, p=2, dim=-1)
        
        # Calculate negative dot product (similarity)
        sim1 = F.pairwise_distance(ts_rep, txt_rep).neg()
        sim2 = F.pairwise_distance(txt_rep, ts_rep).neg()
        
        # Hinge loss with margin
        loss1 = F.relu(self.margin - sim1)
        loss2 = F.relu(self.margin - sim2)
        
        # Mean loss across samples
        mean_loss1 = loss1.mean()
        mean_loss2 = loss2.mean()
        return (mean_loss1+mean_loss2)/2