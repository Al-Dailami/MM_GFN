import os

import torch
import pandas as pd
import numpy as np
from itertools import groupby, islice

import transformers as ppb
import pickle

class MIMICReader(object):

    def __init__(self, data_path, max_len=24, mode='late_bcb', device=None):
        
        self.train = ('train' in data_path)
        self.mode = mode
        self.MaxLens = {'ECG':128,'Echo':128, 'Nursing':512, 'Other':128, 'Physician':128, 'Radiology':512, 'Respiratory': 128}
        self._labels_path = data_path + '/labels.csv'
        self._flat_path = data_path + '/flat.csv'
        
        self._notes_txts_lst_path = data_path + '/notes_txts_lst.json' # 7 note types multiple note (tokens)
        self._notes_ids_cat_path = data_path + '/notes_ids_cat.json' # 7 note types concatnate notetes of each type separatly (input ids)
        
        self._notes_txts_all_path = data_path + '/notes_txts_all.json' # All Notes without category [raw_texts_lst, input_ids_lst, input_ids_cat, tokens_lst, tokens_cat]
        
        self._timeseries_path = data_path + '/timeseries.csv'
        self._device = device
        self._dtype = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor

        self.labels = pd.read_csv(self._labels_path, index_col='patient')
        self.labels_los = self.labels['actualiculos']
        self.labels_mort = self.labels['actualhospitalmortality']
        self.flat = pd.read_csv(self._flat_path, index_col='patient').fillna(0)
        
        if os.path.exists(data_path +'/edge_index.pkl'):
            print('Loading edge index ... ')
            with open(data_path +'/edge_index.pkl', 'rb') as f:
                self.edge_indexs = pickle.load(f)
        else:
            print('Generating edge index ... ')
            self.notes_ids_cat = pd.read_json(self._notes_ids_cat_path, orient='split')
            self.notes_ids_cat.set_index('patient', inplace=True)
            modality_mask = []
            columns = ['ECG','Echo', 'Nursing', 'Other', 'Physician', 'Radiology', 'Respiratory']
            for index, row in self.notes_ids_cat.iterrows():
                mask = []
                for col in columns:
                    if len(row[col]) <= 1:
                        mask.append(0)
                    else:
                        mask.append(1)
                modality_mask.append(mask)
            
            adj_list = []
            for m in modality_mask:
                dst = [0,1] + [(j+2) for j in range(len(m)) if m[j] > 0 ]
                src = [0]*len(dst)
                adj_list.append([src,dst])
                
            self.edge_indexs = [torch.tensor(adj, dtype=torch.long) for adj in adj_list] 
            
            edg_idx = open(data_path +'/edge_index.pkl','wb')
            pickle.dump(self.edge_indexs, edg_idx, -1)
            edg_idx.close()
               
        if os.path.exists(data_path +'/notes_bert_multi_lst.json'):
            print('Loading BioBert input ids and atten mask ... ')    
            self.notes_bert_multi_lst = pd.read_json(data_path +'/notes_bert_multi_lst.json', orient='split')
            self.notes_bert_multi_lst.set_index('patient', inplace=True)
        else:
            print('Generating BioBert input ids and atten mask ... ')
            # Download pre-trained weights from BERT. This takes a while if running for the first time. 
            # model_class, tokenizer_class, pretrained_weights = (ppb.AutoModel, ppb.AutoTokenizer, "./MIMIC-III_mortality_AMTN")
            model_class, tokenizer_class, pretrained_weights = (ppb.AutoModel, ppb.AutoTokenizer, 'emilyalsentzer/Bio_ClinicalBERT')
            tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            
            self.notes_txts_lst = pd.read_json(self._notes_txts_lst_path, orient='split')
            self.notes_txts_lst.set_index('patient', inplace=True)
            self.notes_txts_lst['ECG_BioBERT_lst'] = self.notes_txts_lst['ECG'].apply(lambda x: self.bioBert_tokenize_lst(x, tokenizer, self.MaxLens['ECG']))
            self.notes_txts_lst['Echo_BioBERT_lst'] = self.notes_txts_lst['Echo'].apply(lambda x: self.bioBert_tokenize_lst(x, tokenizer, self.MaxLens['Echo']))
            self.notes_txts_lst['Nursing_BioBERT_lst'] = self.notes_txts_lst['Nursing'].apply(lambda x: self.bioBert_tokenize_lst(x, tokenizer, self.MaxLens['Nursing']))
            self.notes_txts_lst['Other_BioBERT_lst'] = self.notes_txts_lst['Other'].apply(lambda x: self.bioBert_tokenize_lst(x, tokenizer, self.MaxLens['Other']))
            self.notes_txts_lst['Physician_BioBERT_lst'] = self.notes_txts_lst['Physician'].apply(lambda x: self.bioBert_tokenize_lst(x, tokenizer, self.MaxLens['Physician']))
            self.notes_txts_lst['Radiology_BioBERT_lst'] = self.notes_txts_lst['Radiology'].apply(lambda x: self.bioBert_tokenize_lst(x, tokenizer, self.MaxLens['Radiology']))
            self.notes_txts_lst['Respiratory_BioBERT_lst'] = self.notes_txts_lst['Respiratory'].apply(lambda x: self.bioBert_tokenize_lst(x, tokenizer, self.MaxLens['Respiratory']) )
            # saving
            print('Saving BioBert input ids and atten mask ... ')
            notes_txts_m_lst_cols = ['patient','ECG_BioBERT_lst','Echo_BioBERT_lst','Nursing_BioBERT_lst','Other_BioBERT_lst','Physician_BioBERT_lst','Radiology_BioBERT_lst','Respiratory_BioBERT_lst']
            self.notes_txts_lst.reset_index(inplace=True)
            self.notes_txts_lst[notes_txts_m_lst_cols].to_json(data_path + '/notes_bert_multi_lst.json', orient='split', index=False)
            print('Loading BioBert input ids and atten mask ... ')
            self.notes_bert_multi_lst = pd.read_json(data_path +'/notes_bert_multi_lst.json', orient='split')
            self.notes_bert_multi_lst.set_index('patient', inplace=True)
            
        self.no_flat_features = self.flat.shape[1]
        self.no_ts_features = (pd.read_csv(self._timeseries_path, index_col='patient', nrows=1).shape[1])

        self.patients = list(self.labels.index)
        self.no_patients = len(self.patients)

        self.ratio = np.unique(self.labels['actualhospitalmortality'], return_counts=True)[1] #Counter(self.labels_mort)
        self.class_weights = self.get_class_weights(self.labels_mort)
        self.max_len = max_len

    
    def bioBert_tokenize_lst(self, notes, tokenizer, max_len):
        nids = []  
        for note in notes:
            if(len(note) > 0):
                # print(note)
                note = note[0] if (type(note)==list) else note    # to fix the ['note'] problem               
                encoded_note = tokenizer.encode_plus(
                                            text=note,                    # Preprocess sentence
                                            add_special_tokens=True,        # Add [CLS] and [SEP]
                                            max_length=max_len,             # Max length to truncate/pad
                                            padding='max_length',         # Pad sentence to max length
                                            # pad_to_max_length=True,         # Pad sentence to max length
                                            # return_tensors='pt',           # Return PyTorch tensor
                                            return_attention_mask=True,     # Return attention mask
                                            truncation=True,
                                            )
                nids.append([encoded_note.get('input_ids'), encoded_note.get('attention_mask')])
                
        if(len(nids) == 0):
            encoded_note = tokenizer.encode_plus(
                                            text='<pad>',                      # Preprocess sentence
                                            add_special_tokens=True,        # Add [CLS] and [SEP]
                                            max_length=max_len,             # Max length to truncate/pad
                                            padding='max_length',         # Pad sentence to max length
                                            # pad_to_max_length=True,         # Pad sentence to max length
                                            # return_tensors='pt',           # Return PyTorch tensor
                                            return_attention_mask=True,     # Return attention mask
                                            truncation=True,
                                            )
            nids.append([encoded_note.get('input_ids'), encoded_note.get('attention_mask')])
        return np.array(nids)


    
    def get_class_weights(self, train_labels):
        """
        return class weights to handle class imbalance problems
        """
        occurences = np.unique(train_labels, return_counts=True)[1]
        class_weights = occurences.sum() / occurences
        class_weights = torch.Tensor(class_weights).float()
        return class_weights

    def line_split(self, line):
        return [float(x) for x in line.split(',')]

    def pad_sequences(self, ts_batch):
        max_len = self.max_len
        seq_lengths = [len(x[:max_len]) for x in ts_batch]
        padded = [patient[:max_len] + [[0] * (self.no_ts_features)] * (max_len - len(patient[:max_len])) for patient in ts_batch]
        padded = torch.tensor(padded, device=self._device).type(self._dtype)#.permute(0, 2, 1)  # B * (2F + 2) * T
        padded[:, :, 0] /= 24  # scale the time into days instead of hours
        mask = torch.zeros(padded[:, :, 0].shape, device=self._device).type(self._dtype)
        for p, l in enumerate(seq_lengths):
            mask[p, :l] = 1
        return padded, mask, torch.tensor(seq_lengths).type(self._dtype)
    
    def padd_lst(self, n_ids, MaxLen, pad_token):
        nids = []
        for i in range(len(n_ids)):
            if len(n_ids[i]) < MaxLen and len(n_ids[i]) > 1:
                nids.append(n_ids[i] + [pad_token] * (MaxLen - len(n_ids[i])))   
            elif len(n_ids[i]) > MaxLen:
                nids.append(n_ids[i][:MaxLen])
        if(len(nids) == 0):
            nids.append([pad_token] * MaxLen)
        return np.array(nids)

    def padd(self, n_ids, MaxLen, pad_token):
        if len(n_ids) < MaxLen:
            n_ids+=([pad_token] * (MaxLen - len(n_ids)))
        elif len(n_ids) > MaxLen:
            n_ids = n_ids[:MaxLen]
        return np.array(n_ids)
    
    def concat(self, n_ids, MaxLen, pad_token):
        all_nids = []
        for nid in n_ids:
            if len(nid) > 1:
                all_nids+= nid
        if len(all_nids) < MaxLen:
            all_nids += [pad_token] * (MaxLen - len(all_nids))
        elif len(all_nids) > MaxLen:
            all_nids = all_nids[:MaxLen]
        return np.array(all_nids)

    def get_los_labels(self, labels):
        LoS_labels = labels  
        return LoS_labels

    def get_mort_labels(self, labels):
        repeated_labels = labels
        return repeated_labels

    def batch_gen(self, batch_size=8):

        # note that once the generator is finished, the file will be closed automatically
        with open(self._timeseries_path, 'r') as timeseries_file:
            # the first line is the feature names; we have to skip over this
            self.timeseries_header = next(timeseries_file).strip().split(',')
            # this produces a generator that returns a list of batch_size patient identifiers
            patient_batches = (self.patients[pos:pos + batch_size] for pos in range(0, len(self.patients), batch_size))
            # create a generator to capture a single patient timeseries
            ts_patient = groupby(map(self.line_split, timeseries_file), key=lambda line: line[0])
            # we loop through these batches, tracking the index because we need it to index the pandas dataframes
            for i, batch in enumerate(patient_batches):
                ts_batch = [[line[1:] for line in ts] for _, ts in islice(ts_patient, batch_size)]
                padded, mask, seq_lengths = self.pad_sequences(ts_batch)
                los_labels = self.get_los_labels(torch.tensor(self.labels_los.iloc[i*batch_size:(i+1)*batch_size].values, device=self._device).type(self._dtype))
                mort_labels = self.get_mort_labels(torch.tensor(self.labels_mort.iloc[i*batch_size:(i+1)*batch_size].values, device=self._device).type(self._dtype))
                                   
                yield (padded,  # B * (2F + 2) * T
                    mask[:,:],  # B * (T - time_before_pred)
                    torch.tensor(self.flat.iloc[i*batch_size:(i+1)*batch_size].values.astype(float), device=self._device).type(self._dtype),  # B * no_flat_features
                    # BioBert input_ids and atten_mask
                    np.array(self.notes_bert_multi_lst.iloc[i*batch_size:(i+1)*batch_size].values), # B * NTypes * (Note_len*N)
                    self.edge_indexs[i*batch_size:(i+1)*batch_size],
                    los_labels,
                    mort_labels,
                    seq_lengths)
                    

if __name__=='__main__':
    MIMIC_reader = MIMICReader('./MIMIC_data_24H/train')
    print(next(MIMIC_reader.batch_gen()))