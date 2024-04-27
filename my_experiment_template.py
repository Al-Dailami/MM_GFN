import torch
from MIMIC_III_reader import MIMICReader
import numpy as np
import os
from metrics import compute_binary_metrics, compute_regression_metrics
from shuffle_train import shuffle_train

from tqdm import tqdm

def create_folder(parent_path, folder):
    if not parent_path.endswith('/'):
        parent_path += '/'
    folder_path = parent_path + folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path
# view the results by running: python3 -m trixi.browser --port 8080 BASEDIR

def save_to_csv(Logger, data, path, header=None):
    """
        Saves a numpy array to csv in the experiment save dir

        Args:
            data: The array to be stored as a save file
            path: sub path in the save folder (or simply filename)
    """

    folder_path = create_folder(Logger.save_dir, os.path.dirname(path))
    file_path = folder_path + '/' + os.path.basename(path)
    if not file_path.endswith('.csv'):
        file_path += '.csv'
    np.savetxt(file_path, data, delimiter=',', header=header, comments='')
    return

def remove_padding(y, mask, device):
    """
        Filters out padding from tensor of predictions or labels

        Args:
            y: tensor of los predictions or labels
            mask (bool_type): tensor showing which values are padding (0) and which are data (1)
    """
    # note it's fine to call .cpu() on a tensor already on the cpu
    y = y.where(mask, torch.tensor(float('nan')).to(device=device)).flatten().detach().cpu().numpy()
    y = y[~np.isnan(y)]
    return y

class Elog:
    def __init__(self, save_dir, log_file='output.log'):
        self.save_dir = save_dir
        self.log_file = log_file
        

    def print(self, message):
        # Print to console
        print(message)

        # Write to the log file
        with open(self.log_file, 'a') as file:
            file.write(f"{message}\n")

class MyExperimentTemplate:
    def __init__(self, config, n_epochs, exp_name, base_dir):
        self.config = config
        self.n_epochs = n_epochs
        self.exp_name = exp_name
        self.base_dir = base_dir
        self.elog = Elog(self.base_dir,self.base_dir+'/results.log')
    
    def setup_template(self):
        self.elog.print("Config:")
        self.elog.print(self.config)
        if not self.config.disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # set bool type for where statements
        self.bool_type = torch.cuda.BoolTensor if self.device == torch.device('cuda') else torch.BoolTensor

        # get datareader
        self.datareader = MIMICReader
        self.data_path = self.config.data_path
            
        self.train_datareader = self.datareader(self.data_path + 'train', max_len=self.config.window, mode=self.config.fmode, device=self.device)
        self.val_datareader = self.datareader(self.data_path + 'val', max_len= self.config.window, mode=self.config.fmode, device=self.device)
        self.test_datareader = self.datareader(self.data_path + 'test', max_len=self.config.window, mode=self.config.fmode, device=self.device)
        self.no_train_batches = len(self.train_datareader.patients) // self.config.batch_size
        self.no_val_batches = len(self.val_datareader.patients) // self.config.batch_size_test
        self.no_test_batches = len(self.test_datareader.patients) // self.config.batch_size_test
        self.checkpoint_counter = 0

        self.model = None
        self.optimiser = None

        # add a new function to elog (will save to csv, rather than as a numpy array like elog.save_numpy_data)
        self.elog.save_to_csv = lambda data, filepath, header: save_to_csv(self.elog, data, filepath, header)
        self.elog.add_result = lambda data, filepath, header: save_to_csv(self.elog, data, filepath, header)
        self.remove_padding = lambda y, mask: remove_padding(y, mask, device=self.device)
        self.elog.print('Experiment set up.')

        self.max_auroc_mort = 0
        self.max_auprc_mort = 0
        self.max_f1macro_mort = 0
        self.max_auroc_plos = 0
        self.max_auprc_plos = 0
        self.max_f1macro_plos = 0
        self.max_msle = 100
        self.max_r2 = 0
        self.max_kapa = 0

        self.lbd = 1.0
        self.max_clip_norm = 5
        
        self.patience  = 0

        self.file_name_best = '{}/{}_{}_{}_Best.pth'.format(self.config.model_ckpt_path, self.config.dataset, self.config.task, self.config.exp_name)
        self.file_name_last = '{}/{}_{}_{}_Last.pth'.format(self.config.model_ckpt_path, self.config.dataset, self.config.task, self.config.exp_name)
        return
    
    def save_checkpoint(self, file_name, epoch):
        state = {
            'model_net': self.model.state_dict(),
            'optimiser': self.optimiser.state_dict(),
            # 'optimizer_params': optimizer_params.state_dict(),
            'epoch': epoch
        }
        torch.save(state, file_name)
        return

    def train(self, epoch):

        self.model.train()
        if epoch > 0 and self.config.shuffle_train:
            shuffle_train(self.config.mimic_data_path + 'train')  # shuffle the order of the training data to make the batches different, this takes a bit of time
        train_batches = self.train_datareader.batch_gen(batch_size=self.config.batch_size)
        train_loss = []
        train_y_hat_los = np.array([])
        train_y_los = np.array([])
        train_y_hat_mort = np.array([])
        train_y_mort = np.array([])

        for batch_idx, batch in tqdm(enumerate(train_batches), total=self.no_train_batches, desc="Training", unit="batch"):
        # for batch_idx, batch in enumerate(train_batches):
            # print('   ==================> params: ',self.params[0],self.params[1],self.params[2],self.params[3])
            if batch[0].size(0) <= 1:
                print('Empty batch Train: ==========>',batch[0].shape)
                continue
            if batch_idx > (self.no_train_batches // (100 / self.config.percentage_data)):
                break

            # unpack batch
            timeseries, mask, flat, notes, edge_indexs, los_labels, mort_labels, seq_lengths = batch
           
            predction_output, output_sd_ts_rep, output_txt_rep = self.model(flat, timeseries, notes, edge_indexs)
           
            # print('  =======> Flat Preds: ', flat_output)
            beta = 0.2
            self.optimiser.zero_grad()
            
            if str.lower(self.config.task) == 'mortality':
                ce_loss = self.model.loss(predction_output.squeeze(-1), mort_labels)
                cont_loss = self.model.contrastive_loss(output_sd_ts_rep,output_txt_rep)
                loss = ce_loss*(1-beta)+cont_loss*beta
                
            else:    
                msle_loss = self.model.loss(predction_output.squeeze(-1), los_labels)
                cont_loss = self.model.contrastive_loss(output_sd_ts_rep,output_txt_rep)
                loss = msle_loss*(1-beta)+cont_loss*beta
                
            # print(loss)
            
            loss.backward()
            self.optimiser.step()
       
            train_loss.append(loss.item())

            if str.lower(self.config.task) == 'los':
                train_y_hat_los = np.append(train_y_hat_los, predction_output.detach().cpu().numpy())
                train_y_los = np.append(train_y_los, los_labels.detach().cpu().numpy())

            if str.lower(self.config.task) == 'mortality':
                train_y_hat_mort = np.append(train_y_hat_mort, predction_output.detach().cpu().numpy())
                train_y_mort = np.append(train_y_mort, mort_labels.detach().cpu().numpy())

            if self.config.intermediate_reporting and batch_idx % self.config.log_interval == 0 and batch_idx != 0:
                mean_loss_report = sum(train_loss[(batch_idx - self.config.log_interval):-1]) / self.config.log_interval
                self.elog.save_to_csv(np.vstack((mean_loss_report, epoch + (batch_idx // self.no_train_batches))).transpose(), 'Intermediate_Train_Loss.csv', header='Intermediate_Train_Loss, Epoch')
                self.elog.print('Epoch: {} [{:5d}/{:5d} samples] | train loss: {:3.4f}'
                                    .format(epoch,
                                            batch_idx * self.config.batch_size,
                                            self.config.batch_size * self.no_train_batches,
                                            mean_loss_report))
                self.checkpoint_counter += 1

        if not self.config.intermediate_reporting and self.config.mode == 'train':
            print('Train Metrics:')
            mean_train_loss = sum(train_loss) / len(train_loss)
            if str.lower(self.config.task) == 'los':
                los_metrics_list = compute_regression_metrics(train_y_los, train_y_hat_los, verbose=0, elog=self.elog)

            if str.lower(self.config.task) == 'mortality':
                mort_metrics_list = compute_binary_metrics(train_y_mort, train_y_hat_mort, verbose=0, elog=self.elog)

            self.elog.print('Epoch: {} | Train Loss: {:3.4f}'.format(epoch, mean_train_loss))

        if str.lower(self.config.mode) == 'test':
            print('Done epoch {}'.format(epoch))

        if epoch == self.n_epochs - 1:
            if str.lower(self.config.mode) == 'train':
                self.save_checkpoint(file_name=self.file_name_last, epoch=epoch)
            if self.config.save_results_csv:
                if str.lower(self.config.task) == 'los':
                    self.elog.save_to_csv(np.vstack((train_y_hat_los, train_y_los)).transpose(),
                                          'train_predictions_los/epoch{}.csv'.format(epoch),
                                          header='los_predictions, label')

                if str.lower(self.config.task) == 'mortality':
                    self.elog.save_to_csv(np.vstack((train_y_hat_mort, train_y_mort)).transpose(),
                                          'train_predictions_mort/epoch{}.csv'.format(epoch),
                                          header='mort_predictions, label')
        return

    def validate(self, epoch):
        if str.lower(self.config.mode) == 'train':
            # self.model.eval()
            self.model.eval()
            val_batches = self.val_datareader.batch_gen(batch_size=self.config.batch_size_test)
            val_loss = []
            val_y_hat_los = np.array([])
            val_y_los = np.array([])
            val_y_hat_mort = np.array([])
            val_y_mort = np.array([])
            beta = 0.2

            for batch_idx, batch in tqdm(enumerate(val_batches), total=self.no_val_batches, desc="Validation", unit="batch"):
            # for batch in val_batches:
                if batch[0].size(0) <= 1:
                    print('Empty batch Val: ==========>',batch[0].shape)
                    continue

                # unpack batch
                timeseries, mask, flat, notes, edge_indexs, los_labels, mort_labels, seq_lengths = batch

                
                predction_output, output_sd_ts_rep, output_txt_rep = self.model(flat, timeseries, notes, edge_indexs)
                
                if str.lower(self.config.task) == 'mortality':
                    ce_loss = self.model.loss(predction_output.squeeze(-1), mort_labels)
                    cont_loss = self.model.contrastive_loss(output_sd_ts_rep,output_txt_rep)
                    loss = ce_loss*(1-beta)+cont_loss*(beta)
                else:    
                    msle_loss = self.model.loss(predction_output.squeeze(-1), los_labels)
                    cont_loss = self.model.contrastive_loss(output_sd_ts_rep,output_txt_rep)
                    loss = msle_loss*(1-beta)+cont_loss*(beta)

                val_loss.append(loss.item())  

                if str.lower(self.config.task) == 'los':
                    val_y_hat_los = np.append(val_y_hat_los, predction_output.detach().cpu().numpy())
                    val_y_los = np.append(val_y_los, los_labels.detach().cpu().numpy())
                if str.lower(self.config.task) == 'mortality':
                    val_y_hat_mort = np.append(val_y_hat_mort, predction_output.detach().cpu().numpy())
                    val_y_mort = np.append(val_y_mort, mort_labels.detach().cpu().numpy())

            print('Validation Metrics:')
            mean_val_loss = sum(val_loss) / len(val_loss)
            
            if str.lower(self.config.task) == 'los':
                los_metrics_list = compute_regression_metrics(val_y_los, val_y_hat_los, verbose=0, elog=self.elog)
                
                if self.config.save_results_csv:
                    self.elog.save_to_csv(np.vstack((val_y_hat_mort, val_y_mort)).transpose(),
                                        'val_predictions_los/epoch{}.csv'.format(epoch),
                                        header='los_predictions, label')
                    
                ##########################Saving Best Model#####################
                cur_msle = los_metrics_list[4]
                # cur_r2 = los_metrics_list[6]
                # cur_kapa = los_metrics_list[7]
                if cur_msle < self.max_msle:
                    self.max_msle = cur_msle
                    if str.lower(self.config.mode) == 'train':
                        self.save_checkpoint(file_name=self.file_name_best, epoch=epoch)
                    print('\n------------ Save best msle model ------------\n')
                    self.test(epoch)
                    self.patience = 0
                elif epoch == self.n_epochs - 1:
                    self.test(epoch)
                else:
                    self.patience = self.patience + 1
                    
            if str.lower(self.config.task) == 'mortality':
                # print(val_y_mort.shape)
                # print(val_y_hat_mort.shape)
                mort_metrics_list = compute_binary_metrics(val_y_mort, val_y_hat_mort, verbose=0, elog=self.elog)
                if self.config.save_results_csv:
                        self.elog.save_to_csv(np.vstack((val_y_hat_mort, val_y_mort)).transpose(),
                                            'val_predictions_mort/epoch{}.csv'.format(epoch),
                                            header='mort_predictions, label')
                ##########################Saving Best Model#####################
                cur_auroc_mort = mort_metrics_list[5] #5:'auroc', 6:'auprc', 7:'f1macro'
                # cur_auprc_plos = mort_metrics_list[6]
                # cur_f1_score_mort = mort_metrics_list[8]
                
                if cur_auroc_mort > self.max_auroc_mort:
                    self.max_auroc_mort = cur_auroc_mort
                    if str.lower(self.config.mode) == 'train':
                        self.save_checkpoint(file_name=self.file_name_best, epoch=epoch)
                    print('\n------------ Save best auroc model ------------\n')
                    self.test(epoch)
                    self.patience = 0
                elif epoch == self.n_epochs - 1 and self.config.task in ('mortality'):
                    self.test(epoch)
                else:
                    self.patience = self.patience + 1
                
            self.elog.print('Epoch: {} | Validation Loss: {:3.4f}'.format(epoch, mean_val_loss))          

        elif str.lower(self.config.mode) == 'test':
            self.test(epoch)
            
        return

    def test(self, epoch):
        if epoch == self.n_epochs - 1:
            ####################Load Best Checkpoint######################
            print('============> Loading Best Model ... ')
            checkpoint = torch.load(self.file_name_best)
            save_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_net'])
            self.optimiser.load_state_dict(checkpoint['optimiser'])
            ##############################################################           
        self.model.eval()
        test_batches = self.test_datareader.batch_gen(batch_size=self.config.batch_size_test)
        test_loss = []
        test_y_hat_los = np.array([])
        test_y_los = np.array([])
        test_y_hat_mort = np.array([])
        test_y_mort = np.array([])
        beta = 0.2
        
        test_mort_metrics_list = None
        test_los_metrics_list = None

        for batch_idx, batch in tqdm(enumerate(test_batches), total=self.no_test_batches, desc="Testing", unit="batch"):
        # for batch in test_batches:
            if batch[0].size(0) <= 1:
                print('Empty batch Test: ==========>',batch[0].shape)
                continue

            # unpack batch
            timeseries, mask, flat, notes, edge_indexs, los_labels, mort_labels, seq_lengths = batch

            predction_output, output_sd_ts_rep, output_txt_rep = self.model(flat, timeseries, notes, edge_indexs)            
            
            if str.lower(self.config.task) == 'mortality':
                ce_loss = self.model.loss(predction_output.squeeze(-1), mort_labels)
                cont_loss = self.model.contrastive_loss(output_sd_ts_rep,output_txt_rep)
                loss = ce_loss*(1-beta)+cont_loss*(beta)
            else:    
                msle_loss = self.model.loss(predction_output.squeeze(-1), los_labels)
                cont_loss = self.model.contrastive_loss(output_sd_ts_rep,output_txt_rep)
                loss = msle_loss*(1-beta)+cont_loss*(beta)
            
            test_loss.append(loss.item()) 

            if str.lower(self.config.task) == 'los':
                test_y_hat_los = np.append(test_y_hat_los, predction_output.detach().cpu().numpy())
                test_y_los = np.append(test_y_los, los_labels.detach().cpu().numpy())

            if str.lower(self.config.task) == 'mortality':
                test_y_hat_mort = np.append(test_y_hat_mort, predction_output.detach().cpu().numpy())
                test_y_mort = np.append(test_y_mort, mort_labels.detach().cpu().numpy())

        print('Test Metrics:')
        mean_test_loss = sum(test_loss) / len(test_loss)

        if str.lower(self.config.task) == 'los':
            test_los_metrics_list = compute_regression_metrics(test_y_los, test_y_hat_los, verbose=0, elog=self.elog)

        if str.lower(self.config.task) in ('mortality', 'multitask'):
            test_mort_metrics_list = compute_binary_metrics(test_y_mort, test_y_hat_mort, verbose=0, elog=self.elog)

        if self.config.save_results_csv:
            if self.config.task in ('mortality', 'multitask'):
                self.elog.save_to_csv(np.vstack((test_y_hat_mort, test_y_mort)).transpose(), 'val_predictions_mort.csv', header='mort_predictions, label')
        self.elog.print('Test Loss: {:3.4f}'.format(mean_test_loss))

        # write to file
        if epoch == self.n_epochs - 1:
            if str.lower(self.config.task) == 'los':
                with open(self.config.base_dir + '/results.csv', 'a') as f:
                    #test_los_metrics_list ==> [mad, mse, rmse, mape, msle, rmsle, r2, kappa]
                    mad = test_los_metrics_list[0]
                    mse = test_los_metrics_list[1]
                    rmse = test_los_metrics_list[2]
                    mape = test_los_metrics_list[3]
                    msle = test_los_metrics_list[4]
                    rmsle = test_los_metrics_list[5]
                    r2 = test_los_metrics_list[6]
                    kappa = test_los_metrics_list[7]
                    f.write('\n{},{},{},{},{},{},{},{}'.format(mad, mse, rmse, mape, msle, rmsle, r2, kappa))
                    
            elif str.lower(self.config.task) == 'mortality':
                with open(self.config.base_dir + '/results.csv', 'a') as f:
                    # test_mort_metrics_list ==> [acc, prec0, prec1, rec0, rec1, auroc, auprc, minpse, f1macro]
                    acc = test_mort_metrics_list[0]
                    prec0 = test_mort_metrics_list[1]
                    prec1 = test_mort_metrics_list[2]
                    rec0 = test_mort_metrics_list[3]
                    rec1 = test_mort_metrics_list[4]
                    auroc = test_mort_metrics_list[5]
                    auprc = test_mort_metrics_list[6]
                    minpse = test_mort_metrics_list[7]
                    f1macro = test_mort_metrics_list[8]
                    f.write('\n{},{},{},{},{},{},{},{},{}'.format(acc, prec0, prec1, rec0, rec1, auroc, auprc, minpse, f1macro))
        return

    def run(self,):
        for e in range(self.n_epochs):
            self.train(e)
            self.validate(e)
            if self.patience >= 5:
                self.test(self.n_epochs-1)
                break
        return
    
    def resume(self, epoch):
        return