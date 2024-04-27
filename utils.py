import pandas as pd
import numpy as np
from sklearn.utils import shuffle

import torch

import os


def shuffle_stays(stays, seed=9):
    return shuffle(stays, random_state=seed)

def process_table(table_name, table, stays, folder_path):
    table = table.loc[stays].copy()
    table.to_csv('{}/{}.csv'.format(folder_path, table_name))
    return

def shuffle_train(train_path):

    labels = pd.read_csv(train_path + '/labels.csv', index_col='patient')
    flat = pd.read_csv(train_path + '/flat.csv', index_col='patient')
    diagnoses = pd.read_csv(train_path + '/diagnoses.csv', index_col='patient')
    timeseries = pd.read_csv(train_path + '/timeseries.csv', index_col='patient')

    stays = labels.index.values
    stays = shuffle_stays(stays, seed=None)  # No seed will make it completely random
    for table_name, table in zip(['labels', 'flat', 'diagnoses', 'timeseries'],
                                 [labels, flat, diagnoses, timeseries]):
        process_table(table_name, table, stays, train_path)

    with open(train_path + '/stays.txt', 'w') as f:
        for stay in stays:
            f.write("%s\n" % stay)
    return


def create_folder(parent_path, folder):
    if not parent_path.endswith('/'):
        parent_path += '/'
    folder_path = parent_path + folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path
# view the results by running: python3 -m trixi.browser --port 8080 BASEDIR

def save_to_csv(save_dir, data, path, header=None):
    """
        Saves a numpy array to csv in the experiment save dir

        Args:
            data: The array to be stored as a save file
            path: sub path in the save folder (or simply filename)
    """

    folder_path = create_folder(save_dir, os.path.dirname(path))
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


    