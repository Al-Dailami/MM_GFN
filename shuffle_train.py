import pandas as pd
from sklearn.utils import shuffle

def shuffle_stays(stays, seed=666):
    return shuffle(stays, random_state=seed)

def process_table(table_name, table, stays, folder_path):
    table = table.loc[stays].copy()
    table.to_csv('{}/{}.csv'.format(folder_path, table_name))
    return

def shuffle_train(train_path):

    labels = pd.read_csv(train_path + '/labels.csv', index_col='patient')
    flat = pd.read_csv(train_path + '/flat.csv', index_col='patient')
    timeseries = pd.read_csv(train_path + '/timeseries.csv', index_col='patient')

    notes_txts_lst = pd.read_json(train_path + 'notes_txts_lst.json', orient='split')
    notes_txts_lst.set_index('patient', inplace=True)

    stays = labels.index.values
    stays = shuffle_stays(stays, seed=None)  # No seed will make it completely random
    for table_name, table in zip(['labels', 'flat', 'timeseries', 'notes_txts_lst'],
                                 [labels, flat, timeseries, notes_txts_lst]):
        process_table(table_name, table, stays, train_path)

    with open(train_path + '/stays.txt', 'w') as f:
        for stay in stays:
            f.write("%s\n" % stay)
    return