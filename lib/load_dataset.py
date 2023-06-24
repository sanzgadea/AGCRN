import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'METR-LA':
        data_path = os.path.join('../data/METR-LA/metr-la.npz')
        data = np.load(data_path)['data'][:, :]  #METR-LA only has one dimension
    elif dataset == 'METR-LA_small':
        data_path = os.path.join('../data/METR-LA/metr-la_small.npz')
        data = np.load(data_path)['data'][:, :]  #METR-LA only has one dimension
    elif dataset == 'PEMSD4':
        data_path = os.path.join('../data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #only the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PeMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #only the first dimension, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
