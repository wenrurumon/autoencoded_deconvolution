import csv
import sys
import os
from keras.layers import Lambda, Input, Dense, Dropout
from keras.models import Model, load_model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import numpy as np
from datetime import datetime
import pandas as pd

def mse_score(fit,actual):
        return np.mean(np.square(fit-actual))/np.mean(np.square(actual-np.mean(actual)))

bulk_data = pd.read_csv('/lustre/wangjc01/huzixin/deconv/data/bulk.csv')
bulk_data = np.array(bulk_data,dtype='float')
for i in range(bulk_data.shape[1]):
        bulk_data[...,i] = (bulk_data[...,i]-np.min(bulk_data[...,i]))/(np.max(bulk_data[...,i])-np.min(bulk_data[...,i]))

model = '/lustre/wangjc01/huzixin/deconv/log/rlt_rush/ae_25910_1024_250_256_10000_2_20191030224038.decoder'
fits = ['lmy_ae_rush.bulk_fit','stf_ae_rush.bulk_fit','nnls_ae_rush.bulk_fit','own_ae_rush.bulk_fit']
for i in fits:
  fitted = decoder.predict(pd.read_csv(i))
  print(mse_score(fitted,bulk_data))

