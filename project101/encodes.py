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
def mse_score(fit,actual):
        return np.mean(np.square(fit-actual))

data = []
for line in csv.reader(open('/lustre/wangjc01/huzixin/deconv/data/bulk.csv','r')):
        data.append(line)

bulk_data = np.array(data[1:],dtype='float')
for i in range(bulk_data.shape[1]):
        bulk_data[...,i] = (bulk_data[...,i]-np.min(bulk_data[...,i]))/(np.max(bulk_data[...,i])-np.min(bulk_data[...,i]))

bulk_label = data[0]
rlt = []
models = ['ae_25910_1024_250_256_4000_2',
'ae_25910_1024_400_256_4000_2',
'vae_25910_1024_250_256_4000_2',
'vae_25910_1024_401_256_4000_2']

def encoding(m):
  m_encoder = '%s.encoder'%m
  encoder = load_model(m_encoder)
  bulk_encoder = encoder.predict(bulk_data)
  rlt.append(bulk_encoder)

models_encoded = []
for m in models:
  models_encoded.append(encoding(m))



