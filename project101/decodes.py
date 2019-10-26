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
        return np.mean(np.square(fit-actual))

data = []
for line in csv.reader(open('/lustre/wangjc01/huzixin/deconv/data/bulk.csv','r')):
        data.append(line)

bulk_data = np.array(data[1:],dtype='float')
for i in range(bulk_data.shape[1]):
        bulk_data[...,i] = (bulk_data[...,i]-np.min(bulk_data[...,i]))/(np.max(bulk_data[...,i])-np.min(bulk_data[...,i]))

bulk_label = data[0]
data = []
for line in csv.reader(open('/lustre/wangjc01/huzixin/deconv/data/reference.csv','r')):
        data.append(line)

models = ['ae_25910_1024_250_256_4000_2',
'ae_25910_1024_400_256_4000_2',
'vae_25910_1024_250_256_4000_2',
'vae_25910_1024_401_256_4000_2']

def decoding(m,x):
  m_decoder = '/lustre/wangjc01/huzixin/deconv/log/rlt/%s.decoder'%m
  decoder = load_model(m_decoder)
  bulk_decoder = decoder.predict(x)
  return(bulk_decoder)


np.set_printoptions(precision=20)
models_decoded = []
for m in models:
  x = np.loadtxt('%s.fit_encoded'%m,delimiter=',',dtype=str)[1:,...]
  x = np.array(x,dtype='float128')
  models_decoded.append(decoding(m,x))

for i in models_decoded:
  mse_score(i,bulk_data

for i in range(4):
  pd.DataFrame(models_decoded[i]).to_csv('/lustre/wangjc01/huzixin/deconv/log/rlt/%s.bulk_fit'%models[i],index=0)
