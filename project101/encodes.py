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

ref_data = np.array(data[1:],dtype='float')
for i in range(ref_data.shape[1]):
        ref_data[...,i] = (ref_data[...,i]-np.min(ref_data[...,i]))/(np.max(ref_data[...,i])-np.min(ref_data[...,i]))

ref_label = data[0]

models = ['ae_25910_1024_250_256_4000_2',
'ae_25910_1024_400_256_4000_2',
'vae_25910_1024_250_256_4000_2',
'vae_25910_1024_401_256_4000_2']

def encoding(m,x):
  m_encoder = '/lustre/wangjc01/huzixin/deconv/log/rlt/%s.encoder'%m
  encoder = load_model(m_encoder)
  bulk_encoder = encoder.predict(x)
  return(bulk_encoder)

models_encoded = []
for m in models:
  models_encoded.append(encoding(m,bulk_data))

ref_encoded = []
for m in models:
  ref_encoded.append(encoding(m,ref_data))

models_encoded[2] = models_encoded[2][2]
models_encoded[3] = models_encoded[3][2]
ref_encoded[2] = ref_encoded[2][2]
ref_encoded[3] = ref_encoded[3][2]

for i in range(4):
  pd.DataFrame(models_encoded[i]).to_csv('/lustre/wangjc01/huzixin/deconv/log/rlt/%s.bulk_encoded'%models[i],index=0)
  pd.DataFrame(ref_encoded[i]).to_csv('/lustre/wangjc01/huzixin/deconv/log/rlt/%s.ref_encoded'%models[i],index=0)
