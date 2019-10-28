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

data = []
for line in csv.reader(open('/lustre/wangjc01/huzixin/deconv/data/bulk_adni.csv','r')):
        data.append(line)

bulk_data = np.array(data[1:],dtype='float')
for i in range(bulk_data.shape[1]):
        bulk_data[...,i] = (bulk_data[...,i]-np.min(bulk_data[...,i]))/(np.max(bulk_data[...,i])-np.min(bulk_data[...,i]))

bulk_label = data[0]
data = []
for line in csv.reader(open('/lustre/wangjc01/huzixin/deconv/data/ref_adni.csv','r')):
        data.append(line)

ref_data = np.array(data[1:],dtype='float')
for i in range(ref_data.shape[1]):
        ref_data[...,i] = (ref_data[...,i]-np.min(ref_data[...,i]))/(np.max(ref_data[...,i])-np.min(ref_data[...,i]))

ref_label = data[0]

models = ['ae_16055_1024_200_256_4000_2',
'ae_16055_1024_250_256_4000_2',
'ae_16055_1024_300_256_4000_2',
'ae_16055_1024_350_256_4000_2',
'vae_16055_1024_200_256_4000_2',
'vae_16055_1024_250_256_4000_2',
'vae_16055_1024_300_256_4000_2',
'vae_16055_1024_350_256_4000_2']

def encoding(m,x):
  m_encoder = '/lustre/wangjc01/huzixin/deconv/log/rlt_adni/%s.encoder_adni'%m
  encoder = load_model(m_encoder)
  bulk_encoder = encoder.predict(x)
  return(bulk_encoder)

models_encoded = []
for m in models:
  models_encoded.append(encoding(m,bulk_data))

for i in range(4,8):
  models_encoded[i] = models_encoded[i][2]

def decoding(m,x):
  m_decoder = '/lustre/wangjc01/huzixin/deconv/log/rlt_adni/%s.decoder_adni'%m
  decoder = load_model(m_decoder)
  bulk_decoder = decoder.predict(x)
  return(bulk_decoder)

models_decoded = []
for i in range(8):
  models_decoded.append(decoding(models[i],models_encoded[i]))

for i in models_decoded:
  mse_score(i,bulk_data)
