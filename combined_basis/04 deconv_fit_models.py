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

bulk_file = ['/lustre/wangjc01/huzixin/deconv/data/bulk.csv',
             '/lustre/wangjc01/huzixin/deconv/data/bulk_adni.csv',
             '/lustre/wangjc01/huzixin/deconv/data/bulk_lmy.csv']

def getbulk(x):
        bulk_data = pd.read_csv(x)
        bulk_data = np.array(bulk_data,dtype='float')
        for i in range(bulk_data.shape[1]):
                bulk_data[...,i] = (bulk_data[...,i]-np.min(bulk_data[...,i]))/(np.max(bulk_data[...,i])-np.min(bulk_data[...,i]))
        return(bulk_data)

bulk_rush = getbulk(bulk_file[0])
bulk_adni = getbulk(bulk_file[1])
bulk_lmy = getbulk(bulk_file[2])

def patternfiles(x):
        rlt = []
        for i in os.listdir():
                if x in i: rlt.append(i)
        return(rlt)

def fit_bulk(decoder,fits,bulk):
        fits = patternfiles(fits)
        m_decoder = load_model(decoder)
        bulk_fit = []
        for i in range(len(fits)):
                bulk_fit.append([decoder,fits[i],mse_score(m_decoder.predict(np.array(pd.read_csv(fits[i]))),bulk)])
        return(bulk_fit)

model_list = [['vae_25910_1024_250_256_10000_2_20191031130042.decoder','_vae_rush',bulk_rush],
              ['ae_25910_1024_250_256_10000_2_20191030224038.decoder','_ae_rush',bulk_rush],
              ['ae_16055_1024_300_256_10000_2_20191031163134.decoder','_ae_adni',bulk_adni],
              ['vae_16055_1024_300_256_10000_2_20191031175449.decoder','_vae_adni',bulk_adni],
              ['vae_17121_1024_400_128_10000_2_20191109165805.decoder','_vae_lmy',bulk_lmy],
              ['ae_17121_1024_300_128_10000_2_20191109163844.decoder','_ae_lmy',bulk_lmy]]

i = model_list[0]
rlt1 = fit_bulk(i[0],i[1],i[2])
for i in model_list[1:]:
        temp = fit_bulk(i[0],i[1],i[2])
        for j in temp:
                rlt1.append(j)

pd.DataFrame(np.array(rlt1)).to_csv('mse_score.csv')
