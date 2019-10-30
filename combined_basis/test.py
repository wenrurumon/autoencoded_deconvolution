import csv
import sys
import os
from keras.layers import Lambda, Input, Dense, Dropout
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import numpy as np
import pandas as pd
from datetime import datetime
def mse_score(fit,actual):
        return np.mean(np.square(fit-actual))/np.mean(np.square(actual-np.mean(actual)))

def ae(X,Y,intermediate_dim=0,latent_dim=0,batch_size=256,epochs=100,verbose=0,validation_split=0.1):
        if intermediate_dim == 0: intermediate_dim = X.shape[1]
        if latent_dim == 0: latent_dim = int(np.floor(intermediate_dim/20))
        if batch_size == 0: batch_size = X.shape[0]
        input_dim = X.shape[1]
        output_dim = Y.shape[1]
        input_dim = X.shape[1]
        output_dim = Y.shape[1]
        ae_input = Input(shape=(input_dim,), name='encoder_input')
        ae_inter = Dropout(0.2)(Dense(intermediate_dim, activation='sigmoid')(ae_input))
        ae_latent = Dropout(0.2)(Dense(latent_dim, activation='sigmoid')(ae_inter))
        encoder = Model(ae_input,ae_latent,name='encoder')
        latent_input = Input(shape=(latent_dim,), name='latent')
        ae_inter2 = Dropout(0.2)(Dense(intermediate_dim, activation='sigmoid')(latent_input))
        ae_output = Dropout(0.2)(Dense(output_dim, activation='sigmoid')(ae_inter2))
        decoder = Model(latent_input,ae_output,name='decoder')
        ae = Model(ae_input, decoder(encoder(ae_input)), name='autoencoder')
        ae.compile(loss='mse', optimizer='adam')
        history = ae.fit(X,Y,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                verbose=verbose)
        return ae,encoder,decoder,history

def getarg(x):
        return np.int(x)

def minmax(x):
        x = np.array(x)
        for i in range(x.shape[1]):
                x[...,i] = (x[...,i]-np.min(x[...,i]))/(np.max(x[...,i])-np.min(x[...,i]))
        return(x)

bulk = minmax(pd.read_csv('/lustre/wangjc01/huzixin/deconv/data/bulk.csv'))
ref = minmax(pd.read_csv('/lustre/wangjc01/huzixin/deconv/data/reference.csv'))
cdata = np.concatenate((bulk,ref),axis=0)

argv = ['test.py','ae',25910,1024,200,128,50,1]
if argv[1] == 'ae':
  model = ae
elif argv[1] == 'vae':
  model = vae

Zsel = getarg(argv[2])
intermediate_dim = getarg(argv[3])
latent_dim = getarg(argv[4])
batch_size = getarg(argv[5])
epochs = getarg(argv[6])
verbose = getarg(argv[7])
Z = cdata[...,range(Zsel)]

