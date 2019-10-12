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
from datetime import datetime
def mse_score(fit,actual):
        return np.mean(np.square(fit-actual))
def autoencoder(X,Y,intermediate_dim=0,latent_dim=0,batch_size=256,epochs=100,verbose=0,validation_split=0.1):
        if intermediate_dim == 0: intermediate_dim = X.shape[1]
        if latent_dim == 0: latent_dim = int(np.floor(intermediate_dim/20))
        if batch_size == 0: batch_size = X.shape[0]
        input_dim = X.shape[1]
        output_dim = Y.shape[1]
        input_dim = X.shape[1]
        output_dim = Y.shape[1]
        ae_input = Input(shape=(input_dim,), name='encoder_input')
        ae_inter = Dropout(0.1)(Dense(intermediate_dim, activation='sigmoid')(ae_input))
        ae_latent = Dropout(0.1)(Dense(latent_dim, activation='sigmoid')(ae_inter))
        encoder = Model(ae_input,ae_latent,name='encoder')
        latent_input = Input(shape=(latent_dim,), name='latent')
        ae_inter2 = Dropout(0.1)(Dense(intermediate_dim, activation='sigmoid')(latent_input))
        ae_output = Dropout(0.1)(Dense(output_dim, activation='sigmoid')(ae_inter2))
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
def dicts(x,y):
        rlt = {}
        for k,v in x.items():
                rlt[k] = v
        for k,v in y.items():
                rlt[k] = v
        return rlt
data = []
for line in csv.reader(open('bulk.csv','r')):
        data.append(line)
bulk_data = np.array(data[1:],dtype='float')
for i in range(bulk_data.shape[1]):
        bulk_data[...,i] = (bulk_data[...,i]-np.min(bulk_data[...,i]))/(np.max(bulk_data[...,i])-np.min(
bulk_data[...,i]))
bulk_label = data[0]
argv = sys.argv
Zsel = getarg(argv[1])
latent_dim = getarg(argv[2])
batch_size = getarg(argv[3])
epochs = getarg(argv[4])
verbose = getarg(argv[5])
Z = bulk_data[...,range(Zsel)]
t = datetime.now()
model, encoder, decoder, history = autoencoder(Z,Z,latent_dim=latent_dim,batch_size=batch_size,epochs=epochs,verbose=verbose)
history = dicts(history.params,history.history)
history['time'] = (datetime.now()-t).seconds
history['argv'] = argv
fo = open('/lustre/wangjc01/huzixin/deconv/log/log_%s.log' % np.int(np.int(datetime.now().timestamp()*1000000)), "w")
for k in history:
        fo.write('%s: %s\n' % (k, history[k]))
fo.close()
