
###########################################################################
# vae.py
# argv = syntax, Zsel, latent_dim, batch_size, epochs
# argv = ['test.py','250','200','128','10','1']
# python3 vae.py 25910 200 256 200 0&
###########################################################################

import csv
import sys
import os
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import numpy as np
from datetime import datetime
def mse_score(fit,actual):
    return np.mean(np.square(fit-actual))
def getarg(x):
    return np.int(x)
def dicts(x,y):
    rlt = {}
    for k,v in x.items():
        rlt[k] = v
    for k,v in y.items():
        rlt[k] = v
    return rlt
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
def vae(X,Y,intermediate_dim=0,latent_dim=0,batch_size=256,epochs=100,verbose=0,validation_split=0.1):
    if intermediate_dim == 0: intermediate_dim = X.shape[1]
    if latent_dim == 0: latent_dim = int(np.floor(intermediate_dim/20))
    if batch_size == 0: batch_size = X.shape[0] 
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    inputs = Input(shape=(input_dim, ), name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(output_dim, activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    vae = Model(inputs, decoder(encoder(inputs)[2]), name='vae_mlp')
    vae.compile(loss='mse', optimizer='adam')
    history = vae.fit(Z,Z,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=verbose)
    return vae,encoder,decoder,history
data = []
for line in csv.reader(open('bulk.csv','r')):
    data.append(line)
bulk_data = np.array(data[1:],dtype='float')
for i in range(bulk_data.shape[1]):
    bulk_data[...,i] = (bulk_data[...,i]-np.min(bulk_data[...,i]))/(np.max(bulk_data[...,i])-np.min(bulk_data[...,i]))
bulk_label = data[0]
argv = sys.argv
Zsel = getarg(argv[1])
latent_dim = getarg(argv[2])
batch_size = getarg(argv[3])
epochs = getarg(argv[4])
verbose = getarg(argv[5])
Z = bulk_data[...,range(Zsel)]
t = datetime.now()
model, encoder, decoder, history = vae(Z,Z,latent_dim=latent_dim,batch_size=batch_size,epochs=epochs,verbose=verbose)  
history = dicts(history.params,history.history)
history['time'] = (datetime.now()-t).seconds
history['argv'] = argv
fo = open('/lustre/wangjc01/huzixin/deconv/log/vae_%s.log' % np.int(np.int(datetime.now().timestamp()*1000000)), "w")
for k in history:
    fo.write('%s: %s\n' % (k, history[k]))
fo.close()