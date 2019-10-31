import csv
import sys
import os
import time
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
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
def vae(X,Y=0,intermediate_dim=0,latent_dim=0,batch_size=256,epochs=100,verbose=0,validation_split=0.1):
    if intermediate_dim == 0: intermediate_dim = X.shape[1]
    if latent_dim == 0: latent_dim = int(np.floor(intermediate_dim/20))
    if batch_size == 0: batch_size = X.shape[0] 
    input_dim = X.shape[1]
    output_dim = X.shape[1]
    inputs = Input(shape=(input_dim,), name='encoder_input')
    x = Dropout(0.2)(Dense(intermediate_dim, activation='sigmoid')(inputs))
    z_mean = Dropout(0.2)(Dense(latent_dim, name='z_mean')(x))
    z_log_var = Dropout(0.2)(Dense(latent_dim, name='z_log_var')(x))
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dropout(0.2)(Dense(intermediate_dim, activation='sigmoid')(latent_inputs))
    outputs = Dropout(0.2)(Dense(input_dim, activation='sigmoid')(x))
    decoder = Model(latent_inputs, outputs, name='decoder')
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    history = vae.fit(X,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=verbose)
    return vae,encoder,decoder,history
def getarg(x):
        return np.int(x)
def dicts(x,y):
        rlt = {}
        for k,v in x.items():
                rlt[k] = v
        for k,v in y.items():
                rlt[k] = v
        return rlt
def minmax(x):
        x = np.array(x)
        for i in range(x.shape[1]):
                x[...,i] = (x[...,i]-np.min(x[...,i]))/(np.max(x[...,i])-np.min(x[...,i]))
        return(x)
bulk = minmax(pd.read_csv('/lustre/wangjc01/huzixin/deconv/data/bulk.csv'))
ref = minmax(pd.read_csv('/lustre/wangjc01/huzixin/deconv/data/reference.csv'))
cdata = np.concatenate((bulk,ref),axis=0)
#argv = ['test.py','ae',25910,1024,200,128,50,1]
argv = sys.argv
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
t = datetime.now()
model, encoder, decoder, history = model(Z,Z,intermediate_dim=intermediate_dim,latent_dim=latent_dim,batch_size=batch_size,epochs=epochs,verbose=verbose)
history = dicts(history.params,history.history)
history['time'] = (datetime.now()-t).seconds
history['argv'] = argv
bulk = bulk[...,range(Zsel)]
ref = ref[...,range(Zsel)]
if argv[1] = 'ae':
        bulk_encoded = encoder.predict(bulk)
        ref_encoded = encoder.predict(ref)
elif argv[1] == 'vae':
        bulk_encoded = encoder.predict(bulk)[2]
        ref_encoded = encoder.predict(ref)[2]
history['mse'] = [mse_score(model.predict(Z),Z),mse_score(decoder.predict(bulk_encoded),bulk),mse_score(decoder.predict(ref_encoded),ref)]
fo = "%s_%s" % ('_'.join([str(i) for i in argv[1:]]),time.strftime('%Y%m%d%H%M%S',time.localtime(time.time())))
encoder.save('/lustre/wangjc01/huzixin/deconv/log/rlt_rush/%s.encoder' % fo)
decoder.save('/lustre/wangjc01/huzixin/deconv/log/rlt_rush/%s.decoder' % fo)
pd.DataFrame(bulk_encoded).to_csv('/lustre/wangjc01/huzixin/deconv/log/rlt_rush/%s.bulk_encoded'%fo,index=0)
pd.DataFrame(ref_encoded).to_csv('/lustre/wangjc01/huzixin/deconv/log/rlt_rush/%s.ref_encoded'%fo,index=0)
fo = open('/lustre/wangjc01/huzixin/deconv/log/rlt_rush/%s.rlt' % fo, "w")
for k in history:
        fo.write('%s: %s\n' % (k, history[k]))
fo.close()


