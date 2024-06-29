
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, gc, json, getpass, sys

"""
Disclaimer: This code is Grammar VAE with a modification to input in sparse matrix format.
The original code can be found in the following repositories.
https://github.com/mkusner/grammarVAE
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import backend as K

from tqdm import tqdm
from tqdm.keras import TqdmCallback

ALPHA = 0.9
VALID_DIVIDE_NUMBER = 10

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class GramVAE(keras.Model):
    def __init__(self, encoder, decoder, mask, ind_of_ind, **kwargs):
        super(GramVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.mask = mask
        self.ind_of_ind = ind_of_ind
        _, self.used_rule, self.all_rule = self.encoder.input_shape

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, data, *args, **kwargs):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        return reconstruction

    def _conditional(self, x_true, x_pred):
        '''imported from gramvae'''
        most_likely = K.argmax(x_true)
        most_likely = tf.reshape(most_likely,[-1])
        ix2 = tf.expand_dims(tf.gather(self.ind_of_ind, most_likely),1)
        ix2 = tf.cast(ix2, tf.int32)
        M2 = tf.gather_nd(self.mask, ix2)

        M3 = tf.reshape(M2, [-1, self.used_rule, self.all_rule])
        P2 = tf.multiply(K.exp(x_pred),M3)
        P2 = tf.divide(P2,K.sum(P2,axis=-1,keepdims=True))
        return P2

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            x_decoded_mean = self._conditional(data, reconstruction)
            data = K.flatten(data)
            x_decoded_mean = K.flatten(x_decoded_mean)
            
            reconstruction_loss = self.all_rule * ALPHA * keras.losses.binary_crossentropy(data, x_decoded_mean)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = self.all_rule * (1-ALPHA) * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        x_decoded_mean = self._conditional(data, reconstruction)
        data = K.flatten(data)
        x_decoded_mean = K.flatten(x_decoded_mean)
            
        reconstruction_loss = self.all_rule * ALPHA * keras.losses.binary_crossentropy(data, x_decoded_mean)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = self.all_rule * (1-ALPHA) * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

class GramVAEWrapper():
    def __init__(self, input_shape, encode_dim, mask, ind_of_ind) -> None:
        self.n_sample, self.used_rule, self.all_rule = input_shape
        self.encode_dim = encode_dim
        self.loss_table = None
        self.mask = mask
        self.ind_of_ind = ind_of_ind

        self.make_model()
    
    def make_model(self):
        encoder_inputs = tf.keras.Input(shape=(self.used_rule, self.all_rule), batch_size=BATCH_SIZE) #, sparse=True)
        #to_dense = layers.Lambda(tf.sparse.to_dense, name='to_dense')(encoder_inputs)
        #to_dense.set_shape(shape=(1, self.used_rule, self.all_rule))
        x = layers.Conv1D(9, 9, activation='relu', name='conv_1')(encoder_inputs)
        x = layers.Conv1D(9, 9, activation='relu', name='conv_2')(x)
        x = layers.Conv1D(10, 11, activation='relu', name='conv_3')(x)
        x = layers.Flatten(name='flatten_1')(x)
        x = layers.Dense(435, activation='relu', name='dense_1')(x)

        z_mean = layers.Dense(self.encode_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.encode_dim, name="z_log_var")(x)

        z = Sampling(name="z")([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        print(encoder.summary())

        decoder_inputs = keras.Input(shape=(self.encode_dim,))
        x = layers.Dense(self.encode_dim, name='dense_2', activation = 'relu')(decoder_inputs)
        x = layers.RepeatVector(self.used_rule, name='repeat_vector')(x)
        x = layers.GRU(501, return_sequences = True, name='gru_1')(x)
        x = layers.GRU(501, return_sequences = True, name='gru_2')(x)
        x = layers.GRU(501, return_sequences = True, name='gru_3')(x)

        decoder_outputs = TimeDistributed(layers.Dense(self.all_rule), name='decoded_mean')(x)
        decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        #print(decoder.summary())

        self.vae = GramVAE(encoder, decoder, self.mask, self.ind_of_ind)
        self.vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1.0e-6))
        self.vae.build((None, self.used_rule, self.all_rule))
        self.vae.compute_output_shape(input_shape=(None, self.used_rule, self.all_rule))

    def load_model(self, prefix):
        if os.path.isfile(prefix+'_weights.index'):
            zero_grad = [tf.zeros_like(x) for x in self.vae.weights]
            self.vae.optimizer.apply_gradients(zip(zero_grad, self.vae.weights))
            self.vae.load_weights(prefix+'_weights')
            self.loss_table = pd.read_csv(prefix+'_history.csv', header=0, index_col=0)
            #self.vae.load_weights(prefix+'_weights').expect_partial()
        elif os.path.isdir(prefix+'_model'):
            self.vae = keras.models.load_model(prefix+'_model')
            self.loss_table = pd.read_csv(prefix+'_history.csv', header=0, index_col=0)
        
        elif os.path.isdir(prefix+'_checkpoint'):
            zero_grad = [tf.zeros_like(x) for x in self.vae.weights]
            self.vae.optimizer.apply_gradients(zip(zero_grad, self.vae.weights))
            self.vae.load_weights(prefix+'_checkpoint/ckpt')
        
        else:
            print('no trained model:', prefix)
            exit()
        
        self.loss_table = pd.read_csv(prefix+'_history.csv', header=0, index_col=0)
        
    def train(self, train, valid, n_sample, n_epoch=100, save_prefix='vae_model/gvae'):
        fit_callbacks = [
            keras.callbacks.EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1),
            #TqdmCallback(epochs=n_epoch, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='loss', verbose=1, factor=0.2, patience=2, min_lr=1.0e-10, mode='min'),
            keras.callbacks.ModelCheckpoint(filepath=save_prefix, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        ]

        n_valid = n_sample//VALID_DIVIDE_NUMBER
        n_train = n_sample - n_valid

        self.hypparam = {'batch_size': BATCH_SIZE,
                        'optimizer': self.vae.optimizer.get_config(),
                        'encode_dim': self.encode_dim,
                        'train': n_train,
                        'valid': n_valid,
                        'encoder': self.vae.encoder.get_config(),
                        'decoder': self.vae.decoder.get_config()}

        self.his = self.vae.fit(train,
                        epochs=n_epoch,
                        batch_size=BATCH_SIZE,
                        verbose=0,
                        callbacks=fit_callbacks,
                        #shuffle=True,
                        #steps_per_epoch=self.n_sample//BATCH_SIZE
                        steps_per_epoch=n_train//BATCH_SIZE,
                        validation_data=(valid, ),
                        validation_steps=n_valid//BATCH_SIZE
                    )

    def encode(self, data):
        z_mean, z_log_var, z = self.vae.encoder.predict(data)
        return z_mean

    def decode(self, arr):
        return self.vae.decoder.predict(arr)

    def evaluate(self, data):
        z_mean, z_log_var, z = self.vae.encoder.predict(data)
        reconstructed = self.vae.decoder.predict(z)
        return reconstructed
    
    def save(self, prefix='vae_model/gvae') -> None:
        #self.vae.save(prefix+'_model/')
        self.vae.save_weights(prefix+'_weights', save_format="tf")
        if self.loss_table is None:
            his_table = pd.DataFrame(self.his.history)
        else:
            his_table = pd.concat([self.loss_table, pd.DataFrame(self.his.history)])
            his_table.reset_index(inplace=True, drop=True)
        his_table.to_csv(prefix+'_history.csv')
        self.loss_table = his_table

        for k in self.hypparam.keys():
            item = self.hypparam[k]
            self.hypparam[k] = str(item)
        with open(prefix+'_hyperparam.json', mode='w') as f:
            json.dump(self.hypparam, f, indent=4)

    def plot_loss(self):
        if self.loss_table is not None:
            self.loss_table.plot()
        else:
            for k in self.his.history.keys():
                plt.plot(self.his.history[k], label=k)
        plt.savefig('vae_model/g2vcompress_loss.png')
        plt.show()

def load_json(prefix):
    with open('../gram/'+prefix+'/gram.json', mode='r') as f:
        gram = json.load(f)
        shape = (gram['sample_num'], gram['max_used_gram'], gram['all_gram'])

    data = np.load('../gram/'+prefix+'/mask.npz')
    mask = tf.cast(tf.Variable(data['mask'], trainable=False), tf.float32, name='mask')
    ind_of_ind = tf.cast(tf.Variable(data['ind_of_ind'], trainable=False), tf.float32, name='ind_of_ind')
    
    return shape, mask, ind_of_ind

def load_train(prefix, shape, shuffle=2048):
    def parse_train(example):
        description = {
            "data": tf.io.SparseFeature(index_key=['row','col'],
                                        value_key='value', dtype=tf.float32,
                                        size=shape[1:]),
            "label": tf.io.FixedLenFeature([], tf.string)
        }
        parsed = tf.io.parse_example(example, features=description)
        #return parsed['data']
        return tf.sparse.to_dense(parsed['data'])

    n_valid = shape[0]//VALID_DIVIDE_NUMBER
    dir = '../gram/'+prefix
    data = tf.data.TFRecordDataset([dir+'/'+f for f in os.listdir(path=dir) if 'sparse.tfrecords' in f], "GZIP").map(parse_train).shuffle(shuffle)
    train = data.skip(n_valid).batch(BATCH_SIZE).repeat()
    valid = data.take(n_valid).batch(BATCH_SIZE).repeat()

    return train, valid

def load_eval(prefix, shape):

    def parse_eval(example):
        description = {
            "data": tf.io.SparseFeature(index_key=['row','col'],
                                        value_key='value', dtype=tf.float32,
                                        size=shape[1:]),
            "label": tf.io.FixedLenFeature([], tf.string)
        }
        parsed = tf.io.parse_example(example, features=description)
        #return parsed['data'], parsed['label']
        return tf.sparse.to_dense(parsed['data']), parsed["label"]

    dir = '../gram/'+prefix
    eval_data = tf.data.TFRecordDataset([dir+'/'+f for f in os.listdir(path=dir) if 'sparse.tfrecords' in f], "GZIP").map(parse_eval)

    return eval_data

def gramvae(dim):
    tag = 'mc_mnist'
    shape, mask, ind_of_ind = load_json(tag)
    train, valid = load_train(tag, shape, shuffle=512)

    suffix = tag+'_gvae_'+str(dim)+'d'
    gvae = GramVAEWrapper(shape, dim, mask, ind_of_ind)
    exit()
    #gvae.load_model('vae_model/'+suffix)
    gvae.train(train, valid, shape[0], save_prefix='vae_model/'+suffix)
    gvae.save(prefix='vae_model/'+suffix)

    del train, valid
    gc.collect()

    eval_data = load_eval(tag, shape)
    encoded = np.empty((1,dim))
    label = []
    for d, l in eval_data.take(-1):
        e = gvae.encode(d[tf.newaxis, :, :])
        encoded = np.vstack((encoded, e))
        label.append(l.numpy().decode())

    encoded = np.delete(encoded, 0, axis=0)
    np.savez_compressed('encoded_'+suffix, encoded=encoded, label=np.array(label))
    print('FALL IN ONE:',np.all(encoded[0,:]==encoded[1,:]))
    print('NAN IN ARRAY:',np.any(np.isnan(encoded)))

def transition(dim):
    from utils import sort_split_label
    tag = 'mc_mnist'
    suffix = tag+'_gvae_'+str(dim)+'d'

    data = np.load('encoded_'+suffix+'.npz', allow_pickle=True)
    label, encoded = sort_split_label(data['label'], data['encoded'])

    shape, mask, ind_of_ind = load_json(tag)
    gvae = GramVAEWrapper(shape, dim, mask, ind_of_ind)
    gvae.load_model('vae_model/'+suffix)
    os.makedirs('decode_gvae/'+tag+'/transition/'+str(dim)+'d', exist_ok=True)

    def g2g(a,b,n=16):
        ga, gb = np.mean(a, axis=0), np.mean(b, axis=0)
        p = np.array([((15-x)*ga+x*gb)/(n-1) for x in range(n)])
        return p

    for i in range(len(label)):
        for j in range(i+1, len(label)):
            taskA, taskB = label[i], label[j]
            p = g2g(encoded[i], encoded[j])
            recon = gvae.decode(p)
            for k in range(recon.shape[0]):
                byte_arr = recon[k].flatten().tobytes()
                if np.all(np.isnan(recon[i])): print('ALL NAN')
                proto = tf.train.Example(features=tf.train.Features(feature={ \
                    "shape": tf.train.Feature(float_list=tf.train.FloatList(value=shape[1:])),
                    "recon": tf.train.Feature(bytes_list=tf.train.BytesList(value=[byte_arr])),
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=p[k])) \
                }))
                serialized = proto.SerializeToString()
                with tf.io.TFRecordWriter('decode_gvae/'+tag+'/transition/'+str(dim)+'d/'+taskA+'-'+taskB+'.tfrecords.'+str(k), tf.io.TFRecordOptions(compression_type="GZIP")) as writer:
                    writer.write(serialized)

def make_vec_from_z(z):
    shape, mask, ind_of_ind = load_json('mc_mnist')

    gvae = GramVAEWrapper(shape, 2, mask, ind_of_ind)
    gvae.load_model('vae_model/mc_mnist_2d')

    decoded = gvae.decode(z)
    np.savez_compressed('decoded_mnist_mc', decoded=decoded)

def reconstruction():
    tag = 'mc'
    shape, mask, ind_of_ind = load_json(tag)

    gvae = GramVAEWrapper(shape, 3, mask, ind_of_ind)
    gvae.load_model('vae_model/'+tag)

    eval_data = load_eval(tag, shape)
    for d, l in eval_data.take(-1):
        r = gvae.evaluate(d[tf.newaxis, :, :])
        label = l.numpy().decode()
        np.savez_compressed('decoded/mc/reconstructed/'+label, recon=r, label=label)

if __name__ == "__main__":
    gramvae(int(sys.argv[1]))
    transition(int(sys.argv[1]))
