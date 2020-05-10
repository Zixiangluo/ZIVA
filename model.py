# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import regularizers
from tensorflow.keras.layers import InputLayer, Dense, Dropout, BatchNormalization, Activation


# Versatile autoencoder network
class VAE(tf.keras.Model):

    # Build model
    def __init__(self, in_dim, latent_dim, f, lmd):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.in_dim = in_dim
        self.f = f
        self.lmd = lmd
        self.encoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(in_dim)),
                Dropout(0.2),
                Dense(units=512, kernel_regularizer=regularizers.l1(0.01)),
                BatchNormalization(),
                Dense(units=128),
                BatchNormalization(),
                Activation("relu"),
                Dense(units=32),
                BatchNormalization(),
                Activation("relu"),
                Dense(latent_dim + latent_dim)
            ], name="encoder_net")

        self.decoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dim)),
                Dense(units=32),
                BatchNormalization(),
                Activation("relu"),
                Dense(units=128),
                BatchNormalization(),
                Activation("relu"),
                Dense(units=512),
                BatchNormalization(),
                Activation("relu"),
                Dense(units=in_dim),
                # Activation("sigmoid")
                Activation("relu")
            ], name="decoder_net")

    # Get latent variable from encoder
    def encode(self, input):
        mean, logvar = tf.split(self.encoder(input), num_or_size_splits=2, axis=1)
        return mean, logvar

    # Sampling from latent variable
    def sample(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        sampled_var = eps * tf.exp(logvar * .5) + mean
        return sampled_var

    # Get output from decoder
    def decode(self, latent_var):
        return self.decoder(latent_var)

    def mask(self, f, lmd, logits):
        if (f == "nb"):
            p = tf.exp(-lmd * tf.square(logits))
        elif (f == "mm"):
            p = 1 - logits / (lmd + logits)
        else:
            p = 0
        q = 1 - p
        pq = tf.stack([p + 1e-20, q + 1e-20], axis=-1)
        s = tfp.distributions.RelaxedOneHotCategorical(temperature=0.5, probs=pq).sample()
        mask = s[:, :, 1]
        return mask

    def dropout(self, mask, logits):
        return tf.multiply(mask, logits)


def gumbel_softmax(logits):
    eps = 1e-8
    shape = tf.shape(logits)
    u = tf.random.uniform(shape)
    g = -tf.math.log(-tf.math.log(u + eps) + eps)
    z = logits + g
    return tf.nn.softmax(z / 0.5)


'''
    def mask(self, f, lmd, logits):
        if (f == "nb"):
            p = tf.exp(-lmd * tf.square(logits))
        elif (f == "mm"):
            p = 1 - logits / (lmd + logits)
        else:
            p = 0
        q = 1 - p
        logp = tf.math.log(p + 1e-20)
        logq = tf.math.log(q + 1e-20)
        logpq = tf.stack([logp, logq], axis=-1)
        s = gumbel_softmax(logpq)
        mask = s[:, :, 1]
        return mask
'''
