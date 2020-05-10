import tensorflow as tf
import numpy as np


# def count_loss(data_in, data_out, param_count):
#     # Total dataession count in original gene dataession data (input)
#     count_in = tf.math.reduce_sum(data_in, 1)
#     # Total dataession count in recovered gene dataession (with dropout removed)
#     count_out = tf.math.reduce_sum(data_out, 1)
#
#     return tf.math.squared_difference(param_count*count_in, count_out)


# def entropy_loss(data_in, data_out, param_ent):
#     # Entropy of original gene dataession data (input)
#     ent_in = tf.nn.softmax_cross_entropy_with_logits(data_in, data_in, 1)
#     # Entropy of recovered gene dataession (with dropout removed)
#     ent_out = tf.nn.softmax_cross_entropy_with_logits(data_out, data_out, 1)
#
#     return tf.math.squared_difference(param_ent*ent_in, ent_out)


def vae_loss(data_in,
             mean,
             logvar,
             data_out,
             data_out_dropout):
    # (Optional) compute reconstruction loss with ZI model
    # out = data_out_dropout if use_zi else data_out
    # rec_loss = tf.reduce_sum(tf.math.squared_difference(data_in, out), 1)
    #tf.print("data in:")
    #tf.print(data_in)
    #tf.print("data out:")
    #tf.print(data_out)
    rec_loss = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(data_in, data_out), 1))
    #print(tf.shape(data_in))
    #print(tf.shape(data_out))
    #print(tf.shape(tf.keras.losses.binary_crossentropy(data_in, data_out)))
    #print(tf.shape(tf.math.squared_difference(data_in, data_out)))
    #print(tf.shape(tf.reduce_sum(tf.math.squared_difference(data_in, data_out), 1)))
    #rec_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(data_in, data_out)) * tf.cast(tf.shape(data_in)[1], tf.float32)
    # Compute KL divergence loss
    kl_loss = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), 1))
    # (Optional) compute rank loss
    #rank_loss = tf.cast(tf.linalg.trace(data_out_dropout), tf.float32) * 10
    rank_loss = tf.linalg.trace(tf.math.sqrt(tf.matmul(data_out, data_out, transpose_a=True) + 1e-20))
    #rank_loss = np.linalg.norm(data_out, ord="nuc")
    #rank_loss = tf.cast(tf.linalg.matrix_rank(data_out), tf.float32)
    #rank_loss = tf.reduce_sum(tf.linalg.svd(data_out, compute_uv=False))
    # (Optional) compute count loss
    # c_loss = count_loss(data_in, data_out, param_count) if use_cl else 0
    # (Optional) compute entropy loss
    # ent_loss = entropy_loss(data_in, data_out, param_ent) if use_el else 0
    #print("rec_loss: ", rec_loss, " kl_loss: ", kl_loss, " rank_loss: ", rank_loss)
    # return tf.reduce_mean(rec_loss + kl_loss + rank_loss + c_loss + ent_loss)
    #return tf.reduce_mean(rec_loss + kl_loss + rank_loss)
    #print(rec_loss.numpy())
    #print(kl_loss.numpy())
    #print(rank_loss.numpy())
    return rec_loss, kl_loss, rank_loss


# def zi_loss(data_in,
#             data_zi,
#             zi_type,
#             **args):
#     # True dropout from original gene dataession data (0 if dropout, 1 else)
#     condition = tf.math.greater(data_in, 0)
#     data_true_dropout = tf.where(condition, 1, data_in)
#     # Compute loss for ZI model
#     if zi_type == "binary_crossentropy":
#         zi_loss = tf.keras.losses.binary_crossentropy(data_zi, data_true_dropout)
#     if zi_type == "mse":
#         zi_loss = tf.math.reduce_sum(tf.math.squared_difference(data_zi, data_true_dropout), 1)
#     # Compute sparsity loss
#     # sparsity_loss = tf.norm(data_zi, ord=1)
#     sparsity_loss = 0
#
#     return tf.math.reduce_mean(zi_loss + sparsity_loss)
