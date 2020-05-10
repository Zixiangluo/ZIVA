import numpy as np
import tensorflow as tf
import losses
import model
import fitting


def sample(model_vae, expr_in):
    mean, logvar = model_vae.encode(expr_in)
    z = model_vae.sample(mean, logvar)
    expr_out = model_vae.decode(z)
    mask = model_vae.mask(model_vae.f, model_vae.lmd, expr_out)
    expr_out_dropout = model_vae.dropout(mask, expr_out)
    return expr_out, expr_out_dropout, mean, logvar, mask


# Training method
def update_model(model_vae,
                 data_batch,
                 optimizer,
                 loss_func):

    with tf.GradientTape(persistent=True) as tape:
        expr_out, expr_out_dropout, mean, logvar, mask = sample(model_vae, data_batch)
        rec_loss, kl_loss, rank_loss = loss_func(data_batch, mean, logvar, expr_out, expr_out_dropout)
        loss_val = rec_loss + kl_loss + rank_loss*0.001

    # Compute gradients and update model
    grad = tape.gradient(loss_val, model_vae.trainable_variables)
    optimizer.apply_gradients(zip(grad, model_vae.trainable_variables))
    return rec_loss, kl_loss, rank_loss


def train(expr_in,
          vae_lr=1e-4,
          epochs=500,
          info_step=10,
          batch_size=50,
          latent_dim=2,
          f="nb",
          log=True,
          scale=True):
    # Preprocessing
    expr_in[expr_in < 0] = 0.0

    if log:
        expr_in = np.log2(expr_in + 1)
    if scale:
        for i in range(expr_in.shape[0]):
            expr_in[i, :] = expr_in[i, :] / np.max(expr_in[i, :])


    # Number of data samples
    n_sam = expr_in.shape[0]
    # Dimension of input data
    in_dim = expr_in.shape[1]
    # Build VAE model and its optimizer
    lmd = fitting.fit(expr_in, f)
    model_vae = model.VAE(in_dim=in_dim, latent_dim=latent_dim, f=f, lmd=lmd)
    optimizer_vae = tf.keras.optimizers.Adam(vae_lr)

    # Training
    for epoch in range(1, epochs + 1):
        # Minibatch for VAE training
        vae_train_set = tf.data.Dataset.from_tensor_slices(expr_in).shuffle(n_sam).batch(batch_size)
        # Batch training
        for vae_batch in vae_train_set:
            # Update VAE model
            rec_loss, kl_loss, rank_loss = update_model(model_vae,
                                    vae_batch,
                                    optimizer_vae,
                                    losses.vae_loss)
        # Print training info
        if epoch % info_step == 0:
            print("Epoch", epoch, " rec_loss: ", rec_loss.numpy(), " kl_loss: ", kl_loss.numpy(), " rank_loss: ", rank_loss.numpy())

    return model_vae
