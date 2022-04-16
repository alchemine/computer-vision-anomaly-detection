from analysis_tools.common import *
from util import *

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import sklearn
import cv2
import os

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
sklearn.random.seed(RANDOM_STATE)

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# strategy = tf.distribute.MirroredStrategy()

class_sample = 'screw'

@delayed
def load_img(path, size):
    return cv2.resize(cv2.imread(path), (size, size))

SIZE        = 256
INPUT_SHAPE = (SIZE, SIZE, 3)
train_full_data_meta = pd.read_csv(join(PATH.input, 'train_df.csv'), index_col=0)
train_full_data_meta = train_full_data_meta.query(f"`class` == '{class_sample}'")
paths                = train_full_data_meta['file_name']
with ProgressBar():
    X_train_full = np.array(compute(*[load_img(join(PATH.train, path), SIZE) for path in paths]), dtype=np.float32)
    y_train_full = train_full_data_meta['state'].values


from sklearn.model_selection import train_test_split

X_normal   = X_train_full[y_train_full == 'good']
X_abnormal = X_train_full[y_train_full != 'good']
X_train, X_test = train_test_split(X_normal)
X_train, X_val  = train_test_split(X_train)
X_test = np.concatenate([X_test, X_abnormal])

X_train = X_train / 255
X_val   = X_val / 255
X_test  = X_test / 255

print("X_train:", X_train.shape)  # normal
print("X_val:", X_val.shape)      # noraml + abnormal
print("X_test:", X_test.shape)    # noraml + abnormal


BATCH_SIZE = 32

aug_model = keras.models.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

def preprocess(ds, training, batch_size, augment=True):
    ds = ds.cache().batch(batch_size)
    if training:
        ds = ds.shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(lambda x, y: (aug_model(x), aug_model(y)), num_parallel_calls=tf.data.AUTOTUNE)
    return ds

ds_train = preprocess(tf.data.Dataset.from_tensor_slices(X_train), True, BATCH_SIZE, augment=False)
ds_val   = preprocess(tf.data.Dataset.from_tensor_slices(X_val), False, BATCH_SIZE, augment=False)

from tensorflow.keras import layers
from functools import partial


conv = partial(layers.Conv2D, kernel_size=3, strides=2, padding='same', kernel_initializer='lecun_normal', activation='selu')
convt = partial(layers.Conv2DTranspose, kernel_size=3, strides=2, padding='same', kernel_initializer='lecun_normal', activation='selu')

# with strategy.scope():
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


LATENT_DIM = 32
SIZE = 256
INPUT_SHAPE = (SIZE, SIZE, 3)

# Encoder
encoder_input = layers.Input(INPUT_SHAPE)
x = conv(32)(encoder_input)
for filters in (64, 128, 256):
    x = conv(filters)(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='tanh')(x)
z_mean = layers.Dense(LATENT_DIM, name='z_mean')(x)
z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(inputs=encoder_input, outputs=[z_mean, z_log_var, z], name='encoder')

# Decoder
latent_input = layers.Input(LATENT_DIM)
x = layers.Dense(16 * 16 * 64, activation='selu', kernel_initializer='lecun_normal')(latent_input)
x = layers.Reshape((16, 16, 64))(x)
for filters in (256, 128, 64):
    x = convt(filters)(x)
decoder_output = convt(3, activation='sigmoid', kernel_initializer='glorot_normal')(x)
decoder = keras.Model(inputs=latent_input, outputs=decoder_output, name='decoder')

encoder.summary()
decoder.summary()


discriminator_input = layers.Input(INPUT_SHAPE)
x = conv(128)(discriminator_input)
x = conv(64)(x)
x = conv(32)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
discriminator_output = layers.Dense(1, activation='sigmoid')(x)
discriminator = keras.Model(inputs=discriminator_input, outputs=discriminator_output, name='discriminator')
discriminator.summary()


def corr_loss(z):
    corr_matrix = tfp.stats.correlation(z)
    n = corr_matrix.shape[0]
    loss = tf.reduce_sum(corr_matrix**2)
    for i, j in product(range(n), range(n)):
        loss -= corr_matrix[i, i]**2
    return loss


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z, reconstruction

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction       = self.decoder(z)
            reconstruction_loss  = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)))

            kl_loss = -0.5 * (1 + z_log_var - z_mean**2 - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {'loss': self.total_loss_tracker.result(), 'reconstruction_loss': self.reconstruction_loss_tracker.result(), 'kl_loss': self.kl_loss_tracker.result()}


from tensorflow.keras import backend as K

class VAE_GAN(keras.Model):
    def __init__(self, vae, discriminator, opti1=keras.optimizers.Adam(), opti2=keras.optimizers.Adam(), opti3=keras.optimizers.Adam(), **kwargs):
        super().__init__(**kwargs)
        self.vae           = vae
        self.discriminator = discriminator
        self.encoder       = vae.encoder
        self.decoder       = vae.decoder

        self.vae_loss_tracker            = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker             = keras.metrics.Mean(name='kl_loss')
        self.correlation_loss_tracker    = keras.metrics.Mean(name="cr_loss")
        self.disc_loss_tracker           = keras.metrics.Mean(name='disc_loss')
        self.gen_loss_tracker            = keras.metrics.Mean(name='gen_loss')
        self.disc_loss                   = keras.losses.BinaryCrossentropy()

        self.vae_optimizer  = opti1
        self.gen_optimizer  = opti2
        self.disc_optimizer = opti3

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z, reconstruction

    @property
    def metrics(self):
        return [self.vae_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker, self.correlation_loss_tracker, self.disc_loss_tracker, self.gen_loss_tracker]

    def train_step(self, data):
        batch_size = K.shape(data)[0]

        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as disc_tape:
            # VAE
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)))
            kl_loss             = -0.5*(1 + z_log_var - z_mean**2 - tf.exp(z_log_var))
            kl_loss             = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            correlation_loss    = corr_loss(z)

            # GAN
            recon_vect = z
            construction = self.decoder(recon_vect)
            combined_images = tf.concat([data, construction], axis=0)
            data_l, recon_l = tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))  # 0: real, 1: fake
            combined_l = tf.concat([data_l, recon_l], axis=0)
            tot_predictions = self.discriminator(combined_images)
            r_prediction = self.discriminator(construction)

            discr_loss = self.disc_loss(combined_l, tot_predictions)
            gen_loss   = tf.maximum(self.disc_loss(data_l, r_prediction) - discr_loss, 1e-4)
            vae_loss   = reconstruction_loss + kl_loss + gen_loss

            grad_discr = disc_tape.gradient(discr_loss, self.discriminator.trainable_weights)
            grad_vae = enc_tape.gradient(vae_loss, self.vae.trainable_weights)

            self.disc_optimizer.apply_gradients(zip(grad_discr, self.discriminator.trainable_weights))
            self.vae_optimizer.apply_gradients(zip(grad_vae, self.vae.trainable_weights))

            self.vae_loss_tracker.update_state(vae_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            self.correlation_loss_tracker.update_state(correlation_loss)
            self.disc_loss_tracker.update_state(discr_loss)
            self.gen_loss_tracker.update_state(gen_loss)

        return {'vae_loss': self.vae_loss_tracker.result(), 'disc_loss': self.disc_loss_tracker.result(), 'gen_loss': self.gen_loss_tracker.result()}


vae   = VAE(encoder, decoder)
model = VAE_GAN(vae, discriminator)
model.compile(optimizer='adam')


from tensorflow.keras.callbacks import EarlyStopping

H = model.fit(ds_train, validation_data=ds_val, epochs=500, verbose=1, callbacks=[EarlyStopping(patience=20)])
