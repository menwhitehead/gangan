from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, UpSampling2D, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
import keras.utils
import numpy as np




def buildGenerator(img_size, num_channels, noise_shape, img_shape):
    generator = Sequential()
    generator.add(Dense(img_size * img_size * num_channels, input_shape=noise_shape, activation = 'sigmoid'))
    generator.add(Reshape(img_shape))

    # generator.add(Dense(16, activation = 'sigmoid'))
    generator.add(Dense(num_channels, activation = 'sigmoid'))
    # generator.add(Reshape(self.image_shape))

    # generator.add(Conv2D(64, (3, 3), strides = (1,1), padding='same', activation = 'relu'))
    # # generator.add(Conv2D(64, (3, 3), strides = (1,1), padding='same', activation = 'relu'))
    # generator.add(Conv2D(self.number_channels, (3, 3), strides = (1,1), padding='same', activation = 'sigmoid'))

    generator.summary()
    noise = Input(shape=noise_shape)
    img = generator(noise)
    return Model(noise, img)


def buildDiscriminator(img_shape, opt="adam"):
    discriminator = Sequential()
    discriminator.add(Dense(16, activation="sigmoid", input_shape=img_shape))
    # discriminator.add(Conv2D(32, kernel_size=3, strides=1, activation="relu", input_shape=self.image_shape, padding="same"))
    # discriminator.add(Dense(64, activation="relu"))

    # discriminator.add(Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same"))
    # discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
    discriminator.add(Flatten())
    discriminator.add(Dense(16, activation='sigmoid'))
    # discriminator.add(Dense(64, activation='sigmoid'))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.summary()
    img = Input(shape=img_shape)
    validity = discriminator(img)
    disc = Model(img, validity)
    disc.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=["accuracy"])
    return disc
