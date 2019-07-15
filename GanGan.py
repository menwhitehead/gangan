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
import h5py
import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import cv2
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from scipy.misc import imsave

class GanGan:

    def __init__(self, X_train):
        self.X_train = X_train
        self.image_size = self.X_train[0].shape[0]
        self.image_shape = self.X_train[0].shape
        self.number_channels = self.X_train[0].shape[2]
        self.noise_vect_size = 100
        self.noise_shape = (self.noise_vect_size, )
        # self.adam = Adam(0.0002, 0.5)
        self.magnification = 10
        self.seed = np.random.normal(0, 2, size=[1, self.noise_vect_size])

        self.buildGenerator()
        self.buildDiscriminator()
        self.buildFullModel()

    def buildGenerator(self):
        generator = Sequential()
        generator.add(Dense(self.image_size * self.image_size * self.number_channels, input_shape=self.noise_shape, activation = 'sigmoid'))
        generator.add(Reshape(self.image_shape))

        generator.add(Dense(64, activation = 'sigmoid'))
        generator.add(Dense(self.number_channels, activation = 'sigmoid'))
        # generator.add(Reshape(self.image_shape))

        # generator.add(Conv2D(64, (3, 3), strides = (1,1), padding='same', activation = 'relu'))
        # # generator.add(Conv2D(64, (3, 3), strides = (1,1), padding='same', activation = 'relu'))
        # generator.add(Conv2D(self.number_channels, (3, 3), strides = (1,1), padding='same', activation = 'sigmoid'))

        generator.summary()
        noise = Input(shape=self.noise_shape)
        img = generator(noise)
        self.generator = Model(noise, img)


    def buildDiscriminator(self):
        discriminator = Sequential()
        discriminator.add(Dense(32, activation="sigmoid", input_shape=self.image_shape))
        # discriminator.add(Conv2D(32, kernel_size=3, strides=1, activation="relu", input_shape=self.image_shape, padding="same"))
        # discriminator.add(Dense(64, activation="relu"))

        # discriminator.add(Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same"))
        # discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
        discriminator.add(Flatten())
        discriminator.add(Dense(64, activation='sigmoid'))
        # discriminator.add(Dense(64, activation='sigmoid'))
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.summary()
        img = Input(shape=self.image_shape)
        validity = discriminator(img)
        self.discriminator = Model(img, validity)
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = "adam", metrics=["accuracy"])

    def buildFullModel(self):
        gan_input = Input(shape=self.noise_shape)
        discrim_input = self.generator(gan_input)
        self.discriminator.trainable = False
        gan_output = self.discriminator(discrim_input)
        self.gan = Model(gan_input, gan_output)
        self.gan.compile(loss = 'binary_crossentropy', optimizer = "adam")

    def generateImage(self):
        arr = self.generator.predict(self.seed)
        arr = np.reshape(arr, self.image_shape)
        res = cv2.resize(arr, None, fx=self.magnification, fy=self.magnification, interpolation = cv2.INTER_NEAREST)
        return res

    def saveImage(self):
        imsave("test.jpg", self.generateImage())

    def visualizeImage(self):
        res = self.generateImage()
        # res = 0.5 * res + 0.5

        cv2.imshow('Generated Image', res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(0)

    def train(self, epochs=10, batch_size=128):
        half_batch = batch_size / 2

        for e in range(epochs):
            chosen_data_indexes = np.random.randint(0, self.X_train.shape[0], half_batch)
            data_x = self.X_train[chosen_data_indexes]

            generated_x = self.generator.predict(np.random.normal(0, 1, (half_batch, self.noise_vect_size)))

            dloss1 = self.discriminator.train_on_batch(data_x, np.ones((half_batch, 1)))[1]
            dloss2 = self.discriminator.train_on_batch(generated_x, np.zeros((half_batch, 1)))[1]

            gan_x = np.random.normal(0, 1, (batch_size, self.noise_vect_size))
            gan_y = np.ones((batch_size, 1))
            gloss = self.gan.train_on_batch(gan_x, gan_y)

            self.visualizeImage()
            print ("%8d %10.1f%10.1f%10.5f" % (e, dloss1*100, dloss2*100, gloss))





