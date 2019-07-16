import keras.utils
from keras.datasets import mnist
import numpy as np
import h5py
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from GanGan import GanGan

def loadMNIST(dataType):
    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    # X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    # Rescale to 0 to 1
    X_train = X_train.astype(np.float32) / 255.0
    X_train = np.expand_dims(X_train, axis=3)
    return X_train


#grabbing all training inputs and begin training
if __name__ == '__main__':
    epochs = 10000000
    batch_size = 8

    X_train = loadMNIST("train")
    # x_train = loadFaces()
    # x_train = loadShoes()

    gg = GanGan(X_train)
    gg.train(epochs=epochs, batch_size=batch_size)







