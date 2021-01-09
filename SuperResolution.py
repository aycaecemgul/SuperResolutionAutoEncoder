import pickle
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Conv2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from matplotlib import pyplot as plt

pickle_in = open("X.pickle","rb")
X = np.array(pickle.load(pickle_in))

pickle_in = open("y.pickle","rb")
y = np.array(pickle.load(pickle_in))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.2,shuffle=True)


pickle_out = open("X_test.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()


def create_model():
  x = Input(shape=(200, 200, 3))

  # Encoder

  e_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  pool1 = MaxPooling2D((2, 2), padding='same')(e_conv1)
  batchnorm_1 = BatchNormalization()(pool1)
  e_conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(batchnorm_1)
  pool2 = MaxPooling2D((2, 2), padding='same')(e_conv1)
  batchnorm_1 = BatchNormalization()(pool2)
  e_conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(batchnorm_1)
  pool2 = MaxPooling2D((2, 2), padding='same')(e_conv2)


  # Decoder
  d_conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool2)
  up1 = UpSampling2D((2, 2))(d_conv1)
  d_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
  up2 = UpSampling2D((2, 2))(d_conv2)
  d_conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
  up3 = UpSampling2D((2, 2))(d_conv3)


  r = Conv2D(3, (3, 3), activation='softmax', padding='same')(up3)
  model = Model(x, r)
  model.compile(optimizer='adam', loss='mse',metrics=["accuracy"])
  return model

auto_encoder = create_model()

gaussian_history = auto_encoder.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val),shuffle=True)

auto_encoder.save("autoencoder.model")







