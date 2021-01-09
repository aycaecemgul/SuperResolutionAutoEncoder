import numpy as np
import tensorflow as tf
import pickle
from matplotlib import pyplot as plt

pickle_in = open("X_test.pickle","rb")
X_test = np.array(pickle.load(pickle_in))

pickle_in = open("y_test.pickle","rb")
y_test = np.array(pickle.load(pickle_in))

auto_encoder=tf.keras.models.load_model("autoencoder.model")

predict = auto_encoder.predict(X_test)

auto_encoder.evaluate(X_test,y_test)

n = 4
plt.figure(figsize= (20,10))
for i in range(n):
  ax = plt.subplot(3, n, i+1)
  plt.imshow(X_test[i+20])
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax = plt.subplot(3, n, i+1+n)
  plt.imshow(predict[i+20])
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()
