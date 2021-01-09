import skimage
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from numpy import asarray,reshape
import cv2 as cv
from skimage.filters import gaussian

X=[]
y=[]
DIR="data"
IMG_SIZE=200

for file in os.listdir(DIR):
    filename = os.path.join(DIR, file)
    img_array = asarray(Image.open(filename).convert('RGB'))
    image = resize(img_array, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)

    y.append(image)

    blurred_img = gaussian(image, multichannel=False)
    X.append(blurred_img)


X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
y = np.array(y).reshape(-1,IMG_SIZE,IMG_SIZE,3)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
