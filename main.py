import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import cv2 as cv
import PIL
from PIL import Image
from scipy import ndimage as ndi, ndimage
import skimage
from scipy.ndimage import label
from skimage import segmentation, measure
from skimage.exposure import exposure
from skimage.feature import peak_local_max
from skimage.measure import regionprops, find_contours
from skimage.segmentation import watershed
from skimage.morphology import disk, closing, remove_small_objects, area_opening, opening, erosion
from skimage.filters import median,threshold_otsu, sobel, meijering
from skimage.color import rgb2gray

def image1():

    image= rgb2gray(np.asarray(Image.open("flowers.jpg")))
    thresh = threshold_otsu(image)
    binary = image > thresh
    binary=area_opening(binary,area_threshold=3000)
    binary = closing(binary, disk(5))
    binary=erosion(binary)

    return binary

def image2():

    image = rgb2gray(np.asarray(Image.open("magnolia.jpg")))
    elevation_map = meijering(image)

    markers = np.zeros_like(image)
    markers[image < 0.5] = 1
    markers[image > 0.6] = 2

    segment = segmentation.watershed(elevation_map, markers)
    segment=area_opening(segment,area_threshold=3000)
    segment=closing(segment,disk(5))
    segment=erosion(segment)
    #plt.hist(elevation_map, bins = 10)


    # fig, ax = plt.subplots(figsize=(7, 4))
    # ax.set_title('Elevation Map')
    # plt.imshow(elevation_map,cmap="gray")
    # ax.axis("off")



    # fig, ax = plt.subplots(figsize=(7, 4))
    # ax.imshow(markers, cmap=plt.cm.nipy_spectral)
    # ax.set_title('markers')

    # fig, ax = plt.subplots(figsize=(7, 4))
    # ax.set_title('segmentation')
    # plt.imshow(segment,cmap="gray")
    # ax.axis("off")
    #plt.show()
    return segment
#image1()
image2()
