
import numpy as np
import scipy.ndimage
import scipy.stats
import skimage.feature
import skimage.morphology
import sklearn
from sklearn.linear_model import Perceptron
from training import *
import numpy as np
import pandas as pd
import os
from glob import glob
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical # used for converting labels to one-hot-encoding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

def gaussian_deriv_1d(sigma, deriv = 0, min_width = None):
   # compute x range
   if min_width is None:
      width = np.ceil(3.0 * sigma)
   else:
      width = np.ceil(max((3.0 * sigma), min_width))
   x = np.arange(-width, width + 1)
   # compute gaussian derivative
   x2                 = x * x
   sigma2_inv         = 1.0 / (sigma * sigma)
   neg_two_sigma2_inv = -0.5 * sigma2_inv
   if (deriv == 0):
      g = np.exp(x2 * neg_two_sigma2_inv)
   elif (deriv == 1):
      g = np.exp(x2 * neg_two_sigma2_inv) * (-x)
   elif (deriv == 2):
      g = np.exp(x2 * neg_two_sigma2_inv) * (x2 * sigma2_inv - 1)
   else:
      raise ValueError('deriv must be 0, 1, or 2')
   # make zero mean (if deriv > 0)
   if (deriv > 0):
      g = g - np.mean(g)
   # normalize
   filt = np.atleast_2d(g / np.sum(np.abs(g)))
   return filt


def gaussian_deriv_2d( \
      sigma, deriv_x = 0, deriv_y = 0, ori = 0.0, min_width = None):
   gx = gaussian_deriv_1d(sigma, deriv_x, min_width)
   gy = gaussian_deriv_1d(sigma, deriv_y, min_width)
   filt = np.transpose(gx) * gy
   filt = scipy.ndimage.rotate(filt, (180.0 / np.pi) * ori, reshape=False)
   return filt


def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3],[0.29894, 0.58704, 0.11402])
    return gray


def superpixels(img, th_dist = 11):
   # convert image to grayscale
   if (img.ndim == 3):
      img = rgb2gray(img)
   assert img.ndim == 2, 'image should be grayscale'
   # compute edge strength from multiscale local oriented gradients
   edge = np.zeros(img.shape)
   for scale in [2.0, 4.0]:
      for ori in range(8):
         filt = gaussian_deriv_2d(scale, 1, 0, (ori/8)*np.pi)
         grad = scipy.signal.convolve2d( \
                  img, filt, mode = 'same', boundary = 'symm')
         edge = edge + np.abs(grad)
   edge = edge / np.amax(edge)
   # find local minima in edge strength
   e_level = np.round(edge * 1000)
   lm = skimage.feature.peak_local_max( \
      -e_level, min_distance = th_dist, indices=False)
   markers = scipy.ndimage.label(lm)[0]
   # perform watershed oversegmentation
   seg = skimage.morphology.watershed(e_level, markers, compactness=0.5)
   seg = seg - 1  # switch to zero-based indexing of regions
   # randomly permute region labels for easier visualization
   rperm = np.random.permutation(np.amax(seg)+1)
   seg = rperm[seg]
   return seg


def seg2graph(seg):
   # get image size
   assert seg.ndim == 2, 'segmentation should be 2D'
   sx, sy = seg.shape
   # get region count
   n_reg = np.amax(seg) + 1
   # initialize adjacency indicator matrix
   adjacency_mx = np.zeros((n_reg, n_reg))
   # initialize region pixel lists
   region_pixels = []
   for n in range(n_reg):
      region_pixels.append([])
   # detect neighboring regions
   for x in range(sx):
      xa = max(0, x - 1)
      xb = min(x + 2, sx)
      for y in range(sy):
         ya = max(0, y - 1)
         yb = min(y + 2, sy)
         r = seg[x,y]                  # region id at current location
         r_ids = seg[xa:xb, ya:yb]     # ids of neighboring regions
         r_ids = r_ids.flatten()
         # update adjacency matrix
         adjacency_mx[r,r_ids] = 1
         adjacency_mx[r_ids,r] = 1
         # update pixel list
         px_id = x * sy + y
         region_pixels[r].append(px_id)
   return adjacency_mx, region_pixels


def pool_superpixels(tile_df=None):
    if tile_df is None:
        with open('tile_dataframe.pkl', 'rb') as f:
            tile_df = pickle.load(f)
    pos = []
    neg = []

    imgs_train, labels_train, imgs_val, labels_val, imgs_test, labels_test = split_data(tile_df)
    print(len(labels_train))
    labels = list(labels_train)
    print(labels[2])
    imglen = len(imgs_train)
    for i, img in enumerate(imgs_train):
        print(str(i+1) + " / " + str(imglen))
        supers = superpixels(img)
        img = rgb2gray(img)
        adjacency_mx, region_pixels = seg2graph(supers)
        perneg = []
        img = img.ravel()
        for superpix in region_pixels:
            val = float(np.sum(img[superpix])) / len(superpix)
            if labels[i] == 0:
                pos.append(val)
            elif labels[i] == 1:
                perneg.append(val)
        if len(perneg) > 0:
            perneg.sort()
            neg += perneg[0:3]

    jpos = []
    jneg = []
    jmglen = len(imgs_val)
    for j, jmg in enumerate(imgs_val):
        print(str(j + 1) + " / " + str(jmglen))
        supers = superpixels(jmg)
        img = rgb2gray(jmg)
        adjacency_mx, region_pixels = seg2graph(supers)
        perneg = []
        img = img.ravel()
        for superpix in region_pixels:
            val = float(np.sum(img[superpix])) / len(superpix)
            if labels[j] == 0:
                pos.append(val)
            elif labels[j] == 1:
                perneg.append(val)
        if len(perneg) > 0:
            perneg.sort()
            neg += perneg[0:3]
    return pos, neg, jpos, jneg


def test_superpixels(perceptron, imgs):
    outimages = []
    for img in imgs:
        supers = superpixels(img)
        img = rgb2gray(img)
        adjacency_mx, region_pixels = seg2graph(supers)
        values = []
        inbox = []
        img = img.ravel()
        for superpix in region_pixels:
            val = float(np.sum(img[superpix])) / len(superpix)
            values.append(val)
        values.sort()
        labels = perceptron.predict(values)
        for ind, val in enumerate(labels[0]):
            if val == 1:
                inbox.append(region_pixels[ind])
        for superpix in inbox:
            img[superpix] = max(min(values) - 15, min(values))
        img.reshape(300, 300)
        outimages.append(img)
        print(str(len(inbox)) + " / " + str(len(region_pixels)) + " superpixels determined part of lesion.")
    return outimages


tile_df = load_data()
if len(glob('inputdata.pkl')) and len(glob('inputlanels.pkl')) and len(glob('perceptron-parameters.pkl')):
    with open('inputdata.pkl', 'rb') as f:
        inputdata = pickle.load(f)
    with open('inputlabels.pkl', 'rb') as f:
        inputlabels = pickle.load(f)
    with open('validbound.pkl', 'rb') as f:
        jinputdata = pickle.load(f)
    with open('validlab.pkl', 'rb') as f:
        jinputlabels = pickle.load(f)
    with open('perceptron-parameters.pkl', 'rb') as f:
        perceptron = Perceptron().set_params(pickle.load(f))
else:
    pos, neg, jpos, jneg = pool_superpixels(tile_df)
    print(len(pos))
    print(len(neg))
    neglab = [1] * len(neg)
    poslab = [0] * len(pos)
    jneglab = [1] * len(jneg)
    jposlab = [0] * len(jpos)
    jinputdata = np.array(jpos + jneg)
    jinputlabels = jposlab + jneglab
    inputdata = np.array(pos + neg)
    inputlabels = poslab + neglab
    with open('inputdata.pkl', 'wb') as f:
        pickle.dump(inputdata, f)
    with open('inputlabels.pkl', 'wb') as f:
        pickle.dump(inputlabels, f)
    with open('validbound.pkl', 'wb') as f:
        pickle.dump(jinputdata, f)
    with open('validlab.pkl', 'wb') as f:
        pickle.dump(jinputlabels, f)
# with open('inputdata.pkl', 'rb') as f:
#     inputdata = np.array(pickle.load(f))
# with open('inputlabels.pkl', 'rb') as f:
#     inputlabels = pickle.load(f)
    perceptron = Perceptron()
    perceptron.fit(inputdata.reshape(-1, 1), inputlabels)
    with open('perceptron-parameters.pkl', 'wb') as f:
        pickle.dump(perceptron.get_params(), f)

score = perceptron.score(jinputdata.reshape(-1, 1), jinputlabels)
print(score[0])
print(score[1])
print(score)
with open('perceptron-score.pkl', 'wb') as f:
    pickle.dump(score)

file_path = "image_segmentation.pkl"
n_bytes = 2 ** 31
max_bytes = 2 ** 31 - 1
if len(glob(file_path)) > 0:
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
        _, _, _, _, imgs_test, labels_test = pickle.loads(bytes_in)
else:
    _, _, _, _, imgs_test, labels_test = split_data(tile_df)

test_imgs = []
for i, j in enumerate(labels_test):
    if j == 1:
        test_imgs.append(imgs_test[i])
imgs = test_superpixels(perceptron, test_imgs)

bytes_out = pickle.dumps(imgs)
with open(file_path, 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx + max_bytes])

print("DONE!")
