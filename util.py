
import tensorflow
import cv2
import numpy as np
import math

def get_lab_img(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

def get_bw_img(image):
    gray = np.dot(image[..., :3], [0.29894, 0.58704, 0.11402])
    return gray

def get_rgb_img(path):
    return cv2.imread(path)

def split_lab(image):
    return cv2.split(image)

def mirror_border_bw(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   # mirror top/bottom
   top    = image[:wx:,:]
   bottom = image[(sx-wx):,:]
   img = np.concatenate( \
      (top[::-1,:], image, bottom[::-1,:]), \
      axis=0 \
   )
   # mirror left/right
   left  = img[:,:wy]
   right = img[:,(sy-wy):]
   img = np.concatenate( \
      (left[:,::-1], img, right[:,::-1]), \
      axis=1 \
   )
   return img

def mirror_border_rgb(image, wx=1, wy=1):
    assert image.ndim == 3, 'image should be rgb'
    sx, sy, sz = image.shape
    # mirror top/bottom
    top = image[:wx:, :, :]
    bottom = image[(sx - wx):, :, :]
    img = np.concatenate( \
        (top[::-1, :, :], image, bottom[::-1, :, :]), \
        axis=0 \
        )
    # mirror left/right
    left = img[:, :wy, :]
    right = img[:, (sy - wy):, :]
    img = np.concatenate( \
        (left[:, ::-1, :], img, right[:, ::-1, :]), \
        axis=1 \
        )
    return img

def gaussian_1d(sigma = 1.0):
   width = np.ceil(3.0 * sigma)
   x = np.arange(-width, width + 1)
   g = np.exp(-(x * x) / (2 * sigma * sigma))
   g = g / np.sum(g)          # normalize filter to sum to 1 ( equivalent
   g = np.atleast_2d(g)       # to multiplication by 1 / sqrt(2*pi*sigma^2) )
   return g

def conv_2d(image, filt):
   # make sure that both image and filter are 2D arrays
   assert image.ndim == 2, 'image should be grayscale'
   filt = np.atleast_2d(filt)
   ##########################################################################
   # TODO: YOUR CODE HERE
   offset = max(math.ceil(len(filt) / 2) - 1, 1)
   if offset == 0:
      result = image.copy()
      padded = image.copy()
   else:
      result = image.copy()
      padded = mirror_border_bw(image, offset, offset)
   i = 0
   height = len(image)
   length = len(image[0])
   filt = filt.astype(float)
   while i < height:  # for each row in image
      j = 0
      while j < length:  # for each col in row
         k = 0
         filtvals = filt.copy()
         while k < len(filt):  # for each row in filter
            l = 0
            if len(filt.shape) == 1:
               while l < len(filt):
                  filtvals[l] = filt[l] * padded[i - k][j - l]
                  l += 1
            else:
               while l < len(filt[0]):
                  filtvals[k][l] = filt[k][l] * padded[i - k][j - l]
                  l += 1
               k += 1
         result[i][j] = np.sum(filtvals) / np.size(filt)
         j += 1
      i += 1
   return result

def conv_3d(image, filt):
    # make sure that both image and filter are 2D arrays
    assert image.ndim == 3, 'image should be lab'
    filt = np.atleast_2d(filt)
    ##########################################################################
    # TODO: YOUR CODE HERE
    offset = max(math.ceil(len(filt) / 2) - 1, 1)
    if offset == 0:
        result = image.copy()
        padded = image.copy()
    else:
        result = image.copy()
        padded = mirror_border_rgb(image, offset, offset)
    i = 0
    height = len(image)
    length = len(image[0])
    filt = filt.astype(float)
    while i < height:  # for each row in image
        j = 0
        while j < length:  # for each col in row
            a = 0
            while a < 3:
                k = 0
                filtvals = filt.copy()
                while k < len(filt):  # for each row in filter
                    l = 0
                    if len(filt.shape) == 1:
                        while l < len(filt):
                            filtvals[l] = filt[l] * padded[i - k][j - l][a]
                            l += 1
                    else:
                        while l < len(filt[0]):
                            filtvals[k][l] = filt[k][l] * padded[i - k][j - l][a]
                            l += 1
                    k += 1
                result[i][j][a] = np.sum(filtvals) / np.size(filt)
                a += 1
        j += 1
    i += 1
    return result

def gauss_bw(image):
    return conv_2d(image, gaussian_1d(sigma=1.0))

def gauss_rgb(image):
    return conv_3d(image, gaussian_1d(sigma=1.0))