
import cv2
import tensorflow as tf
import numpy as np
import math
import glob

filenames = glob.glob("ddsm-mammography/*.tfrecords")

# Extract features using the keys set during creation
feature = {'label': tf.FixedLenFeature([], tf.int64),
        'label_normal': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string)}


# Pulls a record user TFRecordReader.read, and outputs a lab image with 0-5 label, and 0-1 label
# def load_tfrecord():
#     _, tfrecord = reader.read(filename_queue)
#     # Decode the record read by the reader
#     features = tf.parse_single_example(tfrecord, features=feature)
#     # Convert the image data from string back to the numbers
#     image = tf.decode_raw(features['image'], tf.uint8)
#     # image = tf.image.decode_png(features['image'],channels=1)
#     label = features['label']
#     label_normal = features['label_normal']
#     image = tf.reshape(image, [299, 299, 1])
#     print(np.array(image))
#     image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
#     return (image_lab, label, label_normal)


# Given a image, returns a dictionary with keys of intensities, and values of the number of occurrences of that intensity
def intensity_frequency(image):
    frequencies = dict()
    for y in range(len(image)):
        for x in range(len(image[0])):
            val = image[y][x]
            if val in frequencies:
                frequencies[val] += 1
            else:
                frequencies[val] = 1
    return frequencies


# Given a image, returns a dictionary with keys of brightnesses, and values of the number of occurrences of that brightness
def brightness_frequency(image):
    frequencies = dict()
    for y in range(len(image)):
        for x in range(len(image[0])):
            val = image[y][x]
            if val in frequencies:
                frequencies[val] += 1
            else:
                frequencies[val] = 1
    return frequencies


# Reduces a 2d 299x299 image to 297x297 by trimming
def reduce_2d(image):
    return image[0:297, 0:297]


# Reduces a 3d 299x299 image to 297x297 by trimming
def reduce_3d(image):
    return image[0:297, 0:297, :]


# Returns a list of measures most intense 6x6 blocks in an image
def most_intense_hits1(image):
    most_intense = []
    indices = list(range(295))
    for y in indices:
        yjump = 6
        if y == 294:
            yjump = 5
        for x in indices:
            xjump = 6
            if x == 294:
                xjump = 5
            block = sum(image[y:yjump, x:xjump]) / (yjump * xjump)
            if len(most_intense) < 36:
                most_intense.append(block)
            else:
                if most_intense[35] < block:
                    most_intense[35] = block
                    most_intense.sort(reverse=True)
    return most_intense


# Returns a list of most intense 12x12 blocks in an image
def most_intense_hits2(image):
    most_intense = []
    indices = list(range(0, 289, 2))
    for y in indices:
        yjump = 12
        if y == 288:
            yjump = 11
        for x in indices:
            xjump = 12
            if x == 288:
                xjump = 11
            block = sum(image[y:yjump, x:xjump]) / (yjump * xjump)
            if len(most_intense) < 36:
                most_intense.append(block)
            else:
                if most_intense[35] < block:
                    most_intense[35] = block
                    most_intense.sort(reverse=True)
    return most_intense


# print(load_tfrecord())