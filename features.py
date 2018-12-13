
import cv2
import tensorflow as tf
import numpy as np
import math
import glob
import datetime

filenames = glob.glob("ddsm-mammography/*.tfrecords")
print(filenames)
raw_img_dir = "raw_images/"

# Extract features using the keys set during creation
feature = {'label': tf.FixedLenFeature([], tf.int64),
        'label_normal': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string)}

sess = tf.Session()


def _parse_(serialized_img):
    print(serialized_img)
    features = {'label': tf.FixedLenFeature([], tf.int64, default_value=-1),
               'label_normal': tf.FixedLenFeature([], tf.int64, default_value=-1),
               'image': tf.FixedLenFeature([], tf.string, default_value="")}
    # features = tf.parse_single_example(serialized_img, features=feature)
    parsed_features = tf.parse_example(serialized_img, features)
    print(parsed_features)
    print(parsed_features['label'])
    return parsed_features['image'], parsed_features['label_normal'], parsed_features['label']
    # # Convert the image data from string back to the numbers
    # image = tf.decode_raw(features['image'], tf.uint8)
    # # image = tf.image.decode_png(features['image'],channels=1)
    # #label = features['label']
    # label_normal = features['label_normal']
    # image = tf.reshape(image, [299, 299, 1])
    """
    image_jpeg = tf.image.encode_jpeg(image)
    fname = tf.constant('2.jpg')
    fwrite = tf.write_file(fname, image_jpeg)
    sess = tf.Session()
    result = sess.run(fwrite)
    """

    #fname = tf.constant('image' + ".jpeg")
    #image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    #return (image, label_normal)
    return (dict({'image':image}), label_normal)


def process_tfrecords(batch_size=32):
    tfrecord_dataset = tf.data.TFRecordDataset(filenames).shuffle(True).batch(batch_size)
    tfrecord_dataset = tfrecord_dataset.map(_parse_)
    tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()
    return tfrecord_iterator


# NOTE: this function might need to be tweaked for whatever model we want
def train_model():
    training_data = process_tfrecords(32)
    # print(training_data)

    # TODO: set up a keras estimator to be our model
    # see these links for examples of TFRecords being used as input to keras:
    # https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/
    # https://github.com/Tony607/Keras_catVSdog_tf_estimator/blob/master/keras_estimator_vgg16-cat_vs_dog-TFRecord.ipynb

train_model()


"""
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
"""

# print(load_tfrecord())
