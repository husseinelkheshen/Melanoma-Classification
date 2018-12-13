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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json



img_rows = 300
img_cols = 300
img_channels = 3
metadata_path = r"skin-lesions\HAM10000_metadata.csv"
image_dir = r"skin-lesions\images"


lesion_type_dict = {
    'nv': 'Malignant',
    'mel': 'Malignant',
    'bkl': 'Benign',
    'bcc': 'Malignant',
    'akiec': 'Malignant',
    'vasc': 'Malignant',
    'df': 'Malignant'
}


def load_data():
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x \
                     for x in glob(os.path.join(image_dir, '*.jpg'))}

    #print(imageid_path_dict)
    tile_df = pd.read_csv(metadata_path)
    tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
    tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get)
    tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes
    tile_df['dx'] = np.where(tile_df['dx'] != 'bkl', 'mal', tile_df['dx'])
    #print(tile_df.sample(3))

    # NOTE: This line is only for speeding up development, it should be removed
    # when doing true model testing
    #tile_df = tile_df.sample(100)

    # resize images
    tile_df['image'] = tile_df['path'].map(lambda x: np.asarray(Image\
        .open(x).resize((img_rows, img_cols))))

    #print(tile_df['image'].map(lambda x: x.shape).value_counts())
    # TODO: we should save tile_df to disk so we dont have to import
    # all the images each time, too
    return tile_df


def split_data(tile_df):
    lab = tile_df.cell_type_idx

    # split data into test, train, validation
    # the split is as follows: 80% non-test, 20% test
    # --> then the non-test is split 70-30 into training and validation
    imgs_train, imgs_test, labels_train, labels_test = train_test_split(tile_df, lab, test_size=0.2)

    imgs_train, imgs_val, labels_train, labels_val = train_test_split(imgs_train, labels_train, test_size=0.3)

    # TODO: possibly use Keras' ImageDataGenerator do set up the image data?
    # a friend said it can be used to standardize/normalize image data, but
    # One of the kernels for our dataset subtracted out the mean image value and
    # then divided everything by the standard deviation. Idk enough stats to know
    # why it was being done though lol, I pasted it below:
    """
    x_train = np.asarray(x_train_o['image'].tolist())
    x_test = np.asarray(x_test_o['image'].tolist())

    x_train_mean = np.mean(x_train)
    x_train_std = np.std(x_train)

    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)

    x_train = (x_train - x_train_mean)/x_train_std
    x_test = (x_test - x_test_mean)/x_test_std
    """

    # Perform one-hot encoding on the labels
    labels_train = to_categorical(labels_train, num_classes = 2)
    labels_test = to_categorical(labels_test, num_classes = 2)
    labels_val = to_categorical(labels_val, num_classes = 2)

    return imgs_train, labels_train, imgs_val, labels_val, imgs_test, labels_test


def train_model():
    tile_df = load_data()
    imgs_train, labels_train, imgs_val, labels_val, imgs_test, labels_test = split_data(tile_df)

    # TODO: set up a sequential keras model (there's lots of stuff about this online)

    # TODO: add some convolution layers with MaxPooling2D's & Dropouts inbetween
    # Everything I've seen ends with a Flatten and a Dense too

    # TODO: train our model & save it to disk so we dont have to always retrain


#train_model()
