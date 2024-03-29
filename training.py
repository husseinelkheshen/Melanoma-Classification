import numpy as np
import pandas as pd
import os
from glob import glob
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

from tensorflow.keras.utils import to_categorical # used for converting labels to one-hot-encoding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten


img_rows = 300
img_cols = 300
img_channels = 3
metadata_path = r"skin-lesions/HAM10000_metadata.csv"
image_dir = r"skin-lesions/images"


lesion_type_dict = {
    'mal': 'Malignant',
    'bkl': 'Benign',
}


def load_data():
    file_path = "tile_df.pkl"
    n_bytes = 2 ** 31
    max_bytes = 2 ** 31 - 1

    # If the pandas dataframe has been saved to disk we
    # do not have to go through the process of creating
    # it and loading the image data again
    if len(glob(file_path)) > 0:
        bytes_in = bytearray(0)
        input_size = os.path.getsize(file_path)
        with open(file_path, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        tile_df = pickle.loads(bytes_in)
        return tile_df

    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x \
                     for x in glob(os.path.join(image_dir, '*.jpg'))}

    tile_df = pd.read_csv(metadata_path)
    tile_df['dx'] = np.where(tile_df['dx'] != 'bkl', 'mal', tile_df['dx'])
    tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
    tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get)
    tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes

    # resize images
    tile_df['image'] = tile_df['path'].map(lambda x: np.asarray(Image\
        .open(x).resize((img_rows, img_cols))))

    # Save model to disk
    bytes_out = pickle.dumps(tile_df)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])

    return tile_df


def split_data(tile_df):
    # Split data into test, train, validation
    # The split is as follows: 80% non-test, 20% test,
    # with non-test split 70-30 into training and validation
    imgs_train, imgs_test, labels_train, labels_test = train_test_split(tile_df, tile_df['cell_type_idx'], test_size=0.2)

    imgs_train, imgs_val, labels_train, labels_val = train_test_split(imgs_train, labels_train, test_size=0.3)

    imgs_train = np.asarray(imgs_train['image'].tolist())
    imgs_test = np.asarray(imgs_test['image'].tolist())
    imgs_val = np.asarray(imgs_val['image'].tolist())

    # Save images to disk
    file_path = "images_segmentation.pkl"
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps((imgs_train, labels_train, imgs_val, labels_val, imgs_test, labels_test))
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])

    return imgs_train, labels_train, imgs_val, labels_val, imgs_test, labels_test


def train_model():
    tile_df = load_data()
    imgs_train, labels_train, imgs_val, labels_val, imgs_test, labels_test = split_data(tile_df)

    # Create a Keras Sequential model with a few convolutional layers
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_rows, img_cols, img_channels)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(124, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(124, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    model.compile(optimizer=tf.train.AdamOptimizer(), \
                           loss='binary_crossentropy', \
                           metrics=['accuracy'])

    batch_size = 32
    epochs = 12

    history = model.fit(imgs_train, labels_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(imgs_val, labels_val))

    score = model.evaluate(imgs_test, labels_test, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    # Save model to disk
    model.save('saved_model.h5')
    # load with model = load_model('saved_model.h5')


train_model()
