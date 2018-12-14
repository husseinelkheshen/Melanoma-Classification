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


img_rows = 300
img_cols = 300
img_channels = 3
metadata_path = r"skin-lesions\HAM10000_metadata.csv"
image_dir = r"skin-lesions\images"


lesion_type_dict = {
    'mal': 'Malignant',
    'bkl': 'Benign',
}


def load_data():
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x \
                     for x in glob(os.path.join(image_dir, '*.jpg'))}

    #print(imageid_path_dict)
    tile_df = pd.read_csv(metadata_path)
    tile_df['dx'] = np.where(tile_df['dx'] != 'bkl', 'mal', tile_df['dx'])
    tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
    tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get)
    tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes
    #print(tile_df.sample(3))

    # NOTE: This line is only for speeding up development, it should be removed
    # when doing true model testing
    #tile_df = tile_df.sample(100)

    # resize images
    tile_df['image'] = tile_df['path'].map(lambda x: np.asarray(Image\
        .open(x).resize((img_rows, img_cols))))

    #print(tile_df.sample(1))
    #print(tile_df['image'].map(lambda x: x.shape).value_counts())
    # TODO: we should save tile_df to disk so we dont have to import
    tile_df.to_pickle('tile_dataframe.pkl')
    # load with tile_df = pd.read_pickle('tile_dataframe.pkl')
    # all the images each time, too
    return tile_df


def split_data(tile_df):
    #lab = tile_df.cell_type_idx'

    # split data into test, train, validation
    # the split is as follows: 80% non-test, 20% test
    # --> then the non-test is split 70-30 into training and validation
    imgs_train, imgs_test, labels_train, labels_test = train_test_split(tile_df, tile_df['cell_type_idx'], test_size=0.2)

    imgs_train, imgs_val, labels_train, labels_val = train_test_split(imgs_train, labels_train, test_size=0.3)

    # TODO: possibly use Keras' ImageDataGenerator do set up the image data?
    # a friend said it can be used to standardize/normalize image data, but
    # One of the kernels for our dataset subtracted out the mean image value and
    # then divided everything by the standard deviation. Idk enough stats to know
    # why it was being done though lol, I pasted it below:
    imgs_train = np.asarray(imgs_train['image'].tolist())
    imgs_test = np.asarray(imgs_test['image'].tolist())
    imgs_val = np.asarray(imgs_val['image'].tolist())

    """
    print("TEST LABS")
    print(labels_val)
    print("TEST IMGS")
    print(imgs_val[0])
    """

    return imgs_train, labels_train, imgs_val, labels_val, imgs_test, labels_test


def train_model():
    tile_df = load_data()
    imgs_train, labels_train, imgs_val, labels_val, imgs_test, labels_test = split_data(tile_df)

    # TODO: set up a sequential keras model (there's lots of stuff about this online)
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


    # TODO: save model to disk so we dont have to always retrain
    model.save('saved_model.h5')
    # load with model = load_model('saved_model.h5')


train_model()
