# Importing necessary libraries
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import random
import os
import gc

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models, Sequential
from tensorflow.keras import optimizers

from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D

from tensorflow.keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import imageio

def read_and_process_image(list_of_images, nrows, ncolumns):
    X = []
    y = []
    for image in list_of_images:
        try:
            img = imageio.imread(image)
            if img is None:
                print("Error: Failed to load image:", image)
                continue
            resized_img = cv2.resize(img, (nrows, ncolumns), interpolation=cv2.INTER_CUBIC)
            X.append(resized_img)
            if 'Non_Autistic' in image:
                y.append(0)
            else:
                y.append(1)
        except Exception as e:
            print("Error processing image:", image, e)
    return X, y


def train():
    # Creating file path for our train data and test data
    train_dir = "AutismDataset/test"
    test_dir = "AutismDataset/guifolder"

    # Getting 'Autistic' and 'Non-Autistic' train images from respective file names of train data
    train_non_autistic = []
    train_autistic = []
    for i in os.listdir(train_dir):
        if 'Non_Autistic' in ("AutismDataset/test/{}".format(i)):
            train_non_autistic.append(("AutismDataset/test/{}".format(i)))
        else:
            train_autistic.append(("AutismDataset/test/{}".format(i)))
            
    test_imgs = ["AutismDataset/guifolder/{}".format(i) for i in os.listdir(test_dir)]

    # Concatenate 'Autistic'  and 'Non-Autistic' images and shuffle them as train_images
    train_imgs = train_autistic + train_non_autistic
    random.shuffle(train_imgs)

    # Remove the lists to save space
    del train_autistic
    del train_non_autistic
    gc.collect()

    # Get resized images and labels from train data
    X_train, y_train = read_and_process_image(train_imgs,150,150)

    # Delete train images to save space
    del train_imgs
    gc.collect()

    # Convert the lists to array
    plt.figure(figsize=(12, 8))
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Repeat the above process for validation data to get val_images
    val_autistic = "AutismDataset/valid/Autistic"
    val_non_autistic = "AutismDataset/valid/Non_Autistic"
    val_autistic_imgs = ["AutismDataset/valid/Autistic/{}".format(i) for i in os.listdir(val_autistic)]
    val_non_autistic_imgs = ["AutismDataset/valid/Non_Autistic/{}".format(i) for i in os.listdir(val_non_autistic)]
    val_imgs = val_autistic_imgs + val_non_autistic_imgs
    random.shuffle(val_imgs)

    # Remove the lists to save space
    del val_autistic_imgs
    del val_non_autistic_imgs
    gc.collect()

    # Get resized images and labels from validation data
    X_val, y_val = read_and_process_image(val_imgs,150,150)

    # Delete validation images to save space
    del val_imgs
    gc.collect()

    # Convert the lists to array
    plt.figure(figsize=(12, 8))
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    ntrain = len(X_train)
    nval = len(X_val)
    batch_size = 32

    base_model = VGG16(include_top=False,weights='imagenet',input_shape=(150,150,3))
    for layer in base_model.layers:
        layer.trainable = False

    model = keras.models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = keras.optimizers.Adam(), metrics = ['acc'])

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    rotation_range = 40,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

    # Only rescaling for validation data
    val_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow(X_train, y_train, batch_size = batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size = batch_size)

    # Train the model
    history = model.fit(train_generator,
                                steps_per_epoch=ntrain // batch_size,
                                epochs=1,
                                validation_data=val_generator,
                                validation_steps=nval // batch_size
                                )
    return [train_dir, test_dir, X_train, model]


def evaluate(train_dir, test_dir, X_train, model):
    nrows = 150
    ncolumns  = 150
    channels = 3

    train_non_autistic = []
    train_autistic = []
    for i in os.listdir(train_dir):
        if 'Non_Autistic' in ("AutismDataset/train/{}".format(i)):
            train_non_autistic.append(("AutismDataset/train/{}".format(i)))
        else:
            train_autistic.append(("AutismDataset/train/{}".format(i)))
            
    # Getting test images from test data file path
    test_imgs = ["AutismDataset/guifolder/{}".format(i) for i in os.listdir(test_dir)]
    # test_imgs = ["AutismDataset/customTest/{}".format(i) for i in os.listdir(test_dir)]

    # Concatenate 'Autistic'  and 'Non-Autistic' images and shuffle them as train_images
    train_imgs = train_autistic + train_non_autistic


    #from sklearn.model_selection import train_test_split
    # Read and resize test images
    random.shuffle(test_imgs)

    X_test, y_test = read_and_process_image(test_imgs,nrows,ncolumns)
    # print(X_test)
    #X_test,y_test = train_test_split(test_imgs)
    X = np.array(X_test)
    #test_datagen = ImageDataGenerator(rescale = 1./255)

    output = np.round(model.predict(X_train[1:30]), 3)
    # print(output)


    # Predict label for test images
    pred = model.predict(X)
    threshold = 0.5
    predictions = np.where(pred > threshold, 1,0)
    # print(predictions)

    # Let's check our predcitions against some test images
    # print(' - Prediction: ' +  f"{'Autistic' if predictions[0] == 1 else 'Non-Autistic'}")

    return [predictions[0] == 1, output[0][0]]

data = train()
prediction = evaluate(data[0], data[1], data[2], data[3])
print(prediction)
