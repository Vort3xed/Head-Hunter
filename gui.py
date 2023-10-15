import tkinter as tk
import cv2
from PIL import Image, ImageTk
import PIL.Image
import os
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from FunctionalIdentifier import read_and_process_image, evaluate, train

# Imports for Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Create a GUI window
app = tk.Tk()
app.geometry("600x400")
app.title("HEADHUNTER")

global predictionArray
predictionArray = [False, 0]

global data
data = []

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Function to capture a photo
def capture_photo():
    ret, frame = cap.read()
    if ret:
        # Save the photo to the 'captured_photos' folder
        photo_path = "AutismDataset/guifolder/user.jpg"
        cv2.imwrite(photo_path, frame)

        # Display the captured photo on the GUI
        # display_captured_photo(photo_path)
        gen_gradcam()

        # display_captured_photo('output/heatmap.jpg')
        # display_captured_photo('output/1695968512.jpg')

def save_and_display_gradcam(img_path, heatmap, alpha=0.8):
    # folder to save the image in
    folder_path = 'output'
        
    file_name = 'heatmap.jpg'

    # load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # save the superimposed image
    array_to_img(superimposed_img).save(os.path.join(folder_path, file_name))

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # model that maps the input image to the activations of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # compute the gradient of the top predicted class for our input image with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # gradient of the output neuron (top predicted or chosen) with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # vector where each entry is the mean intensity of the gradient over a specifpipic feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # multiply each channel in the feature map array by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def preprocess_input(image_path, target_size, rescale_value=1./255):
    # load image
    image = load_img(image_path, target_size=target_size)
    # convert to numpy array
    image = img_to_array(image)
    # rescale pixel values
    image /= rescale_value
    # expand dimension to create a batch of 1 image
    image = np.expand_dims(image, axis=0)
    return image

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # add a dimension to transform our array into a "batch" of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def gen_gradcam():
    model = load_model('vgg_model50.h5')
    photo_size = 224
    img_size = (photo_size, photo_size)
    # find the last layer for analysis
    model.summary()

    # save this layer
    last_conv_layer_name = "block5_conv3"

    # load in the image for analysis
    img_path = 'AutismDataset/guifolder/user.jpg'

    img_array = preprocess_input(img_path, target_size=img_size)

    # remove last layer's softmax
    model.layers[-1].activation = None

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    save_and_display_gradcam(img_path, heatmap)


# Function to display the captured photo on the GUI
def display_captured_photo(photo_path):
    captured_image = Image.open(photo_path)
    captured_image = captured_image.resize((400, 300), Image.ANTIALIAS)  # Adjust the size as needed
    captured_photo = ImageTk.PhotoImage(captured_image)

    photo_label.config(image=captured_photo)
    photo_label.image = captured_photo

def display_updated_photo(photo_path):
    captured_image = PIL.Image.open(photo_path)
    captured_image = captured_image.resize((400, 300))  # Adjust the size as needed
    captured_photo = ImageTk.PhotoImage(captured_image)

    photo_label.config(image=captured_photo)
    photo_label.image = captured_photo

# Button to capture a photo
capture_button = tk.Button(app, text="Capture Photo", command=capture_photo)
capture_button.pack()

def train_model():
    global data
    data = train()

def evaluate_model():
    global predictionArray
    predictionArray = evaluate(data[0], data[1], data[2], data[3])
    prediction_label = tk.Label(app, text="Prediction: " + str(predictionArray[0]) + ", Confidence: " + str(predictionArray[1]))
    prediction_label.pack()

def display_image():
    display_updated_photo('output/heatmap.jpg')

train_button = tk.Button(app, text="Train", command=train_model)
train_button.pack()
evaluate_button = tk.Button(app, text="Evaluate", command=evaluate_model)
evaluate_button.pack()
# evaluate_button = tk.Button(app, text="Show GRADCAM", command=display_image)
# evaluate_button.pack()

# Label for displaying the captured photo
photo_label = tk.Label(app)
photo_label.pack()

prediction_label = tk.Label(app, text="Prediction: " + str(predictionArray[0]) + ", Confidence: " + str(predictionArray[1]))
prediction_label.pack()

# Run the GUI
app.mainloop()

# Release the webcam when the GUI is closed
cap.release()