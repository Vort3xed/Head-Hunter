import os
from flask import Flask, render_template, request, send_from_directory, jsonify

# General Imports
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

# Imports for Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import imageio
from threading import Thread

app = Flask(__name__)

# Define the path where captured photos will be saved
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = load_model('AutismImageModel.h5')

def process_image(image, nrows, ncolumns):
    img = imageio.imread(image)
    resized_img = cv2.resize(img, (nrows, ncolumns), interpolation=cv2.INTER_CUBIC)
    if resized_img.shape[-1] != 3:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    nrows = 150
    ncolumns  = 150
    channels = 3

    image_path = 'uploads/captured_photo.jpg'

    X_test = process_image(image_path,nrows,ncolumns)
    X_test = np.expand_dims(X_test, axis=0)
    return resized_img

@app.route('/')
def index():
    return render_template('index.html')

global_prediction = None

@app.route('/predict', methods=('GET', 'POST'))
def predict():

    if request.method == 'POST':
        nrows = 150
        ncolumns  = 150
        channels = 3
        image_path = 'uploads/captured_photo.jpg'

        def predict_worker():
            
            try: 
                with app.app_context():
                    X_test = process_image(image_path,nrows,ncolumns)
                    X_test = np.expand_dims(X_test, axis=0)

                    prediction = model.predict(X_test)

                    global global_prediction
                    global_prediction = prediction
                    print(prediction)
                    
                    print(global_prediction)
                    return render_template('index.html', prediction=global_prediction)
            except Exception as e:
                print(e)
                return render_template('index.html', prediction='Error')


        prediction_thread = Thread(target=predict_worker)
        prediction_thread.start()

        if (global_prediction == None):
            return render_template('index.html', prediction='Loading...')
        else:
            return render_template('index.html', prediction=global_prediction)

    return render_template('index.html', prediction=str(global_prediction))

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

def save_and_display_gradcam(img_path, heatmap, alpha=0.8):
    # folder to save the image in
    folder_path = 'static'
        
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

def gen_gradcam():
    model = load_model('vgg_model50.h5')
    photo_size = 224
    img_size = (photo_size, photo_size)
    # find the last layer for analysis
    model.summary()

    # save this layer
    last_conv_layer_name = "block5_conv3"

    # load in the image for analysis
    img_path = 'uploads/captured_photo.jpg'

    img_array = preprocess_input(img_path, target_size=img_size)

    # remove last layer's softmax
    model.layers[-1].activation = None

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    save_and_display_gradcam(img_path, heatmap)

@app.route('/gengradcam', methods=('GET', 'POST'))
def gengradcam():
    print("Generating GradCAM")
    gen_gradcam()
    return render_template('index.html', prediction='0', image_url='static\heatmap.jpg')

@app.route('/upload', methods=['POST'])
def upload():
    photo = request.files['photo']

    if photo:
        # Ensure the filename is secure to prevent potential security issues
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_photo.jpg')
        photo.save(filename)
        return filename

if __name__ == '__main__':
    app.run(debug=True)
