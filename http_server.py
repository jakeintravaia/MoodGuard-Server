from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import base64
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import glob
import math
import json


# This function loads the image in a way the model can interpret it
def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [256, 256])  # Resize images to a fixed size
    img = img / 255.0  # Normalize pixel values
    return img

# This function deletes all the temp images from our server after processing
def purge_images():
    files = os.listdir('./server-images')
    image_paths = [os.path.join('./server-images/', i) for i in files]
    for image in image_paths:
        os.remove(image)

    
# This function takes base64 encoded images and saves them temporarily into our server for
# processing in our model
def output_images(base64_data):
    try:
        for i in range(0, len(base64_data)):
            image_data = base64.b64decode(base64_data[i])
            with open(f'./server-images/output-{i}.png', 'wb') as file:
                file.write(image_data)
    except Exception as e:
        print(f"Error: {e}")
# This function loads our image sentiment model
def load_h5():
    print('Loading model...')
    model = load_model('./model/main.h5')
    print('Model loaded.')
    return model

# Setting our model variable to our image sentiment model
model = load_h5()

# This function does our predictions and returns a JSON text of the results
def do_predictions():
    print('Predicting...')
    files = os.listdir('server-images') # Get image outputs
    image_paths = [os.path.join('./server-images/', i) for i in files]
    images = []
    predictions = []
    for image in image_paths:
        images.append(load_image(image))

    for image in images:
        image = np.asarray(image)
        pred = model.predict(image.reshape(1, 256, 256, 3))
        pred = pred.flatten()
        rounded = round(pred[0])
        predictions.append(rounded)
    print(predictions)
    response = {
        "positive": predictions.count(1),
        "negative": predictions.count(0)
        }
    print(response)
    return json.dumps(response, indent=4)

# This POST request handler code is derived from the below link
# https://stackoverflow.com/questions/66514500/how-do-i-configure-a-python-server-for-post
# This is a modified version for our HTTP server that works with MoodGuard, but the code
# is not strictly ours, so credits go to the kind user that shared his code.

class handler(BaseHTTPRequestHandler):   
    def do_POST(self):
        print("POST REQUEST RECEIVED.")
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        # Print the POST data to the console

        decoded = json.loads(post_data.decode('utf-8')) # Transform to dict

        base64_data = decoded["base64DataArray"]

        # Clean up base64 data
        
        for i in range(len(base64_data)):
            split_data = base64_data[i].split(',')
            base64_data[i] = split_data[1]

        #print(base64_data)

        output_images(base64_data) # Save images temporarily to server
        message = do_predictions()
        purge_images() # Purge images from server

        #print(base64_data[0])
        
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()

        self.wfile.write(bytes(message, "utf8"))

with HTTPServer(('127.0.0.1', 8000), handler) as server:
    server.serve_forever()
