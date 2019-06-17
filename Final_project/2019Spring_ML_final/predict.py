from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.backend.tensorflow_backend import set_session
import numpy as np
import pylab as plt
import tensorflow as tf
import os
from data_utils import *
from PIL import Image
from generator import *
from keras.models import model_from_json 


# Set session for keras's backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.gpu_options.per_process_gpu_memory_fraction=0.9
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# Load model
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss="binary_crossentropy", optimizer="adam")


# Load data
input_frames = get_test_batch(0, 500)

for i in range(input_frames.shape[0]) :
    os.makedirs('./test_predicted/sequence%03d' % i, exist_ok=True)
    track = input_frames[i]
    for j in range(10) :
        new_pos = loaded_model.predict(track[np.newaxis, ::, ::, ::, ::])
        new = new_pos[::, -1, ::, ::, ::]
        track = np.concatenate((track, new), axis=0)
    
    for t in range(10, 20) :
        img = Image.fromarray((track[t] * 255).astype(np.uint8))
        img.save('./test_predicted/sequence%03d/frame%02d.png' % (i, t-10))
