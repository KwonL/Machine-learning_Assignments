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

# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.

# Set session for keras's backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.gpu_options.per_process_gpu_memory_fraction=0.9
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

seq = Sequential()
seq.add(ConvLSTM2D(filters=64, kernel_size=(1, 1),
                   input_shape=(None, 64, 64, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=64, kernel_size=(2, 2),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=64, kernel_size=(1, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=64, kernel_size=(2, 2),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=3, kernel_size=(1, 1, 1),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='binary_crossentropy', optimizer='adam')


# Train the network
# noisy_movies, shifted_movies = generate_movies(n_samples=1)
sample_input, sample_output = get_train_batch(batch_size = 100)
noisy_movies = np.concatenate((sample_input, sample_output[:, :-1]), axis=1)
shifted_movies = np.concatenate((sample_input[:, 1:], sample_output), axis=1)
# seq.fit(noisy_movies, shifted_movies, batch_size=10,
#         epochs=10, validation_split=0.05)

training_generator = DataGenerator(batch_size=10)
validation_generator = ValidGenerator(batch_size=10)
seq.fit_generator(
    generator=training_generator,
    validation_data=validation_generator,
    use_multiprocessing=True,
    workers=12,
    epochs=40
)

# Save model
model_json = seq.to_json()
with open("./model.json", 'w') as f :
    f.write(model_json)
seq.save_weights('./model.h5')

# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions
which = 4
track = noisy_movies[which][:10, ::, ::, ::]

for j in range(20):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    print("shape: %s, %s" % (track.shape, new.shape))
    track = np.concatenate((track, new), axis=0)


# And then compare the predictions
# to the ground truth
track2 = noisy_movies[which][::, ::, ::, ::]
os.makedirs('./output_example', exist_ok=True)
for i in range(19):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 7:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=20)

    toplot = track[i, ::, ::, :]

    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)

    toplot = track2[i, ::, ::, :]
    if i >= 2:
        toplot = shifted_movies[which][i - 1, ::, ::, :]

    plt.imshow(toplot)
    plt.savefig('./output_example/%i_animate.png' % (i + 1))
