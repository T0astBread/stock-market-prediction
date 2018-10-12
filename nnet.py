import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import datetime

NUMBER_OF_STOCKS = 500
NEURONS_IN_LAYER_1 = 1024
NEURONS_IN_LAYER_2 = 512
NEURONS_IN_LAYER_3 = 256
NEURONS_IN_LAYER_4 = 128
OUTPUT_SIZE = 1

BATCH_SIZE = 256
EPOCHS = 10


def get_log_dir():
    time_str = datetime.datetime.now().isoformat().replace(':', '-')
    return 'logs/' + time_str

tensorboard_callbacks = keras.callbacks.TensorBoard(log_dir=get_log_dir(), histogram_freq=0, write_graph=True)

def get_model():
    return keras.Sequential([
        layers.Dense(NEURONS_IN_LAYER_1, input_shape=(NUMBER_OF_STOCKS,), activation='relu'),
        layers.Dense(NEURONS_IN_LAYER_2, input_shape=(NEURONS_IN_LAYER_1,), activation='relu'),
        layers.Dense(NEURONS_IN_LAYER_3, input_shape=(NEURONS_IN_LAYER_2,), activation='relu'),
        layers.Dense(NEURONS_IN_LAYER_4, input_shape=(NEURONS_IN_LAYER_3,), activation='relu'),
        layers.Dense(OUTPUT_SIZE, input_shape=(NEURONS_IN_LAYER_4,)),
        layers.Lambda(lambda x: tf.transpose(x))
    ])


def compile(model: keras.Sequential):
    model.compile('Adam', loss='mean_squared_error')


def train(model: keras.Sequential, inputs, results):
    model.fit(x=inputs, y=results, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[tensorboard_callbacks])
