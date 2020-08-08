import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras import backend

class NeuralNet:
    def __init__(self, input_shape, output_shape, learning_rate):
        self._input_shape = input_shape
        self._output_shape = output_shape


        self.model = Sequential([
            Dense(100, input_shape=(input_shape,)),
            Activation('tanh'),
            Dense(80),
            Activation("tanh"),
            Dense(60),
            Activation("tanh"),
            Dense(40),
            Activation("tanh"),
            Dense(output_shape),
            Activation('linear')
            ])

        optimizer = optimizers.Adam(lr=learning_rate)

        self.prediction = None
        self.action = None

        self.model.compile(optimizer=optimizer,
                           loss='mse',
                           metrics=['accuracy'])

    def predict(self, observation):
        obs = observation.reshape((1, self._input_shape))
        prediction = self.model.predict(obs)
        self.prediction = prediction

    # The train output is not useful, the model itself will have been updated so this can be lost
    def train_model(self, x, y, verbose):
        if verbose:
            train_output = self.model.fit(x=x, y=y, verbose=1)
        else:
            train_output = self.model.fit(x=x, y=y, verbose=0)

    def keras_clear_session(self):
        backend.clear_session()

    # def save_model
    # def load_model