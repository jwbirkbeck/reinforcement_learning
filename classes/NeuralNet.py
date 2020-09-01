import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
from tensorflow.keras import backend
from tensorflow.compat.v1 import disable_eager_execution

#disable_eager_execution()

class NeuralNet:
    def __init__(self, input_shape, output_shape, learning_rate):
        self._input_shape = input_shape
        self._output_shape = output_shape

        # Hardcoded network structure for now, but this can be parameterised
        self.model = Sequential([
            Dense(30, input_shape=(input_shape,)),
            Activation('relu'),
            Dense(20),
            Activation("relu"),
            Dense(10),
            Activation("relu"),
            Dense(output_shape),
            Activation('linear')
            ])

        optimizer = optimizers.Adam(lr=learning_rate)

        self.model.compile(optimizer=optimizer,
                           loss='mse',
                           metrics=['accuracy'])

    def predict(self, observation):
        obs = observation.reshape((1, self._input_shape))
        prediction = self.model.predict_on_batch(obs)
        return prediction

    # The train output is not useful, the model itself will have been updated so this can be lost
    def train_model(self, x, y, verbose):
        if verbose:
            train_output = self.model.train_on_batch(x=x, y=y)
        else:
            train_output = self.model.train_on_batch(x=x, y=y)

    def keras_clear_session(self):
        backend.clear_session()

    # def save_model
    # def load_model