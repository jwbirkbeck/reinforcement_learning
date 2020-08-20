import numpy as np

# I'll generate some simple data just to test that the neural network and the backprop algorithm is correctly constructed.
# An extension to this would be to use e.g. the MNIST dataset to test a more real world scenario.

# 4 datapoints with 4 inputs to the NN, and then 2 labels.

X = np.array([[0,0,0,1],
              [0,0,1,0],
              [0,1,0,0],
              [1,0,0,0]])
Y = np.array([0,1,0,1])

# A correct NN would see that a '1' in the first or third input corresponds with an output value of 0, and similarly the second
# or fourth input corresponds with an output of 1.

# The design of the neural network will be a 4, 4, 1. This is more complicated than it needs to be (we could do without
# the hidden layer and still train to 100% accuracy) but it tests the backprop algo is correct. Between the hidden layer
# and the output layer I will have a sigmoid activation layer

# With this design of NN, we need a weights matrix which is 4x4 between the input layer and the hidden layer and a
# length 4 vector for weights between the hidden layer and the output layer (which has length 1).

class NN:
    def __init__(self):
        # This is using the simple initialisation from the module notes - N(0,0.1) as small random weights. Mentions He
        # and Xavier but that's beyond the requirements of this example.
        self.W1 = np.random.normal(0,0.1,16).reshape((4,4))
        self.w2 = np.random.normal(0,0.1,4)
        self.b1 = np.zeros(4)
        self.b2 = np.zeros(1)
        self.learning_rate = 0.01

    def acti(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x_vec):
        # Matrix multiply input observation by W1 to get z. Then put z through activation.
        Z1 = np.matmul(self.W1, x_vec) + self.b1
        H1 = self.acti(Z1)
        z2 = np.matmul(self.w2, H1) + self.b2
        prediction = self.acti(z2)
        return prediction

    def backprop(self, x_vec, y):
        prediction = self.predict(x_vec)
        # TODO: complete the below.  We don't want to actually return the logloss, we need to calculate the partial
        #  diff of the logloss w.r.t the weights and then update the weights based on how much they contribute to the
        #  overall logloss.
        logloss = -( (1 - y) * np.log(1 - prediction) + y * np.log(prediction))
        return(logloss)