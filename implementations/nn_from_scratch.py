import numpy as np

# For this work, I'm following lecture 10's module notes from this Stanford module I found online:
# http://cs229.stanford.edu/syllabus-summer2020.html 

# I'll generate some simple data just to test that the neural network and the backprop algorithm is correctly constructed.
# An extension to this would be to use e.g. the MNIST dataset to test a more real world scenario.

# 4 datapoints with 4 inputs to the NN, and then 2 labels.

X = np.array([[0,0,0,1],
              [0,0,1,0],
              [0,1,0,0],
              [1,0,0,0]])
Y = np.array([[0,1],[1,0],[0,1],[1,0]])

# A correct NN would see that a '1' in the first or third input corresponds with an output value of 0, and similarly the second
# or fourth input corresponds with an output of 1.

# The design of the neural network will be a in 4, 4, 1. This is more complicated than it needs to be (we could do without
# the hidden layer and still train to 100% accuracy) but it tests the backprop algo is correct. Between the hidden layer
# and the output layer I will have a sigmoid activation layer

# With this design of NN, we need a weights matrix which is 4x4 between the input layer and the hidden layer and a
# length 4 vector for weights between the hidden layer and the output layer (which has length 1).

W1 = np.random.normal(0,0.1,16).reshape((4,4))
W2 = np.random.normal(0,0.1,8).reshape((4,2))

learning_rate = 0.01
x_vec = X[0]


def acti(x):
    return 1 / (1 + np.exp(-x))

def deriv_acti(x):
    return x * (1-x)

def predict(x_vec):
    # Matrix multiply input observation by W1 to get z. Then put z through activation.
    Z1 = np.matmul(x_vec, W1)
    H1 = acti(Z1)
    Z2 = np.matmul(H1, W2)
    H2 = acti(Z2)
    prediction = acti(H2)
    return prediction

    # def backprop(self, x_vec, y):
    #     prediction = self.predict(x_vec)
    #     # TODO: complete the below.  We don't want to actually return the logloss, we need to calculate the partial
    #     #  diff of the logloss w.r.t the weights and then update the weights based on how much they contribute to the
    #     #  overall logloss.
    #
    #     # * find the error between the network's predictions and the outputs
    #     # * apply the derivative of the activation function (delta output sum)
    #     # * use delta output sum to work out how much error z2 contributed - we do this
    #     # with a dot product
    #     # * do same steps for previous layer
    #     error =  prediction - y
    #     deriv_error = self.deriv_acti(error)
    #     # Incorporate the learning rate:
    #     deriv_error *= self.learning_rate
    #
    #     second_layer_deriv_err = deriv_error.dot
    #     second_layer_delta_output_sum = second_layer_error.dot
    #     return(logloss)


prediction = predict(X[0])
y = Y[0]

o_error =  y - prediction
o_delta = o_error * deriv_acti(prediction)
# Incorporate the learning rate:
# deriv_error *= learning_rate

z2_error = o_delta.dot(W2.T)
z2_delta = z2_error * deriv_acti(Z1)

z1_error = z2_delta.dot(W1)

W1 += X[0]
