
import numpy as np
import os
import random
import gym
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras import backend

backend.clear_session()

# # #
# Initial constants and model setup
# # #

epsilon = 0.1
memoryLength = 1000
episodesTilLearn = 1
frameBatchSize = 32  # 32
learningRate = 0.001  # 0.001
discountRate = 0.99
numWinstreak = 0
numFrames = 0
episodeNum = 0
done = False


np.random.seed(1)

# 3 hidden layers, basic neural network with a linear output activation to 1-1 predict Q of each action
inputLength = 4
outputLength = 2
model = Sequential([
    Dense(100, input_shape=(inputLength,)), # 100
    Activation('tanh'),
    Dense(80), # 80
    Activation("tanh"),
    Dense(60), # 60
    Activation("tanh"),
    Dense(40), # 40
    Activation("tanh"),
    Dense(outputLength),
    Activation('linear')
])


# using ADAM, which is like SGD, but does some magic stuff as well to speed up gradient descent
optimizer = optimizers.Adam(lr=learningRate)

# set the model up (required before prediction or training)
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['accuracy'])

# # #
# Setting up the gym environment
# # #

env = gym.make("CartPole-v0")
observation = env.reset()
# The "memory" is a special array type, it's a double ended queue that's fast to call from. It does mean that there's
# some scrappy code which formats the data to be put into the network so that it'll work. The memory has a max length
# constant. The memory is used later, for now just view it as storing the previous <long-period> of
# games, across episodes
memory = deque(maxlen=memoryLength)

# # #
# Learning:
# # #

# For a robust network, I'd like the model to be able to 'solve' the env n times in a row
nSolved = 50
while numWinstreak < nSolved:

    # Choose an action by using the model:
    QPreds = model.predict(observation.reshape((1, inputLength)))
    if np.random.uniform() < epsilon:
        # epsilon percent of the time, we'll choose a random action to 'explore':
        action = env.action_space.sample()
    else:
        # Otherwise we'll chose the action which the model predicts as having the highest Q (potential reward)
        action = np.argmax(QPreds)

    # Store the current obs before stepping the game:
    prevObs = observation

    # step the game forward - render the next frame of the game:
    observation, reward, done, _ = env.step(action)
    numFrames += 1

    # Assess the predicted Qs for being in the new state, so we can use them to update the previous action's Qs:
    newQPreds = model.predict(observation.reshape(1, inputLength))

    # If the game isn't done, update the Q values while the action hasn't been updated yet:
    if not done:
        QPreds[0, action] = reward + discountRate * np.max(newQPreds)
        # store the previous state with the updated predicted Qs so we can learn from it;
        memory.append([prevObs, QPreds[0]])

    # Or, if the game IS done and not completed, update the QPreds for the last state with a large negative
    # reward, and then perform some additional processing the episode:
    maxNumFrames = 200
    if done or numFrames >= maxNumFrames:
        # (if env solved, count it):
        if numFrames >= maxNumFrames:
            numWinstreak += 1
        else:
            numWinstreak = 0

        # if env not solved, then give a negative reward to the Q predictions:
        if numFrames < maxNumFrames:
            QPreds[0, action] = -1
        # but if the env IS solved, then set the reward to be the standard +1 per frame. This is subtle, but
        # think of it as we're punishing failure rather than rewarding success directly, but over a few thousand
        # episodes this gives the same outcome - carrot vs stick
        else:
            QPreds[0, action] = reward

        # As before, store this last state against the update Q predictions in memory
        memory.append([prevObs, QPreds[0]])

        # Step the number of episodes:
        episodeNum += 1
        epsilon = 50.0 / (50.0 + episodeNum)

        # Every <episodesTilLearn>, have the model learn from a sample of size frameBatchSize:
        if episodeNum % episodesTilLearn == 0:
            # sample the memory to reduce correlation between observations we're learning from. frameBatchSize is tiny
            # compared to the memorySize, so correlation should be very small.
            sample = np.array(random.sample(memory, min([frameBatchSize, memory.__len__()])))

            # console printing for user benefit
            print('{} frames at episode {}, with {} winstreak'.format(numFrames, episodeNum, numWinstreak))

            # sample[:,0] is the previous state; sample[:,1] is the updated predicted Q.
            # We vstack both so that keras can work with it (because I used a double ended queue for speed, it needs
            # this reformatting)
            # Train the model on the sample:
            history = model.fit(x=np.vstack(np.array(sample[:, 0])), y=np.vstack(np.array(sample[:, 1])), verbose=0)
        # At end of each game/episode, reset the env, and start counting the frames from zero again
        observation = env.reset()
        numFrames = 0

    # If we have a streak of n frames consecutively, let's say we've won
    if numWinstreak == nSolved:
        print('Environment Solved!')
