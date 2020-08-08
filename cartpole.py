
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


# Initial constants and model setup
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
# This is a pretty big neural network! It's beyond the scope of this to find a small network that works. This size
# network solves the cartpole environment in around 3000-5000 episodes or games.
inputLength = 4
outputLength = 2
model = Sequential([
    Dense(100, input_shape=(inputLength,)),
    Activation('tanh'),
    Dense(80),
    Activation("tanh"),
    Dense(60),
    Activation("tanh"),
    Dense(40),
    Activation("tanh"),
    Dense(outputLength),
    Activation('linear')
])


# Using ADAM, which is like SGD, but does some magic stuff as well to control gradient descent. Adam is a popular
# choice online for reinforcement learning hence the use here. I'm not going to play around with the extra parameters
# of the ADAM algorithm here.
optimizer = optimizers.Adam(lr=learningRate)

model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['accuracy'])


# Setting up the gym environment
env = gym.make("CartPole-v0")
observation = env.reset()
# The "memory" is a special array type, it's a double ended queue that's fast to call from. It does mean that there's
# some scrappy code which formats the data to be put into the network so that it'll work. The memory has a max length
# constant. The memory is used later, for now just view it as storing the previous <long-period> of
# games, across episodes.
# When max length is reached, the data  at the old end of the deque is lost when new data is written at the new end.
memory = deque(maxlen=memoryLength)

# For a robust network, I'd like the model to be able to 'solve' the cartpole 50 times in a row
nSolved = 50

# This logic is scatterbrained coding, but outlines the logic more directly than the more ordered class based approach does.
while numWinstreak < nSolved:

    # Use the model to predict the best action (1 - epsilon) percent of the time:
    QPreds = model.predict(observation.reshape((1, inputLength)))
    if np.random.uniform() < epsilon:
        # epsilon percent of the time, we'll choose a random action to 'explore':
        action = env.action_space.sample()
    else:
        action = np.argmax(QPreds)

    # Store the current obs before stepping the game:
    prevObs = observation

    # Step the game forward - observing the next step of the game:
    observation, reward, done, _ = env.step(action)
    numFrames += 1

    # Assess the predicted Qs in the new frame of the game, we will use them to update the previous action's Qs:
    # (This is conceptually the hardest part of deep q learning - we're using the model's predictions at the next step
    # to update the model which seems pointless.
    # A way to think about this: At first, the model is only really learning useful info from the final frame, i.e. what a loss
    # state is, and that it should be avoided. As many games are played, it begins to learn to spot potential loss states
    # frames into future based on it's evaluation of the next frame, until it (for cartpole) is avoiding a loss state for 200 frames!
    newQPreds = model.predict(observation.reshape(1, inputLength))

    # If the game isn't done, update the Q values while the action hasn't been updated yet:
    if not done:
        QPreds[0, action] = reward + discountRate * np.max(newQPreds)
        # store the previous state with the updated predicted Qs so we can learn from it;
        memory.append([prevObs, QPreds[0]])


    maxNumFrames = 200 # should be outside loop but this is a working POC
    if done or numFrames >= maxNumFrames:
        # if env is won, count it for the winstreak:
        if numFrames >= maxNumFrames:
            numWinstreak += 1
        else:
            numWinstreak = 0

        # if env is done and the NN didn't win, then give a negative reward to the Q predictions:
        if numFrames < maxNumFrames:
            QPreds[0, action] = -1
        # but if the env IS solved, then set the reward to be the standard +1 per frame. This is subtle, but
        # think of it as we're punishing failure rather than rewarding success directly, but over a few thousand
        # episodes this gives the same outcome - carrot vs stick
        else:
            QPreds[0, action] = reward

        # As before, store this last state against the update Q predictions in memory
        memory.append([prevObs, QPreds[0]])

        # Step the number of episodes/games:
        episodeNum += 1
        # (playing with explore/exploit curves, unneeded really):
        epsilon = 50.0 / (50.0 + episodeNum)

        # Every <episodesTilLearn>, have the model learn from a sample of size frameBatchSize:
        # In this final version, the model learns a little after every game
        if episodeNum % episodesTilLearn == 0:
            # sample the memory to reduce correlation between observations we're learning from. frameBatchSize is tiny
            # compared to the memorySize, so correlation should be very small.
            sample = np.array(random.sample(memory, min([frameBatchSize, memory.__len__()])))

            # console printing for user
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
