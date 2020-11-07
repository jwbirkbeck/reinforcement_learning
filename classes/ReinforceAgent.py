import numpy as np
from classes.NeuralNet import NeuralNet
from collections import deque

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers

class ReinforceNeuralNet:
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
            # softmax turns logs into probs
            Activation('softmax')
            ])

        self.optimizer = optimizers.Adam(lr=learning_rate)

        self.model.compile(optimizer=self.optimizer,
                           loss='mse',
                           metrics=['accuracy'])

    def predict(self, observation):
        obs = observation.reshape((1, self._input_shape))
        prediction = self.model.predict_on_batch(obs)
        return prediction

    # The train output is not useful, the model itself will have been updated so this can be lost
    def train_model(self, gradients_and_vars, verbose):
        self.optimizer.apply_gradients(gradients_and_vars)

class ReinforceMemory:
    def __init__(self, memory_length):
        self._observations = deque(maxlen=memory_length)
        self._probs = deque(maxlen=memory_length)
        self._actions = deque(maxlen=memory_length)
        self._rewards = deque(maxlen=memory_length)
        self.log_probs = None

    def append_observation(self, observation):
        self._observations.append(observation)

    def append_probs(self, probs):
        self._probs.append(probs)

    def append_reward(self, reward):
        self._rewards.append(reward)

    def append_actions(self, action):
        self._actions.append(action)

    def calc_log_probs(self):
        return(np.log(self._probs))

    def calc_discounted_future_rewards(self, discount_rate):
        rewards = np.asarray(self._rewards)

        calcd_rewards = np.zeros(len(rewards))
        # The total future rewards with discount of the end state is just the end state reward
        calcd_rewards[len(rewards) - 1] = rewards[len(rewards) - 1]
        # Step back from the terminal state, discount the future total rewards, and then add the current state's reward
        # TODO: profile this, and replace with the commented out matrix method if the for loop is slower than it
        for i in reversed(range(len(rewards) - 1)):
            calcd_rewards[i] = rewards[i] + (calcd_rewards[i + 1] * discount_rate)

        # The commented out bit below is the alternative to using a for loop - I construct a matrix where each row
        # represents the future discounting factors needed to calculate the future discounted reward at each step. This
        # matrix is then matmuled with the rewards to produce the same numbers as the below for loop. It's probably slower
        # than the for loop as there's _many_ more mults than the above - this can be profiled later, though
        # left = discount_rate ** np.arange(-1, -len(rewards) - 1, -1)
        # right = discount_rate ** np.arange(1, len(rewards) + 1)
        #
        # calcd_rewards = np.matmul(np.triu(np.outer(left, right)), rewards)
        return(calcd_rewards)

    def wipe(self):
        self._observations.clear()
        self._probs.clear()
        self._actions.clear()
        self._rewards.clear()
        self.log_probs = None


class ReinforceAgent:
    def __init__(self, game, input_shape, output_shape,
                 memory_length=1000, discount_rate=0.99,
                 learning_rate=0.01):
        self.brain = ReinforceNeuralNet(input_shape=input_shape,
                               output_shape=output_shape,
                               learning_rate=learning_rate)
        # Note that the current QLearnAgent only uses the observation and qpreds deques in Memory, even though the
        # Memory class allows me to store the rewards and actions as well.
        self.memory = ReinforceMemory(memory_length=memory_length)
        self.game = game
        self.winstreak = 0
        self._discount_rate = discount_rate
        self._mem_sample_obs = None
        self._mem_sample_probs = None
        self._total_wins = 0
        self._game_counter = 0

    def calc_action(self, predictions):
        action = np.argmax(predictions)
        return action

    def _play_game(self, verbose=False):
        self.game.reset_env()
        while not self.game.done:
            current_obs = self.game.observation
            current_prob_preds = self.brain.predict(current_obs)
            self.memory.append_observation(current_obs)
            self.memory.append_probs(current_prob_preds)


            current_action = self.calc_action(current_prob_preds)
            self.memory.append_actions(current_action)
            self.game.take_action(current_action)

            reward = self.game.reward
            self.memory.append_reward(reward)

            # game can now be done as we've taken a s
            if self.game.done:
                # admin first
                self._game_counter += 1
                if self.game.won:
                    self.winstreak += 1
                    self._total_wins += 1
                elif self.game.lost:
                    self.winstreak = 0
                if verbose and self._game_counter % 1 == 0:
                    print('{} frames in game {}, on a winstreak of {}. Total wins {}'.format(self.game.frames,
                                                                                             self._game_counter,
                                                                                             self.winstreak,
                                                                                             self._total_wins))

                # calculate the differential of J:
                discounted_future_rewards = self.memory.calc_discounted_future_rewards(self._discount_rate)
                log_probs = self.memory.calc_log_probs()
                # Now need to label the log probs for the action we didn't take as 0, so that we have a 0 gradient
                # for the action we didn't take
                actions = self.memory._actions
                action_log_probs = log_probs.reshape(len(log_probs), 2)
                action_log_probs[np.arange(len(log_probs)), list(actions)] = 0
                delta_J = discounted_future_rewards.reshape(len(log_probs), 1) * action_log_probs
                # We need to put a minus sign in because policy gradient methods are gradient ASCENT methods, and the
                # ReinforceNeuralNetwork (copied from NeuralNetwork class for now) is implemented to minimise loss rather than maximise -loss
                minus_delta_J = -delta_J
                gradients_and_vars = zip(minus_delta_J, np.asarray(self.memory._observations))
                self.brain.train_model(gradients_and_vars = gradients_and_vars, verbose = False)
                self.memory.wipe()

    def play_games(self, num_games, verbose):
        for _ in range(num_games):
            self._play_game(verbose)