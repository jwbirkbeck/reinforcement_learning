import numpy as np
from classes.NeuralNet import NeuralNet
from classes.Memory import Memory
from pynput.keyboard import Listener
from gym.wrappers import Monitor
import time

class QLearnAgent:

    def __init__(self, game, input_shape, output_shape,
                 epsilon=0.1, memory_length=1000, discount_rate=0.99,
                 learning_rate = 0.001):
        self.brain = NeuralNet(input_shape=input_shape,
                               output_shape=output_shape,
                               learning_rate=learning_rate)
        # Note that the current QLearnAgent only uses the observation and qpreds deques in Memory, even though the
        # Memory class allows me to store the rewards and actions as well.
        self.memory = Memory(memory_length=memory_length)
        self.game = game
        self.winstreak = 0
        self._epsilon = epsilon
        self._discount_rate = discount_rate
        self._mem_sample_obs = None
        self._mem_sample_qpreds = None
        self._total_wins = 0
        self._game_counter = 0

    def wipe_memory(self):
        self.memory._observation.clear()
        self.memory._qpreds.clear()
        self.memory._reward.clear()
        self.memory._action.clear()

    def calc_action(self):
        rand_num = np.random.uniform()
        if rand_num < self._epsilon:
            action = self.game.env.action_space.sample()
        else:
            action = np.argmax(self.brain.predict(self.game.observation))
        return action

    def _play_game(self, verbose = False):
        self.game.reset_env()
        while not self.game.done:
            current_qpreds = self.brain.predict(self.game.observation)
            current_action = self.calc_action()
            current_obs = self.game.observation

            self.game.take_action(current_action)
            next_prediction = self.brain.predict(self.game.observation)

            if not self.game.done:
                current_qpreds[0, current_action] = self.game.reward + self._discount_rate * np.max(next_prediction)
                self.memory.append_observation(current_obs)
                self.memory.append_qpreds(current_qpreds[0])
            # If the game is done, we've either lost or won
            elif self.game.done and self.game.won:
                self._game_counter +=1
                self.winstreak += 1
                self._total_wins += 1
                # we've won, reward the win but do not use the next frame's predictions are they are not relevant
                current_qpreds[0, current_action] = self.game.reward
                self.memory.append_observation(current_obs)
                self.memory.append_qpreds(current_qpreds[0])
                if verbose:
                    print('{} frames in game {}, on a winstreak of {}. Total wins {}'.format(self.game.frames,
                                                                                             self._game_counter,
                                                                                             self.winstreak,
                                                                                             self._total_wins))

            elif self.game.done and self.game.lost:
                self._game_counter += 1
                # we've lost, do stuff
                current_qpreds[0, current_action] = self.game.reward
                self.memory.append_observation(current_obs)
                self.memory.append_qpreds(current_qpreds[0])
                self.winstreak = 0
                if verbose:
                    print('{} frames in game {}, on a winstreak of {}. Total wins {}'.format(self.game.frames,
                                                                                             self._game_counter,
                                                                                             self.winstreak,
                                                                                             self._total_wins))

    def play_games(self, num_games, verbose):
        for _ in range(num_games):
            self._play_game(verbose)

    def _sample_memory(self, sample_length):
        memory_size = len(self.memory._observation)
        sample_index = np.random.choice(range(memory_size), sample_length, replace = True)
        self._mem_sample_obs = np.array(self.memory._observation)[sample_index]
        self._mem_sample_qpreds = np.array(self.memory._qpreds)[sample_index]

    def _flush_memory_samples(self):
        self._mem_sample_obs = None
        self._mem_sample_qpreds = None

    def batch_train(self, sample_size, verbose = False):
        self._sample_memory(sample_size)
        self.brain.train_model(x = self._mem_sample_obs, y = self._mem_sample_qpreds, verbose=verbose)
        self._flush_memory_samples()

    def display_gameplay(self, save_gif = False):
        if save_gif:
            self.game.env = Monitor(self.game.env, 'recording', force=True)
        self.game.reset_env()
        while not self.game.done:
            self.brain.predict(self.game.observation)
            action = self.calc_action()
            self.game.env.render()
            self.game.take_action(action)
            if self.game.done and self.game.won:
                time.sleep(0.75)
                print("game won")
                self.game.env.close()

            if self.game.done and self.game.lost:
                time.sleep(0.75)
                print("game lost")
                self.game.env.close()

    def human_game(self):
        self.game.reset_env()
        while not self.game.done:
            self.brain.predict(self.game.observation)
            self.game.env.render()

            # not calculating the action here because the human will input the action themselves
            current_qpreds = self.brain.predict(self.game.observation)
            current_obs = self.game.observation

            # Inside the game's on_press method, the translation between human key press and game action is made.
            with Listener(
                    on_press=self.game.on_press,
                    on_release=self.game.on_release) as listener:
                listener.join()
            action = self.game.pressed_action

            self.game.take_action(action)
            next_prediction = self.brain.predict(self.game.observation)

            if not self.game.done:
                current_qpreds[0, action] = self.game.reward + self._discount_rate * np.max(next_prediction)
                self.memory.append_observation(current_obs)
                self.memory.append_qpreds(current_qpreds[0])
            # If the game is done, we've either lost or won
            elif self.game.done and self.game.won:
                self.game.env.close()
                self._game_counter += 1
                self.winstreak += 1
                # we've won, reward the win but do not use the next frame's predictions are they are not relevant
                current_qpreds[0, action] = self.game.reward
                self.memory.append_observation(current_obs)
                self.memory.append_qpreds(current_qpreds[0])
                print('{} frames in game {}, on a winstreak of {}'.format(self.game.frames, self._game_counter,
                                                                          self.winstreak))

            elif self.game.done and self.game.lost:
                self.game.env.close()
                self._game_counter += 1
                # we've lost, do stuff
                current_qpreds[0, action] = -1
                self.memory.append_observation(current_obs)
                self.memory.append_qpreds(current_qpreds[0])
                self.winstreak = 0
                print('{} frames in game {}, on a winstreak of {}'.format(self.game.frames, self._game_counter,
                                                                          self.winstreak))
