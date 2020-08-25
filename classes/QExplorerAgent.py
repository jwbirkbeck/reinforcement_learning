


from collections import deque
class StateMemory:
    def __init__(self, memory_length):
        self.current_state = deque(maxlen=memory_length)
        self.current_action = deque(maxlen=memory_length)
        self.next_state = deque(maxlen=memory_length)
        self.state_action_sample = None
        self.next_state_sample = None

    def wipe(self):
        self.current_state.clear()
        self.current_action.clear()
        self.next_state.clear()

    def sample(self, sample_size):
        memory_size = len(self.current_state)
        sample_index = np.random.choice(range(memory_size), sample_size, replace=True)
        state_sample = np.array(self.current_state)[sample_index]
        action_sample = np.array(self.current_action)[sample_index]

        self.state_action_sample = np.asarray([np.append(np.array(self.current_state)[i], np.array(self.current_action)[i]) for i in sample_index])
        self.next_state_sample = np.array(self.next_state)[sample_index]

    def wipe_sample(self):
        self.state_action_sample = None
        self.next_state_sample = None



from collections import deque
class QMemory:
    def __init__(self, memory_length):
        self.state = deque(maxlen=memory_length)
        self.q = deque(maxlen=memory_length)
        self.state_sample = None
        self.q_sample = None

    def wipe(self):
        self.state.clear()
        self.q.clear()

    def sample(self, sample_size):
        memory_size = len(self.state)
        sample_index = np.random.choice(range(memory_size), sample_size, replace=True)
        self.state_sample = np.array(self.state)[sample_index]
        self.q_sample = np.array(self.q)[sample_index]

    def wipe_sample(self):
        self.state_sample = None
        self.q_sample = None


import numpy as np
from classes.NeuralNet import NeuralNet
from classes.Memory import Memory
from pynput.keyboard import Listener
from gym.wrappers import Monitor
import time
class QExplorerAgent:

    def __init__(self, game, learning_rate, memory_length, q_mem_length, epsilon, discount_rate = 0.99):
        self.game = game
        self.state_memory = StateMemory(memory_length)
        self.q_memory = QMemory(memory_length)
        self.state_model = NeuralNet(input_shape=game.observation_space + 1,
                                    output_shape=game.observation_space,
                                     learning_rate=learning_rate)
        self.q_model = NeuralNet(input_shape=game.observation_space,
                                    output_shape=1,
                                     learning_rate=learning_rate)

        self._discount_rate = discount_rate
        self._total_wins = 0
        self._game_counter = 0
        self.winstreak = 0
        self._epsilon = epsilon
        self.exploit_q_mem = deque(maxlen=q_mem_length)
        self.explore_q_mem = deque(maxlen=q_mem_length)
        self.q_mem_length = q_mem_length
        for _ in range(q_mem_length):
            game.reset_env()
            self.exploit_q_mem.append(self.q_model.predict(self.game.observation)[0][0])
            self.explore_q_mem.append(self.q_model.predict(self.game.observation)[0][0])


    def calc_action(self, state_preds, q_preds):
        # if len(self.exploit_q_mem) + len(self.exploit_q_mem) < 2 * self.q_mem_length:
        #     self.exploit_q_mem.append(np.max(q_preds))
        #     self.explore_q_mem.append(np.max(q_preds))
        # sum_explore_q = sum(self.explore_q_mem)
        # sum_exploit_q = sum(self.exploit_q_mem)
        # min_q = min(min(self.explore_q_mem), min(self.exploit_q_mem))
        # # probability is the probability we should pick exploration
        # probability = (sum_explore_q - min_q) / (sum_explore_q + sum_exploit_q - 2 * min_q)
        # # transform to account for epsilon
        # probability = self._epsilon + (1 - 2 * self._epsilon) * probability
        # rand_num = np.random.uniform()
        # if rand_num < probability and self._game_counter > 10:
        #     action = self._calc_explore_action(state_preds=state_preds, q_preds=q_preds)
        # else:
        #     action = self._calc_exploit_action(q_preds=q_preds)
        rand_num = np.random.uniform()
        if rand_num < self._epsilon:
            action = self.game.env.action_space.sample()
        else:
            action = np.argmax(q_preds)
        return(action)

    def _calc_exploit_action(self, q_preds):
        action = np.argmax(q_preds)
        self.exploit_q_mem.append(np.max(q_preds))
        return(action)

    def _calc_explore_action(self, state_preds, q_preds):
        distances = [self._distance(state) for state in state_preds]
        action = np.argmax(distances)
        self.explore_q_mem.append(q_preds[action][0])
        return(action)

    def _distance(self, state):
        # mahalanobis distance
        data = np.asarray(self.state_memory.current_state)
        obs_minus_mean_obs = state - np.mean(data, 0)
        cov = np.cov(data.T)
        # if np.linalg.cond(cov) < 1 / float_info.epsilon & self.inv_covmat is not None:
        #     inv_covmat = np.linalg.inv(cov)
        #     self.inv_covmat = inv_covmat
        # else:
        #     inv_covmat = self.inv_covmat
        inv_covmat = np.linalg.inv(cov)
        left_term = np.dot(obs_minus_mean_obs, inv_covmat)
        mahal = np.dot(left_term, obs_minus_mean_obs.T)
        return mahal[0][0]


    def _play_game(self, verbose = False):
        self.game.reset_env()
        action_range = range(self.game.action_space)
        while not self.game.done:
            current_state = self.game.observation
            # State model: predict next states.
            state_action_stacks = [[] for _ in action_range]
            state_preds = [[] for _ in action_range]
            for i in action_range:
                state_action_stacks[i] = np.append(current_state, i)
                state_preds[i] = self.state_model.predict(state_action_stacks[i])

            # Q model:
            #   predict q value for each state prediction.
            qpreds = [self.q_model.predict(i)[0] for i in state_preds]
            action = self.calc_action(state_preds, qpreds)
            self.game.take_action(action)

            # State model: store the next state for training.
            next_state = self.game.observation
            next_qpred = self.q_model.predict(next_state)[0][0]
            self.state_memory.current_state.append(current_state)
            self.state_memory.current_action.append(action)
            self.state_memory.next_state.append(next_state)

            # Q Model: calculate the q values for each of the states, by predicting forward one more step using
            # the state model

            if not self.game.done:
                q_model_outcome = self.game.reward + self._discount_rate * next_qpred

            elif self.game.done:
                self._game_counter += 1
                q_model_outcome = self.game.reward

                if self.game.won:
                    self.winstreak += 1
                    self._total_wins += 1

                elif self.game.lost:
                    self.winstreak = 0

                if verbose:
                    print('{} frames in game {}, on a winstreak of {}. Total wins {}'.format(self.game.frames,
                                                                                             self._game_counter,
                                                                                             self.winstreak,
                                                                                             self._total_wins))

            self.q_memory.state.append(next_state)
            self.q_memory.q.append(q_model_outcome)

    def play_games(self, num_games, verbose):
        for _ in range(num_games):
            self._play_game(verbose)

    def train_state_model(self, batch_size, verbose = False):
        self.state_memory.sample(sample_size=batch_size)
        self.state_model.train_model(x=self.state_memory.state_action_sample, y=self.state_memory.next_state_sample, verbose=verbose)
        self.state_memory.wipe_sample()

    def train_q_model(self,batch_size, verbose = False):
        self.q_memory.sample(sample_size=batch_size)
        self.q_model.train_model(x=self.q_memory.state_sample, y=self.q_memory.q_sample, verbose=verbose)
        self.q_memory.wipe_sample()

    #def display_gameplay(self, save_gif=False):