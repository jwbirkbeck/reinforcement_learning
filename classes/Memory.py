from collections import deque

class Memory:
    def __init__(self, memory_length):
        self._observation = deque(maxlen=memory_length)
        self._qpreds = deque(maxlen=memory_length)
        self._reward = deque(maxlen=memory_length)
        self._action = deque(maxlen=memory_length)

    def append_observation(self, observation):
        self._observation.append(observation)

    def append_qpreds(self, qpreds):
        self._qpreds.append(qpreds)

    def append_reward(self, reward):
        self._reward.append(reward)

    def append_action(self, action):
        self._action.append(action)