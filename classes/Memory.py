from collections import deque

class Memory:
    def __init__(self, memory_length):
        self._observation = deque(maxlen=memory_length)
        self._qpreds = deque(maxlen=memory_length)

    def append_observation(self, observation):
        self._observation.append(observation)

    def append_qpreds(self, qpreds):
        self._qpreds.append(qpreds)

    def wipe(self):
        self._observation.clear()
        self._qpreds.clear()