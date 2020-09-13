import gym

class Environment:
    def __init__(self):
        self.env = None
        self.observation = None
        self.reward = None
        self.done = False
        self.frames = 0
        self.won = False
        self.lost = False
        self.observation_space = None
        self.action_space = None
        self.pressed_action = None

    def reset_env(self):
        self.observation = self.env.reset()
        self.reward = None
        self.done = False
        self.frames = 0
        self.won = False
        self.lost = False

    # Placeholder methods which warn if these methods (used for a human to interact with the environment) are used
    # without the environment's class having defined them.
    def on_press(self, key):
        raise NotImplementedError('on_press has not been defined for this environment.')

    def on_release(self, key):
        raise NotImplementedError('on_release has not been defined for this environment.')