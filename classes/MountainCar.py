import gym
from pynput.keyboard import Key, Listener

class MountainCar:
    # Init only sets the correct environment but does not start it, there's a method for that
    def __init__(self):
        self.env = gym.make("MountainCar-v0")
        self.observation = None
        self.reward = None
        self.done = False
        self.frames = 0
        self.won = False
        self.lost = False
        self.input_shape = 2
        self.output_shape = 3
        self.pressed_action = None

    # Each time we take an action, we want to store what we did, what happens in the next frame, and count the frame
    def take_action(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.observation = obs
        self.reward = reward
        self.done = done
        self.frames += 1
        if done and self.observation[0] >= 0.5:
            self.done = True
            self.won = True
        elif done and self.observation[0] < 0.5:
            self.lost = True

    # resets the env to the initial state.
    def reset_env(self):
        self.observation = self.env.reset()
        self.reward = None
        self.done = False
        self.frames = 0
        self.won = False
        self.lost = False

    # We define the key listening methods as part of the environment, as the number of actions (e.g. left or right) is
    # environment specific. These are actually called by the QLearnAgent method as part of the human_game() method.
    def on_press(self, key):
        if key == Key.left:
            self.pressed_action = 0
        if key == Key.up:
            self.pressed_action = 1
        if key == Key.right:
            self.pressed_action = 2

    def on_release(self, key):
        if key == Key.left or Key.right or Key.up:
            return False