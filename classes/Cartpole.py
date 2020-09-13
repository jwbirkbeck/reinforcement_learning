import gym
from classes.Environment import Environment
from pynput.keyboard import Key

class Cartpole(Environment):
    def __init__(self):
        super().__init__()
        self.env = gym.make("CartPole-v0")
        self.observation_space = 4
        self.action_space = 2

    def take_action(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.observation = obs
        self.reward = reward
        self.done = done
        self.frames += 1
        if self.frames >= 200:
            self.done = True
            self.won = True
        elif done and self.frames < 200:
            self.lost = True

    # We define the key listening methods as part of the environment, as the number of actions (e.g. left or right) is
    # environment specific. These are actually called by the QLearnAgent method as part of the human_game() method.
    def on_press(self, key):
        if key == Key.left:
            self.pressed_action = 0
        if key == Key.right:
            self.pressed_action = 1

    def on_release(self, key):
        if key == Key.left or Key.right:
            return False