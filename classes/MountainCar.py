import gym
from classes.Environment import Environment
from pynput.keyboard import Key

class MountainCar(Environment):
    def __init__(self):
        super().__init__()
        self.env = gym.make("MountainCar-v0")
        self.observation_space = 2
        self.action_space = 3

    def take_action(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.observation = obs
        self.reward = obs[0]
        self.done = done
        self.frames += 1
        if done and self.observation[0] >= 0.5:
            self.done = True
            self.won = True
        elif done and self.observation[0] < 0.5:
            self.lost = True

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