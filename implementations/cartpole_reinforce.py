from classes.ReinforceAgent import ReinforceAgent
from classes.Cartpole import Cartpole
from tensorflow.keras.backend import clear_session
clear_session()

agent = ReinforceAgent(game = Cartpole(),
                input_shape=Cartpole().observation_space,
                output_shape=Cartpole().action_space,
                learning_rate=0.001)

# agent.brain.keras_clear_session()
while agent.winstreak < 50:
    agent.play_games(1, verbose=True)


import numpy as np

agent.brain.model.predict(np.asarray(agent.game.observation).reshape(1,4))

