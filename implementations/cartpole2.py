from classes.QExplorerAgent import QExplorerAgent
from classes.Cartpole import Cartpole
from keras.backend import clear_session
clear_session()

agent = QExplorerAgent(game = Cartpole(),
                    learning_rate = 0.01,
                    memory_length = 10000,
                       q_mem_length=50,
                       epsilon = 0.2)

# agent.brain.keras_clear_session()
while agent.winstreak < 50:
    agent.play_games(1, verbose=True)
    agent.train_state_model(200, verbose=False)
    agent.train_q_model(200, verbose=True)

agent.q_memory.wipe()
agent.state_memory.wipe()

sum(agent.exploit_q_mem)
sum(agent.explore_q_mem)
min_q = min(min(agent.explore_q_mem), min(agent.exploit_q_mem))
import numpy as np
tmp = agent.state_memory.state_action_sample

(sum(agent.explore_q_mem) - min_q) / (sum(agent.explore_q_mem) + sum(agent.exploit_q_mem) - 2 * min_q)



agent.q_memory.q