from classes.QLearnAgent import QLearnAgent
from classes.MountainCar import MountainCar
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model
import gc
clear_session()

agent = QLearnAgent(game = MountainCar(),
                    input_shape=MountainCar().observation_space,
                    output_shape=MountainCar().action_space,
                    learning_rate = 0.01,
                    memory_length = 10000)

# to avoid the memory leak currently in keras, we'll have to train in groups of 10k before saving, restarting the keras
# session and loading the model back in:
while agent.winstreak < 25:
    agent.play_games(1, verbose=True)
    agent.batch_train(32, verbose=False)
    if agent._game_counter % 5000 == 0:
        agent.brain.model.save('mountaincar.model')



agent.brain.model.save('mountaincar.model')
clear_session()
agent.brain.model = load_model('mountaincar.model')


agent.display_gameplay()

# Alternatively, we can teach the neural net by playing the game ourselves a few times:
agent.memory.wipe()
agent.human_game()
# since the learning rate is low for the neural net, but we want the network to learn a lot from us, we can
# batch learn multiple times on our game. It would be better to play a few different games and monitor the performance
# of the neural net but this hacky approach works for now.
for _ in range(1000):
    agent.batch_train(32, verbose=True)

agent.display_gameplay()


agent.game.reset_env()
while True:
    agent.batch_train(32, verbose=False)