from classes.QLearnAgent import QLearnAgent
from classes.MountainCar import MountainCar
from keras.backend import clear_session
from keras.models import load_model
clear_session()

agent = QLearnAgent(game = MountainCar(),
                    input_shape=MountainCar().input_shape,
                    output_shape=MountainCar().output_shape,
                    learning_rate = 0.01,
                    memory_length = 10000)

# to avoid the memory leak currently in keras, we'll have to train in groups of 10k before saving, restarting the keras
# session and loading the model back in:
while agent._game_counter <= 20000:
    agent.play_games(1, verbose=True)
    agent.batch_train(32, verbose=False)

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
