from classes.QLearnAgent import QLearnAgent
from classes.MountainCar import MountainCar
from keras.backend import clear_session
clear_session()

agent = QLearnAgent(game = MountainCar(),
                    input_shape=MountainCar().input_shape,
                    output_shape=MountainCar().output_shape,
                    learning_rate = 0.001,
                    memory_length = 10000)


while agent.winstreak < 10:
    agent.play_games(1, verbose=True)
    agent.batch_train(32, verbose = False)

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