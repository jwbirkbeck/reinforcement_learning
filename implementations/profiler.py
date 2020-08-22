import cProfile
from classes.QLearnAgent import QLearnAgent
from classes.MountainCar import MountainCar
from keras.backend import clear_session
clear_session()

agent = QLearnAgent(game = MountainCar(),
                    input_shape=MountainCar().input_shape,
                    output_shape=MountainCar().output_shape,
                    learning_rate = 0.01,
                    memory_length = 10000)


def func(agent):
    for _ in range(20):
        agent.play_games(1, verbose=True)
        agent.batch_train(32, verbose=False)

cProfile.run('agent._play_game()')


cProfile.run('func(agent)')