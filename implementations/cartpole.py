from classes.QLearnAgent import QLearnAgent
from classes.Cartpole import Cartpole

agent = QLearnAgent(game = Cartpole(),
                    input_shape=4,
                    output_shape=2,
                    learning_rate = 0.001)

# agent.brain.keras_clear_session()
while agent.winstreak < 50:
    agent.play_games(1, verbose=True)
    agent.batch_train(32, verbose = False)


# Alternatively, we can teach the neural net by playing the game ourselves a few times:
agent.wipe_memory()
agent.human_game()
# since the learning rate is low for the neural net, but we want the network to learn a lot from us, we can
# batch learn multiple times on our game. It would be better to play a few different games and monitor the performance
# of the neural net but this hacky approach works for now.
for _ in range(100):
    agent.batch_train(32)