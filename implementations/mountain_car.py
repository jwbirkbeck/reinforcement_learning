from classes.QLearnAgent import QLearnAgent
from classes.MountainCar import MountainCar
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model

clear_session()

agent = QLearnAgent(game = MountainCar(),
                    input_shape=MountainCar().observation_space,
                    output_shape=MountainCar().action_space,
                    learning_rate = 0.01,
                    memory_length = 10000)

# to avoid the memory leak currently in keras, we'll have to train in groups of 10k before saving, restarting the keras
# session and loading the model back in:
agent.brain.model = load_model('mountaincar.model')


while agent.winstreak < 25:
    agent.play_games(1, verbose=True)
    agent.batch_train(32, verbose=False)
    if agent._game_counter % 100 == 0:
        agent.brain.model.save('mountaincar.model')
        print("model saved!")


agent.display_gameplay()