# Reinforcement Learning

### Purpose:

Solve the cartpole environment from the OpenAI Gym with a proof of concept script, before building generalisable classes to apply to more complicated environments.

I'm reading through _Reinforcement Learning: A Introduction_ by Sutton and Barto as I'm building this code, which I'll refer to as 'the book' below!


### Requirements:

For this project I've used:

* Ryzen 2600 and Vega56 GPU
* [AMD ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)
* [Tensorflow-ROCm](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream)
* Anaconda
* The OpenAI Gym (install via `conda install -c conda-forge gym`).
* A few other python packages (a `requirements.txt` is present)

The model objects should be loadable and executable by another tensorflow install (CPU or GPU) on e.g. an Nvidia GPU, but I haven't tested this. 

# CartPole

### Purpose: Balance the pole on the cart, not falling more than a few degrees of centre, and not Travelling off screen

[CartPole](https://gym.openai.com/envs/CartPole-v0/) is an introductory environment to learn to apply basic reinforcement learning concepts. The specific approach I've taken to solve the environment is:
* A temporal difference method, where the current state's predicted Q values for the chosen action are updated from the observation of the next state's reward and max predicted Q value. 
* Waiting until the end of the episode before training, even though TD learning allows online training.
* Sampling the memory to avoid correlation between observations (seen in the book)
* Epsilon-greedy policy for exploration. The cartpole environment looks well suited to this,as the initial states appear close to an ideal state of balancing. I think epsilon-greedy exploration will result in effective exploration in the high-interest state spaces. 

## Untrained and trained cartpole

* The untrained agent fails within 10-15 frames as the pole's angle exceeds the environment's limits and the game ends. 
* The trained agents succeeds at balancing the pole within the environment's limits for 200 frames, 50 games in a row. 

Untrained:          |  Trained:
:-------------------------:|:-------------------------:
<img src="https://github.com/JWB110123/reinforcement_learning/blob/master/recording/untrained_agent_cartpole.gif" alt="Untrained" width="400">  |  <img src="https://github.com/JWB110123/reinforcement_learning/blob/master/recording/trained_agent_cartpole.gif" alt="Untrained" width="400">

(Note that the pole being unbalanced in the same direction in both gifs is chance - the environment is initialised with a slight perturbation to both angular velocity and position, and so could equally have started with the pole unbalanced to the left)

It took nearly 3000 episodes where `learning_rate=0.01` to train the cartpole agent to win 50 times in a row. To win, the agent must balance the pole for 200 frames. 3000 episodes for such a simple environment is not an outright record by any means but demonstrates that the generalisable classes and class methods can be used for more complex environments!

```
...
200 frames in game 2790, on a winstreak of 47. Total wins 276
200 frames in game 2791, on a winstreak of 48. Total wins 277
200 frames in game 2792, on a winstreak of 49. Total wins 278
200 frames in game 2793, on a winstreak of 50. Total wins 279
```

# Acrobot

### Purpose: Put the second linkage above the black line

Untrained:          |  Trained:
:-------------------------:|:-------------------------:
<img src="https://github.com/JWB110123/reinforcement_learning/blob/master/recording/untrained_agent_acrobot.gif" alt="Untrained" width="400">  |  <img src="https://github.com/JWB110123/reinforcement_learning/blob/master/recording/trained_agent_acrobot.gif" alt="Untrained" width="400">

'Solving' this env takes much longer than CartPole. I have a memory leak in Keraa/Tensorflow ([similar to the latest comments on this issue](https://github.com/tensorflow/tensorflow/issues/33030)), which means I've had to implement basic model saving to avoid losing all progress on overnight training runs. This means, unlike cartpole, I don't currently know how many games it took to train, as I had to restart from a saved intermediate model a few times, but I'd guess it was more than 20,000. An enhancement would be to save the entire agent including the game counter, rather than just the tensorflow model, but the basic saving works for now.

The arm can only input torque around the second joint, not the fixed pivot point. This means that momentum must be built to swing the tip of the arm above the line. 

There's a few things to mention at this point:
1) My agent memory and training logic is very basic; I'm only storing the states and current model's predicted Q values. This means that when the current model is trained via sampling the memory, the current NN weights are being updated on experiences from older versions of the model. This isn't ideal. 
2) My exploration policy is  basic, with it being epsilon greedy, and for this environment where a reward is only observed when the second link/arm is above the line, random actions will hardly ever result in being rewarded in this environment. 
3) My  basic implementation of Q learning means that, even if a reward is obtained randomly, this observation of a single frame's positive reward will only have a small impact on the model weights. Based on the Reinforcement Learning: An Introduction book by Sutton/Barto, implementing a Monte Carlo style approach where the rewards from the whole episode are processed at the end of the episode would probably result in much better learning in this scenario. (Or eligibility traces)

With the above in mind, I'm pleased this simple Deep Q Learning agent managed to learn this environment in a relatively short period of time. 

# MountainCar
[MountainCar](https://gym.openai.com/envs/MountainCar-v0/) is another environment that's more complex than cartpole. After leaving an agent training for some time, some progress is made, but it takes days rather than 10 minutes for the agents to make visible progress. There is still some way to go before the agent would be fully trained, as well. 

As mentioned above in the Acrobot example, a nice extension to try would be [eligibility traces](https://link.springer.com/content/pdf/10.1007/BF00114726.pdf). There's a chapter on eligibility traces in [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf). Now that the basic `QLearnAgent` class has been shown to work, I'll go some tidying up of the classes in that approach before implementing a more complex agent inspried by the eligibility traces concept. 

Untrained:          |  Trained:
:-------------------------:|:-------------------------:
<img src="https://github.com/JWB110123/reinforcement_learning/blob/master/recording/untrained_agent_mountaincar.gif" alt="Untrained" width="400">  |  <img src="https://github.com/JWB110123/reinforcement_learning/blob/master/recording/trained_agent_mountaincar.gif" alt="Untrained" width="400">

