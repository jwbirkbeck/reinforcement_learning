# Reinforcement Learning

Note: I'm reading through _Reinforcement Learning: A Introduction_ by Sutton and Barto as I'm building this repository, which I'll refer to as 'the book' below!


### Purpose:

* [x] Build a simple temporal difference Q learning agent, and use it to solve a few simple environments.
* [ ] Build a more complex agent, potentially using concepts like eligibility traces, or off-policy learning.
* [ ] Compare the performance of the two. Apply the more complex agent to a more complex environment if learning performance is significantly better.



### Requirements:

For this project I've used:

* Ryzen 2600 and Vega56 GPU
* [AMD ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)
* [Tensorflow-ROCm](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream)
* Anaconda
* The OpenAI Gym (install via `conda install -c conda-forge gym`).
* A few other python packages (a `requirements.txt` is present)

The model objects should be loadable and executable by another tensorflow install (CPU or GPU) on e.g. an Nvidia GPU, but I haven't tested this. 

# Approach

The `QLearnAgent` built here is a simple implementation of Deep Q Learning. While temporal difference learning allows online learning, this agent does not learn online - it waits until the end of an episode before training on it's experiences. 

The difference between a Monte Carlo approach and a temporal difference approach is primarily in how the rewards are used. In a Monte Carlo approach, to update a predicted Q<sub>a, t</sub> for the action that was taken, all rewards r<sub>t+1</sub>, ..., r<sub>T</sub> until the termination of the episode at time T are needed. Temporal difference learning uses only the reward at the next time step r<sub>t+1</sub> and the Q predictions **Q**<sub>t+1</sub> for each action in the next step to update the prediction of Q<sub>a, t</sub>. This is referred to as bootstrapping in the book.

I am not making the most of a benefit of temporal difference learning by **choosing not to learn online**, but this is acceptable as the initial environments I use are limited to be short (e.g. 200 frames). The `QLearnAgent` currently implemented does, however, reliably converge on a 'solution' for each of the three environments shown below. Rather than continually update this existing agent, now that it works, these improvements can be implemented on a more complex agent.

The training of this agent uses a variant of stochastic/mini-batch gradient descent, sampling from a 'memory' of previous experiences. Tensorflow provides a number of gradient descent algorithms - one of which being **ADAM**, the algorithm I chose to use here. Researching the relative performance of the various algorithms available was not a part of this project. I found ADAM described as potentially quicker than a normal mini-batch gradient descent so I choice to use it without much more thought than this. I would expect the agent to converge to a similar result with a usual mini-batch GD algorithm as well.

For exploration, an **epsilion greedy** policy, was used. Epsilon is not hardcoded in the `QLearnAgent`, but I have used `0.1` for the cartpole, acrobot, and mountain-car environments below. 

#### Known weaknesses to the `QLearnAgent` implementation:
* The stored experiences are _not_ used properly for experience replay - true experience reply would have an agent rescore the stored observation so that the current model is updated using current Q predictions on historical observations. Currently, only the Q scores from the time that experience was recorded are stored. This means the agent's previous predictions are being used to train the current model. I would expect the agent to 'solve' an environment quicker if the observations were stored and the Q values predicted as part of the training step. For larger 'memory' lengths, this becomes more of a concern.
* The `human_game` method demonstrably works, but is not well suited to the temporal difference approach. Since the agent uses the next timesteps Q values to improve the current The `human_game` method does not fully utilise the value of seeing a human solve an environment. A separate Monte Carlo approach to learning in the `human_game` method would potenially be much more sensible, as for each frame observed, the actual Q value for the set of actions taken is calculated using all rewards from the current state observation until the  the terminal state. 

# CartPole

### Purpose: Balance the pole on the cart, not falling more than a few degrees of centre, and not Travelling off screen

Untrained:          |  Trained:
:-------------------------:|:-------------------------:
<img src="https://github.com/JWB110123/reinforcement_learning/blob/master/recording/untrained_agent_cartpole.gif" alt="Untrained" width="400">  |  <img src="https://github.com/JWB110123/reinforcement_learning/blob/master/recording/trained_agent_cartpole.gif" alt="Untrained" width="400">

[CartPole](https://gym.openai.com/envs/CartPole-v0/) is an introductory environment to learn to apply basic reinforcement learning concepts.

* The untrained agent fails within 10-15 frames as the pole's angle exceeds the environment's limits and the game ends. 
* The trained agents succeeds at balancing the pole within the environment's limits for 200 frames, 50 games in a row. 

(Note that the pole being unbalanced in the same direction in both gifs is chance - the environment is initialised with a slight perturbation to both angular velocity and position, and so could equally have started with the pole unbalanced to the left)

This environment was used while developing the `QLearnAgent` class and related classes such as `NeuralNetwork` and `Memory`. This environment takes a short amount of wall clock time (around 10 minutes for the finished `QLearnAgent`) which allowed relatively quick testing during development.

This is a simple environment to solve, for multiple reasons. The primary reason I beleive it's simple is that the pole starts nearly balanced. 

It took nearly 3000 episodes where `learning_rate=0.01` to train the cartpole agent to win 50 times in a row. To win, the agent must balance the pole for 200 frames.

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

