# Reinforcement Learning

### Purpose:

Solve the cartpole environment from the OpenAI Gym with a proof of concept script, before building generalisable classes to apply to more complicated environments.


### Requirements:

This project uses:

* The amdgpu-pro driver (not required if you're learning via CPU)
* Tensorflow, keras, and plaidML; plaidML will require a pip install and not a conda install (see plaidml documentation)
* Anaconda
* The OpenAI Gym (install via `conda install -c conda-forge gym`).
* A few other python packages (a `requirements.txt` is provided for replicating my environment)

### Current notes
* Learning via human should maybe reward every move the network make that is like a human's, rather than using the normal Q updating logic. 
* Storing all rewards and then discounting them from the end may be quicker, and for environments like mountain car where naive exploration will never reach the positive reward may lead to more robust learning?


# CartPole approach

CartPole is an introductory environment to learn to apply basic reinforcement learning concepts. The specific approach I've taken to solve the environment is:
* A temporal difference method, where the current state's predicted Q values for the chosen action are updated from the observation of the next state's reward and max predicted Q value. 
* Epsilon-greedy policy for exploration (the cartpole environment is well suited to this as random actions are likely to be rewarded as the (semi-random) beginning state space is close to the optimal area in the state space. In other words, the pole starts nearly balanced so random actions will sometimes help to balance the pole, resulting in effective exploration. 

## Untrained and trained cartpole
### Untrained:

Untrained:          |  Trained:
:-------------------------:|:-------------------------:
<img src="https://github.com/JWB110123/reinforcement_learning/blob/master/recording/untrained_agent_cartpole.gif" alt="Untrained" width="400">  |  <img src="https://github.com/JWB110123/reinforcement_learning/blob/master/recording/untrained_agent_cartpole.gif" alt="Untrained" width="400">
