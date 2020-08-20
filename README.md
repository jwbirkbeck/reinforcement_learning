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


# CartPole

[CartPole](https://gym.openai.com/envs/CartPole-v0/) is an introductory environment to learn to apply basic reinforcement learning concepts. The specific approach I've taken to solve the environment is:
* A temporal difference method, where the current state's predicted Q values for the chosen action are updated from the observation of the next state's reward and max predicted Q value. 
* Epsilon-greedy policy for exploration (the cartpole environment is well suited to this as random actions are likely to be rewarded as the (semi-random) beginning state space is close to the optimal area in the state space. In other words, the pole starts nearly balanced so random actions will sometimes help to balance the pole, resulting in effective exploration. 

## Untrained and trained cartpole
### Untrained:

* The untrained agent fails within 10-15 frames as the pole's angle exceeds the environment's limits and the game ends. 
* The trained agents succeeds at balancing the pole within the environment's limits for 200 frames, 50 games in a row. 

Untrained:          |  Trained:
:-------------------------:|:-------------------------:
<img src="https://github.com/JWB110123/reinforcement_learning/blob/master/recording/untrained_agent_cartpole.gif" alt="Untrained" width="400">  |  <img src="https://github.com/JWB110123/reinforcement_learning/blob/master/recording/trained_agent_cartpole.gif" alt="Untrained" width="400">

(Note that the pole being unbalanced in the same direction in both gifs is chance - the environment is initialised with a slight perturbation to both angular velocity and position, and so could equally have started with the pole unbalanced to the left)

It took nearly 3000 episodes to train the cartpole agent to win 50 times in a row. To win, the agent must balance the pole for 200 frames. 3000 episodes for such a simple environment is not a record by any means but demonstrates that the generalisable classes and class methods can be used for more complex environments!

```
...
200 frames in game 2790, on a winstreak of 47. Total wins 276
200 frames in game 2791, on a winstreak of 48. Total wins 277
200 frames in game 2792, on a winstreak of 49. Total wins 278
200 frames in game 2793, on a winstreak of 50. Total wins 279
```

# MountainCar
[MountainCar](https://gym.openai.com/envs/MountainCar-v0/) is a slightly more complex environment than cartpole. On applying the classes developed for cartpole to the new environment, it's apparent that an epsilon-greedy approach to exploration is not effective. Even by viewing [the environment](https://gym.openai.com/envs/MountainCar-v0/) you can see that random actions from the starting position (around the bottom of the valley) will not result in receiving the reward in a reasonable timeframe.

Some googling led to a paper which discusses [eligibility traces in the mountaincar environment](https://link.springer.com/content/pdf/10.1007/BF00114726.pdf). There's a chapter on eligibility traces in [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf). I'll attempt this environment after some reading!
