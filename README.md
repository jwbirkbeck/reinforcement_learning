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
* Learning via human should maybe reward every move like a human, rather than using the normal Q updating logic. 
* Storing all rewards and then discounting them from the end may be quicker, and for environments like mountain car where naive exploration will never reach the positive reward may lead to more robust learning?
