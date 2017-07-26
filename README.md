# gym-reacher2

This repo contains the gym environment `Reacher2-v1`.

Modification of the original `Reacher-v1` environment to allow for dynamic limb length and joint torque.

#### Installation:

on the shell:

    git clone https://github.com/fgolemo/gym-reacher2.git
    cd gym-reacher2
    pip install -e .
(the last command here might need `sudo` beforehand)
    
#### Usage:

in Python 3 (!):

    import gym
    import gym_reacher2   
    env = gym.make("Reacher2-v1")
    env.env._init(
        arm0 = .05,    # length of limb 1
        arm1 = .2,     # length of limb 2
        torque0 = 100, # torque of joint 1
        torque1 = 400  # torque of joint 2
    )
    env.reset()   
    for i in range(100):
        env.render()
        env.step(env.action_space.sample())
    
    env.close()


#### Defaults:

The vanilla Reacher-v1 environment has the following parameters:

    arm0 = .1,
    arm1 = .1,
    torque0 = 200,
    torque1 = 200

If you don't assign some of these parameters they will default to these values.
