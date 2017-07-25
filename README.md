# gym-reacher2

Modification of the original Reacher-v1 environment to allow for dynamic limb length and joint torque.

#### Usage:

on the shell: 

`pip install gym-reacher2`

in Python 3 (!):

    import gym
    import gym_reacher2
    
    env = gym.make("Reacher2-v1")
    env._init(
        arm0 = .05,     # length of limb 1
        arm1 = .2,     # length of limb 2
        torque0 = 100, # torque of joint 1
        torque1 = 400  # torque of joint 2
    )
    env.reset()
    
    for i in range(100):
        env.render()
        env.step(env.action_space.sample())

