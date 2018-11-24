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
    env = gym.make("Reacher2-v0")
    env.unwrapped._init(
        arms = (.05,.2),     # length of limbs
        torques = (400,100), # torque of joints
        fov=70,        # field of view
        colors={
                "arenaBackground": ".9 .0 .5",
                "arenaBorders": "0.1 0.1 0.4",
                "arms": "0.8 0.7 0.1",
            },
        topDown=True   # top-down centered camera?
    )
    env.reset()   
    for i in range(100):
        env.render()
        env.step(env.action_space.sample())
    
    env.close()


#### Defaults:

The vanilla Reacher-v1 environment has the following parameters:

    arms = (.1,.1),
    torques = (200,200),
    fov = 45,
    colors = {
        "arenaBackground": "0.9 0.9 0.9",
        "arenaBorders": "0.9 0.4 0.6",
        "arms": "0.0 0.4 0.6",
    },
    topDown = False
    

If you don't assign some of these parameters they will default to these values.

#### All the DoF:

If you want more than 2 DoF, like ... 3, you can use this code:

    env = gym.make("Reacher2-3Dof-v0")
    env.unwrapped._init(
        armlens=(.07,.07,.06),  # length of limbs, should add up to .2
        torques=(200,200,200),  # torque of joints, 200 is default
    )
    env.reset()

    for i in range(100):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(.05)

