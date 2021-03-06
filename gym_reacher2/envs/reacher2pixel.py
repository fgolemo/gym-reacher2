import gym
import numpy as np
from gym import spaces


class MujocoPixelWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        self.env = env
        super(MujocoPixelWrapper, self).__init__(env)
        self.observation_space = spaces.Tuple((self.observation_space, spaces.Box(0, 255, [500, 500, 3])))

    def get_viewer(self):
        return self.env.unwrapped._get_viewer()

    def _observation(self, observation):
        super_obs = self.env.unwrapped._get_obs()
        fig = self.env.unwrapped.render("rgb_array")
        return [super_obs, fig]


def Reacher2PixelEnv(base_env_id):
    return MujocoPixelWrapper(gym.make(base_env_id))


if __name__ == '__main__':
    import gym_reacher2
    env = gym.make("Reacher2Pixel-v0")
    env.env.env._init(
        arm0=.05,  # length of limb 1
        arm1=.2,  # length of limb 2
        torque0=400,  # torque of joint 1
        torque1=100,  # torque of joint 2
        fov=70, # field of view
        colors={
                "arenaBackground": ".9 .0 .5",
                "arenaBorders": "0.1 0.1 0.4",
                "arm0": "0.8 0.7 0.1",
                "arm1": "0.2 0.5 0.1"
            },
        topDown=True
    )
    obs = env.reset()
    print (len(obs))
    print (obs[0].shape, obs[1].shape)

    for i in range(100):
        env.render()
        obs, _, _, _ = env.step(env.action_space.sample())
