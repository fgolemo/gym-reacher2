import gym
import numpy as np
from gym import spaces


class MujocoPixelWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(MujocoPixelWrapper, self).__init__(env)
        self.observation_space = [self.observation_space, spaces.Box(0, 255, [500, 500, 3])]

    def get_viewer(self):
        return self.env.unwrapped._get_viewer()

    def _observation(self, observation):
        self.get_viewer().render()
        data, width, height = self.get_viewer().get_image()
        return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:]


def Reacher2PixelEnv(base_env_id):
    return MujocoPixelWrapper(gym.make(base_env_id))


if __name__ == '__main__':
    import gym_reacher2
    env = gym.make("Reacher2Pixel-v1")
    env.env.env._init(
        arm0=.05,  # length of limb 1
        arm1=.2,  # length of limb 2
        torque0=400,  # torque of joint 1
        torque1=100  # torque of joint 2
    )
    obs = env.reset()
    print (obs.shape)

    for i in range(100):
        env.render()
        env.step(env.action_space.sample())