import numpy as np
from gym import utils, error, spaces

from gym_reacher2.envs import MujocoReacher2Env

NOT_INITIALIZED_ERR = "Before doing a reset or your first " \
                      "step in the environment " \
                      "please call env._init()."

try:
    import mujoco_py
    from mujoco_py.mjlib import mjlib
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))


class Reacher2Env(MujocoReacher2Env, utils.EzPickle):
    isInitialized = False

    def _init(self, arm0=.1, arm1=.1, torque0=200, torque1=200):
        params = {
            "arm0": arm0,
            "arm1": arm1,
            "torque0": torque0,
            "torque1": torque1
        }
        self.isInitialized = True
        utils.EzPickle.__init__(self)
        MujocoReacher2Env.__init__(self, 'reacher.xml', 2, params)

    def __init__(self):
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 50
        }
        self.obs_dim = 11

        self.action_space = spaces.Box(
            np.array([-1., -1.]),
            np.array([1., 1.])
        )

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()

    def _step(self, a):
        if not self.isInitialized:
            raise Exception(NOT_INITIALIZED_ERR)
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])


if __name__ == '__main__':
    env = Reacher2Env()
    env._init(
        arm0=.05,  # length of limb 1
        arm1=.2,  # length of limb 2
        torque0=400,  # torque of joint 1
        torque1=100  # torque of joint 2
    )
    env.reset()

    for i in range(100):
        env.render()
        env.step(env.action_space.sample())