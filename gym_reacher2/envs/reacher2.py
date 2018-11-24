import gym
import numpy as np
import time
from gym import utils, error, spaces

from gym_reacher2.envs import MujocoReacher2Env

NOT_INITIALIZED_ERR = "Before doing a reset or your first " \
                      "step in the environment " \
                      "please call env._init()."

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))


class Reacher2Env(MujocoReacher2Env, utils.EzPickle):
    isInitialized = False

    def _init(self, armlens=(.1,.1), torques=(200,200), fov=45, colors=None, topDown=False, xml='reacher.xml'):
        if colors is None:
            # color values are "R G B", for red, green, and blue respectively
            # in the range from 0 to 1. For example white it "1 1 1". Red is
            # "1 0 0".

            colors = {
                "arenaBackground": ".9 .9 .9",
                "arenaBorders": "0.9 0.4 0.6",
                "arms": "0.0 0.4 0.6",
            }
        assert len(armlens) == len(torques) == self.dof
        if self.dof != 2 and xml == "reacher.xml":
            xml = "reacher-{}dof.xml".format(self.dof)

        params = {
            "armlens": armlens,
            "torques": torques,
            "fov": fov,
            "colors": colors
        }
        if topDown:
            self.viewer_setup = self.top_down_cam

        self.isInitialized = True
        utils.EzPickle.__init__(self)
        MujocoReacher2Env.__init__(self, xml, 2, params)

    def __init__(self, dof=2):
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 50
        }
        self.obs_dim = 13 # two more at the end for the absolute joint positions

        self.dof = dof

        self.action_space = spaces.Box(
            np.array([-1.]*self.dof),
            np.array([1.]*self.dof),
        dtype=np.float32)

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()

    def step(self, a):
        assert len(a) == self.dof
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

    def top_down_cam(self):
        self.viewer.cam.trackbodyid = 0  # id of the body to track
        self.viewer.cam.distance = self.model.stat.extent * 0.6  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] = 0  # x,y,z offset from the object
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0
        self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 0

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
        # qpos (shape [4,1]) holds the angles of the 2 joints
        # it has the form: [
        # x angle joint 1 in rad,
        # x angle joint 2 in rad,
        # y angle joint 1 (constant)
        # y angle joint 2 (constant)
        # ]
        theta = self.sim.data.qpos.flat[:self.dof]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta), # this is the most meaningful data here,
            # containing the sine of the two joint angles
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:self.dof], # angular momentum of the two joints
            self.get_body_com("fingertip") - self.get_body_com("target"),
            theta
        ])


if __name__ == '__main__':
    import gym_reacher2
    # env = gym.make("Reacher2-v0")
    # env.env._init(
    #     armlens=(.05,.2),  # length of limbs
    #     torques=(100,100),  # torque of joints
    # )
    env = gym.make("Reacher2-3Dof-v0")
    env.unwrapped._init(
        armlens=(.06,.06,.07),  # length of limbs
        torques=(200,200,200),  # torque of joints
    )
    env.reset()

    for i in range(100):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(.05)
