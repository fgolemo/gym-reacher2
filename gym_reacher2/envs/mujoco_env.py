import os
import numpy as np
from gym import error, spaces
from gym.envs.mujoco import MujocoEnv

try:
    import mujoco_py
    from mujoco_py.mjlib import mjlib
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))


class MujocoReacher2Env(MujocoEnv):

    def __init__(self, model_path, frame_skip, model_parameters):
        assert "arm0" in model_parameters
        assert "arm1" in model_parameters
        assert "torque0" in model_parameters
        assert "torque1" in model_parameters

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        modified_xml_path = self._modifyXml(fullpath, model_parameters)
        print ('new xml path: {}'.format(modified_xml_path))
        self.model = mujoco_py.MjModel(modified_xml_path)

        self.frame_skip = frame_skip
        self.data = self.model.data
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.model.data.qpos.ravel().copy()
        self.init_qvel = self.model.data.qvel.ravel().copy()
        observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()

