import datetime
import os
import tempfile
import xml.etree.cElementTree as ET

import numpy as np
from gym import utils, error
from gym.envs.mujoco import ReacherEnv

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


class Reacher2Env(ReacherEnv):
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
        pass
        # super().__init__() that's what we don't wanna do

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

    def _reset(self):
        if not self.isInitialized:
            raise Exception(NOT_INITIALIZED_ERR)

        mjlib.mj_resetData(self.model.ptr, self.data.ptr)
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer.autoscale()
            self.viewer_setup()
        return ob

    def _render(self, mode='human', close=False):
        if not self.isInitialized:
            raise Exception(NOT_INITIALIZED_ERR)

        if close:
            if self.viewer is not None:
                self._get_viewer().finish()
                self.viewer = None
            return

        if mode == 'rgb_array':
            self._get_viewer().render()
            data, width, height = self._get_viewer().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().loop_once()

    def _modifyXml(self, xml_file, model_parameters):

        tree = ET.ElementTree(file=xml_file)
        root = tree.getroot()

        for i in range(2):
            bodies = "/".join(["body"] * (i+1))

            arm_value = model_parameters["arm{}".format(i)]

            arm = root.find('worldbody/{}/geom'.format(bodies))
            arm.set('fromto', '0 0 0 {} 0 0'.format(arm_value))

            body = root.find('worldbody/{}/body'.format(bodies))
            body.set('pos', '{} 0 0'.format(arm_value))


        for i in range(2):
            joint = root.find('actuator/motor[@joint="joint{}"]'.format(i))
            joint.set('gear', str(float(model_parameters["torque{}".format(i)])))


        file_name = os.path.basename(xml_file)
        tmp_dir = tempfile.gettempdir()
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        file_name_with_date = "{}-{}".format(now, file_name)

        new_file_path = os.path.join(tmp_dir, file_name_with_date)

        tree.write(new_file_path, "UTF-8")

        return new_file_path


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
