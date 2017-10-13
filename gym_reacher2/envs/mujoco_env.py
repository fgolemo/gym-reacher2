import datetime
import os
import tempfile
import xml.etree.cElementTree as ET

import numpy as np
from gym import error
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
        assert "fov" in model_parameters
        assert "colors" in model_parameters

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        modified_xml_path = self._modifyXml(fullpath, model_parameters)
        # print('new xml path: {}'.format(modified_xml_path))
        self.model = mujoco_py.MjModel(modified_xml_path)

        self.frame_skip = frame_skip
        self.data = self.model.data
        self.viewer = None

        self.init_qpos = self.model.data.qpos.ravel().copy()
        self.init_qvel = self.model.data.qvel.ravel().copy()

    def _modifyXml(self, xml_file, model_parameters):

        tree = ET.ElementTree(file=xml_file)
        root = tree.getroot()

        for i in range(2):
            bodies = "/".join(["body"] * (i + 1))

            arm_value = model_parameters["arm{}".format(i)]

            arm = root.find('worldbody/{}/geom'.format(bodies))
            arm.set('fromto', '0 0 0 {} 0 0'.format(arm_value))

            if "arm" + str(i) in model_parameters["colors"]:
                arm.set('rgba', '{} 1'.format(model_parameters["colors"]["arm" + str(i)]))

            body = root.find('worldbody/{}/body'.format(bodies))
            body.set('pos', '{} 0 0'.format(arm_value))

        for i in range(2):
            joint = root.find('actuator/motor[@joint="joint{}"]'.format(i))
            joint.set('gear', str(float(model_parameters["torque{}".format(i)])))

        if "arenaBackground" in model_parameters["colors"]:
            arena = root.find('worldbody/geom[@type="plane"]')
            arena.set('rgba', '{} 1'.format(model_parameters["colors"]["arenaBackground"]))

        if "arenaBorders" in model_parameters["colors"]:
            arenaBorders = root.findall('worldbody/geom[@type="capsule"]')
            for arenaBorder in arenaBorders:
                arenaBorder.set('rgba', '{} 1'.format(model_parameters["colors"]["arenaBorders"]))

        fovy = root.find('visual/global')
        fovy.set('fovy', str(model_parameters["fov"]))

        file_name = os.path.basename(xml_file)
        tmp_dir = tempfile.gettempdir()
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        file_name_with_date = "{}-{}".format(now, file_name)

        new_file_path = os.path.join(tmp_dir, file_name_with_date)

        tree.write(new_file_path, "UTF-8")

        return new_file_path

    def _render(self, mode='human', close=False):
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
