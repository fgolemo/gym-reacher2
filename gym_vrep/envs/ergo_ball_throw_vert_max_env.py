import os
import time

import gym
import numpy as np
from gym import error, spaces
import math

try:
    from vrepper.vrepper import vrepper
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you can install VRepper dependencies with 'pip install vrepper.)'".format(e))

import logging

logger = logging.getLogger(__name__)

JOINT_LIMITS = [
    (-150, 150),
    (-90, 125),
    (-90, 90),
    (-90, 90),
    (-90, 90),
    (-90, 90)
]

JOINT_LIMITS_MAXMIN = [-150, 150]

FORCE_LIMITS = [-5.0, 5.0]

REST_POS = [0, -90, 35, 0, 55, -90]

BALL_POS = [.035, .035, .035,  # size
            0, 0.05, .3,  # position
            .01]  # weight)

REWARD_SCALING = 10


class ErgoBallThrowVertMaxEnv(gym.Env):
    vrep_running = False
    max_z = 0

    def __init__(self):
        joint_boxes = spaces.Box(low=JOINT_LIMITS_MAXMIN[0], high=JOINT_LIMITS_MAXMIN[1], shape=6)

        force_boxes = (joint_boxes, spaces.Box(low=FORCE_LIMITS[0], high=FORCE_LIMITS[1], shape=(6)))

        self.observation_space = spaces.Tuple(force_boxes)
        self.action_space = spaces.Tuple(joint_boxes)

        self.minima = [JOINT_LIMITS[i][0] for i in range(6)]
        self.maxima = [JOINT_LIMITS[i][1] for i in range(6)]

    def _actualInit(self, headless=True):
        self._startEnv(headless)
        self.vrep_running = True

    def _startEnv(self, headless):
        self.venv = vrepper(headless=headless)
        self.venv.start()
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.venv.load_scene(current_dir + '/../scenes/poppy_ergo_jr_vanilla_ball.ttt')
        motors = []
        for i in range(6):
            motor = self.venv.get_object_by_name('m{}'.format(i + 1), is_joint=True)
            motors.append(motor)
        self.motors = motors

    def _restPos(self):
        self.venv.stop_simulation()
        self.venv.start_simulation(is_sync=False)

        for i, m in enumerate(self.motors):
            m.set_position_target(REST_POS[i])

        params = self.venv.create_params([], BALL_POS, [], '')

        self.venv.call_script_function('spawnBall', params)
        time.sleep(.5)
        self.ball = self.venv.get_object_by_name("ball")

        self.venv.make_simulation_synchronous(True)

    def _reset(self):
        if not self.vrep_running:
            self._actualInit()

        self._restPos()
        self._self_observe()
        return self.observation

    def _getCurrentBall(self):
        ballPos = self.ball.get_position()
        return ballPos

    def _getTip(self):
        tip = self.motors[-1].get_position()
        return tip

    def _getReward(self):
        tip = self._getTip()
        ball = self._getCurrentBall()
        z = ball[2] - tip[2] # try to maximize height
        if z > self.max_z:
            self.max_z = z
        xy = math.sqrt( (ball[0] - tip[0])**2 + (ball[1] - tip[1])**2 ) # try to minimize x-y distance

        return (self.max_z - xy) * REWARD_SCALING

    def _self_observe(self):
        pos = []
        forces = []
        for m in self.motors:
            pos.append(m.get_joint_angle())
            forces += m.get_joint_force()

        self.observation = np.array(pos + forces).astype('float32')

    def _gotoPos(self, pos):
        for i, m in enumerate(self.motors):
            m.set_position_target(pos[i])

    def _clipActions(self, actions):
        a = []
        for i, action in enumerate(actions):
            a.append(np.clip(action, self.minima[i], self.maxima[i]))
        return np.array(a)

    def _step(self, actions):
        actions = self._clipActions(actions)

        # step
        self._gotoPos(actions)
        self.venv.step_blocking_simulation()

        # observe again
        self._self_observe()

        return self.observation, self._getReward(), False, {}

    def _close(self):
        self.venv.stop_simulation()
        self.venv.end()

    def _render(self, mode='human', close=False):
        pass


def randomAction(obs, step=1):
    tmp = obs[0] + np.random.uniform(-step, step, 6)
    # tmp = tuple(np.array([action]) for action in tmp.tolist())
    return tmp


if __name__ == '__main__':
    env = ErgoBallThrowEnv()
    env._actualInit(headless=False)

    for k in range(3):
        observation = env.reset()
        for _ in range(6):
            action_real = randomAction(observation, step=20)
            observation, reward, done, info = env.step(action_real)
            print (observation, reward)

    env.close()

    print('simulation ended. leaving in 5 seconds...')
    time.sleep(2)
