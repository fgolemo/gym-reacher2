from vrepper.vrepper import vrepper

import os,time
import numpy as np


venv = vrepper(headless=False)
venv.start()
# time.sleep(3)
venv.load_scene(os.getcwd() + '/../scenes/poppy_ergo_jr_vanilla_ball.ttt')
venv.start_blocking_simulation()
# self.slider = venv.get_object_by_name('slider')
# self.cart = venv.get_object_by_name('cart')
# self.mass = venv.get_object_by_name('mass')

print('(CartPoleVREP) initialized')
time.sleep(2)

# cartpos = self.cart.get_position()
# masspos = self.mass.get_position()
# cartvel,cart_angvel = self.cart.get_velocity()
# massvel,mass_angvel = self.cart.get_velocity()
#
# self.observation = np.array([
#     cartpos[0],cartvel[0],
#     masspos[0],masspos[2],
#     massvel[0],massvel[2]
#     ]).astype('float32')
#
# actions = np.clip(actions, -1, 1)
# v = actions[0]
#
# # step
# self.slider.set_velocity(v)
# self.venv.step_blocking_simulation()
#
# # observe again
# self._self_observe()
#
# # cost
# height_of_mass = self.observation[3] # masspos[2]
# cost = - height_of_mass + (v**2) * 0.001
#
#
# def _reset(self):
#     self.venv.stop_blocking_simulation()
#     self.venv.start_blocking_simulation()
#     self._self_observe()
#     return self.observation
#
# def _destroy(self):
#     self.venv.stop_blocking_simulation()
#     self.venv.end()
#
#
# print('simulation ended. leaving in 3 seconds...')
# time.sleep(5)
venv.stop_blocking_simulation()
venv.end()