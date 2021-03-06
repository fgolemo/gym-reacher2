import gym
import numpy as np
import torch
from gym import spaces

relevant_items = [2, 3, 6, 7]


def obs_to_net(obs):
    obs_v = torch.autograd.Variable(torch.from_numpy(obs[relevant_items].astype(np.float32), ), requires_grad=False)
    return obs_v


def net_to_obs(net, obs):
    for i, idx in enumerate(relevant_items):
        obs[idx] = net[0, i].data.numpy()[0]
    return obs


class Reacher2InferenceWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(Reacher2InferenceWrapper, self).__init__(env)
        env.load_model = self.load_model
        self.env = env

    def load_model(self, net, modelPath):
        self.net = net
        checkpoint = torch.load(modelPath, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint['state_dict'])
        print("DBG: MODEL LOADED:", modelPath)

    def _observation(self, observation):
        super_obs = self.env.env._get_obs()
        # print(super_obs)
        obs_v = obs_to_net(super_obs)
        # print(obs_v)
        obs_plus = self.net.forward(obs_v)

        self._set_to_simplus(obs_plus.cpu().data.numpy()[0])

        obs_plus_full = net_to_obs(obs_plus, super_obs)
        # print(obs_plus_full)

        return obs_plus_full

    def _set_to_simplus(self, obs_plus):
        qpos = self.env.env.model.data.qpos.ravel().copy()
        qvel = self.env.env.model.data.qvel.ravel().copy()
        # print (qpos[:2], np.sin(qpos[:2]), obs_plus[:2], np.arcsin(obs_plus[:2]))
        qpos[:2] = np.arcsin(obs_plus[:2])
        qvel[:2] = obs_plus[2:]

        self.env.env.set_state(qpos, qvel)

    def _reset(self):
        self.net.zero_hidden()  # !important
        self.net.hidden[0].detach_()  # !important
        self.net.hidden[1].detach_()  # !important
        # print("DBG: HIDDEN STATE RESET AND DETACHED")
        return super(Reacher2InferenceWrapper, self)._reset()


def Reacher2PlusEnv(base_env_id):
    return Reacher2InferenceWrapper(gym.make(base_env_id))


# if __name__ == '__main__':
#     import gym_reacher2
#     env = gym.make("Reacher2Plus-v1")
#     env.env.env._init(
#         arm0=.05,  # length of limb 1
#         arm1=.2,  # length of limb 2
#         torque0=400,  # torque of joint 1
#         torque1=100,  # torque of joint 2
#         fov=70, # field of view
#         colors={
#                 "arenaBackground": ".9 .0 .5",
#                 "arenaBorders": "0.1 0.1 0.4",
#                 "arm0": "0.8 0.7 0.1",
#                 "arm1": "0.2 0.5 0.1"
#             },
#         topDown=True
#     )
#     obs = env.reset()
#     print (len(obs))
#     print (obs[0].shape, obs[1].shape)
#
#     for i in range(100):
#         env.render()
#         env.step(env.action_space.sample())
