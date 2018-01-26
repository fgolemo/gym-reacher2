import gym
import gym_reacher2
import time

env = gym.make("Reacher2-v0")
env.env._init(
    arm0=.1,
    arm1=.1,
    torque0=200,
    torque1=200,
    fov=45,
    topDown=False
)
env.reset()

env2 = gym.make("Reacher2-v0")
env2.env._init(
    arm0=.1,
    arm1=.1,
    torque0=200,
    torque1=200,
    fov=45,
    topDown=False,
    xml="reacher-backlash.xml"
)
env2.reset()

import matplotlib.pyplot as plt
import numpy as np

plt.ion()
img = np.random.uniform(0, 255, (1000, 500, 3))
plt_img = plt.imshow(img, interpolation='none', animated=True, label="blah")
plt_ax = plt.gca()

for i in range(100):
    render = env.render(mode="rgb_array")
    render2 = env2.render(mode="rgb_array")

    img[:500, :, :] = render
    img[500:, :, :] = render2

    plt_img.set_data(img)
    plt_ax.plot([0])
    plt.pause(0.001)  # I found this necessary - otherwise no visible img

    if i % 10 == 0:
        action = env.action_space.sample()
    env.step(action)
    env2.step(action)

    time.sleep(0.01)

env.close()
