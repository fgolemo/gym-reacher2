from gym.envs.registration import register

register(
    id='Reacher2-v0',
    entry_point='gym_reacher2.envs:Reacher2Env',
    # Set to 50 for learning, might have to set higher for collecting data
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='Reacher2Pixel-v0',
    entry_point='gym_reacher2.envs:Reacher2PixelEnv',
    kwargs={'base_env_id': 'Reacher2-v0'}
)

register(
    id='Reacher2-3Dof-v0',
    entry_point='gym_reacher2.envs:Reacher2Env',
    # Set to 50 for learning, might have to set higher for collecting data
    max_episode_steps=50,
    reward_threshold=-3.75,
    kwargs={'dof': 3}
)

# register(
#     id='Reacher2Plus-v0',
#     entry_point='gym_reacher2.envs:Reacher2PlusEnv',
#     kwargs={'base_env_id': 'Reacher2-v0'}
# )
