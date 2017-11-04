from gym.envs.registration import register

register(
    id='Reacher2-v0',
    entry_point='gym_reacher2.envs:Reacher2Env',
    max_episode_steps=500,
    reward_threshold=-3.75,
)

register(
    id='Reacher2Pixel-v0',
    entry_point='gym_reacher2.envs:Reacher2PixelEnv',
    kwargs={'base_env_id': 'Reacher2-v0'}
)

register(
    id='Reacher2Plus-v0',
    entry_point='gym_reacher2.envs:Reacher2PlusEnv',
    kwargs={'base_env_id': 'Reacher2-v0'}
)
