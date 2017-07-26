from gym.envs.registration import register

register(
    id='Reacher2-v1',
    entry_point='gym_reacher2.envs:Reacher2Env',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='Reacher2Pixel-v1',
    entry_point='gym_reacher2.envs:Reacher2PixelEnv',
    kwargs={'base_env_id': 'Reacher2-v1'}
)