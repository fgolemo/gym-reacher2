import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ErgoBall-v0',
    entry_point='gym_vrep.envs:ErgoBallEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic = False,
)

register(
    id='ErgoBallDyn-v0',
    entry_point='gym_vrep.envs:ErgoBallDynEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic = True,
)

register(
    id='ErgoBallDyn-v1',
    entry_point='gym_vrep.envs:ErgoBallDynRewEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic = True,
)

