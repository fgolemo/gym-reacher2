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

register(
    id='ErgoBallThrow-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic = True,
)

register(
    id='ErgoBallThrowRandom-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowRandEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic = True,
)

register(
    id='ErgoBallThrowVert-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowVertEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic = True,
)

register(
    id='ErgoBallThrowVertRand-v0',
    entry_point='gym_vrep.envs:ErgoBallThrowVertRandEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic = True,
)

register(
    id='ErgoBallThrowVert-v1',
    entry_point='gym_vrep.envs:ErgoBallThrowVertMaxEnv',
    timestep_limit=100,
    reward_threshold=10.0,
    nondeterministic = True,
)

