from gym.envs.registration import register

register(
    id='MAOC-v0',
    entry_point='gym_MAOC.envs:MAOCEnv',
)