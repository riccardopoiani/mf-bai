from gym.envs.registration import register

register(
    id='yahtzee-single-v0',
    entry_point='yahtzee.gym_yahtzee.envs:YahtzeeSingleEnv',
)
