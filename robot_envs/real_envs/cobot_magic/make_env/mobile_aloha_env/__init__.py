from gym.envs.registration import register

register(
    id="mobile_aloha_env/MobileAloha-v0",
    entry_point="mobile_aloha_env.mobile_aloha_env:MobileAlohaEnv",
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True
)