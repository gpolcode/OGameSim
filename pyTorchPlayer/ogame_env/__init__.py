from gymnasium.envs.registration import register

register(
    id="ogame_env/GridWorld-v0",
    entry_point="ogame_env.envs:GridWorldEnv",
)
