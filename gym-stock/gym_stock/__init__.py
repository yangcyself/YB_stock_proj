from gym.envs.registration import register

register(
    id='stock-v0',
    entry_point='gym_stock.envs:StockEnv',
)
# register(
#     id='stock-extrahard-v0',
#     entry_point='gym_stock.envs:StockExtraHardEnv',
# )