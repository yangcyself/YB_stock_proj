## Project Task introduction
This task involves reinforcement learning and will no longer use the tags in task 1. To simplify the problem, this task sets that each tick has at most 5 hand long positions and 5 hand short positions. Long positions and short positions cannot be held at the same time. A tick can only have one action at a time. Positions can be increased or decreased (with unit equals one hand) through buying and selling, and the absolute value of change in the number of positions of one action cannot exceed one hand. The current state can be maintained by an idle action. When the buying action is executed, the purchase will be successful and will not have any impact on the market. The price is AskPrice1 of the current tick. When the selling action is executed, the sell will be successful and will have no eﬀect on the market. The price is BidPrice1 of the current tick. Finally, you should include in your report: the number of buying and spelling on testing set, the average price to buy and the average price to sell. Besides, attach action selection for each tick on testing set for submission. 

## code reference
[VIN](https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks)

[TCN](https://github.com/Songweiping/TCN-TF)

## Use Openai base line based algorithms
The [gym-stock](gym-stock\gym_stock\envs\stock_env.py) environment is slightly different from the [stockenv](env\stockenv.py) as the return observation is a 138-d vector containing the hands count. Not a tuple with 137-d observation and hand count
### install gym-stock
``` bash
pip install -e gym-stock
```
Then can build the gym-stock in the gym environment by 
``` python
import gym
import gym_stock
gym.make('stock-v0')
```

add the following line in the [run.py line 52](https://github.com/openai/baselines/blob/master/baselines/run.py#L52) of openai baseline
```python
_game_envs['custom_type']={'stock-v0'}
```

add the following line in [cmd_util](https://github.com/openai/baselines/tree/master/baselines/common/cmd_tuil.py#L65)
```python
from importlib import import_module
        import_module("gym_stock")
```
Then can train the policy with gym's algorithms
``` bash
python -m baselines.run --alg=deepq --env=stock-v0 --save_path=./cartpole_model.pkl --num_timesteps=1e5 --print_freq=10
```

<!-- add two lines and change the line 75 in cmd_util.py  in baseline to enable monitor's info keywords  -->