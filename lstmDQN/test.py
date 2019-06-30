from stockenv import *
from dqn import *
import matplotlib.pyplot as plt
import pandas as pd
import gym
import gym_stock

if __name__ == "__main__":
    env = gym.make('stock-v0')
    state_size = env.observation_space.shape
    action_size = 3

    agent = LSTMDQNAgent(state_size, action_size)
    agent.load("lstm_model.h5")
    for i in range(10):
        state,hands = env.reset()
        # print(state.shape)
        state = state[np.newaxis,:,:]
        for k in range(3000):
            action = agent.act([state,hands])
            (next_state,next_hands), reward, done,_ = env.step(action)
            next_state = next_state[np.newaxis,:,:]

            # agent.remember(state, action, reward, next_state, done)
            state = next_state
            hands = next_hands
            if done :
                break

  