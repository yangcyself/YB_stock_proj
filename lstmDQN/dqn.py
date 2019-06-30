# -*- coding: utf-8 -*-
import random
# from stockenv import *
import gym
import gym_stock
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,concatenate,Input
from keras.models import Model
from keras.optimizers import Adam

EPISODES = 100

class LSTMDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        inFeatures = Input(shape=self.state_size)
        inHands = Input(shape =(1,) )
        lstmmodel = Sequential()
        lstmmodel.add(LSTM(units=128, return_sequences=True, input_shape=self.state_size))
        lstmmodel.add(Dropout(0.15))
        lstmmodel.add(LSTM(units=128, return_sequences=True))
        lstmmodel.add(Dropout(0.15))
        lstmmodel.add(LSTM(units=32))
        lstmmodel.add(Dropout(0.15))
        lstmmodel.add(Dense(24, activation='relu'))
        outFeatures = lstmmodel(inFeatures)
        mixx = concatenate([outFeatures,inHands])
        tailmodel = Sequential() 
        print(mixx.shape)
        tailmodel.add(Dense(24, activation='relu',input_shape = (25,)))
        tailmodel.add(Dense(self.action_size, activation='linear'))
        model = Model(inputs = [inFeatures,inHands],outputs = [tailmodel(mixx)])
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # print("self.model.predict(state)",state[0].shape,state[1])
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for (state,hand), action, reward, (next_state,next_hand), done in minibatch:
            # print("reward",reward)
            target = reward
            # print("state,hand",state.shape,hand)
            # print("nextstate,nexthand",next_state.shape,next_hand)
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict([next_state,next_hand])[0]))
            target_f = self.model.predict([state,hand])
            target_f[0][action] = target
            self.model.fit([state,hand], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # x, a, b = get_data_set()
    # env = StockEnv(x,a,b)
    env = gym.make('stock-v0')
    state_size = env.observation_space.shape
    action_size = 3
    print(state_size,action_size)
    agent = LSTMDQNAgent(state_size, action_size)

    done = False
    batch_size = 32

    for e in range(EPISODES):
        (state,hands) = env.reset()
        # print(state.shape)
        state = state[np.newaxis,:,:]

        for time in range(300):
            # print(time)
            action = agent.act([state,hands])
            (next_state,next_hands), reward, done,_ = env.step(action)
            next_state = next_state[np.newaxis,:,:]

            agent.remember([state,hands], action, reward, [next_state,next_hands], done)
            state = next_state
            hands = next_hands
            if done or time == 299:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break

            if len(agent.memory) % batch_size == 0 and len(agent.memory) != 0:
                agent.replay(batch_size)
    agent.save("lstm_model.h5")