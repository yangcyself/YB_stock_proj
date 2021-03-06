#-^- coding: utf-8 -^-

import logging
import numpy as np
import random
import pandas as pd
import os

import gym
from gym import error, spaces, utils
from gym.utils import seeding

Feature_num = 137
filePath = os.path.join(os.path.dirname(os.path.realpath(__file__)),"envdata")
allNumbers = set([i for i in range(100)])
testNumbers = set([i for i in range(0,100,7)])
trainNumbers = allNumbers - testNumbers
inputDataNames = [os.path.join(filePath,"%denv.pkl"%i) for i in trainNumbers]
ChangFile_num = 50

class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    # all the data files are in the form of pickle, saved as pandas dataframe
    # at each time, randomly select a pickle to load, and randomly choose a time to start

    def __init__(self,inputdataNames = inputDataNames):
        
        self.hands = 0
        self.inputDataNames = inputDataNames
        self.current = None
        self.index = 0
        
        self.action_hand = [-1,0,1] # mapping the change of hand and the action 
        self.action_prize = [119,108,118] # the index of bidprice1, midprice, Askprice
        
        self.deltaTimeThresh = pd.Timedelta('600s')
        
        self.change_file_count = 0
    
    def step(self,action):
        old_hand = self.hands
        self.hands += self.action_hand[action]
        self.hands = min(10,self.hands)
        self.hands = max(0,self.hands)
        self.index += 1
        tmp_obs = self.current.iloc[self.index:self.index+1]
        tmp_obs = tmp_obs.drop(columns = ["UpdateMillisec","UpdateTime"])
        # print(tmp_obs.columns[119],tmp_obs.columns[108],tmp_obs.columns[118])
        stock_obs = tmp_obs.values[0]
        done = (self.index >= len(self.current.index) or self.current.iloc[self.index]["UpdateTime"] - 
                self.current.iloc[self.index-1 ]["UpdateTime"] >  self.deltaTimeThresh )
        rwd = (old_hand-self.hands) * stock_obs[self.action_prize[action]]
        hand_obs = np.array(self.hands)

        if(done):
            rwd +=  self.hands* stock_obs[self.action_prize[0]]

        #rwd can be too large
        rwd /=5000

        return (stock_obs,hand_obs), rwd, done, None

    def reset(self):
        if(self.current is None or self.change_file_count > ChangFile_num): 
            selectedFile = random.choice(self.inputDataNames)
            pd.read_pickle(selectedFile)
            self.current = pd.read_pickle(selectedFile)
            self.change_file_count = 0
            print("loaded: ", selectedFile)

        self.change_file_count += 1
        self.index = random.randint(0,len(self.current.index)-1)
        # find the begining of the day
        forward_count = 0
        while(self.index>0 and forward_count < 200 and 
            self.current.iloc[self.index]["UpdateTime"] - 
                self.current.iloc[self.index - 1]["UpdateTime"] <  self.deltaTimeThresh ):
            self.index = self.index - 1
            forward_count +=1
        
        print("start from: ",self.index)
        # self.hands = 5
        self.hands = random.randint(0,10)
        print("epoInitHand:",self.hands)
        tmp_obs = self.current.iloc[self.index:self.index+1]
        tmp_obs = tmp_obs.drop(columns = ["UpdateMillisec","UpdateTime"])
        # print(tmp_obs.columns.values)
        stock_obs = tmp_obs.values[0]
        # print(stock_obs.shape)
        return stock_obs, np.array(self.hands)
    def render(self, mode='human'):
        pass           


    # self.lines = 
    def close(self):
        pass

class fannyEnv():
    # this env tests whether the agent can learn something
    # when the state is positive, the price will go up next time
    def __init__(self):
        self.hands = 5
        self.price = 100
        self.obs = 0
        self.stepcount = 0
        self.action_hand = [-1,0,1] # mapping the change of hand and the action 
    def generage_obs(self):
        return np.array([self.obs]*Feature_num)

    def step(self,action):
        old_hand = self.hands
        self.hands += self.action_hand[action]
        self.hands = min(10,self.hands)
        self.hands = max(0,self.hands)
        rwd = (old_hand-self.hands) * self.price
        self.stepcount +=1
        
        done = self.stepcount == 100
        self.price += self.obs
        self.obs += 3*(random.random()-0.5)

        if(random.random()<0.2):
            self.obs = random.randint(-3,3)
        stock_obs = self.generage_obs()
        hand_obs = np.array(self.hands)

        if(done):
            rwd += self.hands*self.price
        return (stock_obs,hand_obs), rwd, done, None

    def reset(self):
        # self.hands = 5
        self.hands = random.randint(0,10)
        self.price = 0
        self.obs = 10 * (random.random()-0.5) #this is the over all tendency
        self.stepcount = 0
        return self.generage_obs(),self.hands
