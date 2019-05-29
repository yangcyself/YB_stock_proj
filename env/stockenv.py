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

class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    # all the data files are in the form of pickle, saved as pandas dataframe
    # at each time, randomly select a pickle to load, and randomly choose a time to start

    def __init__(self,inputdataNames = inputDataNames):
        
        self.hands = 0
        self.inputDataNames = inputDataNames
        self.current = None
        self.index = 0
        
        self.action_hand = [-1,0,1] # mapping the change of hand and the action self.action_prise = [] def step(self, action): assert(self.current is not None) self.index += 1 stock_obs = self.current[self.index] done = self.index == len(self.current-1) 
        self.action_prize = [119,108,118] # the index of bidprice1, midprice, Askprice
        
        self.deltaTimeThresh = pd.Timedelta('600s')
    
    def step(self,action):
        old_hand = self.hands
        self.hands += self.action_hand[action]
        self.hands = min(10,self.hands)
        self.hands = max(0,self.hands)
        self.index += 1
        tmp_obs = self.current.iloc[self.index]
        tmp_obs.drop(columns = ["UpdateMillisec","UpdateTime"])
        stock_obs = tmp_obs.values
        rwd = (old_hand-self.hands) * self.action_prize[action]
        done = (self.current.iloc[self.index+1]["UpdateTime"] - 
                self.current.iloc[self.index ]["UpdateTime"] >  self.deltaTimeThresh )
        hand_obs = self.hands

        if(done):
            self.current = None
        return (stock_obs,hand_obs), rwd, done, None

    def reset(self):
        
        selectedFile = random.choice(self.inputDataNames)
        pd.read_pickle(selectedFile)

        self.current = pd.read_pickle(selectedFile)

        self.index = random.randint(0,len(self.current.index))
        # find the begining of the day
        while(self.index>0 and 
            self.current.iloc[self.index]["UpdateTime"] - 
                self.current.iloc[self.index - 1]["UpdateTime"] <  self.deltaTimeThresh ):
            self.index = self.index - 1
        self.hands = 5
        tmp_obs = self.current.iloc[self.index]
        tmp_obs.drop(columns = ["UpdateMillisec","UpdateTime"])
        stock_obs = tmp_obs.values
        return stock_obs,self.hands
    def render(self, mode='human'):
        pass           


    # self.lines = 
    def close(self):
        pass