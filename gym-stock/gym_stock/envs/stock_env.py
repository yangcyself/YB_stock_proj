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
maxstep = 200
filePath = os.path.join(os.path.dirname(os.path.realpath(__file__)),"envdata")
outfilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)),"envoutdata")
allNumbers = set([i for i in range(100)])
testNumbers = set([i for i in range(0,100,7)])
trainNumbers = allNumbers - testNumbers
MODE = "TEST"
# MODE = "TRAIN"
inputDataNames = [os.path.join(filePath,"%denv.pkl"%i) for i in (trainNumbers if MODE != "TEST" else testNumbers)] 
ChangFile_num = 50
class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    # all the data files are in the form of pickle, saved as pandas dataframe
    # at each time, randomly select a pickle to load, and randomly choose a time to start

    def __init__(self,inputdataNames = inputDataNames):
        # the length of the observation is 137 (not including hand)
        high = np.ones((137,))
        high = np.array(list(high)+[5])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low = -high,high = high,dtype = np.float32)

        self.hands = 0
        self.inputDataNames = inputDataNames
        self.current = None
        self.index = 0
        
        self.action_hand = [-1,0,1] # mapping the change of hand and the action 
        self.action_prize = [119,108,118] # the index of bidprice1, midprice, Askprice
        
        self.deltaTimeThresh = pd.Timedelta('600s')
        
        self.change_file_count = 0
        self.history = None
        self.fileind = 0
    def _concateObs(self,stock,hand):
        return np.array(list(stock)+[int(hand)])

    def step(self,action):
        old_hand = self.hands
        self.hands += self.action_hand[action]
        self.hands = min(10,self.hands)
        self.hands = max(0,self.hands)
        self.index += 1
        tmp_obs = self.current.iloc[self.index:self.index+1]
        tmp_obs = tmp_obs.drop(columns = ["UpdateMillisec","UpdateTime"])
        self.actionCounts[action] += 1
        # print(tmp_obs.columns[119],tmp_obs.columns[108],tmp_obs.columns[118])
        # if(len(tmp_obs.values)==0):
        # print(self.index,len(self.current.index))
        stock_obs = tmp_obs.values[0]
        self.maxstep -= 1
        done = (self.index+1 >= len(self.current.index) or self.current.iloc[self.index]["UpdateTime"] - 
                self.current.iloc[self.index-1 ]["UpdateTime"] >  self.deltaTimeThresh 
                or (self.maxstep<=0 and MODE!="TEST"))
        rwd = (old_hand-self.hands) * stock_obs[self.action_prize[action]]
        hand_obs = np.array(self.hands)

        if(done):
            rwd +=  self.hands* stock_obs[self.action_prize[0]]

        #rwd can be too large
        rwd /=5000

        info = {}
        totalActions = sum([self.actionCounts[i] for i in range(3)])
        info["EpisodeHoldPercentage"] = self.actionCounts[1]
        info["EpisodeSellPercentage"] = self.actionCounts[2]
        info["EpisodeBuyPercentage"] = self.actionCounts[0]
        self.history["midprice"].append(self.stock_obs[self.action_prize[1]]) # the midprice
        self.history["action"].append(action)
        self.history["hands"].append(hand_obs)
        return self._concateObs(stock_obs,hand_obs-5), rwd, done, info

    def reset(self):
        if(MODE=="TEST"):
            with open(os.path.join(outfilePath,"%dhistory.pkl" %self.fileind),"wb") as f:
                f.dump(self.history)
            self.history = {k:[] for k in ["midprice","action","hands"] }
            selectedFile = self.inputDataNames[self.fileind]
            self.fileind = (self.fileind + 1) % len(self.inputDataNames)
        else:
            if(self.current is None or (self.change_file_count > ChangFile_num)): 
                selectedFile = random.choice(self.inputDataNames)
                # pd.read_pickle(selectedFile)
                self.change_file_count = 0
        self.current = pd.read_pickle(selectedFile)
        print("loaded: ", selectedFile)
        self.maxstep = maxstep
        self.change_file_count += 1
        self.index = random.randint(0,len(self.current.index)-1)
        self.actionCounts = [0]*3
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
        return self._concateObs(stock_obs, np.array(self.hands-5))
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
        self.price = 100
        self.obs = 0
        self.stepcount = 0
        return self.generage_obs(),self.hands