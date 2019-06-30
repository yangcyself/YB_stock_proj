#-^- coding: utf-8 -^-

import logging
import numpy as np
import random
import pandas as pd
import os

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pickle as pkl
from sklearn.decomposition import PCA 

Feature_num = 137
maxstep = 200
filePath = os.path.join(os.path.dirname(os.path.realpath(__file__)),"envdata")
outfilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)),"envoutdata")
allNumbers = set([i for i in range(100)])
testNumbers = set([i for i in range(0,100,7)])
trainNumbers = allNumbers - testNumbers
MODE = "TEST"
# MODE = "TRAIN"
# ACCUMUREWARD = True
# ACCUMUREWARD = False
REWARDKIND = ["DIRECT","ACCUMULATE","ASSET"][2]
initHand = 0
obsLenth = 20
inputDataNames = [os.path.join(filePath,"%denv.pkl"%i) for i in (trainNumbers if MODE != "TEST" else testNumbers)] 
ChangFile_num = 50
OBS_HANDS = True
OBS_HANDS = ["NO","CONCATE","TUPLE"][2]
trainAugLength = 200 # the number of data steps used to train the Augmentor

class Augmentor():
    """
    The dataAugmentor class
    used to augment the observation data
    """
    def fit(self,X):
        """
        fit a bunch of data to learn the parameter of aumentor
        """
        pass
    def transform(self,X):
        """
        input the observation X and return the augmented observation
        """
        pass

class NormAugmentor(Augmentor):
    """
        Normalize the observation so that the average is 0 and the variation is 1
    """
    def __init__(self):
        self.outputdim = Feature_num
    def fit(self,X):
        """
        X is a two dim N*P array
        """
        self.ave = np.mean(X, axis = 0)
        self.std = np.var(X,axis = 0) +0.5
    def transform(self,X):
        X = (X-self.ave)/self.std
        return X

class PCAAugmentor(Augmentor):
    """
        Use PCA to do the dimention reduction, after Normalized the data
    """
    def __init__(self,outputdim = 42):
        self.outputdim = outputdim
        self.normalizer = NormAugmentor()
        self.pca = PCA(n_components = outputdim)
    def fit(self,X):
        self.normalizer.fit(X)
        normed = self.normalizer.transform(X)
        self.pca.fit(normed)
    def transform(self,X):
        normed = self.normalizer.transform(X)
        return self.pca.transform(normed)

class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    # all the data files are in the form of pickle, saved as pandas dataframe
    # at each time, randomly select a pickle to load, and randomly choose a time to start

    def __init__(self,inputdataNames = inputDataNames):
        # the length of the observation is 137 (not including hand)
        obsnum = 42
        high = np.ones((obsLenth,obsnum))
        # high = np.array(list(high)+[5])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low = -high,high = high,dtype = np.float32)

        self.hands = 0
        self.inputDataNames = inputDataNames
        self.current = None
        self.index = 0
        
        self.action_hand = [-1,0,1] # mapping the change of hand and the action 
        self.action_prize = [119-137,108-137,118-137] # the index of bidprice1, midprice, Askprice
        
        self.deltaTimeThresh = pd.Timedelta('600s')
        
        self.change_file_count = 0
        self.history = None
        self.fileind = 0
        # self.augmentor = NormAugmentor()
        self.augmentor = PCAAugmentor(obsnum)

    def _concateObs(self,stock,hand):
        return np.array(list(stock)+[int(hand)])

    def step(self,action):
        old_hand = self.hands
        self.hands += self.action_hand[action]
        self.hands = min(10,self.hands)
        self.hands = max(0,self.hands)
        self.index += 1
        tmp_obs = self.current.iloc[self.index:self.index+obsLenth]
        tmp_obs = tmp_obs.drop(columns = ["UpdateMillisec","UpdateTime"])
        self.actionCounts[action] += 1
        # print(tmp_obs.columns[119],tmp_obs.columns[108],tmp_obs.columns[118])
        # if(len(tmp_obs.values)==0):
        # print(self.index,len(self.current.index))
        stock_obs = tmp_obs.values#.reshape(-1)
        self.maxstep -= 1
        done = (self.index+obsLenth >= len(self.current.index) or self.current.iloc[self.index]["UpdateTime"] - 
                self.current.iloc[self.index-1 ]["UpdateTime"] >  self.deltaTimeThresh 
                or (self.maxstep<=0 and MODE!="TEST"))
        rwd = (old_hand-self.hands) * stock_obs[-1][self.action_prize[action]]
        hand_obs = np.array(self.hands)

        if(done):
            rwd +=  self.hands* stock_obs[-1][self.action_prize[0]]

        #rwd can be too large
        rwd /=5000

        if(REWARDKIND=="ACCUMULATE"):
            self.accumulated_reward += rwd
            rwd = self.accumulated_reward
        elif(REWARDKIND=="ASSET"):
            rwd += self.hands*stock_obs[-1][self.action_prize[0]]/6000 # add the bid price as the asset saved in the market

        info = {}
        totalActions = sum([self.actionCounts[i] for i in range(3)])
        info["EpisodeHoldPercentage"] = self.actionCounts[1] /sum(self.actionCounts)
        info["EpisodeSellPercentage"] = self.actionCounts[0]/sum(self.actionCounts)
        info["EpisodeBuyPercentage"] = self.actionCounts[2]/sum(self.actionCounts)
        if(MODE=="TEST"):
            self.history["bidprice"].append(stock_obs[-1][self.action_prize[0]])
            self.history["midprice"].append(stock_obs[-1][self.action_prize[1]]) # the midprice
            self.history["askprice"].append(stock_obs[-1][self.action_prize[2]]) 
            self.history["action"].append(action)
            self.history["hands"].append(hand_obs)
        # print("len(self._concateObs(stock_obs,hand_obs-5))",len(self._concateObs(stock_obs,hand_obs-5)))
        if(self.augmentor is not None):
            stock_obs = self.augmentor.transform(stock_obs)

        if(OBS_HANDS=="TUPLE"):
            return (stock_obs,(hand_obs-5).reshape((1,))), rwd, done, info
        elif(OBS_HANDS=="CONCATE"):
            return self._concateObs(stock_obs,hand_obs-5), rwd, done, info
        else:
            return stock_obs, rwd, done, info


    def reset(self):
        if(MODE=="TEST"):
            with open(os.path.join(outfilePath,"%dhistory.pkl" %self.fileind),"wb") as f:
                pkl.dump(self.history,f)
            self.history = {k:[] for k in ["midprice","action","hands","bidprice","askprice"] }
            selectedFile = self.inputDataNames[self.fileind]
            self.fileind = (self.fileind + 1) % len(self.inputDataNames)
            self.current = pd.read_pickle(selectedFile)
            self.index = 0
        else:
            if(self.current is None or (self.change_file_count > ChangFile_num)): 
                selectedFile = random.choice(self.inputDataNames)
                # pd.read_pickle(selectedFile)
                self.change_file_count = 0
                self.current = pd.read_pickle(selectedFile)
                print("loaded: ", selectedFile)

            self.index = random.randint(0,len(self.current.index)-1)
        self.maxstep = maxstep
        self.change_file_count += 1
        self.actionCounts = [0]*3
        # find the begining of the day
        forward_count = 0
        while(self.index>0 and forward_count < 200 and 
            self.current.iloc[self.index]["UpdateTime"] - 
                self.current.iloc[self.index - 1]["UpdateTime"] <  self.deltaTimeThresh ):
            self.index = self.index - 1
            forward_count +=1
        if(self.augmentor is not None):
            trainAug = self.current.iloc[self.index:self.index + trainAugLength].drop(columns = ["UpdateMillisec","UpdateTime"]).values
            self.augmentor.fit(trainAug)
            self.index += trainAugLength
        print("start from: ",self.index)
        # self.hands = 5
        self.hands = random.randint(0,10) if initHand<0 else initHand
        print("epoInitHand:",self.hands)
        tmp_obs = self.current.iloc[self.index:self.index+obsLenth]
        tmp_obs = tmp_obs.drop(columns = ["UpdateMillisec","UpdateTime"])
        # print(tmp_obs.columns.values)
        stock_obs = tmp_obs.values#.reshape(-1)
        # print(stock_obs.shape)
        if(REWARDKIND=="ACCUMULATE"):
            self.accumulated_reward = 0
        if(self.augmentor is not None):
            stock_obs = self.augmentor.transform(stock_obs)
        if(OBS_HANDS=="TUPLE"):
            return (stock_obs, np.array(self.hands-5).reshape((1,)))
        elif(OBS_HANDS=="CONCATE"):
            return self._concateObs(stock_obs, np.array(self.hands-5))
        else:
            return stock_obs
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