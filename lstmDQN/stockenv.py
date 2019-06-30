import numpy as np
import pandas as pd
import gym
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold


def get_data_set(data_file):
    dataset_df = pd.read_csv(data_file)
    UpdateTime = dataset_df.UpdateTime.values
    RTime = UpdateTime.copy()
    timeArray = []
    for i in range(len(UpdateTime)):
        time_s = '2019-04-08' + ' ' + UpdateTime[i]
        timeArray.append(time.strptime(time_s,'%Y-%m-%d %H:%M:%S'))
    for i in range(1,len(UpdateTime)):
        RTime[i] = int(time.mktime(timeArray[i]))
    RTime[0] = RTime[1]-3

    n = 10
    i_drop = []
    for i in range(len(UpdateTime)-n):
        if abs(RTime[i+n] - RTime[i]) > 3000:
            i_drop.append(i)
    dataset_df.drop(i_drop, inplace = True)
    data = dataset_df.drop(['UpdateMillisec','UpdateTime', 'midPrice','LastVolume'],axis=1)[:-n]

    scaler = MinMaxScaler()
    trans_data = scaler.fit_transform(data)


    fea1 = VarianceThreshold(threshold=0.0287).fit_transform(trans_data)  # train data threshold 0.01
    UpdateTime = dataset_df.UpdateTime.values
    RTime = UpdateTime.copy()
    timeArray = []
    for i in range(len(UpdateTime)):
        time_s = '2019-04-08' + ' ' + UpdateTime[i]
        timeArray.append(time.strptime(time_s,'%Y-%m-%d %H:%M:%S'))

    for i in range(1,len(UpdateTime)):
        RTime[i] = int(time.mktime(timeArray[i]))
    RTime[0] = RTime[1]-3
    sep_time = []
    for i in range(len(UpdateTime)-1):
        if abs(RTime[i+1] - RTime[i]) > 3000:
            sep_time.append(i)
    print(sep_time)
    m = 20
    X_train = []
    originAsk1 = []
    originBid1 = []
    i = m
    while i < len(x_train):
        x_mat = []
        if i in sep_time:
            i += 20
            continue
        originAsk1.append(data['AskPrice1'].iloc[i])
        originBid1.append(data['BidPrice1'].iloc[i])
        for j in range(m):
            x_mat.append(x_train[i+j-m])
        X_train.append(x_mat)
        i += 1
    X_train = np.array(X_train)
    print(X_train.shape)
    X_train = X_train.reshape(len(X_train),20,42)
    return X_train, originAsk1, originBid1


class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, inputData, Ask, Bid):
        self.X = inputData
        self.A = Ask
        self.B = Bid
        self.reward = 0
        self.begin_pos = -1000
        self.index = 0
        self.state_size = inputData.shape[1:]
        self.action_size = 3


    def step(self, action):
        # 0 do nothing, 1 buy, 2 sell.
        if action == 0:
            self.reward += 0
        elif action == 1:
            self.reward += -self.A[self.index]
        elif action == 2:
            self.reward += self.B[self.index]
        else:
            print("No such action.")
        self.index += 1
        # print("env reward:", self.reward)

        return self.X[self.index],self.reward,False


    def reset(self):
        self.begin_pos += 300
        self.reward = 0
        self.index = self.begin_pos
        return self.X[self.index]

    def render(self, mode='human'):
        pass


    def close(self):
        pass

