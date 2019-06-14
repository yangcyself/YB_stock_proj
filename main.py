# -^- coding: utf-8 -^-

from model import Actor_Critic
from env.stockenv import StockEnv,fannyEnv
import tensorflow as tf
import numpy  as np


################ Logger ###################

class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)


sess = tf.Session()
actor = Actor_Critic(sess, 3, 0.001, dict(name='soft', tau=0.01))
sess.run(tf.global_variables_initializer())
e = StockEnv()
#e = fannyEnv()
EPISODES = 100000
MAXSTEPS = 200
WAIT = 20
CHECKPOINTRATE = 1000
saver=tf.train.Saver(max_to_keep=1)
logger = Logger("./logs")

all_reward = 0

def bn(s):
    # print(s.shape)
    ave = s.mean(axis = 0, keepdims = True)
    var = s.var(axis = 0, keepdims  =True) + 1
    # print(var)
    # print(s.shape)
    s = (s-ave)/var
    return s

for episode in range(EPISODES):
    s,h  = e.reset()
    s = s.reshape(1,-1)
    obs = s
    ep_s, ep_h, ep_a, ep_s_, ep_h_, ep_r = [],[],[],[],[],[]
    epoTotalReward = -(s[0][108]*h)/5000 # make cost of invest of hands at first, thus, when this greater than 0, it means agent earded money
    epoTotalBuy ,epoTotalSell, epostartHand = 0,0,0
    for i in range(MAXSTEPS):
        if(obs.shape[0]<WAIT):
            (s_,h_),r,d,_ = e.step(1) 
            epoTotalReward = -(s_[0][108]*h_)/5000
        else:
            a = actor.choose_action( bn(obs[-WAIT:]),h)
            a = int(a)
            if(a == 0 and h>0):
                epoTotalSell +=1
            elif(a==2 and h<10):
                epoTotalBuy += 1
            (s_,h_),r,d,_ =e.step(a)
        
        obs = np.concatenate([obs,s_.reshape(1,-1)])
        if(obs.shape[0]>WAIT):
            ep_h.append(h)
            ep_s.append(bn(obs[-WAIT-1:-1]))
            ep_a.append(a)
            ep_h_.append(h_)
            ep_s_.append(bn(obs[-WAIT:]))
            ep_r.append(r)

        s = s_
        h = h_
        epoTotalReward += r 
        if(d):
            break
    # print(ep_r)
    print("epoTotalReward:",epoTotalReward)
    print("epoTotalBuy",epoTotalBuy,"epoTotalSell",epoTotalSell)
    ep_s = np.array(ep_s)
    ep_s_ = np.array(ep_s_)
    ep_a = np.array(ep_a)
    ep_h = np.array(ep_h)
    ep_h_ = np.array(ep_h_)
    ep_r = np.array(ep_r)
    # print(ep_s.shape,ep_s_.shape,ep_r.shape)
    transitions = (ep_s,ep_h,ep_a,ep_s_,ep_h_, ep_r)
    actor.learn(transitions)
    all_reward += epoTotalReward
    print("episode:",episode)
    if(((episode+1) % CHECKPOINTRATE)==0):
        
        savedfile = saver.save(sess, 'checkpoints/tcn_vin.ckpt', global_step=episode + 1)
        print("SAVED AT:", savedfile)
        info = {'averageTotalReward': all_reward/CHECKPOINTRATE}
        all_reward = 0
        for tag, value in info.items():
            logger.scalar_summary(tag, value, episode)