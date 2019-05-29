# -^- coding: utf-8 -^-

from model import Actor_Critic
from env.stockenv import StockEnv
import tensorflow as tf
import numpy  as np

sess = tf.Session()
actor = Actor_Critic(sess, 3, 0.001, dict(name='soft', tau=0.01))
sess.run(tf.global_variables_initializer())
e = StockEnv()
EPISODES = 100
MAXSTEPS = 200
WAIT = 20

for episode in range(EPISODES):
    s,h  = e.reset()
    s = s.reshape(1,-1)
    obs = s
    ep_s, ep_h, ep_s_, ep_h_, ep_r = [],[],[],[],[]
    for i in range(MAXSTEPS):
        if(obs.shape[0]<WAIT):
            (s_,h_),_,d,_ = e.step(1) 
        else:
            a = actor.choose_action(obs[-WAIT:],h)
            a = int(a)
            (s_,h_),r,d,_ =e.step(a)
        
        obs = np.concatenate([obs,s_.reshape(1,-1)])
        if(obs.shape[0]>WAIT):
            ep_h.append(h)
            ep_s.append(obs[-WAIT-1:-1])
            ep_h_.append(h_)
            ep_s_.append(obs[-WAIT:])
            ep_r.append(r)

        s = s_
        h = h_
    
        if(d):
            break
    ep_s = np.array(ep_s)
    ep_s_ = np.array(ep_s_)
    ep_h = np.array(ep_h)
    ep_h_ = np.array(ep_h_)
    ep_r = np.array(ep_r)
    # print(ep_s.shape,ep_s_.shape,ep_r.shape)
    transitions = (ep_s,ep_h,ep_s_,ep_h_, ep_r)
    actor.learn(transitions)
