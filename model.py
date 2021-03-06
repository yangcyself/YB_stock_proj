# -^- coding:utf-8 -^-
"""
This is a model that input a time state(a series of features) and out put an action selection
first planning to train it with simple policy gradient, 
"""
from TCN.tcn import TemporalConvNet
import tensorflow as tf
import tensorflow.nn as tfnn
import numpy as np
from tensorflow import losses

Feature_num = 137
modelDebug = False
def build_tcn(inputs,tcn_dropout,kernel_size,num_channels):
    # inputs = placeholder
    # self.dropout = tf.placeholder_with_default(0., shape=())

    # num_channels = [hidden1, hidden2, ...., outputchannel]
    # kernel_size
    tcn = TemporalConvNet(num_channels, stride=1, kernel_size=kernel_size, dropout=tcn_dropout)
    outputs = tcn(inputs)
    return outputs




class Actor_Critic(object):
    def __init__(self, sess, action_dim, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a,self.optimizer,self.value = self._build_net(S,H, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_,_ , self.value_ = self._build_net(S_,H_, scope='target_net', trainable=False)
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s,h, scope, trainable,ent_coef = 0.01, vf_coef = 0.5):
        # s is the state of the current market
        # h is the number of hand 0-11
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 30, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)

            

            with tf.variable_scope('tcn'):
                tcndropout = tf.placeholder_with_default(0., shape=())
                value_map = build_tcn(s,tcndropout,kernel_size=3,num_channels=[256,64,32,10])
                if(modelDebug):
                    print("value_map shape",value_map.shape) #value_map shape (?, 20, 10)
            with tf.variable_scope('vin'):
                v = value_map[:,-1,tf.newaxis,:] # get the values of the last time step
                vi_w = tf.get_variable('vi_w', [3,1,3], initializer=init_w, trainable=trainable)
                for i in range(-2,-5,-1):
                    q = tf.pad(v,tf.constant([[0,0],[0,0],[1,1]]))
                    q = tfnn.conv1d(q,vi_w,1,"VALID",data_format="NCW")
                        #v: [?,1,1,12] vi_w:[1,3,1,3]
                    if(modelDebug):
                        print("q shape",q.shape) # q shape (?, 3, 10)
                    v = tf.reduce_max(q, axis=1, keepdims=True, name="v%d"%i)
                    v = v + value_map[:,i,tf.newaxis,:]
                # print(v.shape)
            with tf.variable_scope('a'):
                v = v[:,0,:] # reshape v into rank2
                paddings = tf.constant([[0, 0],[3,3]])
                v = tf.pad(v,paddings,"SYMMETRIC")
                h_pos = tf.one_hot(h,depth=10)
                # att_v = v[:,0,h:h+7]# the attentioned value function
                att_v = tf.concat([v, h_pos], 1) # concat the onehot position 
                if(modelDebug):
                    print("att_v",att_v.shape) #att_v (?, 26)
                action = tf.layers.dense(att_v, self.a_dim,  kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                action = tf.nn.softmax(action) #action (?, 3)
                if(modelDebug):
                    print("action",action.shape)

                value = tf.layers.dense(att_v,1,
                        kernel_initializer = init_w,bias_initializer = init_b,name = "v",trainable = trainable)
                if(modelDebug):
                    print("value :",value.shape)

                a = tf.argmax(action,axis = 1) # the optimal action selected by algorithm for inference
                if(modelDebug):
                    print("a:",a.shape)
                a_hot = tf.one_hot(A,depth = 3) # the one_hot vector from A(place holder of explored action) for training
                prob = tf.reduce_sum(tf.multiply(action, a_hot),reduction_indices=[1])
                eligibility = tf.log(prob) * (R - value ) 
                loss = -tf.reduce_sum(eligibility)

                entropy = tf.reduce_mean( tf.multiply( tf.log(action), action  )) # the entropy term promotes exploration
                if(modelDebug):
                    print(" tf.multiply( tf.log(action), action  )", tf.multiply( tf.log(action), action  ).shape)
                    print("entropy",entropy.shape)
                loss += entropy * ent_coef

                vf_loss = losses.mean_squared_error(value, R)
                loss -= vf_loss * vf_coef

                optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

        return a,optimizer,value


    def learn(self, transitions,gamma = 0.95):   # batch update
        s,h,a,s_,h_,r = transitions
        # print("rshape",r.shape)
        v = self.sess.run(self.value_, feed_dict = {S_:s_, H_:h_})
        # print("vshape",v.shape)
        #v = r + gamma * v
       	r = r.reshape(-1,1)
        v = r + gamma*v
        # print(s.shape, h.shape, v.shape)
        self.sess.run(self.optimizer, feed_dict={S: s,H:h,R:v,A:a})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s,h, epsilon = 0.1):
        s = s[np.newaxis, :]    # single state
        h = h.reshape(1,*(h.shape))
        a = self.sess.run(self.a, feed_dict={S: s,H:h})
        if(np.random.rand()<epsilon):
            a = np.random.randint(0,3)
        return a



state_dim = (20,Feature_num) # num_steps, num_features
# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, *state_dim], name='s')
    H = tf.placeholder(tf.int32, shape=[None,], name='h')

with tf.name_scope("A") :
    A = tf.placeholder(tf.int32, shape=[None,], name='a')

with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')

with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, *state_dim], name='s_')
    H_ = tf.placeholder(tf.int32, shape=[None,], name='h_')


if __name__ == '__main__':
    modelDebug  = True
    sess = tf.Session()
    actor = Actor_Critic(sess, 3, 0.001, dict(name='soft', tau=0.01))
    sess.run(tf.global_variables_initializer())
    from env.stockenv import StockEnv
    e = StockEnv()
    s,h = e.reset()

    s = np.concatenate([s.reshape((1,-1))]*20)
    
    print(s.shape,h.shape)
    print(actor.choose_action(s,h)) #input s: 20*137 h 1 When choose action, we don't have to add batch axis of s
    s = s.reshape(1,*(s.shape))
    h = h.reshape(1,*(h.shape))
    s,h,r,_ = e.step(1)
    # actor.learn(s,h,np.array([[-10000]]))

    # s = s.reshape(*(s.shape[1:]))
    # h = h.reshape(*(h.shape[1:]))
    # print(actor.choose_action(s,h)) #input s: 20*137 h 1 When choose action, we don't have to add batch axis of s
