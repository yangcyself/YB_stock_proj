# Task 3 experiment Report

The task is to train an agent to invest in the market and earn as much profit as possible. The raw observation in this task is the features of the market, and the action space is three discrete choices, namely:  to hold, to sell one hand, and to buy one hand. As the task is simply choose actions according to observation, a naïve method is the train a predictor to predict the rising and falling trends in the near future, and then simply let the agent to buy when the predictor predicts price rising. We use this method as a baseline to test the ability to earn money of our agent. In the following sections, we will compare different reward function. And we will test the effect of adding in different data augmentation methods. Then we will test different network architectures. After this, we will compare two sets of reinforcement learning algorithms--Value function based methods like DQN or Policy Gradient based algorithms like DDPG, TRPO or PPO. At last, we adopt a new kind of network architecture, namely value iteration network in this task. The biggest characteristic of this algorithm is it is model based. As we can know exactly how many hands we will hold after a certain action, we hope this model based algorithm is suitable for the task.



## The Stock Simulation Environment

The stock simulation environment takes in the raw data as simulate the rises and falls of the stock market according to the raw data. Each step is a line in the data. In each episode, it is guaranteed that the difference of time stamps of each neighboring steps is less than 1 minute. 

### Dataset

Although generally reinforcement learning have no dataset, but for this typical task, we have to use the real dataset to represent a real stock market. Together with the notion of dataset, we also have to divide the train and test dataset. In our experiments, we divided the whole dataset into 100 sections, and use 70% of them as train datasets and 30% of them as test datasets.

### rewards

In reinforcement learning, as the agent tries its best to maximize the reward, the reward has huge impact on its behavior. We have designed some rewards and tested their effect.

Note that in the following experiments, we have divided the earned reward by 5000 to keep the loss and gradient of neural network training in a reasonable range

#### direct rewards

The most straight forward reward that comes to our mind is to simulate the money payed and earned in the action of buy and sell stocks. The reward can be expressed as following:
$$
R_{direct}(t) = (Hand(t-1)-Hand(t))*(price(t,a(t)))
$$
The $a(t)$ is the action at time $t$, and the $price(t,a)$ is the price related to the action--sell price for sell action and buy price for buy action. Note that the reward is negative if the agent buys, this might make the agent not willing to buy stock in the following experiments. 

But this reward is still suitable as the aim of agent is to maximize the total discounted reward.
$$
R = \sum_t\gamma^tR_{direct}(t)
$$
We choose $\gamma = 1$ and then the reward of each action is the total income by a series of buy and sell actions.

#### accumulated rewards

From the discussion with other groups, we found that some groups got good results with accumulated reward functions.  
$$
R_{accumulate}(t) = \sum_{\tau = 0}^{t} R_{direct}(\tau)
$$
We cannot understand why this kind of reward can yield a good performance, as the rewards do not represent the ability of earning money at present or in the future, but the money earned in the past. However we do tested this reward for the sake of completeness and practicality .

#### asset rewards

In the above reward functions, the number of stocks currently hold by the agent does not contribute directly to the reward. We can change this by introducing asset rewards.
$$
R_{asset}(t) = R_{direct}(t) +\lambda Hand(t)*bidprice(t)
$$
This means the when the agent takes a "buy" action, it does not actually pay all the money, its money is just changed into a another form of assets.

### observations

Another key issue about the training a good agent is the observation.  As shown in the first and second tasks, simply feed the raw features to the network might is not a feasible solution. Hence, we tried data augmentation based on the techniques explored in the first two tasks. However, we have to keep in mind that the data augmentation should not use the future data, otherwise, we are using future to predict the future. 



## Base Line



## Basic Case

The most basic experiment is DQN with a two hidden layer MLP network structure, with no data augmentation, and direct reward(pay money for buying and earn money for selling). The learning curve is as following.

![1561823347301](D:\yangcy\UNVjunior\EE359\Projects\stockRL\DQN-reward-0.png)

On one typical test dataset, the price and hand number scatter graph is following, where the x and y axes of the points means the time and price of the market, and the color means the number of hands hold at that moment. The lighter the color, the more hands the agent holds.

![1561827962147](D:\yangcy\UNVjunior\EE359\Projects\stockRL\pics\DQN-0-hand-scatter0.png)

We can see that generally the agent holds more hands when the price is low(at about 4000 step)  in the hope of a bounce back. And we can zoom in to find that the agent can correctly buy in stocks before a rise and hold till the market goes down. 

![1561827909927](D:\yangcy\UNVjunior\EE359\Projects\stockRL\pics\DQN-0-hand-scatter.png)

At the step about 3500 the agent is more likely to buy in as the price is low. The zoomed view is as following. We can see that the agent holds stock at about 3500 steps with the price goes down. This mistake can cost the agent money, but later on it correctly bought stocks before a summit.

![1561828428147](D:\yangcy\UNVjunior\EE359\Projects\stockRL\pics\DQN-0-hand-scatter1.png)

We observed that the network prone to converge to a local maxima as the trained steps increases. Here the local maxima is to sell the initial hands but seldom buy any stock. This phenomenon might because  the direct reward gives positive reward every time the agent choose a sell action, and give a negative reward every time the agent choose a buy action. Although the total reward is the actual money earned by the agent, the reward directly penalize the buy action and makes the agent less likely to buy. Or maybe this is because the agent's poor ability to make predictions, which can be alleviated by data augmentation, or a more expressive network structure.

### Basic case with accumulated rewards

Using the accumulated rewards described above, we get the following learning curve.

![1561879501856](D:\yangcy\UNVjunior\EE359\Projects\stockRL\pics\DQN-accuR1.png)

This looks similar to the curve with direct rewards, but the curve of the sell action percentage is interesting.

![1561879630542](D:\yangcy\UNVjunior\EE359\Projects\stockRL\pics\DQN-accuRSellPerc.png)

This is the plot about the percentage of the agent to choose "Sell" action. Note that the agent can still chose sell action even when its short hands reaches its limit, the effect in this case is just same as "hold". This means to things, the first is that the agent still converge to the local maximum of not buy at all.  Another is that the agent some how learned to buy some stocks or to delay the "sell" action, but it found at last that this kind of action cannot increase the reward. This is very simple to explain, this is because if the agent can sell the stocks as soon as possible, it the earned reward is added in all the following steps. 

### Basic case with asset rewards

With the asset reward, the agent prone to buy stock as soon  and as much as possible. The learning curve is as following

![1561886426862](D:\yangcy\UNVjunior\EE359\Projects\stockRL\pics\DQNassetR.png)

Although the learning curve looks good, the curve of percentage of "buy" and "sell" action is:

![1561886538219](D:\yangcy\UNVjunior\EE359\Projects\stockRL\pics\DQNassetR_buy.png)![1561886566650](D:\yangcy\UNVjunior\EE359\Projects\stockRL\pics\DQNassetR_sell.png)

we can find that the agent seldom sells.

Thus, from the comparison above, we conclude the direct reward is the best and we will use the direct reward in the following experiments.

## LSTM Model

Although some people say that using Recurrent Neural Networks in reinforcement learning tasks is a violation of the Markov assumption, using LSTM in reinforcement learning neural network architecture is a common practice. Thus we tried to use LSTM as the network structure.

The behavior of LSTM  agent on test dataset can be shown as the following:

![1561884768057](D:\yangcy\UNVjunior\EE359\Projects\stockRL\pics\LSTMDQNhand0.png)

From which we can see that it learnt to not hold stocks when the market has a falling trend. And to hold stocks when the stock is going up.

However in this episode, the agent failed to money

>PROFIT: -979.0
>AveBuyPrice: 5611.262403528115
>AveSellPrice: 5610.193832599119

We looked at some other episodes, and found the model fails to earn money in most cases

## Data Augmentation 





## DDPG





## Value Iteration Networks

