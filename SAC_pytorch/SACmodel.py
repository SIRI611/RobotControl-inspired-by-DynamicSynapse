import torch
import torch.nn as nn
from parameters import *
import torch.optim as optim

#Actor网络
class Actor(nn.Module):
    def __init__(self,N_S,N_A):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(N_S,64)
        self.fc2 = nn.Linear(64,64)
        self.sigma = nn.Linear(64,N_A)
        self.mu = nn.Linear(64,N_A)
        self.distribution = torch.distributions.Normal
        
    #初始化网络参数
    def set_init(self,layers):
        for layer in layers:
            nn.init.normal_(layer.weight,mean=0.,std=0.1)
            nn.init.constant_(layer.bias,0.)

    def forward(self,s):
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        mu = self.mu(x)
        sigma = self.sigma(x)
        log_sigma = torch.clamp(sigma,-20.0,2.0)
        log_sigma = torch.exp(log_sigma)
        return mu,log_sigma

    def choose_action(self,s):
        mu,log_sigma = self.forward(s)
        # print(mu)
        # print(log_sigma)
        Pi = self.distribution(mu,log_sigma)    # 定义了一个正态分布
        
        return torch.tanh(Pi.sample()).numpy()

#Critic网洛
class Critic(nn.Module):
    def __init__(self,N_S):
        super(Critic,self).__init__()
        # Q1
        self.fc1 = nn.Linear(N_S,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,1)

        # Q2
        self.fc4 = nn.Linear(N_S,64)
        self.fc5 = nn.Linear(64,64)
        self.fc6 = nn.Linear(64,1)

    def set_init(self,layers):
        for layer in layers:
            nn.init.normal_(layer.weight,mean=0.,std=0.1)
            nn.init.constant_(layer.bias,0.)

    def forward(self,s):
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        q1 = self.fc3(x)
        
        y = torch.tanh(self.fc4(s))
        y = torch.tanh(self.fc5(y))
        q2 = self.fc6(y)
        return q1,q2
