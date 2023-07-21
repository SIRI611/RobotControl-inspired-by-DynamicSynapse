from SACmodel import Actor,Critic
import torch.optim as optim
from parameters import *
import torch
import numpy as np

class SaC:
    def __init__(self,N_S,N_A):
        self.actor_net =Actor(N_S,N_A)
        self.critic_net = Critic(N_S)
        self.actor_optim = optim.Adam(self.actor_net.parameters(),lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic_net.parameters(),lr=lr_critic,weight_decay=l2_rate)
        self.critic_loss_func = torch.nn.MSELoss()

    def train(self,memory):
        memory = np.array(memory)
        states = torch.tensor(np.vstack(memory[:,0]),dtype=torch.float32)
        actions = torch.tensor(list(memory[:,1]),dtype=torch.float32)
        rewards = torch.tensor(list(memory[:,2]),dtype=torch.float32)
        states_ = torch.tensor(list(memory[:,3]),dtype=torch.float32)
        masks = torch.tensor(list(memory[:,4]),dtype=torch.float32)
        q1,q2 = self.critic_net(states)
              
        old_mu,old_std = self.actor_net(states)     
        pi = self.actor_net.distribution(old_mu,old_std)    
        old_log_prob = pi.log_prob(actions).sum(1,keepdim=True)

        n = len(states)
        arr = np.arange(n)
        for epoch in range(1):
            np.random.shuffle(arr) # 对arr进行打乱
            for i in range(n//batch_size):  # 每一个batch
                # print("batch:",i)
                batch_index = arr[batch_size*i:batch_size*(i+1)]    #随机选择memory中的记录放在一个batch
                b_states = states[batch_index]
                b_actions = actions[batch_index]
                b_rewards = rewards[batch_index].unsqueeze(1)
                b_states_ = states_[batch_index]
                b_masks = masks[batch_index].unsqueeze(1)

                mu,std = self.actor_net(b_states)
                pi = self.actor_net.distribution(mu,std)
                
                old_prob = old_log_prob[batch_index].detach()  # detach()从计算图中脱离，最后没有grad
                
                new_prob = pi.log_prob(b_actions).sum(1,keepdim=True)       # 按行相加，保持原数组维度
                
                q1,q2 = self.critic_net(b_states)               
                q1_,q2_ = self.critic_net(b_states_)

                qt = torch.min(q1_,q2_) - alpha * new_prob
                qt = b_rewards + gamma * torch.mul(b_masks , qt)
                critic_loss = self.critic_loss_func(q1,qt) + self.critic_loss_func(q2,qt)
                 
                actor_loss = -((alpha * old_prob)-torch.min(q1,q2)).mean()
                # print(critic_loss)
                # print(actor_loss)             
                self.actor_optim.zero_grad()    # 清除grad
                actor_loss.backward(retain_graph=True)           # 反向传播
                self.actor_optim.step()
                
                self.critic_optim.zero_grad()
                critic_loss.backward()
                # critic_loss.backward()
                self.critic_optim.step()

                #print(self.critic_net.fc1.weight)
