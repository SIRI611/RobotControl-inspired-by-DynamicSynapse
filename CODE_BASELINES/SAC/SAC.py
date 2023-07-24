import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

# 定义SAC算法的神经网络模型
class SACModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SACModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义SAC算法
class SAC:
    def __init__(self, state_dim, action_dim, alpha=0.2, gamma=0.99, tau=0.005):
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = SACModel(state_dim, action_dim)
        self.critic1 = SACModel(state_dim, 1)
        self.critic2 = SACModel(state_dim, 1)
        self.target_critic1 = SACModel(state_dim, 1)
        self.target_critic2 = SACModel(state_dim, 1)

        # 初始化目标网络和加载参数
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        for param in self.target_critic1.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False

        # 定义优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=0.001)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.001)

    def select_action(self, state):
        # 根据当前状态选择动作
        with torch.no_grad():
            action = self.actor(state)
        return action.numpy()

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1)

        # 计算目标Q值
        with torch.no_grad():
            next_actions = self.actor(next_states)
            # print(next_actions)
            # print(next_states)
            target_q1 = self.target_critic1(next_states)
            target_q2 = self.target_critic2(next_states)
            target_q = torch.min(target_q1, target_q2) - self.alpha * torch.log(self.actor(next_states))

        target_q = rewards + self.gamma * target_q * (1 - dones)

        # 更新Critic网络
        current_q1 = self.critic1(states)
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        current_q2 = self.critic2(states)
        critic2_loss = nn.MSELoss()(current_q2, target_q)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 更新Actor网络
        actions = self.actor(states)
        actor_loss = -(self.critic1(states) - self.alpha * torch.log(self.actor(states))).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新目标网络
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# 创建Mujoco humanoid-v4环境
env = gym.make('Humanoid-v4')

# 获取状态和动作空间维度
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# 初始化SAC算法
sac_algorithm = SAC(state_dim, action_dim)

# 进行训练
num_episodes = 1000
max_steps_per_episode = 1000
for episode in range(num_episodes):
    state = env.reset()[0]
    total_reward = 0
    for t in range(max_steps_per_episode):
        # 在环境中执行动作
        # print(state)
        action = sac_algorithm.select_action(torch.tensor(state).float())
        next_state, reward, done, _, _ = env.step(action)

        # 存储经验并更新算法
        experience = (state, action, reward, next_state, done)
        sac_algorithm.update(experience)

        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# 评估训练后的模型
# ...

# 保存模型
# ...
