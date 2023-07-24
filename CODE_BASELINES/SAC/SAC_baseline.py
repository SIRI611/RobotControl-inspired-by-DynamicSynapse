import sys
import os
sys.path.append("~/桌面/Code/Code_BASELINES/")
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# 创建Mujoco humanoid-v4环境
# env = gym.make('Humanoid-v4', render_mode="human")
env = gym.make('Humanoid-v4')
# 初始化SAC算法
model = SAC('MlpPolicy', env, verbose=1)

# 使用SAC算法对环境进行训练
model.learn(total_timesteps=1000)

# 训练完成后，您可以将模型保存到文件
model.save("sac_humanoid")

# 加载保存的模型
loaded_model = SAC.load("sac_humanoid.pkl")

# 使用训练好的模型进行评估
mean_reward, _ = evaluate_policy(loaded_model, env, n_eval_episodes=10)
print("Mean reward:", mean_reward)
