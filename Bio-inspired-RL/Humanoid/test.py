# import gym
# from gym.wrappers import RecordVideo, capped_cubic_video_schedule
# env = gym.make("CartPole-v1")
# env.unwrapped.render_mode = "rgb_array"
# # env = RecordVideo(env, "videos")
# # the above is equivalent as
# env = RecordVideo(env, "videos", episode_trigger=capped_cubic_video_schedule)
# observation = env.reset()
# for _ in range(1000):
#     env.render()
#     action = env.action_space.sample() # your agent here (this takes random actions)
#     observation, reward, done, info = env.step(action)
#     if done:
#         observation = env.reset()
# env.close()

# import pickle 
# f = open(r'/home/robot/Documents/SimulationResult/OpenAIGym/Humanoid-v4/2023-08-13_21-27-59/Trace/Weights.pkl','rb')
# data = pickle.load(f)
# print(data)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    sns.set_style()
    dt = 3
    NeuronSensitivity = 0.5
    NeuronSensitivityUpdateRate = 0.00001
    start_time = 0
    stop_time = 100000
    tl = np.linspace(start_time, stop_time, num=int(stop_time / dt + 1))
    t = 0
    input_value_list = []
    output_value_list = []
    #   input_value = np.log(t+1) * (np.sin(t / 1000)) + np.random.randn(3, t.shape[0])
    for i in range(len(tl)):
        t = t+dt
        input_value = np.sin(t / 10000 + 1) * 3 * (np.sin(t / 1000)) + np.random.randn() + np.sin(
            t / 20000) * 2
        input_value *= 10
        input_value_list.append(input_value)
        output_value = input_value * NeuronSensitivity
        output_value_list.append(output_value)
        print(output_value)
        NeuronSensitivity += ((0.3-np.abs(0-input_value))*NeuronSensitivityUpdateRate*dt)*(2-np.log10(NeuronSensitivity))*(np.log10(NeuronSensitivity)-(-2))/4
    plt.plot(tl, input_value_list, "-")
    plt.plot(tl,output_value_list)
    plt.show()