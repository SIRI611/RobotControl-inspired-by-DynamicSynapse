#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
import DynamicSynapseArray2DRandomSin as DSA
import CPGMerge 
#import DynamicSynapseArray2DLinear as DSA
import numpy as np
np.set_printoptions(threshold=np.inf)
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import platform
import shutil
import FitzHughNagumo as FN
from collections import deque
import dill
import copy
import argparse
# from Adapter.RangeAdapter import RangeAdapter
# from Adapter.tracelogger import TraceLogger
# from Adapter.tracereader import TraceReader


PlotPath = '/home/robot/Documents/SimulationResult/CPGDynamic/'
def ChooseRecordingPath(experiment, TimeOfRecording):
    if platform.node() == 'LAPTOP-68CSC593':
        path='J:/recording/OpenAIGym/'+experiment+'/'+TimeOfRecording+'/'
    elif platform.node() == 'robot-GALAX-B760-METALTOP-D4':
        path='/home/robot/Documents/SimulationRecord/OpenAIGym/'+experiment+'/'+TimeOfRecording+'/'
    else:
        path=''
    if not os.path.exists(path):
        os.makedirs(path)        
    return path

def ChooseResultPath(experiment, TimeOfRecording):
    if platform.node() == 'LAPTOP-68CSC593':
        path='J:/SimulationResult/OpenAIGym/'+experiment+'/'+TimeOfRecording+'/'
    elif platform.node() == 'robot-GALAX-B760-METALTOP-D4':
        path='/home/robot/Documents/SimulationResult/OpenAIGym/'+experiment+'/'+TimeOfRecording+'/'
    else:
        path = ''
    if not os.path.exists(path):
        os.makedirs(path)        
    return path


def softMax(AList):
    AList=np.array(AList)
    posibility = AList/(np.sum(AList))
    roll = np.random.rand()
    accum = 0
    for i1 in range(len(posibility)):        
        if roll>accum and roll<accum+posibility[i1]:
            return i1
        accum += posibility[i1]
def sigmoid(x):
    return 1 / (1 + np.exp(-x))        
def relu(x):
    return np.maximum(x,0,x)
def ObPreProcess(ob):
    # factors=[1/2*np.pi,5,2,2,1,
    #          0.5,1,0.3,1,1,
    #          0.5,1,0.25,1,2,
    #          1,1,1,1,1,
    #          1,1,1,1]
    # NormaledOb=ob*factors
    NormaledOb=ob
    co=relu(NormaledOb[0:45])
    re=-relu(-NormaledOb[0:45])
    return np.hstack((co,re,NormaledOb[45:],1))
def MergeInputs(Ob, output):
    return np.hstack((Ob,output))



def key_press(key, mod):
    global human_reward, human_wants_restart, human_sets_pause, human_stop, human_render
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    if key == ord('s'): human_stop = True
    if key == ord('r'): human_render = True        
    a = int( key - ord('0') )
    if a <= 0 or a >= 9: return
    human_reward = 2**((a-5))

def key_release(key, mod):
    global human_reward, human_render
    if key == ord('r'): human_render = False        
    a = int( key - ord('0') )
    if a <= 0 or a >= 9: return
    if human_reward ==  2**((a-5)):
        human_reward = 0
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def judge_record(i_episode):
    if i_episode%100 == 0:
        return True
    else:
        return False     
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Repulsive', nargs='?', type=str2bool,
                        choices=[True, False], 
                        default=True, 
                        help='Enable or disable repulsive learning')
    parser.add_argument('--StepRecording', nargs='?', type=str2bool,
                        choices=[True, False], 
                        default=True, 
                        help='Enable or disable step recording, at least 16 GB memory is required')    
    parser.add_argument('--Adaption', nargs='?', 
                        choices=['None', 'Linear', 'Nonlinear'], 
                        default='Nonlinear', 
                        help='Choose a type of adaptions')
    args = parser.parse_args()
    global human_reward, human_wants_restart, human_sets_pause, human_stop, human_render
    human_reward = 0
    human_wants_restart = False
    human_sets_pause = False  
    human_stop = False  
    human_render = False
    TimeOfRecording=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            # %%
    new = True #False # 
    Info = 0
    StepRecording = args.StepRecording
    ForceNoRendering=True
    Rendering=0
    AllRendering=0
    HumanRewarding=0
    LoadData=False
    ERSum = 0
    Solved = False
    SolveEpisode = 0
    RandomSeed = 10
    np.random.seed(RandomSeed) # choose a number for comparation between simulation with different arguments
    
    RepulsiveLearning = args.Repulsive
    Adaption = args.Adaption
    if new:
        experiment= 'Humanoid-v4'#'InvertedPendulum-v1' #'InvertedDoublePendulum-v1' #'Walker2d-v1' #'BipedalWalker-v2'# 'RoboschoolWalker2d-v1' #'Humanoid-v1' #'InvertedPendulum-v1' #specify environments here
        env = gym.make(experiment,terminate_when_unhealthy=True)
        env.unwrapped.render_mode = "human"
        # env = gym.wrappers.RecordVideo(env, ChooseRecordingPath(experiment, TimeOfRecording)+"video/", episode_trigger=lambda x : x%1000==0)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
            
        # env.render()
        # env.unwrapped.viewer.window.on_key_press = key_press
        # env.unwrapped.viewer.window.on_key_release = key_release

        num_states = env.observation_space.shape[0]
        # print(num_states)
        num_actions = env.action_space.shape[0]    
        observation = env.reset()[0]
        NumberOfSynapses = np.array([56, (num_states-331)*2+(331)+1+num_actions])

        dt = 3*5
        N_episode = 10000
        T = 0
        # Range_Adapter = RangeAdapter()
        # Range_Adapter.init_recording()
        WeightersCentre = 0 * (np.random.rand(*NumberOfSynapses)-0.5)
        # print(WeightersCentre.shape)

        #l1
        # WeightersCentre[0:8,0:19] = (np.random.rand()-0.5)*0.005
        WeightersCentre[0:8,19:21] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[0:8, 21:42] = (np.random.rand()-0.5)*0.005
        WeightersCentre[0:8, 42:44] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[0:8, 44:64] = (np.random.rand()-0.5)*0.005
        WeightersCentre[0:8, 64:66] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[0:8, 66:87] = (np.random.rand()-0.5)*0.005
        WeightersCentre[0:8, 87:89] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[0:8, 89:90] = (np.random.rand()-0.5)*0.005
        # WeightersCentre[0:8, 425:436] = (np.random.rand()-0.5)*0.005
        WeightersCentre[0:8, 436:438] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[0:8, 438:439] = (np.random.rand()-0.5)*0.005

        #r1
        # WeightersCentre[8:16, 0:16] = (np.random.rand()-0.5)*0.005
        WeightersCentre[8:16, 16:18] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[8:16, 18:39] = (np.random.rand()-0.5)*0.005
        WeightersCentre[8:16, 39:41] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[8:16, 41:61] = (np.random.rand()-0.5)*0.005
        WeightersCentre[8:16, 61:63] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[8:16, 63:84] = (np.random.rand()-0.5)*0.005
        WeightersCentre[8:16, 84:86] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[8:16, 86:90] = (np.random.rand()-0.5)*0.005
        WeightersCentre[8:16, 87:89] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[8:16, 425:433] = (np.random.rand()-0.5)*0.005
        WeightersCentre[8:16, 433:435] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[8:16, 435:439] = (np.random.rand()-0.5)*0.005

        #l2
        # WeightersCentre[16:24, 0:12] = (np.random.rand()-0.5)*0.005
        WeightersCentre[16:24, 12:15] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[16:24, 15:35] = (np.random.rand()-0.5)*0.005
        WeightersCentre[16:24, 35:38] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[16:24, 38:57] = (np.random.rand()-0.5)*0.005
        WeightersCentre[16:24, 57:60] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[16:24, 60:80] = (np.random.rand()-0.5)*0.005
        WeightersCentre[16:24, 80:83] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[16:24, 83:90] = (np.random.rand()-0.5)*0.005
        # WeightersCentre[16:24, 425:429] = (np.random.rand()-0.5)*0.005
        WeightersCentre[16:24, 429:432] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[16:24, 432:439] = (np.random.rand()-0.5)*0.005

        #r2
        # WeightersCentre[24:32, 0:8] = (np.random.rand()-0.5)*0.005
        WeightersCentre[24:32, 8:11] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[24:32, 11:31] = (np.random.rand()-0.5)*0.005
        WeightersCentre[24:32, 31:34] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[24:32, 34:53] = (np.random.rand()-0.5)*0.005
        WeightersCentre[24:32, 53:56] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[24:32, 56:76] = (np.random.rand()-0.5)*0.005
        WeightersCentre[24:32, 76:79] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[24:32, 79:90] = (np.random.rand()-0.5)*0.005
        WeightersCentre[24:32, 425:428] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[24:32, 428:439] = (np.random.rand()-0.5)*0.005

        #l3
        # WeightersCentre[32:40, 0:15] = (np.random.rand()-0.5)*0.005
        WeightersCentre[32:40, 15:16] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[32:40, 16:38] = (np.random.rand()-0.5)*0.005
        WeightersCentre[32:40, 38:39] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[32:40, 39:60] = (np.random.rand()-0.5)*0.005
        WeightersCentre[32:40, 60:61] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[32:40, 61:83] = (np.random.rand()-0.5)*0.005
        WeightersCentre[32:40, 83:84] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[32:40, 84:90] = (np.random.rand()-0.5)*0.005
        # WeightersCentre[32:40, 425:428] = (np.random.rand()-0.5)*0.005
        WeightersCentre[32:40, 428:429] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[32:40, 429:439] = (np.random.rand()-0.5)*0.005

        #r3
        # WeightersCentre[40:48, 0:11] = (np.random.rand()-0.5)*0.005
        WeightersCentre[40:48, 11:12] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[40:48, 12:34] = (np.random.rand()-0.5)*0.005
        WeightersCentre[40:48, 34:35] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[40:48, 35:56] = (np.random.rand()-0.5)*0.005
        WeightersCentre[40:48, 56:57] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[40:48, 57:79] = (np.random.rand()-0.5)*0.005
        WeightersCentre[40:48, 79:80] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[40:48, 80:90] = (np.random.rand()-0.5)*0.005
        # WeightersCentre[40:48, 425:432] = (np.random.rand()-0.5)*0.005
        WeightersCentre[40:48, 432:433] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[40:48, 433:439] = (np.random.rand()-0.5)*0.005


        #abdemon
        # WeightersCentre[56:64, 0:18] = (np.random.rand()-0.5)*0.005
        WeightersCentre[48:56, 5:8] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[56:64, 19:41] = (np.random.rand()-0.5)*0.005
        WeightersCentre[48:56, 28:31] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[56:64, 42:63] = (np.random.rand()-0.5)*0.005
        WeightersCentre[48:56, 50:53] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[56:64, 64:86] = (np.random.rand()-0.5)*0.005
        WeightersCentre[48:56, 73:76] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[56:64, 87:90] = (np.random.rand()-0.5)*0.005
        # WeightersCentre[56:64, 425:435] = (np.random.rand()-0.5)*0.005
        WeightersCentre[48:56, 422:425] = (np.random.rand()-0.5)*0.1
        # WeightersCentre[56:64, 436:439] = (np.random.rand()-0.5)*0.005



        ADSA=DSA.DynamicSynapseArray(NumberOfSynapses, Period=20000, tInPeriod=None, PeriodVar=0.1,\
                 Amp=0.2, WeightersCentre = WeightersCentre, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100, Amp2=0.1) #Amp=0.2
        
        NumberOfSynapses2 = np.array([14, 56+1])

        WeightersCentre2 = np.zeros(NumberOfSynapses2)

        WeightersCentre2[0:2, 0:8] = (np.random.rand()-0.5)*0.12
        WeightersCentre2[2:4, 8:16] = (np.random.rand()-0.5)*0.12
        WeightersCentre2[4:6, 16:24] = (np.random.rand()-0.5)*0.12
        WeightersCentre2[6:8, 24:32] = (np.random.rand()-0.5)*0.12
        WeightersCentre2[8:10, 32:40] = (np.random.rand()-0.5)*0.12
        WeightersCentre2[10:12, 40:48] = (np.random.rand()-0.5)*0.12
        WeightersCentre2[12:14, 48:56] = (np.random.rand()-0.5)*0.12

        # WeightersCentre2[:, 64:65] = 0.1

        ADSA2=DSA.DynamicSynapseArray(NumberOfSynapses2 , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                 Amp=0.2, WeightersCentre = WeightersCentre2, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100, Amp2=0.12) #Amp=0.2
        
        # CPG = CPGMerge.CPGCombiened()
        # CPG.InitRecording()


        ADSA_l1 = DSA.DynamicSynapseArray(np.array([1,18]) , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                 Amp=0.2, WeightersCentre = None, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)
        ADSA_r1 = DSA.DynamicSynapseArray(np.array([1,18]) , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                 Amp=0.2, WeightersCentre = None, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)
        ADSA_l2 = DSA.DynamicSynapseArray(np.array([1,17]) , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                 Amp=0.2, WeightersCentre = None, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)
        ADSA_r2 = DSA.DynamicSynapseArray(np.array([1,17]) , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                 Amp=0.2, WeightersCentre = None, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)
        ADSA_l3 = DSA.DynamicSynapseArray(np.array([1,16]) , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                 Amp=0.2, WeightersCentre = None, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)
        ADSA_r3 = DSA.DynamicSynapseArray(np.array([1,16]) , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                 Amp=0.2, WeightersCentre = None, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)
        ADSA_abdomen = DSA.DynamicSynapseArray(np.array([1,19]) , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                 Amp=0.2, WeightersCentre = None, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)
        
        NumberOfSynapses3 = np.array([num_actions, 7+22+1])
        InitWeights = 0.8
        WeightersCentre3 = np.zeros(NumberOfSynapses3)
        # WeightersCentre3[0:3,13:16] = -InitWeights/3
        WeightersCentre3[0:3, 6:7] = InitWeights/2
        WeightersCentre3[0:3, 12:15] = -InitWeights/3
        # WeightersCentre3[:, 53:54] = 0.02
        WeightersCentre3[3:6, 3:4] = InitWeights/2
        WeightersCentre3[3:6, 15:18] = -InitWeights/3
        # WeightersCentre3[3:6, 39:42] = -0.02

        WeightersCentre3[6:7, 5:6] = InitWeights/2
        WeightersCentre3[6:7, 18:19] = -InitWeights
        # WeightersCentre3[6:7, 42:43] = -0.02

        WeightersCentre3[7:10, 2:3] = InitWeights/2
        WeightersCentre3[7:10, 19:22] = -InitWeights/3
        # WeightersCentre3[7:10, 43:46] = -InitWeights

        WeightersCentre3[10:11, 4:5] = InitWeights/2
        WeightersCentre3[10:11, 22:23] = -InitWeights
        # WeightersCentre3[10:11, 46:47] = -InitWeights

        WeightersCentre3[11:13, 1:2] = InitWeights/2
        WeightersCentre3[11:13, 23:25] = -InitWeights/2
        # WeightersCentre3[11:13, 47:49] = -InitWeights


        WeightersCentre3[14:16, 0:1] = InitWeights/2
        WeightersCentre3[14:16, 26:28] = -InitWeights/2
        # WeightersCentre3[14:16, 50:52] = -InitWeights

        # WeightersCentre3[16:17, 52:53] = -InitWeights
        Amp3 = np.ones(NumberOfSynapses3) * 0.1
        ADSA3=DSA.DynamicSynapseArray(NumberOfSynapses3 , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                 Amp=0.15, WeightersCentre = WeightersCentre3, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)
        
        ADSA.InitRecording()
        ADSA2.InitRecording()
        ADSA_l1.InitRecording()
        ADSA_l2.InitRecording()
        ADSA_l3.InitRecording()
        ADSA_r1.InitRecording()
        ADSA_r2.InitRecording()
        ADSA_r3.InitRecording()
        ADSA_abdomen.InitRecording()
        ADSA3.InitRecording()

        CPG = FN.FHNN(NumberOfNeurons=7, scale=0.02)
        CPG.InitRecording()        
    #%%
#        NeuronSensitivity = np.ones(num_actions*2)*0.005
        NeuronSensitivity = np.ones(7)*0.5
        NeuronSensitivityUpdateRate = 0.00001
        NeuronSensitivity2 = np.ones(376)*0.5
        NeuronSensitivity2[22:45] = 0.2
        NeuronSensitivityUpdateRate2 = 0.00001
    #%%
        highscore = 0
        lastAngle=0
        WhenRender=-100
        WhenRenderStep = 10
        
        points = 0
        rewardAdaption=0
        rewardAdaptionRate=0.001
        nb_frames=N_episode*300
        recordingSample=1000
        ContinousMoving=0
        
        Trace={'episode':deque(),
               't':deque(),
               'action':deque(),
               'reward':deque(),
               'points':deque(),
               'rewardIncrease':deque(),
               'rewardAdaption':deque(),
                'info':deque(),
                'NeuronSensitivity':deque(),
                'NeuronSensitivity2':deque()}
        EpisodeTrace={'Number':deque(),
                      't':deque(),
                      'EpisodeReward':deque(),
                      'AverageEpisodeReward':deque(),
                      'Weights1':deque(),
                      'Weights2':deque(),
                      'Weights3':deque(),
                }
        step=0
        action=np.zeros(num_actions)
        # output=np.zeros(num_actions*2)
            
    for i_episode in range(N_episode): # run 20 episodes
        if i_episode == 100:
            # print(CPG.AFHNN_l1.Trace["Vn"])
            CPG.Plot(0)
            CPG.Plot(1)
            CPG.Plot(2)
            CPG.Plot(3)
            CPG.Plot(4)
            CPG.Plot(5)
            CPG.Plot(6)
        if human_stop :
            break            
        points = 0 # keep track of the reward each episode
        points2 = 0
        points1 = 0
        t=0
        PointsLast=0
        EpisodeReward=0
        print("i_episode: %d"%(i_episode))
        print("T: %d"%(T))
        while True: # run until episode is done
            step+=1
            Printing=step%100
            T+=dt
            t+=dt
            AllPossibleRender = human_render or ((not ForceNoRendering) and (HumanRewarding == 1 or AllRendering == 1 or Rendering == 1 or  i_episode<10 or i_episode>N_episode-50))
            if AllPossibleRender:
                env.render()
            if Info:    
                env.render()
            Weights=ADSA.Weighters[:, :]
            observation = observation * NeuronSensitivity2
            # print("observation: ", observation)
            # print("NeuronSensitivity2: ", NeuronSensitivity2)

            NeuronSensitivity2 += ((1 - np.abs(0-observation))*NeuronSensitivityUpdateRate2*dt)*(2-np.log10(np.abs(NeuronSensitivity2)))*(np.log10(np.abs(NeuronSensitivity2))-(-2))/4
            # # Range_Adapter.recording()
            ob = copy.deepcopy(observation)
            Neuron1Out=np.tanh(np.dot(Weights, np.tanh(MergeInputs(ObPreProcess(ob),action))))
            # print('Neuron1Out:', Neuron1Out)
            print("obrightknee: ", observation[11])
            print("obleftknee: ", observation[15])
            Weights2=ADSA2.Weighters[:, :]

            Neuron2Out = np.tanh(np.dot(Weights2, np.hstack((Neuron1Out,1))))
            # print(CPGInput)   
            CPGl1Input = np.dot(ADSA_l1.Weighters, np.hstack((CPG.Vp[1], CPG.Vp[2], CPG.Vp[6], Neuron2Out,1)))
            CPGr1Input = np.dot(ADSA_r1.Weighters, np.hstack((CPG.Vp[0], CPG.Vp[3], CPG.Vp[6], Neuron2Out,1)))
            CPGl2Input = np.dot(ADSA_l2.Weighters, np.hstack((CPG.Vp[0], CPG.Vp[3], Neuron2Out,1)))
            CPGr2Input = np.dot(ADSA_r2.Weighters, np.hstack((CPG.Vp[1], CPG.Vp[5], Neuron2Out,1)))
            CPGl3Input = np.dot(ADSA_l3.Weighters, np.hstack((CPG.Vp[2], Neuron2Out,1)))
            CPGr3Input = np.dot(ADSA_r3.Weighters, np.hstack((CPG.Vp[3], Neuron2Out,1)))
            CPGabdomenInput = np.dot(ADSA_abdomen.Weighters, np.hstack((CPG.Vp[0],CPG.Vp[1], CPG.Vp[2], CPG.Vp[3], Neuron2Out, 1)))


            CPGInput  = np.squeeze(np.array([CPGl1Input, CPGr1Input, CPGl2Input, CPGr2Input, CPGl3Input, CPGr3Input, CPGabdomenInput])) * NeuronSensitivity
            # print(np.array([CPGl1Input, CPGr1Input, CPGl2Input, CPGr2Input, CPGl3Input, CPGr3Input, CPGleftInput, CPGrightInput]))
            # print(NeuronSensitivity)
            # print(CPGInput)
            if Adaption == "Nonlinear":
                NeuronSensitivity += ((0.3-np.abs(0-CPGInput))*NeuronSensitivityUpdateRate*dt)*(2-np.log10(np.abs(NeuronSensitivity)))*(np.log10(np.abs(NeuronSensitivity))-(-2))/4
            elif Adaption == "Linear":
                NeuronSensitivity += ((0.3-np.abs(0-CPGInput))*NeuronSensitivityUpdateRate*dt) #*(2-np.log10(NeuronSensitivity))*(np.log10(NeuronSensitivity)-(-2))/4
            I = np.array(CPGInput)
            # print("NeuronSensitivity: ", NeuronSensitivity)
            # print("CPGInput: ", I)

            
            # print(ADSA_l1.Weighters)
            # print(ADSA_l2.Weighters)
            # Range_Adapter.update()
            # print(I)

            CPGOutput = CPG.StepDynamics(dt, I)
            # print(CPGOutput)
            print("CPGOutput: ", CPGOutput)
            # print(ADSA3.Weighters)
            action = np.dot(ADSA3.Weighters , np.hstack((CPGOutput, observation[0:22],1)))#/10*dt #+NeuronSensitivity2*dt 
            print("action:", action)           
            observation, reward, done, _, info = env.step(action)
            print(reward)
            reward = reward * 0.1
            # print(reward)
            EpisodeReward += reward
            if Printing==0 or AllPossibleRender:
                print("i_episode: %d"%(i_episode))        
                # print('NeuronSensitivity:', NeuronSensitivity)
                print('reward: %.6f, EpisodeReward=%.6f, 100ERSum = %.6f'%(reward,EpisodeReward, ERSum))

                
            if HumanRewarding == 1:  
                print('Human reward: %.6f'%(human_reward))
                ModulatorAmount = human_reward
            else:
                ModulatorAmount=reward
                if not RepulsiveLearning:
                    if ModulatorAmount<0:
                        ModulatorAmount=0 

            ADSA.StepSynapseDynamics(dt, T, reward)    
            ADSA2.StepSynapseDynamics(dt, T, reward)    
            ADSA3.StepSynapseDynamics(dt, T, reward)
            ADSA_l1.StepSynapseDynamics(dt, T, reward)
            ADSA_l2.StepSynapseDynamics(dt, T, reward)
            ADSA_l3.StepSynapseDynamics(dt, T, reward)
            ADSA_r1.StepSynapseDynamics(dt, T, reward)
            ADSA_r2.StepSynapseDynamics(dt, T, reward)
            ADSA_r3.StepSynapseDynamics(dt, T, reward)
            ADSA_abdomen.StepSynapseDynamics(dt, T, reward)
            # ADSA.Recording()
            # ADSA2.Recording()
            # ADSA3.Recording()
            # print(ADSA.Amp[0,0])
            # print(ADSA2.Amp[0,0])
            # print(ADSA3.Amp[0,0])
            # print("rightknee_CPG_weight", ADSA3.Weighters[6:7, 5:6])
            # print("rightknee_ob_weight", ADSA3.Weighters[6:7, 19:20])

            # print("leftknee_CPG_weight", ADSA3.Weighters[10:11, 4:5])
            # print("leftknee_ob_weight", ADSA3.Weighters[10:11, 23:24])

            rewardLast=reward
            if StepRecording:
                Trace['t'].append(T)
                Trace['episode'].append(i_episode )
                Trace['action'].append(copy.deepcopy(action))
                Trace['reward'].append(reward)
                Trace['points'].append(points)
                Trace['info'].append(points)
                Trace['NeuronSensitivity'].append(copy.deepcopy(NeuronSensitivity))
                Trace['NeuronSensitivity2'].append(copy.deepcopy(NeuronSensitivity2))
                CPG.Recording()
                # print("Vn:  ", CPG.AFHNN_l1.Trace["I"])
            CPG.Update()

            if done:
                if  i_episode>100:
                    ERSum += EpisodeReward
                    ERSum -= EpisodeTrace['EpisodeReward'][-100]
                    AERSum = ERSum/100
                    if ERSum>=500*100:
                        if Solved == False:
                            SolveEpisode = i_episode
                        Solved = True
                else:
                    ERSum += EpisodeReward
                    AERSum = ERSum/i_episode
                    
                print("i_episode: %d"%(i_episode))        
                # print('action:', action)
                # print('NeuronSensitivity:', NeuronSensitivity)
                print('reward: %.6f, EpisodeReward=%.6f, 100ERSum = %.6f'%(reward,EpisodeReward, ERSum))
                

                EpisodeTrace['Number'].append(i_episode)
                EpisodeTrace['t'].append(t)
                EpisodeTrace['EpisodeReward'].append(EpisodeReward)
                EpisodeTrace['AverageEpisodeReward'].append(AERSum)
                EpisodeTrace['Weights1'].append(ADSA.Weighters)
                EpisodeTrace['Weights2'].append(ADSA2.Weighters)
                EpisodeTrace['Weights3'].append(ADSA3.Weighters)
                PointsLast=points
                # ADSA.Recording() 

                ContinousMoving = 0
                # env.render()
                if Rendering ==1:
                    Rendering=0
#                    WhenRender+=WhenRenderStep
                if  EpisodeReward > WhenRender: # or PointsLast>30:
                    Rendering = 1
                    WhenRender=EpisodeReward
                env.reset()
                points=0
                ModulatorAmount=0
                EpisodeReward=0
                rewardAdaption=0
                t=0
                break

#                if points > highscore: # record high score
#                    highscore = points
#                    Rendering = 1
#                    break
#    for key in Trace:
#        Trace[key]=np.delete(Trace[key], np.s_[step/recordingSample::], axis=0)    

    env.close()

    dataPath = ChooseResultPath(experiment, TimeOfRecording) +'Trace/'
    if not os.path.exists(dataPath):
        os.makedirs(dataPath) 
    with open(dataPath+'EpisodeTrace.pkl', 'ab') as fTraces:
        dill.dump(EpisodeTrace, fTraces,protocol=dill.HIGHEST_PROTOCOL)
    with open(dataPath+'StepTrace.pkl', 'ab') as fTraces:
        dill.dump(Trace, fTraces,protocol=dill.HIGHEST_PROTOCOL)

    with open(dataPath+'Weights.pkl', 'ab') as fTraces:
        WeightsDict={'Weights1':copy.deepcopy(ADSA3.Weighters),
                     'Weights2':copy.deepcopy(ADSA2.Weighters),
                     'Weights3':copy.deepcopy(ADSA3.Weighters),
                     'ADSA1':copy.deepcopy(ADSA),
                     'ADSA2':copy.deepcopy(ADSA2),
                     'ADSA3':copy.deepcopy(ADSA3),
                     }
        dill.dump(WeightsDict, fTraces,protocol=dill.HIGHEST_PROTOCOL)
    plotpath= ChooseResultPath(experiment, TimeOfRecording) +'plot/'
    if not os.path.exists(plotpath):
        os.makedirs(plotpath)  
    FigureDict=dict()
#     if StepRecording:
#         FigureDict['action'] = plt.figure()  
#         labels = [str(i) for i in range(17)]
#         figure11lines = plt.plot(Trace['t'], Trace['action'])
#         plt.legend(figure11lines, labels)
#         plt.xlabel('Time (s)')
#         plt.title('Action')
        
#         FigureDict['Reward'] = plt.figure()  
#         figure13lines = plt.plot(Trace['t'], Trace['reward'])
# #        plt.legend(figure11lines, labels)
#         plt.xlabel('Time (s)')
#         plt.title('Reward')  

#         FigureDict['RewardZone in'] = plt.figure()  
#         figure13lines = plt.plot(Trace['t'], Trace['reward'])
# #        plt.legend(figure11lines, labels)
#         plt.ylim(-1.5,1.5)
#         plt.xlabel('Time (s)')
#         plt.title('Reward')  
        
        # FigureDict['NeuronSensitivity'] = plt.figure()  
        # figure14lines = plt.plot(Trace['t'], Trace['NeuronSensitivity'])
        # plt.legend(figure14lines, labels)
        # plt.xlabel('Time (s)')
        # plt.title('Neuron Sensitivity')  
        
    FigureDict['EpisodeReward'] = plt.figure()  
    figure12lines1, = plt.plot(EpisodeTrace['EpisodeReward'])
    figure12lines2, = plt.plot(EpisodeTrace['AverageEpisodeReward'])
    plt.legend([figure12lines1, figure12lines2], ['Episode Reward', 'AverageEpisodeReward'], loc=2)
    plt.xlabel('Episode')
    plt.title('Episode Reward')   
    # plt.savefig(plotpath+"Result.png")

#if savePlots == True:
    if not os.path.exists(plotpath):
        os.makedirs(plotpath) 
    pp = PdfPages(plotpath+"Emviroment"+TimeOfRecording+'.pdf')
    for key in FigureDict:
        FigureDict[key].savefig(pp, format='pdf')
    pp.close()
    
    plt.show()    


#dataPath = ChooseResultPath(experiment, TimeOfRecording) +'Trace/'
#if not os.path.exists(dataPath):
#    os.makedirs(dataPath) 
#with open(dataPath+'EpisodeTrace.pkl', 'ab') as fTraces:
#    dill.dump(EpisodeTrace, fTraces,protocol=dill.HIGHEST_PROTOCOL)
#    
#with open(dataPath+'StepTrace.pkl', 'ab') as fTraces:
#    dill.dump(Trace, fTraces,protocol=dill.HIGHEST_PROTOCOL)
#
#
#with open(dataPath+'Weights.pkl', 'ab') as fTraces:
#    WeightsDict={'Weights1':copy.deepcopy(Weights),
#                 'Weights2':copy.deepcopy(Weights2),
#                 'Weights3':copy.deepcopy(ADSA3.Weighters),
#                 'ADSA1':copy.deepcopy(ADSA),
#                 'ADSA2':copy.deepcopy(ADSA2),
#                 'ADSA3':copy.deepcopy(ADSA3),
#                 }
#    dill.dump(WeightsDict, fTraces,protocol=dill.HIGHEST_PROTOCOL)
#plotpath= ChooseResultPath(experiment, TimeOfRecording) +'plot/'
#if not os.path.exists(plotpath):
#    os.makedirs(plotpath)
#    
#    
#FigureDict=dict()
#if recording:
#    FigureDict,ax = ADSA.plot(path=plotpath, savePlots=False, linewidth= 0.2, DownSampleRate=10, NameStr=TimeOfRecording) #path=
#    FNFigure = AFHNN.Plot(0,DownSampleRate=10)
#    IArray=np.array(AFHNN.Trace['I'])
#    FigureDict['FNPhasePortait'], axFNPhasePortait = AFHNN.PlotPhasePortrait([IArray[0][0],IArray[0][-1]], xlim=[-1.5,1.5], ylim=[-1+IArray[0].min(),1+IArray[0].max()], DownSampleRate=10)
#    FigureDict['action'] = plt.figure()  
#    labels = [str(i) for i in range(4)]
#    figure11lines = plt.plot(Trace['t'], Trace['action'])
#    plt.legend(figure11lines, labels)
#    plt.xlabel('Time (s)')
#    plt.title('Action')
#    
#    FigureDict['Reward'] = plt.figure()  
#    figure13lines = plt.plot(Trace['t'], Trace['reward'])
#    plt.legend(figure11lines, labels)
#    plt.xlabel('Time (s)')
#    plt.title('Reward')  
#
#FigureDict['RewardZone in'] = plt.figure()  
#figure13lines = plt.plot(Trace['t'], Trace['reward'])
#plt.legend(figure11lines, labels)
#plt.ylim(-1.5,1.5)
#plt.xlabel('Time (s)')
#plt.title('Reward')  
#
#FigureDict['NeuronSensitivity'] = plt.figure()  
#figure14lines = plt.plot(Trace['t'], Trace['NeuronSensitivity'])
#plt.legend(figure14lines, labels)
#plt.xlabel('Time (s)')
#plt.title('Neuron Sensitivity')  
#    
#FigureDict['EpisodeReward'] = plt.figure()  
#figure12lines = plt.plot(EpisodeTrace['EpisodeReward'])
#plt.legend(figure12lines, labels)
#plt.xlabel('Episode')
#plt.title('Episode Reward')   
#
#
##if savePlots == True:
#if not os.path.exists(plotpath):
#    os.makedirs(plotpath) 
#pp = PdfPages(plotpath+"Emviroment"+TimeOfRecording+'.pdf')
#for key in FigureDict:
#    FigureDict[key].savefig(pp, format='pdf')
#pp.close()
#
#plt.show()      