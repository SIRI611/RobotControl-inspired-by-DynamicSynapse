#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
#import DynamicSynapseArray2D as DSA
#import DynamicSynapseArray2DLimitedDiffuse as DSA
import DynamicSynapseArray2DRandomSin as DSA
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
    #TODO factors adjustment

    # factors=[1/2*np.pi,5,2,2,1,
    #          0.5,1,0.3,1,1,
    #          0.5,1,0.25,1,2,
    #          1,1,1,1,1,
    #          1,1,1,1]
    factors = np.ones(45)
    NormaledOb=ob*factors
    # NormaledOb=ob
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
    Info =0
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
        env = gym.make(experiment, render_mode="human")
        # env = gym.wrappers.Monitor(env,ChooseRecordingPath(experiment, TimeOfRecording)) #'/home/chitianqilin/recording/OpenAIGym/cartpole-experiment-1/'+TimeOfRecording+'/')#'/media/archive2T/chitianqilin/recording/OpenAIGym/cartpole-experiment-1/')
        # steps= env.spec.timestep_limit #steps per episode  
#        if HumanRewarding == 1:
            
        env.render()
        # env.unwrapped.viewer.window.on_key_press = key_press
        # env.unwrapped.viewer.window.on_key_release = key_release

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]    
        observation = env.reset()[0]
        NumberOfSynapses = np.array([64, (num_states-321)*2+(321)+1+num_actions])
#      Weighters= np.ones((NumberOfNeuron,NumberOfSynapses))*0.2+ 0.1 * np.random.rand(NumberOfNeuron,NumberOfSynapses)#np.random.rand(NumberOfNeuron,NumberOfSynapses) #0.5*np.ones(NumberOfSynapses) +0.001*np.random.rand(NumberOfSynapses)  #

        dt = 33
        N_episode = 100000 #550000#1000# 300000
        T = 0
        
#        Weighters= np.ones(NumberOfSynapses)*0+ 0.4* (np.random.rand(*NumberOfSynapses)-0.5)
        # WeightersCentre = np.ones(NumberOfSynapses)*0.1+ 0.2* (np.random.rand(*NumberOfSynapses)-0.5)
        WeightersCentre= np.ones(NumberOfSynapses)*0+ 0.4* (np.random.rand(*NumberOfSynapses)-0.5)
        WeightersCentre[]
        ADSA=DSA.DynamicSynapseArray(NumberOfSynapses , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                 Amp=0.2, WeightersCentre = None, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100) #Amp=0.2

        NumberOfSynapses2 = np.array([8, num_actions*2+1])
        ADSA2=DSA.DynamicSynapseArray(NumberOfSynapses2 , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                 Amp=0.2, WeightersCentre = None, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100) #Amp=0.2

        AFHNN = FN.FHNN(8,scale = 0.02)
        AFHNN.InitRecording()        
        
        NumberOfSynapses3 = np.array([num_actions, 16+45+1])
        # original
        # WeightersCentre=np.array([[0,0,-0.5,0,0,0,0,0,-0.2,-0.2,0,0,0,0,0,0,0,0,0],
        #                            [0.5,0,0,0,0,0,0,0,0,0,-0.2,-0.2,0,0,0,0,0,0,0],
        #                            [0,0,0,-0.5,0,0,0,0,0,0,0,0,0,-0.2,-0.2,0,0,0,0],
        #                            [0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.2,-0.2,0,0]])
#        WeightersCentre=np.array([[-0.5,0,0,0,0,0,0,0,-0.2,-0.2,0,0,0,0,0,0,0,0,0],
#                                   [0,0,0.5,0,0,0,0,0,0,0,-0.2,-0.2,0,0,0,0,0,0,0],
#                                   [0,-0.5,0,0,0,0,0,0,0,0,0,0,0,-0.2,-0.2,0,0,0,0],
#                                   [0,0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,-0.2,-0.2,0,0]])
        ADSA3=DSA.DynamicSynapseArray(NumberOfSynapses3 , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                 Amp=0.2, WeightersCentre = WeightersCentre, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)
        ADSA.InitRecording()
        ADSA2.InitRecording()
        ADSA3.InitRecording()        
    #%%
#        NeuronSensitivity = np.ones(num_actions*2)*0.005
        NeuronSensitivity = np.ones(num_actions*2)*0.5
        NeuronSensitivityUpdateRate = 0.000001
        NeuronSensitivity2 = np.ones(num_actions)*0.005
        NeuronSensitivityUpdateRate2 = 0.000001
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
                'NeuronSensitivity':deque()}
        EpisodeTrace={'Number':deque(),
                      't':deque(),
                      'EpisodeReward':deque(),
                      'AverageEpisodeReward':deque(),
                      'Weights1':deque(),
                      'Weights2':deque(),
                      'Weights3':deque()
                }
        step=0
        action=np.zeros(num_actions)
        output=np.zeros(num_actions*2)

    if LoadData:
        Trainedpath=ChooseResultPath('BipedalWalker-v2', '2018-02-23_04-11-02') +'Trace/'
        with open(Trainedpath+'Weights.pkl', 'rb') as in_strm:
            WeightsData = dill.load(in_strm)
            ADSA=WeightsData['ADSA1']
            ADSA2=WeightsData['ADSA2'] 
            ADSA3=WeightsData['ADSA3']  
            for key in ADSA.Trace:
                ADSA.Trace[key].clear()
            for key in ADSA2.Trace:
                ADSA2.Trace[key].clear()                
            for key in ADSA3.Trace:
                ADSA3.Trace[key].clear()    
            
    for i_episode in range(N_episode): # run 20 episodes
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

            Neuron1Out=relu(np.tanh(np.dot(Weights, np.tanh(MergeInputs(ObPreProcess(observation),action)))*NeuronSensitivity))
            if Adaption == "Nonlinear":
                NeuronSensitivity += ((0.3-np.abs(0-Neuron1Out))*NeuronSensitivityUpdateRate*dt)*(2-np.log10(NeuronSensitivity))*(np.log10(NeuronSensitivity)-(-2))/4
            elif Adaption == "Linear":
                NeuronSensitivity += ((0.3-np.abs(0-Neuron1Out))*NeuronSensitivityUpdateRate*dt) #*(2-np.log10(NeuronSensitivity))*(np.log10(NeuronSensitivity)-(-2))/4
            # print('Neuron1Out:', Neuron1Out)
            if Info:
                print('Neuron1Out')
                print(Neuron1Out)
            Weights2=ADSA2.Weighters[:, :]
            CPGInput=np.tanh(np.dot(Weights2, np.hstack((Neuron1Out,1))))
            CPGOutput = np.array((AFHNN.Vn, AFHNN.Wn)).ravel()

            action = np.tanh(np.dot(ADSA3.Weighters,np.hstack((CPGOutput, observation[0:45],1))))#/10*dt #+NeuronSensitivity2*dt            
            observation, reward, done, _, info = env.step(action)
            EpisodeReward += reward
            if Printing==0 or AllPossibleRender:
                print("i_episode: %d"%(i_episode))        
                print('action:', action)
                print('NeuronSensitivity:', NeuronSensitivity)
                print('reward: %.6f, EpisodeReward=%.6f, 100ERSum = %.6f'%(reward,EpisodeReward, ERSum))
                print('ADSA.Amp:' + str(ADSA.Amp[0,0]))
                print('info:'+str(info))
                print('RepulsiveLearning = %r, Adaption = %s' %(args.Repulsive, args.Adaption))
                if Solved == True:
                    print('Solved = True, SolveEpisode = %d' %(SolveEpisode))

            if Info:
                print(info)
                print('Weights')
                print(Weights)
                print('Observation')
                print(observation)
#            points += reward*np.log(t+1)
#            ModulatorAmount = ((np.log(reward+1) if reward>0 else 0)+ np.log((points-PointsLast)/100 if points-PointsLast>0 else 0))
#            if ModulatorAmount <= 0:
#                  ModulatorAmount = 0
#            else:
#                print('Modulator AmountRepulsiveLearning: %.9f'%(ModulatorAmount))
            if HumanRewarding == 1:  
                print('Human reward: %.6f'%(human_reward))
                ModulatorAmount = human_reward
            else:
                ModulatorAmount=reward
                if not RepulsiveLearning:
                    if ModulatorAmount<0:
                        ModulatorAmount=0 
#           if Printing==0:
#                print('rewardAfterAdapt=%.5f, points=%.5f,  observation[2]=%.5f'%(rewardAfterAdapt,points,  observation[2])    )
#                print('Modulator Amount: %.9f'%(ModulatorAmount) + 'ADSA.Amp:' + str(ADSA.Amp[0,0]))
            ADSA.StepSynapseDynamics(dt, T, reward)    
            ADSA2.StepSynapseDynamics(dt, T, reward)    
            ADSA3.StepSynapseDynamics(dt, T, reward)  
            AFHNN.StepDynamics(dt, CPGInput)
            AFHNN.Update()
            #print(ModulatorAmount)              
            #print(AmplitudeLast)
            #print(Amplitude)
            rewardLast=reward
            if StepRecording and step%recordingSample==0:
                Trace['t'].append(T)
                Trace['episode'].append(i_episode )
                Trace['action'].append(copy.deepcopy(action))
                Trace['reward'].append(reward)
                Trace['points'].append(points)
                Trace['info'].append(points)
                Trace['NeuronSensitivity'].append(copy.deepcopy(NeuronSensitivity))
                ADSA.Recording()
                ADSA2.Recording()
                ADSA.Recording()
                AFHNN.Recording()
                if HumanRewarding == 1: 
                    pass


            if done:
                if  i_episode>100:
                    ERSum += EpisodeReward
                    ERSum -= EpisodeTrace['EpisodeReward'][-100]
                    AERSum = ERSum/100
                    if ERSum>=300*100:
                        if Solved == False:
                            SolveEpisode = i_episode
                        Solved = True
                else:
                    ERSum += EpisodeReward
                    AERSum = ERSum/i_episode
                    
                print("i_episode: %d"%(i_episode))        
                print('action:', action)
                print('NeuronSensitivity:', NeuronSensitivity)
                print('reward: %.6f, EpisodeReward=%.6f, 100ERSum = %.6f'%(reward,EpisodeReward, ERSum))
                
                print('Weights'+str(Weights))
                print('Weights2'+str(Weights2))
                print('Weights3'+str(ADSA3.Weighters))
                print('info'+str(info))

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
                env.render()
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

#%%


    dataPath = ChooseResultPath(experiment, TimeOfRecording) +'Trace/'
    if not os.path.exists(dataPath):
        os.makedirs(dataPath) 
    with open(dataPath+'EpisodeTrace.pkl', 'ab') as fTraces:
        dill.dump(EpisodeTrace, fTraces,protocol=dill.HIGHEST_PROTOCOL)
    with open(dataPath+'StepTrace.pkl', 'ab') as fTraces:
        dill.dump(Trace, fTraces,protocol=dill.HIGHEST_PROTOCOL)

    with open(dataPath+'Weights.pkl', 'ab') as fTraces:
        WeightsDict={'Weights1':copy.deepcopy(Weights),
                     'Weights2':copy.deepcopy(Weights2),
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
    if StepRecording:
        FigureDict,ax = ADSA.plot(path=plotpath, savePlots=False, linewidth= 0.2, DownSampleRate=10, NameStr=TimeOfRecording) #path=
        FNFigure = AFHNN.Plot(0,DownSampleRate=10)
        IArray=np.array(AFHNN.Trace['I'])
        FigureDict['FNPhasePortait'], axFNPhasePortait = AFHNN.PlotPhasePortrait([IArray[0][0],IArray[0][-1]], xlim=[-1.5,1.5], ylim=[-1+IArray[0].min(),1+IArray[0].max()], DownSampleRate=10)
        FigureDict['action'] = plt.figure()  
        labels = [str(i) for i in range(4)]
        figure11lines = plt.plot(Trace['t'], Trace['action'])
        plt.legend(figure11lines, labels)
        plt.xlabel('Time (s)')
        plt.title('Action')
        
        FigureDict['Reward'] = plt.figure()  
        figure13lines = plt.plot(Trace['t'], Trace['reward'])
#        plt.legend(figure11lines, labels)
        plt.xlabel('Time (s)')
        plt.title('Reward')  

        FigureDict['RewardZone in'] = plt.figure()  
        figure13lines = plt.plot(Trace['t'], Trace['reward'])
#        plt.legend(figure11lines, labels)
        plt.ylim(-1.5,1.5)
        plt.xlabel('Time (s)')
        plt.title('Reward')  
        
        FigureDict['NeuronSensitivity'] = plt.figure()  
        figure14lines = plt.plot(Trace['t'], Trace['NeuronSensitivity'])
        plt.legend(figure14lines, labels)
        plt.xlabel('Time (s)')
        plt.title('Neuron Sensitivity')  
        
    FigureDict['EpisodeReward'] = plt.figure()  
    figure12lines1, = plt.plot(EpisodeTrace['EpisodeReward'])
    figure12lines2, = plt.plot(EpisodeTrace['AverageEpisodeReward'])
    figure12lines3, = plt.axhline(y=300, color='black', linestyle='-', label='Threshold', linewidth=0.5)  
    plt.legend([figure12lines1, figure12lines2], ['Episode Reward', 'AverageEpisodeReward'], loc=2)
    plt.xlabel('Episode')
    plt.title('Episode Reward')   
    plt.savefig()

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