import numpy as np
import FitzHughNagumo as FHN
import DynamicSynapseArray2DRandomSin as DSA
class CPGCombiened():
    def __init__(self, t=0):
        self.AFHNN_l1 = FHN.FHNN(1, scale=0.02)
        self.AFHNN_r1 = FHN.FHNN(1, scale=0.02)
        self.AFHNN_l2 = FHN.FHNN(1, scale=0.02)
        self.AFHNN_r2 = FHN.FHNN(1, scale=0.02)
        self.AFHNN_l3 = FHN.FHNN(1, scale=0.02)
        self.AFHNN_r3 = FHN.FHNN(1, scale=0.02)
        self.AFHNN_leftarm = FHN.FHNN(1, scale=0.02)
        self.AFHNN_rightarm = FHN.FHNN(1, scale=0.02)
        self.t=t

    def Output(self):
        return np.array([self.AFHNN_l1.Vn, \
                         self.AFHNN_r1.Vn, \
                         self.AFHNN_l2.Vn, \
                         self.AFHNN_r2.Vn, \
                         self.AFHNN_l3.Vn, \
                         self.AFHNN_r3.Vn, \
                         self.AFHNN_leftarm.Vn, \
                         self.AFHNN_rightarm.Vn] ).ravel()

    def StepDynamics(self, dt, I):
        # print(I[0])
        self.Out_l1 = self.AFHNN_l1.StepDynamics(dt, np.array(I[0]))
        self.Out_r1 = self.AFHNN_r1.StepDynamics(dt, np.array(I[1]))
        self.Out_l2 = self.AFHNN_l2.StepDynamics(dt, np.array(I[2]))
        self.Out_r2 = self.AFHNN_r2.StepDynamics(dt, np.array(I[3]))
        self.Out_l3 = self.AFHNN_l3.StepDynamics(dt, np.array(I[4]))
        self.Out_r3 = self.AFHNN_r3.StepDynamics(dt, np.array(I[5]))
        self.Out_leftarm = self.AFHNN_leftarm.StepDynamics(dt, np.array(I[6]))
        self.Out_rightarm = self.AFHNN_rightarm.StepDynamics(dt, np.array(I[7]))
        self.Out = np.array([self.Out_l1, self.Out_r1, self.Out_l2, self.Out_r2, self.Out_l3, self.Out_r3, self.Out_leftarm, self.Out_rightarm]).ravel()
        return self.Out
        
    def Update(self):

        self.AFHNN_l1.Update()
        self.AFHNN_l2.Update()
        self.AFHNN_l3.Update()
        self.AFHNN_r1.Update()
        self.AFHNN_r2.Update()
        self.AFHNN_r3.Update()
        self.AFHNN_leftarm.Update()
        self.AFHNN_rightarm.Update()

    def Plot(self):
        self.AFHNN_l1.Plot(0)
        self.AFHNN_l2.Plot(0)
        self.AFHNN_l3.Plot(0)
        self.AFHNN_leftarm.Plot(0)
        self.AFHNN_r1.Plot(0)
        self.AFHNN_r2.Plot(0)
        self.AFHNN_r3.Plot(0)
        self.AFHNN_rightarm.Plot(0)
    def InitRecording(self):
        self.AFHNN_l1.InitRecording()
        self.AFHNN_r1.InitRecording()
        self.AFHNN_l2.InitRecording()
        self.AFHNN_r2.InitRecording()
        self.AFHNN_l3.InitRecording()
        self.AFHNN_r3.InitRecording()
        self.AFHNN_leftarm.InitRecording()
        self.AFHNN_rightarm.InitRecording()

    def Recording(self):
        self.AFHNN_l1.Recording()
        self.AFHNN_r1.Recording()
        self.AFHNN_l2.Recording()
        self.AFHNN_r2.Recording()
        self.AFHNN_l3.Recording()
        self.AFHNN_r3.Recording() 
        self.AFHNN_leftarm.Recording()
        self.AFHNN_rightarm.Recording()       


        


        
        
