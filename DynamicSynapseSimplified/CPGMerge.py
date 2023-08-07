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
        self.t=t

    def StepDynamic(self, dt, I):

        self.Out_l1 = self.AFHNN_l1.StepDynamics(dt, I[0])
        self.Out_r1 = self.AFHNN_r1.StepDynamics(dt, I[1])
        self.Out_l2 = self.AFHNN_l2.StepDynamics(dt, I[2])
        self.Out_r2 = self.AFHNN_r2.StepDynamics(dt, I[3])
        self.Out_l3 = self.AFHNN_l3.StepDynamics(dt, I[4])
        self.Out_r3 = self.AFHNN_r3.StepDynamics(dt, I[5])
        self.Out = np.array([self.Out_l1, self.Out_r1, self.Out_l2, self.Out_r2, self.Out_l3, self.Out_l3])
        return self.Out
        
    def Update(self):

        self.AFHNN_l1.Update()
        self.AFHNN_l2.Update()
        self.AFHNN_l3.Update()
        self.AFHNN_r1.Update()
        self.AFHNN_r2.Update()
        self.AFHNN_r3.Update()

    def Plot(self):
        return 0
    
    def InitRecording(self):
        self.AFHNN_l1.InitRecording()
        self.AFHNN_r1.InitRecording()
        self.AFHNN_l2.InitRecording()
        self.AFHNN_r2.InitRecording()
        self.AFHNN_l3.InitRecording()
        self.AFHNN_r3.InitRecording()

    def Recording(self):
        self.AFHNN_l1.Recording()
        self.AFHNN_r1.Recording()
        self.AFHNN_l2.Recording()
        self.AFHNN_r2.Recording()
        self.AFHNN_l3.Recording()
        self.AFHNN_r3.Recording()        


        


        
        
