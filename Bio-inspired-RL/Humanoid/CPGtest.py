import DynamicSynapseArray2DRandomSin as DSA
import CPGMerge 
import numpy as np

if __name__ == "__main__":
    T = 0
    dt = 3
    ADSA_l1 = DSA.DynamicSynapseArray(np.array([1,3]) , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                Amp=0.2, WeightersCentre = np.ones(np.array([1,3])) * 0.2, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)
    
    ADSA_r1 = DSA.DynamicSynapseArray(np.array([1,3]) , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                Amp=0.2, WeightersCentre = np.ones(np.array([1,3])) * 0.2, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)
   
    ADSA_l2 = DSA.DynamicSynapseArray(np.array([1,2]) , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                Amp=0.2, WeightersCentre = np.ones(np.array([1,2])) * 0.2, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)

    ADSA_r2 = DSA.DynamicSynapseArray(np.array([1,2]) , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                Amp=0.2, WeightersCentre = np.ones(np.array([1,2])) * 0.2, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)
 
    ADSA_l3 = DSA.DynamicSynapseArray(np.array([1,2]) , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                Amp=0.2, WeightersCentre = np.ones(np.array([1,2])) * 0.2, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)

    ADSA_r3 = DSA.DynamicSynapseArray(np.array([1,2]) , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                Amp=0.2, WeightersCentre = np.ones(np.array([1,2])) * 0.2, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)

    ADSA_leftarm = DSA.DynamicSynapseArray(np.array([1,1]) , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                Amp=0.2, WeightersCentre = np.ones(np.array([1,1])) * 0.2, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)

    ADSA_rightarm = DSA.DynamicSynapseArray(np.array([1,1]) , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                Amp=0.2, WeightersCentre = np.ones(np.array([1,1])) * 0.2, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100)

    
    while True:
        T += dt
        CPG = CPGMerge.CPGCombiened()

        CPGl1Input = np.tanh(np.dot(ADSA_l1.Weighters, np.hstack((CPG.AFHNN_l2.Vn, CPG.AFHNN_r1.Vn, CPG.AFHNN_leftarm.Vn))))
        CPGr1Input = np.tanh(np.dot(ADSA_r1.Weighters, np.hstack((CPG.AFHNN_l1.Vn, CPG.AFHNN_r2.Vn, CPG.AFHNN_rightarm.Vn)))) 
        CPGl2Input = np.tanh(np.dot(ADSA_l2.Weighters, np.hstack((CPG.AFHNN_l1.Vn, CPG.AFHNN_r2.Vn))))
        CPGr2Input = np.tanh(np.dot(ADSA_r2.Weighters, np.hstack((CPG.AFHNN_l2.Vn, CPG.AFHNN_r1.Vn))))
        CPGl3Input = np.tanh(np.dot(ADSA_l3.Weighters, np.hstack((CPG.AFHNN_l2.Vn, CPG.AFHNN_r3.Vn))))
        CPGr3Input = np.tanh(np.dot(ADSA_r3.Weighters, np.hstack((CPG.AFHNN_l3.Vn, CPG.AFHNN_r2.Vn))))
        CPGleftInput = np.tanh(np.dot(ADSA_leftarm.Weighters, np.hstack((CPG.AFHNN_l1.Vn))))
        CPGrightInput = np.tanh(np.dot(ADSA_rightarm.Weighters, np.hstack((CPG.AFHNN_r1.Vn))))

        I = np.array([CPGl1Input, CPGr1Input, CPGl2Input, CPGr2Input, CPGl3Input, CPGr3Input, CPGleftInput, CPGrightInput])
        print(I)    

        CPGOutput = CPG.StepDynamics(dt, I)
        CPG.Update()