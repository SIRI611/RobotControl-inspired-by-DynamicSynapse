import FitzHughNagumo as FHN
class CPGCombiened():
    def __init__(self, scale) -> None:
        pass
        self.AFHNN_l1 = FHN.FHNN(1, scale=0.02)
        self.AFHNN_r1 = FHN.FHNN(1, scale=0.02)
        self.AFHNN_l2 = FHN.FHNN(1, scale=0.02)
        self.AFHNN_r2 = FHN.FHNN(1, scale=0.02)
        self.AFHNN_l3 = FHN.FHNN(1, scale=0.02)
        self.AFHNN_r3 = FHN.FHNN(1, scale=0.02)

