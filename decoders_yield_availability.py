
community_map = {'Accumbens nucleus, core region':'NAcc', # PL, BLA , Hipp, VTA
    'Accumbens nucleus, shell region':'NAcsh', #wider inputs
    'Lateral accumbens, shell region':'NAcsh', #wider inputs
    'Lateral stripe of the striatum':'NAcsh',
        #Caudal
    'Caudoputamen- caudal, dorsal, dorsolateral':'Caudal_Dorsal',  # MOs, ACA, AUD,ORBmSSmVisam, SS, RSP
    'Caudoputamen- caudal, dorsal, dorsomedial':'Caudal_Dorsal', # MOs, ACA, AUD,ORBmSSmVisam, SS, RSP
    'Caudoputamen- caudal, dorsal, ventromedial':'Caudal_Dorsal', # MOs, ACA, AUD,ORBmSSmVisam, SS, RSP
    'Caudoputamen- caudal, intermediate, dorsal':'Caudal_Int.', # MOp,MOs, SSs
    'Caudoputamen- caudal, intermediate, ventromedial':'Caudal_Int.', # MOp,MOs, SSs
    'Caudoputamen- caudal, ventral':'Caudal_Ventral', # PIR, SS, VISC (visceral), GU
    'Caudoputamen- caudal, dorsal':'Caudal_Dorsal',
        # Intermediate
    'Caudoputamen- intermediate, dorsolateral, dorsal':'Int_DorsoLateral', # MOp. MOs, SSs
    'Caudoputamen- intermediate, dorsolateral, intermedial dorsal':'Int_DorsoLateral',  # MOp. MOs, SSs
    'Caudoputamen- intermediate, dorsomedial, central dorsal':'Int_DorsoMedial', # ACA, ORB,VISP, BLA, PL
    'Caudoputamen- intermediate, dorsomedial, dorsal tip':'Int_DorsoMedial', # ACA, ORB,VISP, BLA, PL
    'Caudoputamen- intermediate, dorsomedial, dorsolateral':'Int_DorsoMedial', # ACA, ORB,VISP, BLA, PL
    'Caudoputamen- intermediate, ventrolateral, central ventrolateral':'Int_VentroLateral', # MOp, MOs, SS, VISc, 
    'Caudoputamen- intermediate, ventrolateral, intermedial ventral':'Int_VentroLateral',  # MOp, MOs, SS, VISc, 
    'Caudoputamen- intermediate, ventrolateral, ventral tip':'Int_VentroLateral',  # MOp, MOs, SS, VISc, 
    'Caudoputamen- intermediate, ventromedial, ventral':'Int_VentroMedial', # ILA, AUD, PL, BLA, VISC, ILA, Temporal association area
    'Caudoputamen- intermediate, ventromedial, ventromedial':'Int_VentroMedial', # ILA, AUD, PL, BLA, VISC, ILA, Temporal association area
    'Caudoputamen- intermediate, ventrolateral, ventral':'Int_VentroLateral',
    'Caudoputamen- intermediate, ventromedial, central ventromedial':'Int_VentroMedial',
    # Rostral
    'Caudoputamen- rostra,l lateral, ventromedial':'Rostral_Lateral', #AUDM GU< MU, PERI, SS, Temporal association, VISC, BLA
    'Caudoputamen- rostral, lateral, lateral strip':'Rostral_Lateral', 
    'Caudoputamen- rostral, intermediate, dorsal':'Rostral_Int.', # ORB, PTL, VISal, Visam, MO, ACA
    'Caudoputamen- rostral, intermediate, ventral':'Rostral_Int.', #PL ORB, BLA
    'Caudoputamen- rostral, medial':'Rostral_Medial', # AUD, ENT, ILA, ORB, RSP, VISp, ILA
    'Caudoputamen- rostral extreme':'Rostral_Ext'}


level_map = {
    'excluded':'excluded',
    'ACB':'excluded',
     'CP':'excluded',
     'Island of Calleja, major island':'excluded',
     'STR':'excluded',
     'Lateral stripe of the striatum':'NAc',
     'Caudoputamen- caudal, intermediate, ventrolateral':'Int',
    'Accumbens nucleus, core region':'NAc', # PL, BLA , Hipp, VTA
    'Accumbens nucleus, shell region':'NAc', #wider inputs
    'Lateral accumbens, shell region':'NAc', #wider inputs
    'Lateral stripe of the striatum':'NAc',
    'Lateral stripe of the striatum ':'NAc',
        #Caudal
    'Caudoputamen- caudal, dorsal, dorsolateral':'Caudal',  # MOs, ACA, AUD,ORBmSSmVisam, SS, RSP
    'Caudoputamen- caudal, dorsal, dorsomedial':'Caudal', # MOs, ACA, AUD,ORBmSSmVisam, SS, RSP
    'Caudoputamen- caudal, dorsal, ventromedial':'Caudal', # MOs, ACA, AUD,ORBmSSmVisam, SS, RSP
    'Caudoputamen- caudal, intermediate, dorsal':'Caudal', # MOp,MOs, SSs
    'Caudoputamen- caudal, intermediate, ventromedial':'Caudal', # MOp,MOs, SSs
    'Caudoputamen- caudal, ventral':'Caudal', # PIR, SS, VISC (visceral), GU
    'Caudoputamen- caudal, dorsal':'Caudal',
        # Intermediate
    'Caudoputamen- intermediate, dorsolateral, dorsal':'Int', # MOp. MOs, SSs
    'Caudoputamen- intermediate, dorsolateral, intermedial dorsal':'Int',  # MOp. MOs, SSs
    'Caudoputamen- intermediate, dorsomedial, central dorsal':'Int', # ACA, ORB,VISP, BLA, PL
    'Caudoputamen- intermediate, dorsomedial, dorsal tip':'Int', # ACA, ORB,VISP, BLA, PL
    'Caudoputamen- intermediate, dorsomedial, dorsal tip ':'Int', # ACA, ORB,VISP, BLA, PL
    'Caudoputamen- intermediate, dorsomedial, dorsolateral':'Int', # ACA, ORB,VISP, BLA, PL
    'Caudoputamen- intermediate, ventrolateral, central ventrolateral':'Int', # MOp, MOs, SS, VISc, 
    'Caudoputamen- intermediate, ventrolateral, intermedial ventral':'Int',  # MOp, MOs, SS, VISc, 
    'Caudoputamen- intermediate, ventrolateral, ventral tip':'Int',  # MOp, MOs, SS, VISc, 
    'Caudoputamen- intermediate, ventromedial, ventral':'Int', # ILA, AUD, PL, BLA, VISC, ILA, Temporal association area
    'Caudoputamen- intermediate, ventromedial, ventromedial':'Int', # ILA, AUD, PL, BLA, VISC, ILA, Temporal association area
    'Caudoputamen- intermediate, ventrolateral, ventral':'Int',
    'Caudoputamen- intermediate, ventromedial, central ventromedial':'Int',
    # Rostral
    'Caudoputamen- rostra,l lateral, ventromedial':'Rostral', #AUDM GU< MU, PERI, SS, Temporal association, VISC, BLA
    'Caudoputamen- rostral, lateral, lateral strip':'Rostral', 
    'Caudoputamen- rostral, intermediate, dorsal':'Rostral', # ORB, PTL, VISal, Visam, MO, ACA
    'Caudoputamen- rostral, intermediate, ventral':'Rostral', #PL ORB, BLA
    'Caudoputamen- rostral, medial':'Rostral', # AUD, ENT, ILA, ORB, RSP, VISp, ILA
    'Caudoputamen- rostral extreme':'Rostral'}  

custom_map = {'Accumbens nucleus, core region':'NAc', # PL, BLA , Hipp, VTA
    'Accumbens nucleus, shell region':'NAc', #wider inputs
    'Lateral accumbens, shell region':'NAc', #wider inputs
    'Lateral stripe of the striatum':'NAc',
        #Caudal
    'Caudoputamen- caudal, dorsal, dorsolateral':'Caudal_Dorsal',  # MOs, ACA, AUD,ORBmSSmVisam, SS, RSP
    'Caudoputamen- caudal, dorsal, dorsomedial':'Caudal_Dorsal', # MOs, ACA, AUD,ORBmSSmVisam, SS, RSP
    'Caudoputamen- caudal, dorsal, ventromedial':'Caudal_Dorsal', # MOs, ACA, AUD,ORBmSSmVisam, SS, RSP
    'Caudoputamen- caudal, intermediate, dorsal':'Caudal_Dorsal', # MOp,MOs, SSs
    'Caudoputamen- caudal, intermediate, ventromedial':'Caudal_Ventral', # MOp,MOs, SSs
    'Caudoputamen- caudal, ventral':'Caudal_Ventral', # PIR, SS, VISC (visceral), GU
    'Caudoputamen- caudal, dorsal':'Caudal_Dorsal',
        # Intermediate
    'Caudoputamen- intermediate, dorsolateral, dorsal':'Int_Dorsal', # MOp. MOs, SSs
    'Caudoputamen- intermediate, dorsolateral, intermedial dorsal':'Int_Dorsal',  # MOp. MOs, SSs
    'Caudoputamen- intermediate, dorsomedial, central dorsal':'Int_Dorsal', # ACA, ORB,VISP, BLA, PL
    'Caudoputamen- intermediate, dorsomedial, dorsal tip':'Int_Dorsal', # ACA, ORB,VISP, BLA, PL
    'Caudoputamen- intermediate, dorsomedial, dorsolateral':'Int_Dorsal', # ACA, ORB,VISP, BLA, PL
    'Caudoputamen- intermediate, ventrolateral, central ventrolateral':'Int_Ventral', # MOp, MOs, SS, VISc, 
    'Caudoputamen- intermediate, ventrolateral, intermedial ventral':'Int_Ventral',  # MOp, MOs, SS, VISc, 
    'Caudoputamen- intermediate, ventrolateral, ventral tip':'Int_Ventral',  # MOp, MOs, SS, VISc, 
    'Caudoputamen- intermediate, ventromedial, ventral':'Int_Ventral', # ILA, AUD, PL, BLA, VISC, ILA, Temporal association area
    'Caudoputamen- intermediate, ventromedial, ventromedial':'Int_Ventral', # ILA, AUD, PL, BLA, VISC, ILA, Temporal association area
    'Caudoputamen- intermediate, ventrolateral, ventral':'Int_Ventral',
    'Caudoputamen- intermediate, ventromedial, central ventromedial':'Int_Ventral',
    # Rostral
    'Caudoputamen- rostra,l lateral, ventromedial':'Rostral_Lateral', #AUDM GU< MU, PERI, SS, Temporal association, VISC, BLA
    'Caudoputamen- rostral, lateral, lateral strip':'Rostral_Lateral', 
    'Caudoputamen- rostral, intermediate, dorsal':'Rostral_Lateral', # ORB, PTL, VISal, Visam, MO, ACA
    'Caudoputamen- rostral, intermediate, ventral':'Rostral_Medial', #PL ORB, BLA
    'Caudoputamen- rostral, medial':'Rostral_Medial', # AUD, ENT, ILA, ORB, RSP, VISp, ILA
    'Caudoputamen- rostral extreme':'Rostral_Extreme'}
