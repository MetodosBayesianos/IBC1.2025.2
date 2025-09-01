# -*- coding: utf-8 -*-
import pandas as pd
from collections import defaultdict
from Evento import *

data_wta = pd.read_csv("../data/kaggle-ATP-WTA-archive/df_wta.csv")

habilidad = defaultdict(lambda: [Gaussian()])
GAMMA = 0.06
forget = Gaussian(0,GAMMA)
for w, l in zip(data_wta["Winner"],data_wta["Loser"]):
    evento = Evento([[habilidad[w][-1]+forget],
                     [habilidad[l][-1]+forget]])
    posterior_w, posterior_l = evento.posterior
    habilidad[w].append(posterior_w[0])
    habilidad[l].append(posterior_l[0])
    print(w, " vs. ", l)


