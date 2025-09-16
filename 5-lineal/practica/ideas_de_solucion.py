import ModeloLineal as ml

from statsmodels.api import OLS # Para selección de hipótesis
from scipy.stats import norm    # La distribución gaussiana
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 14})
np.random.seed(1) # Para reproducir los datos

cmap = plt.get_cmap("tab10") # Colores para los modelos
# ###########


# Ejercicio 1


N = 20 # Cantidad de datos
D = 10 # Cantidad de modelos (de 0 al 9)

BETA = (1/0.04) # Precisión de los datos, el inverso de su varianza
ALPHA = (10e-6) # Precisión de la creencia a prior, el inverso de su varianza

# Realidad causal subyacente
def realidad_causal_subyacente(X, beta =  BETA):
    return np.sin(2 * np.pi * X) + np.random.normal(0,np.sqrt(1/beta),X.shape)

# Las transformaciones de X que hace el modelo de grado D
def phi(X, complejidad = D):
    return(pd.DataFrame({f'X{d}': X[:, 0]**d for d in range(complejidad+1)}))

X = np.random.rand(N,1)-0.5
Y = realidad_causal_subyacente(X)

# Itero por modelos Md
modelos_OLS = []
for d in range(0,D):
    # Ajusto el modelo de compeljidad d
    modelos_OLS.append(OLS(Y, phi(X,d)).fit())

modelos_OLS[1].params # Las hipótesis seleccionadas
modelos_OLS[1].llf # El likelihood de la hipótesis seleccionada (en escala log).

modelos_BAY = []
for d in range(D):
    MU_d, COV_d = ml.posterior(Y,phi(X, d))
    log_evidence_d = ml.log_evidence(Y,phi(X, d))[0][0]
    modelos_BAY.append({"mean":MU_d.reshape(1,d+1)[0], "cov":COV_d, "log_evidence": log_evidence_d})


#
# 3.1 Data
#

X = np.random.rand(N,1)-0.5
Y = realidad_causal_subyacente(X)
# Grilla
X_grilla = np.linspace(0, 1, 100).reshape(-1, 1)-0.5
Y_grilla = realidad_causal_subyacente(X_grilla, np.inf)


# Ejercicio 2

import pandas as pd
Alturas = pd.read_csv("datos/alturas.csv")
Alturas.head()


N, _ = Alturas.shape
Y_alturas = Alturas.altura
X_base = pd.DataFrame({"Base": [1 for _ in range(N)],    # Origen
                       "Altura": Alturas.altura_madre,  # Pendiente
             })

ml.log_evidence(Y_alturas, X_base)

# Ejercicio 3

M = 1000
z1 = np.random.uniform(-3,3, size=M)
w1 = 3*z1 + np.random.normal(size=M,scale=1)
z2 = np.random.uniform(-3,3, size=M)
w2 = 2*z2 + np.random.normal(size=M,scale=1)
z3 = -2*z1 + 2*z2 + np.random.normal(size=M,scale=1)
x = -1*w1 + 2*z3 + np.random.normal(size=M,scale=1)
w3 = 2*x + np.random.normal(size=M,scale=1)
y = 2 - 1*w3 - z3 + w2 + np.random.normal(size=M,scale=1)


X_3_1= pd.DataFrame({
    "w_0": [1 for _ in range(M)],    # Origen
    "w_x": x,
    "w_z3": z3,
    "w_w2": w2,
    })

MU_3_1, COV_3_1 = ml.posterior(y,X_3_1)
MU_3_1, COV_3_1 = ml.posterior(y,X_3_1)

model_ols_3_1 = OLS(y, X_3_1).fit()
model_ols_3_1.summary()
MU_3_1_ols = model_ols_3_1.params



