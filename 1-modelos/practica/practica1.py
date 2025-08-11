import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import inspect

H = np.arange(3) # Posibles valores de las hipótesis
# Como estamos trabajando en python vamos empezar con 0
# Es decir, la posición del regalo r \in {0,1,2} y de
# la misma forma con el resto de las variables.

def pr(r): # P(r)
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def pc(c): # P(c)
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def ps_rM0(s,r): # P(s|r,M=0)
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def ps_rcM1(s,r,c): # P(s|r,c,M=1)
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def prcs_M(r,c,s,m): # P(r,c,s|M)
    # producto de las condicionales
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def ps_cM(s,c,m):
    # Predicción del segundo dato dado el primero
    num = 0 # P(s,c|M) = sum_r P(r,c,s|M)
    den = 0 # P(c|M) = sum_{rs} P(r,c,s|M)
    res = num/den # P(s|c,M) = P(s,c|M)/P(c|M)
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def pr_csM(r,c,s,m):
    # Predicción del segundo dato dado el primero
    num = 0 # p(r,c,s|M)
    den = 0 # p(c,s|M) = sum_r P(r,c,s|M)
    res = num/den # P(r|c,s,M) = P(r,c,s|M)/p(c,s|M)
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def pEpisodio_M(c,s,r,m): # P(Datos = (c,s,r) | M)
    # Predicción del conjunto de datos P(c,s,r|M)
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

# 1.2 Simular datos con Monty Hall

def simular(T=16,seed=0):
    np.random.seed(seed)
    Datos = []
    for t in range(T):
        r = np.random.choice(3, p=[pr(hr) for hr in H])
        c = None
        s = None
        Datos.append((c,s,r))
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

T =16
Datos = simular()

# 1.3 Predicción P(Datos = {(c0,s0,r0),(c1,s1,r1),...} | M )

def _secuencia_de_predicciones(Datos, m):
    # Si se guarda la lista de predicciones de cada uno
    # de los episodios [P(Episodio0|M),P(Episodio1|M),... ]
    # esto va a servir tanto para calcular la predicción
    # P(Datos = {(c0,s0,r0),(c1,s1,r1),...} | M ),
    # pero también va a servir después para graficar como
    # va cambiando el posterior de los modelos en el tiempo
    return None

def pDatos_M(Datos, m):
    # P(Datos = {(c0,s0,r0),(c1,s1,r1),...} | M )
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

pDatos_M(Datos, m=0) # 8.234550899283273e-21
pDatos_M(Datos, m=1) # 3.372872048346429e-17

# 1.4 Calcular predicción de los datos P(Datos)

def pM(m):
    # Prior de los modelos
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def pDatos(Datos):
    # sum_m P(Datos,M=m)
    # sum_m P(Datos|M=m)P(M=m)
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

# 1.5 Posterior de los modelos

def pM_Datos(m,Datos)
    # P(M|Datos = {(c0,s0,r0),(c1,s1,r1),...})
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def lista_pM_Datos(m,Datos):
    # [P(M | (c0,s0,r0) ), P(M | (c0,s0,r0),(c1,s1,r1) ), ... ]
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")


plt.plot(lista_pM_Datos(m=0, Datos), label="M0: Base")
plt.plot(lista_pM_Datos(m=1, Datos), label="M1: Monty Hall")
plt.legend()
plt.show()



# 2.1

def pp_Datos(p,Datos):
    # P(p | Datos = {(c0,s0,r0),(c1,s1,r1), ... })
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")


# 2.2

def pEpisodio_DatosMa(Episodio, Datos):
    # P(EpisodioT = (cT, sT, rT) | Datos = {(c0,s0,r0),(c1,s1,r1), ... })
    cT, sT, rT = Episodio
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

# 2.3

# Actualizar pDatos_M(Datos, m, log = False) agregándole un parámetro log

# 2.4

def log_Bayes_factor(log_pDatos_Mi,log_pDatos_Mj):
    # Recibe la predicción de los datos en ordenes de magnitud
    # y devuelve el logaritmo del Bayes factor, es decir,
    # la diferencia de predicciones en órdenes de magnitud
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

# 2.5

def geometric_mean(Datos, m, log = False)
    # Dado los datos y el modelo devuelve la media geométrica
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

# 2.6

# actualizar pM_Datos(m,Datos) para que soporte al modelo alternativo








