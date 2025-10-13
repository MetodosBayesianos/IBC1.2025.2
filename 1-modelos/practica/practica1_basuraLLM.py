"""
Práctica 1: Inferencia Bayesiana Causal - Problema de Monty Hall

Esta práctica implementa modelos causales para el problema de Monty Hall y evalúa
cuál modelo explica mejor los datos observados.

Variables:
- r: posición del regalo (0, 1, 2)
- c: caja elegida por el participante (0, 1, 2)  
- s: caja señalada/abierta por Monty (0, 1, 2)
- M: modelo causal (0=Base, 1=Monty Hall)

Modelos:
- M=0 (Base): Monty abre una caja aleatoria
- M=1 (Monty Hall): Monty siempre abre una caja vacía diferente a la elegida
"""

import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import inspect

# =============================================================================
# CONFIGURACIÓN INICIAL
# =============================================================================

# Posibles valores de las hipótesis (r, c, s ∈ {0, 1, 2})
H = np.arange(3)

# =============================================================================
# SECCIÓN 1: MODELOS DE PROBABILIDAD
# =============================================================================

def pr(r):
    """
    Probabilidad marginal de que el regalo esté en la posición r.
    
    Args:
        r (int): Posición del regalo (0, 1, 2)
        
    Returns:
        float: P(r)
        
    Nota: Asumir distribución uniforme inicialmente
    """
    # TODO: Implementar P(r)
    # Sugerencia: distribución uniforme sobre las 3 posiciones
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def pc(c):
    """
    Probabilidad marginal de que el participante elija la caja c.
    
    Args:
        c (int): Caja elegida (0, 1, 2)
        
    Returns:
        float: P(c)
        
    Nota: Asumir distribución uniforme inicialmente
    """
    # TODO: Implementar P(c)
    # Sugerencia: distribución uniforme sobre las 3 cajas
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def ps_rM0(s, r):
    """
    Probabilidad de que Monty señale la caja s dado que el regalo está en r y M=0.
    
    En el modelo base (M=0), Monty abre una caja aleatoria.
    
    Args:
        s (int): Caja señalada (0, 1, 2)
        r (int): Posición del regalo (0, 1, 2)
        
    Returns:
        float: P(s|r, M=0)
        
    Nota: Monty puede abrir cualquier caja, incluso la que tiene el regalo
    """
    # TODO: Implementar P(s|r, M=0)
    # Sugerencia: distribución uniforme sobre las 3 cajas
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def ps_rcM1(s, r, c):
    """
    Probabilidad de que Monty señale la caja s dado r, c y M=1.
    
    En el modelo Monty Hall (M=1), Monty siempre abre una caja vacía diferente a la elegida.
    
    Args:
        s (int): Caja señalada (0, 1, 2)
        r (int): Posición del regalo (0, 1, 2)
        c (int): Caja elegida (0, 1, 2)
        
    Returns:
        float: P(s|r, c, M=1)
        
    Nota: Monty nunca abre la caja elegida ni la que tiene el regalo
    """
    # TODO: Implementar P(s|r, c, M=1)
    # Sugerencia: si s != c y s != r, entonces probabilidad uniforme sobre cajas restantes
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def prcs_M(r, c, s, m):
    """
    Probabilidad conjunta P(r, c, s|M) usando el producto de las condicionales.
    
    Args:
        r (int): Posición del regalo (0, 1, 2)
        c (int): Caja elegida (0, 1, 2)
        s (int): Caja señalada (0, 1, 2)
        m (int): Modelo (0=Base, 1=Monty Hall)
        
    Returns:
        float: P(r, c, s|M)
        
    Nota: P(r,c,s|M) = P(r) * P(c) * P(s|r,c,M)
    """
    # TODO: Implementar P(r, c, s|M)
    # Sugerencia: usar las funciones pr(), pc() y ps_rcM1() o ps_rM0()
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def ps_cM(s, c, m):
    """
    Predicción del segundo dato dado el primero: P(s|c, M).
    
    Args:
        s (int): Caja señalada (0, 1, 2)
        c (int): Caja elegida (0, 1, 2)
        m (int): Modelo (0=Base, 1=Monty Hall)
        
    Returns:
        float: P(s|c, M)
        
    Nota: P(s|c,M) = P(s,c|M) / P(c|M) = sum_r P(r,c,s|M) / sum_{r,s} P(r,c,s|M)
    """
    # TODO: Implementar P(s|c, M)
    # Sugerencia: usar marginalización sobre r
    num = 0  # P(s,c|M) = sum_r P(r,c,s|M)
    den = 0  # P(c|M) = sum_{r,s} P(r,c,s|M)
    res = num/den  # P(s|c,M) = P(s,c|M)/P(c|M)
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def pr_csM(r, c, s, m):
    """
    Predicción del primer dato dado los otros dos: P(r|c, s, M).
    
    Args:
        r (int): Posición del regalo (0, 1, 2)
        c (int): Caja elegida (0, 1, 2)
        s (int): Caja señalada (0, 1, 2)
        m (int): Modelo (0=Base, 1=Monty Hall)
        
    Returns:
        float: P(r|c, s, M)
        
    Nota: P(r|c,s,M) = P(r,c,s|M) / P(c,s|M) = P(r,c,s|M) / sum_r P(r,c,s|M)
    """
    # TODO: Implementar P(r|c, s, M)
    # Sugerencia: usar marginalización sobre r
    num = 0  # P(r,c,s|M)
    den = 0  # P(c,s|M) = sum_r P(r,c,s|M)
    res = num/den  # P(r|c,s,M) = P(r,c,s|M)/P(c,s|M)
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def pEpisodio_M(c, s, r, m):
    """
    Probabilidad de un episodio completo: P(Datos = (c, s, r) | M).
    
    Args:
        c (int): Caja elegida (0, 1, 2)
        s (int): Caja señalada (0, 1, 2)
        r (int): Posición del regalo (0, 1, 2)
        m (int): Modelo (0=Base, 1=Monty Hall)
        
    Returns:
        float: P(c, s, r|M)
        
    Nota: Esto es simplemente P(r,c,s|M) con los argumentos en orden diferente
    """
    # TODO: Implementar P(c, s, r|M)
    # Sugerencia: usar prcs_M() con los argumentos en el orden correcto
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

# =============================================================================
# SECCIÓN 2: SIMULACIÓN DE DATOS
# =============================================================================

def simular(T=16, seed=0):
    """
    Simula T episodios del problema de Monty Hall.
    
    Args:
        T (int): Número de episodios a simular
        seed (int): Semilla para reproducibilidad
        
    Returns:
        list: Lista de tuplas (c, s, r) con los datos simulados
        
    Nota: Implementar la lógica completa de simulación según el modelo elegido
    """
    np.random.seed(seed)
    Datos = []
    
    for t in range(T):
        # TODO: Implementar simulación completa
        # 1. Elegir posición del regalo r según P(r)
        r = np.random.choice(3, p=[pr(hr) for hr in H])
        
        # 2. Elegir caja c según P(c)
        c = None  # TODO: implementar
        
        # 3. Elegir caja señalada s según P(s|r,c,M)
        s = None  # TODO: implementar
        
        Datos.append((c, s, r))
    
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

# Configuración de simulación
T = 16
Datos = simular()

# =============================================================================
# SECCIÓN 3: PREDICCIÓN DE DATOS
# =============================================================================

def _secuencia_de_predicciones(Datos, m):
    """
    Calcula la secuencia de predicciones P(Episodio_i|M) para cada episodio.
    
    Args:
        Datos (list): Lista de episodios [(c0,s0,r0), (c1,s1,r1), ...]
        m (int): Modelo (0=Base, 1=Monty Hall)
        
    Returns:
        list: Lista de probabilidades [P(Episodio0|M), P(Episodio1|M), ...]
        
    Nota: Esto sirve para calcular P(Datos|M) y para graficar la evolución temporal
    """
    # TODO: Implementar secuencia de predicciones
    # Sugerencia: usar pEpisodio_M() para cada episodio
    return None

def pDatos_M(Datos, m):
    """
    Probabilidad de observar los datos dados el modelo M: P(Datos|M).
    
    Args:
        Datos (list): Lista de episodios [(c0,s0,r0), (c1,s1,r1), ...]
        m (int): Modelo (0=Base, 1=Monty Hall)
        
    Returns:
        float: P(Datos|M)
        
    Nota: P(Datos|M) = producto de P(Episodio_i|M) para todos los episodios
    """
    # TODO: Implementar P(Datos|M)
    # Sugerencia: usar _secuencia_de_predicciones() y calcular el producto
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

# Ejemplos de uso esperados:
# pDatos_M(Datos, m=0)  # Debería dar aproximadamente 8.234550899283273e-21
# pDatos_M(Datos, m=1)  # Debería dar aproximadamente 3.372872048346429e-17

# =============================================================================
# SECCIÓN 4: INFERENCIA BAYESIANA
# =============================================================================

def pM(m):
    """
    Probabilidad previa del modelo M: P(M).
    
    Args:
        m (int): Modelo (0=Base, 1=Monty Hall)
        
    Returns:
        float: P(M)
        
    Nota: Asumir distribución uniforme sobre los modelos
    """
    # TODO: Implementar P(M)
    # Sugerencia: distribución uniforme sobre los 2 modelos
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def pDatos(Datos):
    """
    Probabilidad marginal de los datos: P(Datos).
    
    Args:
        Datos (list): Lista de episodios [(c0,s0,r0), (c1,s1,r1), ...]
        
    Returns:
        float: P(Datos)
        
    Nota: P(Datos) = sum_m P(Datos|M=m) * P(M=m)
    """
    # TODO: Implementar P(Datos)
    # Sugerencia: usar pDatos_M() y pM() para ambos modelos
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def pM_Datos(m, Datos):
    """
    Probabilidad posterior del modelo M dado los datos: P(M|Datos).
    
    Args:
        m (int): Modelo (0=Base, 1=Monty Hall)
        Datos (list): Lista de episodios [(c0,s0,r0), (c1,s1,r1), ...]
        
    Returns:
        float: P(M|Datos)
        
    Nota: P(M|Datos) = P(Datos|M) * P(M) / P(Datos)
    """
    # TODO: Implementar P(M|Datos)
    # Sugerencia: usar el teorema de Bayes
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def lista_pM_Datos(m, Datos):
    """
    Secuencia de probabilidades posteriores P(M|Datos_t) para t=1,2,...,T.
    
    Args:
        m (int): Modelo (0=Base, 1=Monty Hall)
        Datos (list): Lista de episodios [(c0,s0,r0), (c1,s1,r1), ...]
        
    Returns:
        list: [P(M|Datos_1), P(M|Datos_2), ..., P(M|Datos_T)]
        
    Nota: Útil para visualizar cómo evoluciona la creencia en el modelo
    """
    # TODO: Implementar secuencia de posteriores
    # Sugerencia: calcular P(M|Datos_t) para cada t usando los primeros t episodios
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

# Visualización de la evolución de las probabilidades posteriores
# plt.plot(lista_pM_Datos(m=0, Datos), label="M0: Base")
# plt.plot(lista_pM_Datos(m=1, Datos), label="M1: Monty Hall")
# plt.legend()
# plt.show()

# =============================================================================
# SECCIÓN 5: ANÁLISIS AVANZADO
# =============================================================================

def pp_Datos(p, Datos):
    """
    Probabilidad posterior del parámetro p dado los datos: P(p|Datos).
    
    Args:
        p (float): Valor del parámetro
        Datos (list): Lista de episodios [(c0,s0,r0), (c1,s1,r1), ...]
        
    Returns:
        float: P(p|Datos)
        
    Nota: Implementar según el modelo alternativo propuesto
    """
    # TODO: Implementar P(p|Datos)
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def pEpisodio_DatosMa(Episodio, Datos):
    """
    Predicción de un episodio futuro dado los datos observados: P(EpisodioT|Datos).
    
    Args:
        Episodio (tuple): Episodio futuro (cT, sT, rT)
        Datos (list): Lista de episodios observados [(c0,s0,r0), (c1,s1,r1), ...]
        
    Returns:
        float: P(EpisodioT|Datos)
        
    Nota: P(EpisodioT|Datos) = sum_M P(EpisodioT|M) * P(M|Datos)
    """
    cT, sT, rT = Episodio
    # TODO: Implementar P(EpisodioT|Datos)
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def log_Bayes_factor(log_pDatos_Mi, log_pDatos_Mj):
    """
    Calcula el logaritmo del factor de Bayes entre dos modelos.
    
    Args:
        log_pDatos_Mi (float): log P(Datos|Mi)
        log_pDatos_Mj (float): log P(Datos|Mj)
        
    Returns:
        float: log BF(Mi, Mj) = log P(Datos|Mi) - log P(Datos|Mj)
        
    Nota: Útil para comparar modelos cuando las probabilidades son muy pequeñas
    """
    # TODO: Implementar log Bayes factor
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

def geometric_mean(Datos, m, log=False):
    """
    Calcula la media geométrica de las probabilidades de los episodios.
    
    Args:
        Datos (list): Lista de episodios [(c0,s0,r0), (c1,s1,r1), ...]
        m (int): Modelo (0=Base, 1=Monty Hall)
        log (bool): Si True, devuelve el logaritmo de la media geométrica
        
    Returns:
        float: Media geométrica de P(Episodio_i|M) para todos los episodios
        
    Nota: Útil para comparar modelos independientemente del número de episodios
    """
    # TODO: Implementar media geométrica
    return NotImplementedError(f"La función {inspect.currentframe().f_code.co_name}() no está implementada")

# =============================================================================
# FUNCIONES AUXILIARES Y VALIDACIONES
# =============================================================================

def validar_episodio(episodio):
    """
    Valida que un episodio tenga el formato correcto.
    
    Args:
        episodio (tuple): Episodio (c, s, r)
        
    Returns:
        bool: True si el episodio es válido
        
    Raises:
        ValueError: Si el episodio no es válido
    """
    if not isinstance(episodio, (tuple, list)) or len(episodio) != 3:
        raise ValueError("Un episodio debe ser una tupla de 3 elementos (c, s, r)")
    
    c, s, r = episodio
    if not all(isinstance(x, int) and 0 <= x <= 2 for x in [c, s, r]):
        raise ValueError("c, s, r deben ser enteros entre 0 y 2")
    
    return True

def cargar_datos_reales(archivo="data/NoMontyHall.csv"):
    """
    Carga los datos reales desde el archivo CSV.
    
    Args:
        archivo (str): Ruta al archivo CSV
        
    Returns:
        list: Lista de episodios [(c, s, r), ...]
    """
    try:
        df = pd.read_csv(archivo)
        datos = [(row['c'], row['s'], row['r']) for _, row in df.iterrows()]
        return datos
    except FileNotFoundError:
        print(f"Archivo {archivo} no encontrado. Usando datos simulados.")
        return simular()

# =============================================================================
# EJEMPLOS DE USO Y PRUEBAS
# =============================================================================

def ejemplo_basico():
    """
    Ejemplo básico de uso de las funciones principales.
    """
    print("=== Ejemplo Básico ===")
    
    # Crear un episodio de ejemplo
    episodio = (0, 1, 2)  # c=0, s=1, r=2
    validar_episodio(episodio)
    
    # Calcular probabilidades para ambos modelos
    for m in [0, 1]:
        prob = pEpisodio_M(*episodio, m)
        print(f"P(Episodio|M={m}) = {prob}")
    
    print("=== Fin del Ejemplo ===")

def prueba_consistencia():
    """
    Prueba de consistencia: las probabilidades deben sumar 1.
    """
    print("=== Prueba de Consistencia ===")
    
    # Probar que P(s|c,M) suma 1 para cada c y M
    for c in range(3):
        for m in [0, 1]:
            suma = sum(ps_cM(s, c, m) for s in range(3))
            print(f"Suma P(s|c={c},M={m}) = {suma}")
    
    print("=== Fin de la Prueba ===")

# Descomentar para ejecutar ejemplos:
# ejemplo_basico()
# prueba_consistencia()

# =============================================================================
# INSTRUCCIONES PARA EL ESTUDIANTE
# =============================================================================

"""
INSTRUCCIONES PARA COMPLETAR LA PRÁCTICA:

1. IMPLEMENTAR LAS FUNCIONES BÁSICAS (Sección 1):
   - pr(r): Probabilidad marginal del regalo
   - pc(c): Probabilidad marginal de la elección
   - ps_rM0(s, r): Probabilidad de señalización en modelo base
   - ps_rcM1(s, r, c): Probabilidad de señalización en modelo Monty Hall

2. IMPLEMENTAR LAS FUNCIONES DERIVADAS (Sección 1):
   - prcs_M(r, c, s, m): Probabilidad conjunta
   - ps_cM(s, c, m): Predicción condicional
   - pr_csM(r, c, s, m): Inferencia inversa
   - pEpisodio_M(c, s, r, m): Probabilidad de episodio

3. IMPLEMENTAR LA SIMULACIÓN (Sección 2):
   - simular(T, seed): Generar datos sintéticos

4. IMPLEMENTAR LA PREDICCIÓN (Sección 3):
   - _secuencia_de_predicciones(Datos, m): Secuencia de probabilidades
   - pDatos_M(Datos, m): Probabilidad de los datos dados el modelo

5. IMPLEMENTAR LA INFERENCIA (Sección 4):
   - pM(m): Probabilidad previa del modelo
   - pDatos(Datos): Probabilidad marginal de los datos
   - pM_Datos(m, Datos): Probabilidad posterior del modelo
   - lista_pM_Datos(m, Datos): Evolución temporal de la posterior

6. IMPLEMENTAR ANÁLISIS AVANZADO (Sección 5):
   - pp_Datos(p, Datos): Posterior del parámetro
   - pEpisodio_DatosMa(Episodio, Datos): Predicción futura
   - log_Bayes_factor(log_pDatos_Mi, log_pDatos_Mj): Comparación de modelos
   - geometric_mean(Datos, m, log): Media geométrica

CONSEJOS:
- Comienza con las funciones básicas y ve construyendo las más complejas
- Usa las funciones auxiliares para validar tus resultados
- Ejecuta los ejemplos para verificar que todo funciona correctamente
- Recuerda que las probabilidades deben sumar 1 donde corresponda
- Usa logaritmos para evitar problemas de precisión numérica
"""