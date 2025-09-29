# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: global
#     language: python
#     name: python3
# ---

# %%
import math
import random
import inspect
import warnings

respuestas = {}

# %% [markdown]
# # Parcial
# ##
#
# Este notebook contiene una lista de preguntas junto con una lista exhaustiva de respuestas mutuamente contradictorias. A diferencia de los enunciados de tipo "multiple choise" en los que se pide seleccionar una única opción, aquí se pide que distribuyan creencias entre las diferentes opciones, asegurándose que el valor asignado sea positivo y la suma sea 1. La evaluación será el producto de las creencias asiganadas a las respuestas correctas. En caso de que la respuesta sea una variable aleatoria, se considerará la predicción típica a largo plazo, es decir, su media gométrica. Notar que asignar cero a una posible respuesta correcta hace que el producto sea cero. Por ello, en caso de duda, no conviene que concentren toda su creencia en una sola opción, sino distribuir algo de creencia en todas las opciones que consideran posibles. Noten también que conviene asignar más a la opción en la que más creen, porque distribuir creencias en partes iguales entre todas las opciones no es mucho mejor que el azar (baseline).


# %% [markdown]
# ### 1 Variable
#
# En probabilidad una variable es un conjunto de hipótesis mutuamente contradictorias.
#
# 0. No
# 1. Sí

# %%
respuestas[(1,"Variable")] = [
0.0, # 0. No
0.0, # 1. Sí
"Justifique en profundidad."
]


# %% [markdown]
# ### 2 Conjunta
#
# La distribución de probabilidad conjunta es creencia a priori.
#
# 0. No
# 1. Sí

# %%
respuestas[(2,"Conjunta")] = [
0.0, # 0. No
0.0, # 1. Sí
"Justifique en profundidad."
]


# %% [markdown]
# ### 3 Universos
#
# Hay tres cajas idénticas. Detrás de una de ellas hay un regalo. El resto están vacías. Nos permiten reservar una caja. Luego, una persona elige una de las cajas que no contenga el regalo y no haya sido reservada. Supongamos que reservamos la caja 1. ¿Cuál de todos los universos paralelos va a ocurrir? ¿El regalo está en la caja 1 y nos muestran la caja 1? ¿El regalo está en la caja 1 y nos muestran la caja 2? ... ¿El regalo está en la caja 3 y nos muestran la caja 2? ¿El regalo está en la caja 3 y nos muestran la caja 3?
#
# 0. Regalo = 1, Abren = 1
# 1. Regalo = 1, Abren = 2
# 2. Regalo = 1, Abren = 3
# 3. Regalo = 2, Abren = 1
# 4. Regalo = 2, Abren = 2
# 5. Regalo = 2, Abren = 3
# 6. Regalo = 3, Abren = 1
# 7. Regalo = 3, Abren = 2
# 8. Regalo = 3, Abren = 3
#

# %%
respuestas[(3,"Universos")] = [
0.0, # 0. Regalo = 1, Abren = 1
0.0, # 1. Regalo = 1, Abren = 2
0.0, # 2. Regalo = 1, Abren = 3
0.0, # 3. Regalo = 2, Abren = 1
0.0, # 4. Regalo = 2, Abren = 2
0.0, # 5. Regalo = 2, Abren = 3
0.0, # 6. Regalo = 3, Abren = 1
0.0, # 7. Regalo = 3, Abren = 2
0.0, # 8. Regalo = 3, Abren = 3
"Justifique brevemente",
]


# %% [markdown]
# ### 4 Overfitting
#
# En el área de aprendizaje automático e inteligencia artificial se ha descubierto un problema que se conoce con el nombre de overfitting. ¿El overfitting es/era un problema propio del sistema de razonamiento para contextos de incertidumbre?
#
# 0. No
# 1. Sí

# %%
respuestas[(4,"Overfitting")] = [
0.0, # 0. No
0.0, # 1. Sí
"Justifique en profundidad.",
]

# %% [markdown]
# ### 5 Evaluación
#
# En el área de aprendizaje automático e inteligencia artificial existe una gran cantidad de métricas distintas para evaluar los modelos alternativos. ¿En principio, existe una forma correcta, universal, de evaluar los modelos? 
#
# 0. No
# 1. Sí

# %%
respuestas[(5,"Evaluación")] = [
0.0, # 0. No
0.0, # 1. Si
"Justifique en profundidad.",
]

# %% [markdown]
# ### 6 Predicción
#
# Históricamente todas las ciencias con datos, desde la física hasta las ciencias sociales, explicaron el mundo a través de teorías causales. Los recientes avances en el área de aprendizaje automático e inteligencia artificial, sin embargo, se produjeron por el desarrollo de algoritmos altamente predictivos sin ninguna interpretación causal. ¿Por qué?
#
# 0. El modelo causal correcto nunca puede ser mejor prediciendo que los complejos algoritmos de AI/ML.
# 1. El modelo causal correcto a veces puede ser mejor, y a veces peor, que los complejos algoritmos de AI/ML.
# 2. El modelo causal correcto nunca puede ser peor prediciendo que los complejos algoritmos de AI/ML.
# 3. Los modelos causales solo explican, no predicen.
# 4. Ninguna de las anteriores

# %%
respuestas[(6,"Predicción")] = [
0.0, # 0. El modelo causal correcto nunca puede ser mejor prediciendo que los complejos algoritmos de AI/ML.
0.0, # 1. El modelo causal correcto a veces puede ser mejor, y a veces peor, que los complejos algoritmos de AI/ML.
0.0, # 2. El modelo causal correcto nunca puede ser peor prediciendo que los complejos algoritmos de AI/ML.
0.0, # 3. Los modelos causales solo explican, no predicen.
0.0, # 4. Ninguna de las anteriores
"Justifique en profundidad.",
]


# %% [markdown]
# ### Factor graph
#
# Un factor graph es una grafo bipartito entre dos tipo de nodos: variables y funciones (distribuciones de probabilidad condicional). Los ejes representan ``la variable $v$ es parámetro de la función $f$''.
#
# 0. Falso
# 1. Verdadero

# %%
respuestas[(7,"Factor graph")] = [
0.0, # 0. False
0.0, # 1. Verdadero
"Justifique en profundidad.",
]

# %% [markdown]
# ### do-operator
#
# Cuando se aplica un do-operator a una variable se reemplaza la distribución de probabilidad condicional que naturalmente tiene esa variable por una distribución de probabilidad indicadora (determinista). ¿Es posible especificar do-operators usando la notación de factor graphs? ¿Cómo?
#
# 0. No se puede
# 1. Sí se puede

# %%
respuestas[(8,"do-operator")] = [
0.0, # 0. No se puede
0.0, # 1. Sí se puede
"Justifique en profundidad.",
]


# %% [markdown]
# ### Sum-product marginal
#
# El sum-product algorithm descompone las reglas de probabilidad como pasaje de mensajes entre los nodos de un factor graph. La distribución marginal de una variable es el producto de los mensajes que recibe esa variable.
#
# 0. Falso
# 1. Verdadero

# %%
respuestas[(9,"Sum-product marginal")] = [
0.0, # 0. False
0.0, # 1. Verdadero
"Justifique en profundidad.",
]


# %% [markdown]
# ### Estructura básica
#
# Dada la siguiente estructura causal $P(X,Y,M,W) = P(X)P(Y)P(M|X,Y)P(W|M)$. ¿$X$ es independiente de $Y$ dado $W$?
#
# 0. No son independientes
# 1. Sí son independientes


# %%
respuestas[(10,"Estructura básica")] = [
0.0, # 0. No son independientes
0.0, # 1. Sí son independientes
"Justifique en profundidad.",
]


# %% [markdown]
# ### Predicción causal
#
# Es posible predecir el impacto causal que una variable $X$ tiene sobre una variable $Y$ usando datos observados sin intervenciones sin conocer la estructura causal subyacente.
#
# 0. Falso
# 1. Verdadero

# %%
respuestas[(11,"Predicción causal")] = [
0.0, # 0. Falso
0.0, # 1. Verdadero
"Justifique en profundidad.",
]


# %% [markdown]
# ### d-separation
#
# Hay flujo de inferencia entre los extremos de una cadena si y solo si se condiciona únicamente en todas las consecuencias comunes.
#
# 0. Falso
# 1. Verdadero

# %%
respuestas[(12,"d-separation")] = [
0.0, # 0. Falso
0.0, # 1. Verdadero
"Justifique en profundidad.",
]

# %% [markdown]
# ### Backdoor
#
# Si un conjunto de variable $Q$ cierra el flujo de asociación en todos los caminos ascendentes de $X$ a $Y$ necesariamente cumple con el criterio backdoor.
#
# 0. Falso
# 1. Verdadero

# %%
respuestas[(13,"Adjustment formula")] = [
0.0, # 0. Falso
0.0, # 1. Verdadero
"Justifique en profundidad.",
]


# %% [markdown]
# ### Adjustment formula
#
# Sea $Q$ variables de control que cumplen con el criterio backdoor de $X$ a $Y$. Y sea $M_x$ el modelo intervenido en el que se le asigna aleatoriamente un valor a la variable $X$. ¿Es cierta la siguiente igualdad?
#
#\begin{equation}
#P(Y|\text{do}(X)) = P_{M_x}(Y|X) = \sum_{Q} P(Q|X)P(Y|X,Q)
#\end{equation}
#
# 0. No es cierta
# 1. Sí es cierta

# %%
respuestas[(14,"Adjustment formula")] = [
0.0, # 0. No es cierta
0.0, # 1. Sí es cierta
"Justifique en profundidad.",
]



# %% [markdown]
# ### Ignorar intervención
# Sea $Q$ variables de control que cumplen con el criterio backdoor de $X$ a $Y$. Y sea $M_x$ el modelo intervenido en el que se le asigna aleatoriamente un valor a la variable $X$. ¿Puede ocurrir que $P_{M_x}(Y|X,Q) \neq P(Y|X,Q)$?
#
# 0. No puede ocurrir
# 1. Sí puede ocurrir

# %%
respuestas[(15,"Ignorar intervención")] = [
0.0, # 0. No es cierta
0.0, # 1. Sí es cierta
"Justifique en profundidad.",
]



# %% [markdown]
# ### Independencia
# Sea $Q$ variables de control que cumplen con el criterio backdoor de $X$ a $Y$. Y sea $M_x$ el modelo intervenido en el que se le asigna aleatoriamente un valor a la variable $X$. ¿Es cierta la siguiente igualdad, $P_{M_x}(Q|X) = P_{M_x}(Q)$? ¿Por qué?
#
# 0. No es cierta
# 1. Sí es cierta

# %%
respuestas[(16,"Independencia")] = [
0.0, # 0. No es cierta
0.0, # 1. Sí es cierta
"Justifique en profundidad.",
]


# %% [markdown]
# ### Variables de control
#
# Supongamos que tenemos un conjunto de datos observados sin intervenciones. Nos interesa conocer el efecto causal que una variable $T_i$ tiene sobre una variable objetivo $Y_i$. Sabemos que existe una variable oculta $U_i$ que es causa de las variables $T_i$ e $Y_i$. Sabemos además que el efecto causal de $T_i$ sobre $Y_i$ está mediado por $M_i$. En resumen la estructura causal está determinada por los mecanismos causales, $P(U_i)$, $P(T_i|U_i)$, $P(M_i|T_i)$ y $P(Y_i|M_i,U_i)$. ¿Podemos estimar el efecto causal únicamente con la información de $T_i$, $M_i$ e $Y_i$? Explique cómo.
#
# 0. No cumple backdoor
# 1. Sí cumple backdoor

# %%
respuestas[(17,"Variables de control")] = [
0.0, # 0. No cumple backdoor
0.0, # 1. Sí cumple backdoor
"Justifique en profundidad.",
]


# %% [markdown]
# ### Causa común oculta
#
# Supongamos que tenemos un conjunto de datos observados sin intervenciones. Nos interesa conocer el efecto causal que una variable $T_i$ tiene sobre una variable objetivo $Y_i$. Sabemos que existe una variable oculta $U_i$ que es causa de las variables $T_i$ e $Y_i$. Sabemos además que el efecto causal de $T_i$ sobre $Y_i$ está mediado por $M_i$. En resumen la estructura causal está determinada por los mecanismos causales, $P(U_i)$, $P(T_i|U_i)$, $P(M_i|T_i)$ y $P(Y_i|M_i,U_i)$. ¿Podemos estimar el efecto causal únicamente con la información de $T_i$, $M_i$ e $Y_i$? Explique cómo.
#
# 0. No es posible
# 1. Sí es posible

# %%
respuestas[(18,"Causa común oculta")] = [
0.0, # 0. No es posible
0.0, # 1. Sí es posible
"Justifique en profundidad.",
]

# %% [markdown]
# ### Experimento sin cumplimiento
#
# Supongamos que diseñamos un experimento aleatorizados, asignando el grupo al que pertenece cada persona mediante una variable aleatoria $Z_i$ tal que $P(Z_i) = \text{Bernoulli}(Z_i| 0.5)$. Supongamos que la aplicación efectiva del tratamiento $T_i$ no se cumple estrictamente sino que varía en función de las características ocultas de las personas $C_i$, $P(T_i|Z_i,C_i)$. Finalmente supongamos que la variable objetivo depende tanto del tratamiento aplicado $T_i$ y de las características ocultas, $P(Y_i|T_i, C_i)$. ¿Es posible estimar el efecto causal? Explique cómo.
#
# 0. No es posible
# 1. Sí es posible

# %%
respuestas[(19,"Experimento sin cumplimiento")] = [
0.0, # 0. No es posible
0.0, # 1. Sí es posible
"Justifique en profundidad.",
]

# %% [markdown]
# ### Diversificación
#
# Una casa de apuestas paga $3$ por Cara y $1.2$ por Sello con cada lanzamiento de una moneda que tiene $0.5$ de probabilidad de que salga Cara y Sello. Supongamos que nos ofrecen jugar 1000 veces, apostando en cada paso temporal todos nuestros recursos. ¿Qué proporción apostaría a Cara? Notar que el resto se asigna a Sello. Notar además que si apostamos todo a Cara y sale Sello perdemos todos los recursos y no podemos volver a jugar.
#
# 0. Recursos asignados a Cara: 0.0
# 1. Recursos asignados a Cara: 0.1
# 2. Recursos asignados a Cara: 0.2    
# 3. Recursos asignados a Cara: 0.3
# 4. Recursos asignados a Cara: 0.4
# 5. Recursos asignados a Cara: 0.5
# 6. Recursos asignados a Cara: 0.6
# 7. Recursos asignados a Cara: 0.7
# 8. Recursos asignados a Cara: 0.8
# 9. Recursos asignados a Cara: 0.9
# 10. Recursos asignados a Cara: 1.0

# %%
respuestas[(20,"Diversificación")] = [
0.0, # 0. Recursos asignados a Cara: 0.0
0.0, # 1. Recursos asignados a Cara: 0.1
0.0, # 2. Recursos asignados a Cara: 0.2
0.0, # 3. Recursos asignados a Cara: 0.3
0.0, # 4. Recursos asignados a Cara: 0.4
0.0, # 5. Recursos asignados a Cara: 0.5
0.0, # 6. Recursos asignados a Cara: 0.6
0.0, # 7. Recursos asignados a Cara: 0.7
0.0, # 8. Recursos asignados a Cara: 0.8
0.0, # 9. Recursos asignados a Cara: 0.9
0.0, # 10. Recursos asignados a Cara: 1.0
"Justifique en profundidad.",
]

# %% [markdown]
# ### Apuesta individual
#
# Una casa de apuestas paga $3$ por Cara y $1.2$ por Sello con cada lanzamiento de una moneda que tiene $0.5$ de probabilidad de que salga Cara y Sello. Supongamos que nos ofrecen jugar 1000 veces apostando en cada paso temporal 50\% de los recursos a Cara y 50\% de los recursos a Sello. Notar que cuando sale Cara los recursos aumentan 50\% ($3\times50\% + 0\times50\% =150\%$), y si sale Sello los recursos se reducen 40\% ($0\times50\% + 1.2\times50\% =60\%$). Es decir, crecemos más de lo que caemos. Y efectivamente, si calculamos la riqueza promedio de una población muy muy grande veremos que crece a una tasa de 5\% por paso temporal. ¿Nos conviene jugar?
#
# 0. No conviene
# 1. Sí conviene
# 2. Es indistinto

# %%
respuestas[(21,"Apuesta individual")] = [
0.0, # 0. No conviene
0.0, # 1. Sí conviene
0.0, # 2. Es indistinto
"Justifique en profundidad.",
]

# %% [markdown]
# ### Fondo común
#
# Una casa de apuestas paga $3$ por Cara y $1.2$ por Sello con cada lanzamiento de una moneda que tiene $0.5$ de probabilidad de que salga Cara y Sello. Supongamos que estamos apostando 50\% de los recursos a Cara y 50\% de los recursos a Sello en cada paso temporal. Supongamos que nos proponen participar de un fondo común, en el que en cada paso temporal los recursos de todas las personas que lo integran se redistribuyen en partes iguales. Es decir, en cada paso temporal cada persona tira su propia moneda, actualiza sus propios recursos individuales, los pone en el fondo común, se dividen en partes iguales y volvemos a empezar. ¿Nos conviene participar del fondo común?
#
# 0. No conviene
# 1. Sí conviene
# 2. Es indistinto


# %%
respuestas[(22,"Fondo común")] = [
0.0, # 0. No conviene
0.0, # 1. Sí conviene
0.0, # 2. Es indistinto
"Justifique en profundidad.",
]

# %% [markdown]
# ### Tragedia de los comunes
#
# Una casa de apuestas paga $3$ por Cara y $1.2$ por Sello con cada lanzamiento de una moneda que tiene $0.5$ de probabilidad de que salga Cara y Sello. Supongamos que estamos apostando 50\% de los recursos a Cara y 50\% de los recursos a Sello en cada paso temporal y que participamos de un fondo común, en el que en cada paso temporal los recursos de todas las personas que lo integran se redistribuyen en partes iguales. ¿En términos estrictamente monetarios, nos convendría dejar de aportar al fondo común en caso de que sigamos recibiendo la cuota en partes iguales del fondo común?
#
# 0. No conviene
# 1. Sí conviene
# 2. Es indistinto

# %%
respuestas[(23,"Tragedia de los comunes")] = [
0.0, # 0. No conviene
0.0, # 1. Sí conviene
0.0, # 2. Es indistinto
"Justifique en profundidad.",
]


