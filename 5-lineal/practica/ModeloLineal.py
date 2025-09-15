import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from numpy.random import normal as noise
from scipy.stats import multivariate_normal as normal
import math

def moments_posterior(alpha, beta, t, Phi):
    """
    Calcula los momentos (media y covarianza) de la distribución posterior
    para los parámetros del modelo lineal.

    Parámetros:
    -----------
    alpha : float
        Precisión de la distribución prior (inversa de la varianza).
    beta : float
        Precisión del ruido (inversa de la varianza del ruido).
    t : array
        Vector de valores objetivo observados.
    Phi : array
        Matriz de diseño con las bases de características aplicadas a los datos de entrada.

    Retorna:
    --------
    m_N : array
        Media de la distribución posterior.
    S_N : array
        Matriz de covarianza de la distribución posterior.
    """
    S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T.dot(Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N.dot(Phi.T).dot(t)
    return m_N, S_N


def moments_prior(alpha, beta, M):
    """
    Calcula los momentos (media y covarianza) de la distribución prior
    para los parámetros del modelo lineal.

    Parámetros:
    -----------
    alpha : float
        Precisión de la distribución prior (inversa de la varianza).
    beta : float
        Precisión del ruido (no utilizado en esta función).
    M : int
        Dimensión del espacio de parámetros.

    Retorna:
    --------
    m_0 : array
        Media de la distribución prior (vector de ceros).
    S_0 : array
        Matriz de covarianza de la distribución prior.
    """
    S_0_inv = alpha * np.eye(M)
    S_0 = np.linalg.inv(S_0_inv)
    m_0 = np.zeros((1, M))
    return m_0, S_0

def likelihood(w, t, Phi, beta):
    """
    Calcula la verosimilitud (likelihood) para un conjunto de parámetros dado.

    Parámetros:
    -----------
    w : array
        Vector de parámetros del modelo.
    t : array
        Vector de valores objetivo observados.
    Phi : array
        Matriz de diseño con las bases de características aplicadas a los datos de entrada.
    beta : float
        Precisión del ruido (inversa de la varianza del ruido).

    Retorna:
    --------
    float
        Valor de la función de verosimilitud.
    """
    res = 1
    for i in range(len(t)):
        mean = w.T.dot(Phi[i])
        sigma = np.sqrt(beta**(-1))
        res = res * norm.pdf(t[i], mean, sigma)
    return res

def phi(x, bf, args=None):
    """
    Aplica funciones base a los datos de entrada y añade un término constante.

    Parámetros:
    -----------
    x : array
        Vector de entradas.
    bf : function
        Función base a aplicar a los datos.
    args : tuple, opcional
        Argumentos adicionales para la función base.

    Retorna:
    --------
    array
        Matriz de diseño con las bases aplicadas y una columna de unos añadida.
    """
    if args is None:
        return np.concatenate([np.ones(x.shape), bf(x)], axis=1)
    else:
        return np.concatenate([np.ones(x.shape)] + [bf(x, *args)], axis=1)

def moments_predictive(Phi_posteriori, beta, alpha, t_priori=None, Phi_priori=None):
    """
    Calcula los momentos (media y covarianza) de la distribución predictiva.

    Parámetros:
    -----------
    Phi_posteriori : array
        Matriz de diseño para los puntos donde se quiere hacer predicción.
    beta : float
        Precisión del ruido (inversa de la varianza del ruido).
    alpha : float
        Precisión de la distribución prior (inversa de la varianza).
    t_priori : array, opcional
        Vector de valores objetivo usados para calcular la distribución posterior.
    Phi_priori : array, opcional
        Matriz de diseño de los datos de entrenamiento.

    Retorna:
    --------
    mu : array
        Media de la distribución predictiva.
    sigma2 : array
        Matriz de covarianza de la distribución predictiva.
    """
    N, D = Phi_posteriori.shape

    if t_priori is None:
        t_priori, Phi_priori = np.zeros((0,1)), np.zeros((0,D))

    m_prior, S_prior = moments_posterior(alpha, beta, t_priori, Phi_priori)

    # Cálculo de la covarianza predictiva
    sigma2 = Phi_posteriori.dot(S_prior.dot(Phi_posteriori.T)) + (1/beta)*np.eye(Phi_posteriori.shape[0])
    # Cálculo de la media predictiva
    mu = Phi_posteriori.dot(m_prior)
    return mu, sigma2

def predictive(t_posteriori, Phi_posteriori, beta, alpha, t_priori=None, Phi_priori=None):
    """
    Calcula la densidad de probabilidad de la distribución predictiva en puntos específicos.

    Parámetros:
    -----------
    t_posteriori : array
        Valores de salida donde evaluar la distribución predictiva.
    Phi_posteriori : array
        Matriz de diseño para los puntos donde se quiere hacer predicción.
    beta : float
        Precisión del ruido (inversa de la varianza del ruido).
    alpha : float
        Precisión de la distribución prior (inversa de la varianza).
    t_priori : array, opcional
        Vector de valores objetivo usados para calcular la distribución posterior.
    Phi_priori : array, opcional
        Matriz de diseño de los datos de entrenamiento.

    Retorna:
    --------
    float
        Valor de la función de densidad de probabilidad predictiva.
    """
    m, S = moments_predictive(Phi_posteriori, beta, alpha, t_priori, Phi_priori)
    return normal.pdf(t_posteriori.ravel(), m.ravel(), S)

def log_evidence(t, Phi, alpha, beta):
    """
    Calcula la evidencia logarítmica del modelo.

    Parámetros:
    -----------
    t : array
        Vector de valores objetivo observados.
    Phi : array
        Matriz de diseño con las bases de características aplicadas a los datos de entrada.
    alpha : float
        Precisión de la distribución prior (inversa de la varianza).
    beta : float
        Precisión del ruido (inversa de la varianza del ruido).

    Retorna:
    --------
    float
        Valor de la evidencia logarítmica.
    """
    N, M = Phi.shape
    m_N, S_N = moments_posterior(alpha, beta, t, Phi)
    A = np.linalg.inv(S_N)
    A_det = np.linalg.det(A)
    E_mN = (beta/2) * (t - Phi.dot(m_N)).T.dot(t - Phi.dot(m_N)) \
         + (alpha/2) * m_N.T.dot(m_N)
    res = (M/2) * np.log(alpha)   \
        + (N/2) * np.log(beta)    \
        - E_mN                    \
        - (1/2) * np.log(A_det)   \
        - (N/2) * np.log(2*np.pi)
    return res

def sinus_model(X, variance):
    """
    Modelo de seno con ruido añadido.

    Parámetros:
    -----------
    X : array
        Vector de entradas.
    variance : float
        Varianza del ruido.

    Retorna:
    --------
    array
        Valores de la función seno con ruido añadido.
    """
    return np.sin(2 * np.pi * X) + noise(0, np.sqrt(variance), X.shape)

def polynomial_basis_function(x, degree=1):
    """
    Función de base polinómica.

    Parámetros:
    -----------
    x : array
        Vector de entradas.
    degree : int, opcional
        Grado del polinomio.

    Retorna:
    --------
    array
        Vector con los valores de la función polinómica aplicada.
    """
    return x ** degree
