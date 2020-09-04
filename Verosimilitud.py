from scipy.stats import truncnorm

import numpy as np

def _log_normal_truncada_desplazada(x, limite_inferior, limite_superior, media, desvio):
    """
    Devuelve el log de la densidad de una normal truncada entre los límites indicados
    con la media y desvío recibidos
    """
    a, b = (limite_inferior - media) / desvio, (limite_superior - media) / desvio
    return truncnorm.logpdf(x, a, b, loc = media, scale = desvio) # TODO La versión del paper devuelve -Inf si x == limite_inferior o x == limite_superior

def _log_prior_gamma_beta(x):
    """
    Calcula el log de la pdf para el parámetro gamma | beta
    Gamma | beta se comporta como una normal en el rango [1, 4] con media 2.5 y desvío estándar 1.5
    Los valores se toman a partir del estudio de Li, Q et al
    """
    limite_inferior, limite_superior = 1, 4
    media, desvio = 2.5, 1.5

    return _log_normal_truncada_desplazada(x, limite_inferior, limite_superior, media, desvio)

def _log_prior_gamma(x):
    """
    Calcula el log de la pdf para el parámetro gamma
    Gamma se comporta como una normal en el rango [3.4, 9.4] con media 6.4 y desvío estándar 1.5
    La media se obtuvo del paper de Ferguson et al. Esto nos da un período infeccioso que varía de aproximadamente 3 a 9 días
    """
    limite_inferior, limite_superior = 3.4, 9.4
    media, desvio = 6.4, 1.5

    return _log_normal_truncada_desplazada(x, limite_inferior, limite_superior, media, desvio)

def _log_prior_phi(x):
    """
    Calcula el log de la pdf para el parámetro phi
    Phi es uniforme en el rango [0.01, 0.99]
    """
    if x >= 0.01 and x <= 0.99:
        return 0
    return -np.Inf

def _log_prior_T0(x):
    """
    Calcula el log de la pdf para el parámetro T0
    T0 es uniforme en el rango [1, 60] (1 = 1/1/2020, 60 = 29/2/2020)
    """
    if x >= 1 and x <= 60:
        return 0
    return -np.Inf

def log_prior(beta, gamma, T0, phi):
    return _log_prior_gamma_beta(beta/gamma) + _log_prior_gamma(1/gamma) + _log_prior_T0(T0) + _log_prior_phi(phi)