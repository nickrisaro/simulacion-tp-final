from datetime import date
from datetime import datetime
from scipy.integrate import solve_ivp
from scipy.stats import poisson
from scipy.stats import truncnorm
from Theta import calcular_theta

import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import statsmodels.api as sm
import sys

import Verosimilitud as ll

p = 0.015

ITERACIONES = 50000
PERIODO_ADAPTACION = 10000
VARIANZA_INICIAL = [0.002**2, 0.002**2, 0.75**2, 0.002**2]
FACTOR_ESCALA = (2.38 / 4)

T0_INICIAL = 30 # Inicio epidemia
T1 = 73 # Inicio cuarentena
POBLACION = 39937489

R0_INICIAL = 2.3
GAMMA_INICIAL = 1/6
BETA_INICIAL = GAMMA_INICIAL * R0_INICIAL
PHI_INICIAL = 0.4

def SIR(t, y, beta, gamma):
    """
    Esta función será invocada por el método de resolución de ecuaciones diferenciales
    Para cada instante de tiempo t calcula los nuevos valores de S, I y R con los parámetros beta y gamma
    """
    S, I, R = y

    dsdt = -beta*S*I
    didt = beta*S*I - gamma*I
    drdt = gamma*I
    return dsdt, didt, drdt

def simular_con_cuarentena(BETA, GAMMA, PHI, DIAS, T0):
    """
    Realiza una simulación con cuarentena del modelo SIR
    Toda la población es susceptible de contagiarse
    Retorna todos los valores de S, I y R para cada instante de tiempo
    """

    # Primero simulamos desde el instante en el que se detecta el primer caso (T0)
    # hasta el momento en el que se implementa la cuarentena (T1)
    I0 = 1/POBLACION
    R0 = 0
    S0 = 1 - I0 - R0
    y0 = [S0, I0, R0]

    TIEMPO = np.arange(T0, T1, 1)

    # Usamos LSODA porque es el método de resolución que usan en el paper,
    # con RK45 los resultados son ligeramente distintos
    ret = solve_ivp(SIR, [T0, T1], y0, t_eval=TIEMPO, method='LSODA', args=(BETA, GAMMA))
    S, I, R = ret.y

    # Del día 0 al T0 S, I y R permanecen constantes
    S = np.insert(S, 0, np.repeat(S0, T0+1), axis=0)
    I = np.insert(I, 0, np.repeat(I0, T0+1), axis=0)
    R = np.insert(R, 0, np.repeat(R0, T0+1), axis=0)

    # Calculamos la variación de infectades día a día
    Nus = S * POBLACION
    Nus = np.subtract(Nus[0:Nus.size -1], Nus[1:Nus.size])

    # Ahora simulamos con la implementación de medidas de aislamiento
    # Los parámetros iniciales corresponden a los finales de la simulación anterior
    S0 = S[S.size - 1]
    I0 = I[I.size - 1]
    R0 = R[R.size - 1]
    y0 = [S0, I0, R0]

    # Simulamos hasta DIAS+1 para que la lista de Nus tenga la misma longitud que la de I
    # Descartamos el último valor de S, I y R
    TIEMPO = np.arange(T1, DIAS+1, 1)

    ret = solve_ivp(SIR, [T1, DIAS+1], y0, t_eval=TIEMPO, method='LSODA', args=(BETA*PHI, GAMMA))
    S1, I1, R1 = ret.y

    Nus1 = S1 * POBLACION
    Nus1 = np.subtract(Nus1[0:Nus1.size -1], Nus1[1:Nus1.size])

    # Juntamos los resultados de ambas simulaciones
    Nus = np.insert(Nus1, 0, Nus, axis=0)
    S = np.append(S, S1[1:S1.size - 1])
    I = np.append(I, I1[1:I1.size - 1])
    R = np.append(R, R1[1:R1.size - 1])

    return [S, I, R, Nus]

def proponer_parametros_adaptacion(beta, gamma, T0, phi):
    """
    Propone nuevos valores para los parámetros de la simulación
    Considera que el algoritmo está en la fase de adaptación
    """
    return np.random.normal(loc=(beta, gamma, T0, phi), scale=np.sqrt(VARIANZA_INICIAL), size=4)

def proponer_parametros_fin_adaptacion(iteracion, propuestas_anteriores, beta, gammar, t0, phi):
    """
    Propone la nueva tupla de parámetros para la primera iteración luego del período de adaptación
    """
    matriz_propuestas = np.array(propuestas_anteriores[0: iteracion - 1])
    xbar = matriz_propuestas.mean(0)
    xbar = np.reshape(xbar, [4,1])
    sXXt = np.dot(matriz_propuestas.transpose(), matriz_propuestas)

    sx = (1/(iteracion-2))*sXXt - ((iteracion-1)/(iteracion-2))*np.dot(xbar, xbar.transpose())
    sx = FACTOR_ESCALA * sx

    prop = np.random.multivariate_normal([beta, gammar, t0, phi], sx, 1)
    (beta_propuesto, gammar_propuesto, t0_propuesto, phi_propuesto) = (prop[0][0], prop[0][1], prop[0][2], prop[0][3])
    return (beta_propuesto, gammar_propuesto, t0_propuesto, phi_propuesto, sXXt, xbar)

def proponer_parametros_luego_adaptacion(iteracion, beta, gammar, t0, phi, sXXt, xbar):
    """
    Propone la nueva tupla de parámetros para las iteraciones posteriores al período de adaptación
    """
    x = np.reshape([beta, gammar, t0, phi], [4,1])
    xbar = ((iteracion - 2) * xbar +x) / (iteracion - 1)

    sXXt = sXXt + np.dot(x, x.transpose())

    sx = (1/(iteracion-2))*sXXt - ((iteracion-1)/(iteracion-2))*np.dot(xbar, xbar.transpose())
    sx = FACTOR_ESCALA * sx

    prop = np.random.multivariate_normal([beta, gammar, t0, phi], sx, 1)
    (beta_propuesto, gammar_propuesto, t0_propuesto, phi_propuesto) = (prop[0][0], prop[0][1], prop[0][2], prop[0][3])
    return (beta_propuesto, gammar_propuesto, t0_propuesto, phi_propuesto, sXXt, xbar)

def proponer_nuevos_parametros(theta, beta, gammar, t0, phi, dias_epidemia, Ds, verosimilitud_actual, sXXt, xbar, estado_inicial, iteracion, propuestas_anteriores):

    aceptado = False
    (beta_propuesto, gammar_propuesto, t0_propuesto, phi_propuesto) = (beta, gammar, t0, phi)

    if (iteracion < PERIODO_ADAPTACION):
        (beta_propuesto, gammar_propuesto, t0_propuesto, phi_propuesto) = proponer_parametros_adaptacion(beta, gammar, t0, phi)

    # En la primera iteración luego del período de adaptación hacemos el setup de las matrices de covarianza
    if iteracion == PERIODO_ADAPTACION:
        (beta_propuesto, gammar_propuesto, t0_propuesto, phi_propuesto, sXXt, xbar) = proponer_parametros_fin_adaptacion(iteracion, propuestas_anteriores, beta, gammar, t0, phi)

    if iteracion > PERIODO_ADAPTACION:
        (beta_propuesto, gammar_propuesto, t0_propuesto, phi_propuesto, sXXt, xbar) = proponer_parametros_luego_adaptacion(iteracion, beta, gammar, t0, phi, sXXt, xbar)

    log_prior_actual = ll.log_prior(beta_propuesto, gammar_propuesto, t0, phi_propuesto)

    if(not np.isinf(log_prior_actual)):
        estado_simulado = simular_con_cuarentena(beta_propuesto, gammar_propuesto, phi_propuesto, dias_epidemia, t0_propuesto)

        verosimilitud_propuesta = probabilidades_muertes(Ds, estado_simulado[3], theta) + log_prior_actual

        aceptado = np.exp(verosimilitud_propuesta - verosimilitud_actual) > np.random.uniform(0, 1)
        if aceptado:
            (beta, gammar, t0, phi, verosimilitud_actual, estado_inicial) = (beta_propuesto, gammar_propuesto, t0_propuesto, phi_propuesto, verosimilitud_propuesta, estado_simulado)
    else:
        aceptado = False

    return (beta, gammar, t0, phi, estado_inicial, sXXt, xbar, verosimilitud_actual, aceptado)

def ejecutar_mcmc(theta, beta, gammar, dias_epidemia, Ds, phi):

    t0 = T0_INICIAL
    aceptados = 0
    rechazados = 0

    betas_propuestos = np.zeros(ITERACIONES)
    gammas_propuestos = np.zeros(ITERACIONES)
    T0s_propuestos = np.zeros(ITERACIONES)
    phis_propuestos = np.zeros(ITERACIONES)
    propuestas = [None] * ITERACIONES

    sXXt = []
    xbar = []

    # S, I, R y Nu de cada iteración
    estados_SIR_propuestos = np.empty(ITERACIONES, dtype=np.ndarray)

    estado_inicial = simular_con_cuarentena(beta, gammar, phi, dias_epidemia, t0)

    verosimilitud_actual = probabilidades_muertes(Ds, estado_inicial[3], theta) + ll.log_prior(beta, gammar, t0, phi)

    inicio = datetime.now()
    print("Inicio iteraciones metropolis {0}".format(inicio))
    for iteracion in range(ITERACIONES):
        (beta, gammar, t0, phi, estado_propuesta, sXXt, xbar, verosimilitud_actual, aceptado) = proponer_nuevos_parametros(theta, beta, gammar, t0, phi, dias_epidemia, Ds, verosimilitud_actual, sXXt, xbar, estado_inicial, iteracion, propuestas)

        betas_propuestos[iteracion] = beta
        gammas_propuestos[iteracion] = gammar
        T0s_propuestos[iteracion] = t0
        phis_propuestos[iteracion] = phi
        propuestas[iteracion] = [beta, gammar, t0, phi]
        estados_SIR_propuestos[iteracion] = estado_propuesta

        if(aceptado):
            aceptados += 1
        else:
            rechazados += 1

        if iteracion % (ITERACIONES/10) == 0:
            check = datetime.now()
            print("{0} iteraciones, tiempo transcurrido {1}, aceptados {2}, rechazados {3}".format(iteracion, check - inicio, aceptados, rechazados))

    fin = datetime.now()
    print("Fin    iteraciones metropolis {0}".format(fin))
    print("Duración {0}".format(fin - inicio))
    print("Tasa aceptación {0}".format(aceptados/(aceptados+rechazados)))
    print("Verosimilitud final {0}".format(verosimilitud_actual))

    path_base_archivos = sys.path[0] + "/salida/mcmc/"
    if not os.path.exists(path_base_archivos):
        os.makedirs(path_base_archivos)

    np.save(path_base_archivos + "betas", betas_propuestos)
    np.save(path_base_archivos + "gammas", gammas_propuestos)
    np.save(path_base_archivos + "phis", phis_propuestos)
    np.save(path_base_archivos + "T0s", T0s_propuestos)
    np.save(path_base_archivos + "estados_SIR_propuestos", estados_SIR_propuestos)

    resultado = {}
    resultado["beta"] = limpiar_parametro(betas_propuestos)
    resultado["gamma"] = limpiar_parametro(gammas_propuestos)
    resultado["T0"] = limpiar_parametro(T0s_propuestos)
    resultado["phi"] = limpiar_parametro(phis_propuestos)
    resultado["poblacion"] = POBLACION
    resultado["inicio_cuarentena"] = T1

    print("beta {0}".format(resultado["beta"]))
    print("gammar {0}".format(resultado["gamma"]))
    print("t0 {0}".format(resultado["T0"]))
    print("phi {0}".format(resultado["phi"]))

    with open(path_base_archivos + "parametros.json", "w") as file:
        file.write(json.dumps(resultado))

    print("Se guardaron los valores de los parámetros en {0}".format(path_base_archivos))

def limpiar_parametro(parametro):
    parametro_sin_adaptacion = parametro[PERIODO_ADAPTACION:ITERACIONES]
    parametro_medio = np.mean(parametro_sin_adaptacion)
    return parametro_medio

def medias_muertes_diarias(theta, nus):
    dias_epidemia = nus.size
    # FIXME mi cálculo de theta no incluye las probabilidades para 0, 1 y 2
    # en el paper le asignan la misma probabilidad que a 3
    prob = np.append([0.0000100911228392383,0.0000100911228392383,0.0000100911228392383], list(theta.values())[0:len(theta.values()) -1])
    nus = np.append(np.zeros(prob.size + 1), nus)
    nus = nus * p
    y = sm.tsa.filters.convolution_filter(nus, prob, nsides=1)
    return y[-dias_epidemia:]

def probabilidades_muertes(Ds, Nus, theta):
    mus = medias_muertes_diarias(theta, Nus)
    pois = poisson.logpmf(Ds, mus)
    return np.sum(pois)

def cargar_muertes_reales_por_dia_del_estado(nombre_estado):
    """
    Buscamos para el estado cuantas muertes hubo por día desde el 1/1/2020
    Tomamos los datos de https://github.com/nytimes/covid-19-data al 18/04/2020
    En este archivo se computan las muertes acumuladas al día; para nuestra simulación precisamos las nuevas muertes por día
    """

    path = "entrada/us-states-20200418.csv"
    fecha_inicio = date(2020, 1, 1)
    dia_primer_registro = 0
    muertes_dia_anterior = 0
    nuevas_muertes_por_dia = np.array([])
    with open(path) as f:
        reader = csv.reader(f, quotechar='"')

        for row in f:
            split_list = row.split(",")
            fecha = split_list[0].strip()
            muertes_str = split_list[4].strip()

            if nombre_estado == split_list[1].strip().replace('"', ''):

                muertes_totales = int(muertes_str)
                muertes = muertes_totales - muertes_dia_anterior
                muertes_dia_anterior = muertes_totales

                nuevas_muertes_por_dia = np.append(nuevas_muertes_por_dia, muertes)
                if dia_primer_registro == 0:
                    fecha_dt = datetime.strptime(fecha, '%Y-%m-%d')
                    dia_primer_registro = abs(fecha_dt.date() - fecha_inicio).days

    nuevas_muertes_por_dia = np.insert(nuevas_muertes_por_dia, 0, np.zeros(dia_primer_registro), axis=0)
    return nuevas_muertes_por_dia

def main():

    theta = calcular_theta()

    muertes_reales_por_dia = cargar_muertes_reales_por_dia_del_estado("California")
    dias_epidemia = muertes_reales_por_dia.size

    ejecutar_mcmc(theta, BETA_INICIAL, GAMMA_INICIAL, dias_epidemia, muertes_reales_por_dia, PHI_INICIAL)

if __name__ == "__main__":
    main()