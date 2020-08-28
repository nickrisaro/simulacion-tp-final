from datetime import date
from datetime import datetime
from scipy.integrate import solve_ivp

import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

ITERACIONES = 50000
PERIODO_ADAPTACION = ITERACIONES/5
VARIANZA_INICIAL = [0.002**2, 0.002**2, 0.75**2, 0.002**2]
FACTOR_ESCALA = (2.38 / 4) * 0.5
LIMITES_T0 = [1, 60] # TODO Ver si arranco en 0
LIMITES_EFECTIVIDAD_CUARENTENA = [0.01, 0.99]

T0 = 30 # Inicio epidemia
T1 = 73 # Inicio cuarentena
POBLACION = 39937489

R0_INICIAL = 2/3
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
    S0 = 1
    I0 = 1/POBLACION
    R0 = 1.953490*10**-17 # TODO usar cero? Comparar con paper
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

def ejecutar_mcmc(theta, beta, gammar, dias_epidemia, Ds, xi, zeta, primer_contagio, inicio_cuarentena, efectividad_cuarentena):

    aceptados = 0
    rechazados = 0

    betas_propuestos = np.zeros(ITERACIONES)
    gammas_propuestos = np.zeros(ITERACIONES)
    T0s_propuestos = np.zeros(ITERACIONES)
    phis_propuestos = np.zeros(ITERACIONES)
    propuestas = np.empty(ITERACIONES, dtype=np.ndarray) # TODO parece redundante con los valores propuestos desagregados

    nus_propuestos = np.empty(ITERACIONES, dtype=np.ndarray)
    estados_SIR_propuestos = np.empty(ITERACIONES, dtype=np.ndarray)

    estado_inicial = simular_con_cuarentena(beta, gammar, efectividad_cuarentena, dias_epidemia, T0)
    print(estado_inicial)

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

    muertes_reales_por_dia = cargar_muertes_reales_por_dia_del_estado("California")
    dias_epidemia = muertes_reales_por_dia.size
    ejecutar_mcmc(None, BETA_INICIAL, GAMMA_INICIAL, dias_epidemia, muertes_reales_por_dia, None, None, None, None, PHI_INICIAL)
    pass

if __name__ == "__main__":
    main()