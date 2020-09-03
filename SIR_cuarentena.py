import json
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sys

# Parámetros de la enfermedad
BETA = 0.3833333
GAMMA = 0.1666667
# Efectividad de la cuarentena
PHI = 0.01
# Inicio epidemia
T0 = 10

# Parámetros del estado analizado
# Inicio cuarentena
T1 = 73
# Población
N = 39937489

# Cantidad de días a simular
DIAS = 300

TIEMPO_GRAFICO = np.linspace(0, DIAS, DIAS)

# Si está en True muestra el gráfico y frena la ejecución hasta que se cierra
MOSTRAR_GRAFICO = False

def SIR(t, y, beta, gamma):
    """
    Esta función será invocada por el método de resolución de ecuaciones diferenciales
    Para cada instante de tiempo t calcula los nuevos valores de S, I y R
    Este modelo asume que no hay cuarentena
    """
    S, I, R = y

    dsdt = -beta*S*I
    didt = beta*S*I - gamma*I
    drdt = gamma*I
    return dsdt, didt, drdt

def simular_con_cuarentena(beta, gamma, t0, phi, t1, n):
    """
    Realiza una simulación con cuarentena del modelo SIR
    Toda la población es susceptible de contagiarse
    Retorna todos los valores de S, I y R para cada instante de tiempo
    """
    I0 = 1/n
    R0 = 0
    S0 = 1 - I0 - R0

    y0 = [S0, I0, R0]

    TIEMPO = np.arange(t0, t1, 1)

    ret = solve_ivp(SIR, [t0, t1], y0, t_eval=TIEMPO, method='LSODA', args=(beta, gamma))
    S, I, R = ret.y

    S = np.insert(S, 0, np.repeat(S0, t0+1), axis=0)
    I = np.insert(I, 0, np.repeat(I0, t0+1), axis=0)
    R = np.insert(R, 0, np.repeat(R0, t0+1), axis=0)

    Nus = S * n
    Nus = np.subtract(Nus[0:Nus.size -1], Nus[1:Nus.size])

    S0 = S[S.size - 1]
    I0 = I[I.size - 1]
    R0 = R[R.size - 1]
    y0 = [S0, I0, R0]

    TIEMPO = np.arange(t1, DIAS+1, 1)

    ret = solve_ivp(SIR, [t1, DIAS+1], y0, t_eval=TIEMPO, method='LSODA', args=(beta*phi, gamma))
    S1, I1, R1 = ret.y

    Nus1 = S1 * n
    Nus1 = np.subtract(Nus1[0:Nus1.size -1], Nus1[1:Nus1.size])
    Nus = np.insert(Nus1, 0, Nus, axis=0)

    S = np.append(S, S1[1:S1.size - 1])
    I = np.append(I, I1[1:I1.size - 1])
    R = np.append(R, R1[1:R1.size - 1])

    print(S*n)
    return [S, I, R, Nus]

def simular_sin_cuarentena(beta, gamma, t0, phi, t1, n):
    """
    Realiza una simulación con cuarentena del modelo SIR
    Toda la población es susceptible de contagiarse
    Retorna todos los valores de S, I y R para cada instante de tiempo
    """
    I0 = 1/n
    R0 = 0
    S0 = 1 - I0 - R0

    y0 = [S0, I0, R0]

    TIEMPO = np.arange(t0, DIAS, 1)

    ret = solve_ivp(SIR, [t0, DIAS], y0, t_eval=TIEMPO, method='LSODA', args=(beta, gamma))
    S, I, R = ret.y

    S = np.insert(S, 0, np.repeat(S0, t0+1), axis=0)
    I = np.insert(I, 0, np.repeat(I0, t0+1), axis=0)
    R = np.insert(R, 0, np.repeat(R0, t0+1), axis=0)

    Nus = S * n
    print(Nus)
    Nus = np.subtract(Nus[0:Nus.size -1], Nus[1:Nus.size])

    print(Nus)
    return [S, I, R, Nus]

def graficar(SIR, modelo):
    """
    Realiza un gráfico con la evolución de las 3 variables del modelo respecto del tiempo
    """
    S, I, R, Nus = SIR
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(TIEMPO_GRAFICO, S*100, 'b', alpha=0.5, lw=2, label='Susceptibles')
    ax.plot(TIEMPO_GRAFICO, I*100, 'r', alpha=0.5, lw=2, label='Infectados')
    ax.plot(TIEMPO_GRAFICO, R*100, 'g', alpha=0.5, lw=2, label='Recuperados')

    ax.set_xlabel('Tiempo / días')
    ax.set_ylabel('% población')
    ax.set_ylim(0,100)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.title(modelo)
    plt.savefig("{0}/salida/SIR-{1} - phi {2} - T0 {3} - beta {4} - gamma {5}.png".format(sys.path[0], modelo, PHI, T0, BETA, GAMMA))
    print("El gráfico se guardó en {0}/salida/SIR-{1} - phi {2} - T0 {3} - beta {4} - gamma {5}.png".format(sys.path[0], modelo, PHI, T0, BETA, GAMMA))
    if MOSTRAR_GRAFICO:
        plt.show()

def main():

    print("TP final - grafico SIR con cuarentena")

    if os.path.isfile(sys.path[0] + "/salida/mcmc/parametros.json"):
        parametros = {}
        with open(sys.path[0] + "/salida/mcmc/parametros.json", "r") as file:
            parametros = json.loads(file.read())

        beta = parametros["beta"]
        gamma = parametros["gamma"]
        t0 = int(parametros["T0"])
        phi = parametros["phi"]
        t1 = parametros["inicio_cuarentena"]
        n = parametros["poblacion"]
    else:
        beta = BETA
        gamma = GAMMA
        t0 = T0
        phi = PHI
        t1 = T1
        n = N

    graficar(simular_sin_cuarentena(beta, gamma, t0, phi, t1, n), "Sin cuarentena")


if __name__ == "__main__":
    main()