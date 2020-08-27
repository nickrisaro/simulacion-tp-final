import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sys

# Parámetros de la enfermedad
BETA = 0.3833333
GAMMA = 0.1666667

PHI = 0.4
T0 = 30 # Inicio epidemia
T1 = 73 # Inicio cuarentena
N = 39937489

DIAS = 108
PORCENTAJE_CONTAGIO_INICIAL = 0.03
PORCENTAJE_FIN_EPIDEMIA = 0.01
PORCENTAJE_CAMAS_POBLACION = 0.3
PORCENTAJE_POBLACION_EN_CUARENTENA = 0.35

MOSTRAR_GRAFICO = False
TIEMPO = np.arange(T0, T1, 1)
TIEMPO_GRAFICO = np.linspace(0, DIAS, DIAS)

def SIR_pre_cuarentena(t, y, beta, gamma):
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

def SIR_post_cuarentena(t, y):
    """
    Esta función será invocada por el método de resolución de ecuaciones diferenciales
    Para cada instante de tiempo t calcula los nuevos valores de S, I y R
    Este modelo asume que comenzó una cuarentena y que la efectividad de esta es PHI
    """
    S, I, R = y

    dsdt = -BETA*PHI*S*I
    didt = BETA*PHI*S*I - GAMMA*I
    drdt = GAMMA*I
    return dsdt, didt, drdt

def simular_con_cuarentena():
    """
    Realiza una simulación con cuarentena del modelo SIR
    Toda la población es susceptible de contagiarse
    Retorna todos los valores de S, I y R para cada instante de tiempo
    """
    S0 = 1
    I0 = 1/N
    R0 = 1.953490*10**-17

    y0 = [S0, I0, R0]

    TIEMPO = np.arange(T0, T1, 1)

    ret = solve_ivp(SIR_pre_cuarentena, [T0, T1], y0, t_eval=TIEMPO, method='LSODA', args=(BETA*PHI, GAMMA))
    S, I, R = ret.y

    S = np.insert(S, 0, np.repeat(S0, T0+1), axis=0)
    I = np.insert(I, 0, np.repeat(I0, T0+1), axis=0)
    R = np.insert(R, 0, np.repeat(R0, T0+1), axis=0)

    Nus = S * N
    Nus = np.subtract(Nus[0:Nus.size -1], Nus[1:Nus.size])

    S0 = S[S.size - 1]
    I0 = I[I.size - 1]
    R0 = R[R.size - 1]
    y0 = [S0, I0, R0]

    TIEMPO = np.arange(T1, DIAS+1, 1)

    ret = solve_ivp(SIR_pre_cuarentena, [T1, DIAS+1], y0, t_eval=TIEMPO, method='LSODA', args=(BETA, GAMMA))
    S1, I1, R1 = ret.y

    Nus1 = S1 * N
    Nus1 = np.subtract(Nus1[0:Nus1.size -1], Nus1[1:Nus1.size])
    Nus = np.insert(Nus1, 0, Nus, axis=0)

    S = np.append(S, S1[1:S1.size - 1])
    I = np.append(I, I1[1:I1.size - 1])
    R = np.append(R, R1[1:R1.size - 1])
    print(S*N)
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
    plt.savefig("{0}/SIR-{1} - phi {2} - T1 {3}.png".format(sys.path[0], modelo, PHI, T1))
    print("El gráfico se guardó en {0}/SIR-{1} - phi {2} - T1 {3}.png".format(sys.path[0], modelo, PHI, T1))
    if MOSTRAR_GRAFICO:
        plt.show()

def main():

    print("TP final - grafico SIR con cuarentena")

    graficar(simular_con_cuarentena(), "Con cuarentena")


if __name__ == "__main__":
    main()