import matplotlib.pyplot as plt
import numpy as np
import sys

MOSTRAR_GRAFICO = True
DIAS = 300
TIEMPO = np.linspace(0, DIAS, DIAS)

def graficarSIR(S, I, R):
    """
    Realiza un gráfico con la evolución de las 3 variables del modelo respecto del tiempo
    """
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(TIEMPO, S, 'b', alpha=0.5, lw=2, label='Susceptibles')
    ax.plot(TIEMPO, I, 'r', alpha=0.5, lw=2, label='Infectados')
    ax.plot(TIEMPO, R, 'g', alpha=0.5, lw=2, label='Recuperados')

    ax.set_xlabel('Tiempo / días')
    ax.set_ylabel('Población')
    ax.set_ylim(0,1100000)
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

    plt.savefig("{0}/salida/SIR.png".format(sys.path[0]))
    print("El gráfico se guardó en {0}/salida/SIR.png".format(sys.path[0]))
    if MOSTRAR_GRAFICO:
        plt.show()

def graficarD(D):
    """
    Realiza un gráfico con la evolución de la variable D respecto del tiempo
    """
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(TIEMPO, D[0:DIAS], 'b', alpha=0.5, lw=2, label='D')

    ax.set_xlabel('Tiempo / días')
    ax.set_ylabel('Población')
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

    plt.savefig("{0}/salida/D.png".format(sys.path[0]))
    print("El gráfico se guardó en {0}/salida/D.png".format(sys.path[0]))
    if MOSTRAR_GRAFICO:
        plt.show()

def graficarNu(Nu):
    """
    Realiza un gráfico con la evolución de la variable Nu respecto del tiempo
    """
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(TIEMPO, Nu, 'b', alpha=0.5, lw=2, label='Nu')

    ax.set_xlabel('Tiempo / días')
    ax.set_ylabel('Población')
    # ax.set_ylim(0,1000000)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

    plt.savefig("{0}/salida/Nu.png".format(sys.path[0]))
    print("El gráfico se guardó en {0}/salida/Nu.png".format(sys.path[0]))
    if MOSTRAR_GRAFICO:
        plt.show()

def simularSIR(poblacion, beta, gammar, infectados_iniciales, dias, maximo_tiempo_muerte, probabilidades_tiempo_muerte, tasa_infeccion):

    S = poblacion - infectados_iniciales
    I = infectados_iniciales
    R = 0
    Nu = I

    Nus = np.full(dias, 0.0)
    Ss = np.full(dias, 0.0)
    Is = np.full(dias, 0.0)
    Rs = np.full(dias, 0.0)
    Ds = np.full(dias + maximo_tiempo_muerte, 0.0)

    Nus[0] = Nu
    Ss[0] = S
    Is[0] = I
    Rs[0] = R

    for t in range(1, dias):
        nu = (Ss[t-1]*Is[t-1]*beta)/poblacion

        muertes_futuras = np.random.poisson(tasa_infeccion * nu)

        tiempo_muerte = np.random.choice(list(probabilidades_tiempo_muerte.keys()), size=muertes_futuras, replace=True, p=list(probabilidades_tiempo_muerte.values()))

        for dia in tiempo_muerte:
            if t + dia < Ds.size:
                Ds[t + dia] = Ds[t + dia] + 1

        Rs[t] = Rs[t-1] + (gammar*(Is[t-1] - Ds[t -1])) + Ds[t-1]
        Ss[t] = Ss[t-1] - nu
        Is[t] = Is[t-1] + nu - gammar * (Is[t - 1] - Ds[t - 1]) - Ds[t - 1]
        Nus[t] = nu

    graficarSIR(Ss, Is, Rs)
    graficarD(Ds)
    graficarNu(Nus)

def main():
    # TODO Reutilizar la función definida en Theta.py
    incubacion = np.random.poisson(np.random.gamma(size=100000, shape=5.5, scale=1/1.1))
    print(incubacion)

    # El tiempo desde los primeros síntomas hasta la muerte es una poisson-gamma con α = 27.75 y β = 1.5
    # Zhou, F., Yu, T., Du, R., Fan, G., Liu, Y., Liu, Z., Xiang, J., Wang, Y., Song, B.,Gu, X., et al. (2020). Clinical course and risk factors for mortality of adult in patients with covid-19 in wuhan, china: a retrospective cohort study. The Lancet.
    # Se toman 100000 muestras
    sintomas_a_muerte = np.random.poisson(np.random.gamma(size=100000, shape=27.75, scale=1/1.5))
    print(sintomas_a_muerte)

    # Luego de la incubación se evidencian los síntomas y luego de ellos la muerte
    theta = incubacion + sintomas_a_muerte
    print(theta)

    print("El promedio de θ es {0:.2f}".format(np.average(theta)))


    # Para graficar sólo consideramos el cuantil del 99%
    cuantil = np.quantile(theta, .99)
    theta = theta[(theta <= cuantil)]

    # Obtenemos el máximo de días hasta la muerte (dentro del cuantil 99%)
    maximo_tiempo_muerte = np.max(theta) - 1

    resultadosUnicos, frecuencias = np.unique(theta, return_counts=True)
    # Escalamos a 1 los resultados
    frecuencias = frecuencias/len(theta)

    tabla_probabilidades_dias_muerte = dict(zip(resultadosUnicos, frecuencias))

    simularSIR(1000000, 0.2, 0.1, 10, DIAS, maximo_tiempo_muerte, tabla_probabilidades_dias_muerte, 0.01)

if __name__ == "__main__":
    main()