import matplotlib.pyplot as plt
import numpy as np
import sys

def main():

    print("TP final - gráfico de θ")
    # θ representa la probabilidad de morir s días después del contagio

    # El tiempo de incubación es una poisson-gamma con α = 5.5 y β = 1.1
    # Se toman 100000 muestras
    incubacion = np.random.poisson(np.random.gamma(size=100000, shape=5.5, scale=1/1.1))
    print(incubacion)

    # El tiempo desde los primeros síntomas hasta la muerte es una poisson-gamma con α = 27.75 y β = 1.5
    # Se toman 100000 muestras
    sintomas_a_muerte = np.random.poisson(np.random.gamma(size=100000, shape=27.75, scale=1/1.5))
    print(sintomas_a_muerte)

    theta = incubacion + sintomas_a_muerte
    print(theta)

    print("El promedio de θ es {0:.2f}".format(np.average(theta)))

    cuantil = np.quantile(theta, .99)
    theta = theta[(theta <= cuantil)]

    resultadosUnicos, frecuencias = np.unique(theta, return_counts=True)
    frecuencias = frecuencias/len(theta)

    plt.bar(resultadosUnicos, frecuencias)
    plt.title("Probabilidad de morir a los n días de ser infectado")
    plt.savefig("{0}/salida/theta.png".format(sys.path[0]))
    print("El gráfico se guardó en {0}/salida/theta.png".format(sys.path[0]))
    plt.show()


if __name__ == "__main__":
    main()