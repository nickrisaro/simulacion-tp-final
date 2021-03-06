import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Si está en True muestra el gráfico y frena la ejecución hasta que se cierra
MOSTRAR_GRAFICO = False

def calcular_theta():
    """
    θ representa la probabilidad de morir s días después del contagio
    Retorna un diccionario con los días desde el primer síntoma hasta al muerte
    y la probabilidad de que la persona muera en esa cantidad de días
    """
    # Sólo se contabilizan las personas infectadas que efectivamente fallecen

    # El tiempo de incubación es una poisson-gamma con α = 5.5 y β = 1.1
    # Lauer, S. A., Grantz, K. H., Bi, Q., Jones, F. K., Zheng, Q., Meredith, H. R., Azman, A. S., Reich, N. G., and Lessler, J. (2020). The incubation period of coronavirus disease 2019 (covid-19) from publicly reported confirmed cases: Estimation and application. Annals of Internal Medicine.
    # Se toman 100000 muestras
    incubacion = np.random.poisson(np.random.gamma(size=100000, shape=5.5, scale=1/1.1))

    # El tiempo desde los primeros síntomas hasta la muerte es una poisson-gamma con α = 27.75 y β = 1.5
    # Zhou, F., Yu, T., Du, R., Fan, G., Liu, Y., Liu, Z., Xiang, J., Wang, Y., Song, B.,Gu, X., et al. (2020). Clinical course and risk factors for mortality of adult in patients with covid-19 in wuhan, china: a retrospective cohort study. The Lancet.
    # Se toman 100000 muestras
    sintomas_a_muerte = np.random.poisson(np.random.gamma(size=100000, shape=27.75, scale=1/1.5))

    # Luego de la incubación se evidencian los síntomas y luego de ellos la muerte
    theta = incubacion + sintomas_a_muerte

    print("El promedio de θ es {0:.2f}".format(np.average(theta)))

    # Sólo consideramos el cuantil del 99%
    cuantil = np.quantile(theta, .99)
    theta = theta[(theta <= cuantil)]

    resultadosUnicos, frecuencias = np.unique(theta, return_counts=True)
    # Escalamos a 1 los resultados
    frecuencias = frecuencias/len(theta)

    return dict(zip(resultadosUnicos, frecuencias))

def main():

    print("TP final - gráfico de θ")
    theta = calcular_theta()

    plt.bar(theta.keys(), theta.values())
    plt.title("Probabilidad de morir a los n días de ser infectado")

    if not os.path.exists("{0}/salida".format(sys.path[0])):
        os.makedirs("{0}/salida".format(sys.path[0]))

    plt.savefig("{0}/salida/theta.png".format(sys.path[0]))
    print("El gráfico se guardó en {0}/salida/theta.png".format(sys.path[0]))
    if MOSTRAR_GRAFICO:
        plt.show()


if __name__ == "__main__":
    main()