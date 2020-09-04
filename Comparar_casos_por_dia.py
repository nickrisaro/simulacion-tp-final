import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from datetime import datetime
from datetime import date
from scipy import stats

MOSTRAR_GRAFICO = False

def cargar_casos_estado(nombre_estado):
    path = "entrada/us-states-20200418.csv"
    fecha_inicio = date(2020, 1, 1)
    dia_primer_registro = 0
    casos_dia_anterior = 0
    nuevos_casos_por_dia = np.array([])
    with open(path) as f:
        reader = csv.reader(f, quotechar='"')

        for row in f:
            split_list = row.split(",")
            fecha = split_list[0].strip()
            casoss_str = split_list[3].strip()

            if nombre_estado == split_list[1].strip().replace('"', ''):

                casoss_totales = int(casoss_str)
                casos = casoss_totales - casos_dia_anterior
                casos_dia_anterior = casoss_totales

                nuevos_casos_por_dia = np.append(nuevos_casos_por_dia, casos)
                if dia_primer_registro == 0:
                    fecha_dt = datetime.strptime(fecha, '%Y-%m-%d')
                    dia_primer_registro = abs(fecha_dt.date() - fecha_inicio).days

    nuevos_casos_por_dia = np.insert(nuevos_casos_por_dia, 0, np.zeros(dia_primer_registro), axis=0)
    return nuevos_casos_por_dia

def cargar_casos_simulacion():
    estados = np.load(sys.path[0] + "/salida/mcmc/estados_SIR_propuestos.npy", allow_pickle=True)
    casos = np.empty(108, dtype=np.ndarray)

    for j in range(casos.size):
        casos[j] = np.empty(39999, dtype=np.ndarray)

    for i in range(estados.size):
        if i > 10000:
            casos_dia_simulacion= estados[i][3]
            for j in range(casos_dia_simulacion.size):
                casos[j][i - 10000 - 1] = casos_dia_simulacion[j]

    media_casos  = np.empty(108, dtype=np.ndarray)
    for i in range(casos.size):
        cuantil = np.quantile(casos[i], .90, axis=0)
        media_casos[i] = casos[i][(casos[i] <= cuantil)].mean()

    return media_casos

def main():
    print("Lectura de información de archivos para comparar con los resultados de MCMC")

    if not os.path.isfile(sys.path[0] + "/salida/mcmc/estados_SIR_propuestos.npy"):
        print("No hay datos de la simulación registrados")
        print("Ejecute python MCMC.py y vuelva a ejecutar este programa")
        return

    casos_dia_reales_CA = cargar_casos_estado("California")
    casos_dia_simulacion_CA = cargar_casos_simulacion()

    plt.title("Comparación de casos reportados con los simulados")
    plt.plot(casos_dia_reales_CA, "b", alpha=0.5, lw=2, label='Reportados')

    plt.plot(casos_dia_simulacion_CA, "r", alpha=0.5, lw=2, label='Simulados')

    plt.legend(loc='best');

    plt.savefig("{0}/salida/comparacion_casos_reportados_simulados.png".format(sys.path[0]))
    print("Se guardó el gráfico en {0}/salida/comparacion_casos_reportados_simulados.png".format(sys.path[0]))
    if MOSTRAR_GRAFICO:
        plt.show()

if __name__ == "__main__":
    main()