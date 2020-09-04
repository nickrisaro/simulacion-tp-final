# Análisis del paper "Estimating the number of SARS-CoV-2 infections and the impact of social distancing in the United States"

En este repositorio se encuentra el código con el que se intentó replicar los resultados de este paper https://arxiv.org/abs/2004.02605

El análisis de dicho estudio es el trabajo práctico final de la materia simulación de FIUBA.

Se recomienda ejecutarlo con python 3.6.9 o superior. Las dependencias necesarias pueden instalarse con

    pip install -r requirements.txt

A continuación se detalla el propósito de cada uno de los scripts de este repositorio.

* `MCMC.py` es el script principal, al ejecutarlo se corren las simulaciones propuestas en el paper y se guardan los resultados en la carpeta `salida/mcmc`
* `Comparar_casos_dia.py` toma los resultados de la ejecución anterior y realiza una gráfica en la que se comparan los casos reportados con los simulados.
* `SIR_Cuarentena.py` parte de los parámetros obtenidos en la simulación y realiza una simulación de la enfermedad utilizando el modelo SIR propuesto por los y las autoras del paper.
* `Theta.py` genera las probabilidades de muerte luego del contagio. Puede ejecutarse sólo y genera un histograma o desde `MCMC.py` como dato de entrada para la simulación.
* `SIR.py` realiza una simulación básica del modelo SIR en la que se consideran los nuevos casos por día y se hace una predicción de las muertes que ocurrirán por día utilizando los valores de Theta.