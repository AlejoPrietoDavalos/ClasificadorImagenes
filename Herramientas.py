import numpy as np
import matplotlib.pyplot as plt
from jupyterthemes import jtplot

class UtilsPlot():
    """ En esta clase voy a guardar todas las utilidades para hacer plots de los datos."""
    def __init__(self):
        pass
    
    def PlotInGrid(self, n_filas: int, n_cols: int, list_images: list) -> None:
        """ Recibe como parámetro una lista con todas las imágenes convertidas en matriz con la intensidad de los píxeles entre 0 y 255.
            Genera un gráfico con el número de filas y columnas especificadas.
            
            Args:
                n_filas (int): Número de filas del plot.
                n_cols (int): Número de columnas del plot."""
        def PlotByPixelMatrix(image):
            """ Recibe una matriz de pixeles con la intensidad entre 0 y 255. Y retorna la gráfica de la imagen."""
            jtplot.style(grid=False)
            plt.imshow(image.squeeze(), cmap=plt.get_cmap('gray'))
        assert len(list_images) <= n_filas*n_cols, "El número de imagenes que se quieren graficar, sobrepasa la matriz de gráficos planteada. Agrandar número de filas y columnas."      # Condiciones para que esto funcione.
        assert type(n_filas) == int and type(n_cols) == int, "El número de filas y columnas es un número entero."

        # Recorremos la lista de imágenes y hacemos un plot en grilla de cada uno.
        for i, image in enumerate(list_images):
            plt.subplot(n_filas, n_cols, i+1)
            PlotByPixelMatrix(image)
            plt.xticks([])
            plt.yticks([])
        plt.show()