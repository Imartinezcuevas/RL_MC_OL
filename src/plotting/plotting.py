"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/25

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_episode_lengths(episode_lengths):
    """
    Grafica la longitud de los episodios y muestra una curva de tendencia.

    Args:
        episode_lengths: Lista con la cantidad de pasos de cada episodio.
    """
    episodes = np.arange(len(episode_lengths))

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, episode_lengths, label="Longitud del episodio", alpha=0.5)
    
    # Agregar curva de tendencia (ajuste polinómico de grado 1)
    if len(episode_lengths) > 1:
        z = np.polyfit(episodes, episode_lengths, 1)
        p = np.poly1d(z)
        plt.plot(episodes, p(episodes), "r--", label="Tendencia")

    plt.xlabel("Episodio")
    plt.ylabel("Longitud del episodio")
    plt.title("Longitud de los episodios y curva de tendencia")
    plt.legend()
    plt.grid()
    plt.show()

def plot_reward_ratio(list_stats):
    # Creamos una lista de índices para el eje x
    indices = list(range(len(list_stats)))

    # Creamos el gráfico
    plt.figure(figsize=(6, 3))
    plt.plot(indices, list_stats)

    # Añadimos título y etiquetas
    plt.title('Proporción de recompensas')
    plt.xlabel('Episodio')
    plt.ylabel('Proporción')

    # Mostramos el gráfico
    plt.grid(True)
    plt.show()

def plot_training_comparation(algorithms_data, same=False):
    """
    Recibe un diccionario con la estructura:
        {
          "nombre_algoritmo": (lista_recompensas, lista_longitudes),
          "otro_algoritmo":   (lista_recompensas, lista_longitudes)
        }
    y genera:
      - Si same=False (por defecto), un gráfico con dos subplots:
          * Recompensa por episodio
          * Duración de los episodios
      - Si same=True, solo se muestra la gráfica de la duración de los episodios.
    """
    if not same:
        fig, (ax_reward, ax_length) = plt.subplots(2, 1, figsize=(10, 8))
        for alg_name, (rewards, lengths) in algorithms_data.items():
            episodes = np.arange(1, len(rewards) + 1)
            # Gráfica de recompensas
            ax_reward.plot(episodes, rewards, label=f"{alg_name}")
            # Gráfica de duración de episodios
            ax_length.plot(episodes, lengths, label=f"{alg_name}")
        
        ax_reward.set_title("Recompensa por Episodio")
        ax_reward.set_xlabel("Episodio")
        ax_reward.set_ylabel("Recompensa")
        ax_reward.legend()
        ax_reward.grid(True)
        
        ax_length.set_title("Duración de los Episodios")
        ax_length.set_xlabel("Episodio")
        ax_length.set_ylabel("Número de pasos")
        ax_length.legend()
        ax_length.grid(True)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        for alg_name, (rewards, lengths) in algorithms_data.items():
            episodes = np.arange(1, len(lengths) + 1)
            # Solo se grafica la duración
            ax.plot(episodes, lengths, label=f"{alg_name}")
        
        ax.set_title("Duración de los Episodios")
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Número de pasos")
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()