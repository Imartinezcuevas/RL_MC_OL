"""
Module: agentes/qlearning_agent.py
Description: Implementación de la clase base para agentes SARSA.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/26

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from agentes.tabular_agent import TabularAgent
from typing import Any, Dict
import numpy as np

class QLearningAgent(TabularAgent):
    """
    Implementación del algoritmo Q-Learning (off-policy).
    Este algoritmo aprende la política óptima independientemente 
    de las acciones que el agente esté tomando.
    """
    
    def _init_algorithm_params(self, **kwargs):
        """
        Inicializa parámetros específicos para Q-Learning
        
        Args:
            **kwargs: Parámetros adicionales
        """
        super()._init_algorithm_params(**kwargs)
        
        # Tasa de aprendizaje
        self.alpha = kwargs.get('alpha', 0.1)
        
        # Para exploración decreciente
        self.min_alpha = kwargs.get('min_alpha', 0.01)
    
    def update(self, state: Any, action: int, next_state: Any, reward: float, 
               done: bool, info: Dict = None) -> None:
        """
        Actualiza los valores Q usando la regla de actualización Q-Learning
        
        Args:
            state: Estado actual
            action: Acción tomada
            next_state: Estado siguiente
            reward: Recompensa recibida
            done: Indicador de fin de episodio
            info: Información adicional
        """
        # Obtiene el valor Q actual
        current_q = self.Q[state, action]
        
        # Calcula el valor objetivo para la actualización
        if done:
            # Si el episodio ha terminado, no hay estado siguiente
            target = reward
        else:
            # En Q-Learning, siempre se elige la mejor acción para el siguiente estado
            # independientemente de la política de comportamiento (por eso es off-policy)
            max_next_q = np.max(self.Q[next_state])
            target = reward + self.gamma * max_next_q
        
        # Actualiza el valor Q usando la tasa de aprendizaje (alpha)
        self.Q[state, action] += self.alpha * (target - current_q)
    
    def decay_learning_rate(self, episode, total_episodes):
        """
        Reduce la tasa de aprendizaje a medida que avanza el entrenamiento
        
        Args:
            episode: Episodio actual
            total_episodes: Número total de episodios
        """
        decay_factor = 1 - (episode / total_episodes)
        self.alpha = max(self.min_alpha, self.alpha * decay_factor)