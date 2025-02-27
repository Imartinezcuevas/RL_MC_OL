"""
Module: politicas/epsilon_greedy.py
Description: Implementación de la política epsilon-greedy.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/24

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from politicas.policy import Policy
from typing import Any
import gymnasium as gym
import numpy as np

class EpsilonGreedyPolicy(Policy):
    """
    Implementación de la política epsilon-greedy para selección de acciones.
    Esta política selecciona la acción con mayor valor Q con probabilidad 1-epsilon,
    y con probabilidad epsilon selecciona una acción aleatoria.
    """
    
    def __init__(self, action_space: gym.spaces, epsilon: float = 0.1):
        """
        Inicializa la política epsilon-greedy
        
        Args:
            action_space: Espacio de acciones del entorno
            epsilon: Probabilidad de seleccionar una acción aleatoria
        """
        super().__init__(action_space)
        self.epsilon = epsilon
    
    def get_action_probabilities(self, state: Any, action_values: np.ndarray):
        """
        Calcula las probabilidades de seleccionar cada acción según la política epsilon-greedy
        
        Args:
            state: Estado actual
            action_values: Matriz Q de valores de acción
            
        Returns:
            Array con probabilidades para cada acción
        """
        # Implementación de la política epsilon-soft
        pi_A = np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        best_action = np.argmax(action_values[state])
        pi_A[best_action] += (1.0 - self.epsilon)
        return pi_A
    
    def select_action(self, state: Any, action_values: np.ndarray) -> int:
        """
        Selecciona una acción basada en el estado actual y los valores Q
        siguiendo la política epsilon-greedy
        
        Args:
            state: Estado actual
            action_values: Matriz Q de valores de acción
            
        Returns:
            La acción seleccionada
        """
        pi_A = self.get_action_probabilities(state, action_values)
        return np.random.choice(np.arange(self.n_actions), p=pi_A)
    
    def decay_epsilon(self):
        """Aplica el decaimiento a epsilon"""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)