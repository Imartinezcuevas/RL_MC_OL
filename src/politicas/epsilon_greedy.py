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

class Policy(ABC):
    """
    Clase base abstracta para todas las políticas de selección de acciones.
    """

    def __init__(self, action_space: gym.spaces):
        """
        Inicializa la política
        
        Args:
            action_space: Espacio de acciones del entorno
        """
        self.action_space = action_space
        self.n_actions = action_space.n

    @abstractmethod
    def select_action(self, state: gym.spaces.Box):
        """
        Selecciona una acción basada en el estado actual y los valores de acción
        
        Args:
            state: Estado actual
            action_values: Valores de acción (Q) o cualquier estructura que use el agente
            
        Returns:
            La acción seleccionada
        """
        pass

    @abstractmethod
    def get_action_probabilities(self, state: Any, action_values: Any):
        """
        Obtiene las probabilidades de seleccionar cada acción
        
        Args:
            state: Estado actual
            action_values: Valores de acción (Q) o cualquier estructura que use el agente
            
        Returns:
            Array con probabilidades para cada acción
        """
        pass