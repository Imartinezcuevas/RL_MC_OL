"""
Module: agentes/agent.py
Description: Implementación de la clase base para Agentes.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/24

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from abc import ABC, abstractmethod
import gymnasium as gym
from politicas import Policy
from typing import Any


class Agent(ABC):
    """
    Clase base abstracta para todos los agentes.
    """

    def __init__(self, env: gym.Env, policy: Policy = None, gamma: float = 0.99, **kwargs):
        """
        Inicializa el agente
        
        Args:
            env: Entorno de gymnasium
            policy: Política para seleccionar acciones
            gamma: Factor de descuento para recompensas futuras
            **kwargs: Argumentos adicionales específicos de cada algoritmo
        """
        self.env = env
        self.gamma = gamma
        
        # Configura el espacio de acciones y observaciones
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
        # Política
        self.policy = policy
        
        # Estadísticas de aprendizaje
        self.episode_rewards = []
        self.total_steps = 0
        self.episode_count = 0
        
        # Inicialización específica según el tipo de algoritmo
        self._init_algorithm_params(**kwargs)

    @abstractmethod
    def _init_algorithm_params(self, **kwargs):
        """
        Inicializa parámetros específicos del algoritmo
        
        Args:
            **kwargs: Parámetros específicos
        """
        pass

    def get_action(self, state: Any) -> int:
        """
        Obtiene una acción basada en la política actual
        
        Args:
            state: Estado actual
            
        Returns:
            Acción seleccionada
        """
        return self.policy.select_action(state, self.get_action_values())
    
    @abstractmethod
    def get_action_values(self) -> Any:
        """
        Devuelve la estructura de valores de acción (Q) que usa el agente
        
        Returns:
            Valores Q o estructura equivalente
        """
        pass

    @abstractmethod
    def get_action_values(self) -> Any:
        """
        Devuelve la estructura de valores de acción (Q) que usa el agente
        
        Returns:
            Valores Q o estructura equivalente
        """
        pass
